//! H.264 NAL-aware streaming with hybrid FEC+ARQ.
//!
//! This example properly parses H.264 Annex B NAL units and sends each
//! NAL (or access unit) as a discrete FEC-protected frame. This ensures
//! that FEC frame boundaries align with video frame boundaries.
//!
//! # Features
//!
//! - **NAL-aware**: Parses Annex B start codes (00 00 01 / 00 00 00 01)
//! - **Adaptive FEC**: Parity count adjusts based on packet loss
//! - **ARQ Fallback**: Retransmits if FEC cannot recover
//! - **Access Unit grouping**: Optionally groups NALs into access units
//!
//! # Usage
//!
//! ## Receiver (start first):
//! ```bash
//! cargo run --release --example h264_streaming -- recv | ffplay -f h264 -probesize 32 -
//! ```
//!
//! ## Sender:
//! ```bash
//! ffmpeg -re -i input.mp4 -c:v libx264 -tune zerolatency -f h264 - 2>/dev/null | \
//!     cargo run --release --example h264_streaming -- send
//! ```

use cm256::reliable::{RecvResult, ReliableDecoder, ReliableEncoder, ReliableParams};
use std::io::{self, Read, Write};
use std::net::{SocketAddr, UdpSocket};
use std::time::{Duration, Instant};

// ============================================================================
// H.264 Annex B Parser
// ============================================================================

/// H.264 NAL unit types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum NalType {
    Unspecified = 0,
    NonIdrSlice = 1,
    SlicePartA = 2,
    SlicePartB = 3,
    SlicePartC = 4,
    IdrSlice = 5,
    Sei = 6,
    Sps = 7,
    Pps = 8,
    AccessUnitDelimiter = 9,
    EndOfSeq = 10,
    EndOfStream = 11,
    Filler = 12,
    SpsExt = 13,
    Prefix = 14,
    SubsetSps = 15,
    Reserved16 = 16,
    Reserved17 = 17,
    Reserved18 = 18,
    AuxSlice = 19,
    SliceExt = 20,
    Reserved21 = 21,
    Reserved22 = 22,
    Reserved23 = 23,
    Unknown = 255,
}

impl From<u8> for NalType {
    fn from(b: u8) -> Self {
        match b & 0x1F {
            0 => NalType::Unspecified,
            1 => NalType::NonIdrSlice,
            2 => NalType::SlicePartA,
            3 => NalType::SlicePartB,
            4 => NalType::SlicePartC,
            5 => NalType::IdrSlice,
            6 => NalType::Sei,
            7 => NalType::Sps,
            8 => NalType::Pps,
            9 => NalType::AccessUnitDelimiter,
            10 => NalType::EndOfSeq,
            11 => NalType::EndOfStream,
            12 => NalType::Filler,
            13 => NalType::SpsExt,
            14 => NalType::Prefix,
            15 => NalType::SubsetSps,
            16 => NalType::Reserved16,
            17 => NalType::Reserved17,
            18 => NalType::Reserved18,
            19 => NalType::AuxSlice,
            20 => NalType::SliceExt,
            21 => NalType::Reserved21,
            22 => NalType::Reserved22,
            23 => NalType::Reserved23,
            _ => NalType::Unknown,
        }
    }
}

impl NalType {
    /// Returns true if this NAL type starts a new access unit
    fn starts_access_unit(self) -> bool {
        matches!(
            self,
            NalType::AccessUnitDelimiter
                | NalType::Sps
                | NalType::Pps
                | NalType::Sei
                | NalType::IdrSlice
        )
    }

    /// Returns true if this is a VCL NAL (contains video data)
    fn is_vcl(self) -> bool {
        matches!(
            self,
            NalType::NonIdrSlice
                | NalType::SlicePartA
                | NalType::SlicePartB
                | NalType::SlicePartC
                | NalType::IdrSlice
                | NalType::AuxSlice
                | NalType::SliceExt
        )
    }
}

/// Annex B NAL unit parser
struct AnnexBParser {
    buffer: Vec<u8>,
    pos: usize,
}

impl AnnexBParser {
    fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(1024 * 1024), // 1MB buffer
            pos: 0,
        }
    }

    /// Add data to the parser buffer
    fn push(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
    }

    /// Find next start code position (00 00 01 or 00 00 00 01)
    fn find_start_code(&self, from: usize) -> Option<(usize, usize)> {
        let buf = &self.buffer[from..];
        let len = buf.len();

        if len < 3 {
            return None;
        }

        for i in 0..len - 2 {
            if buf[i] == 0 && buf[i + 1] == 0 {
                if buf[i + 2] == 1 {
                    // 00 00 01
                    return Some((from + i, 3));
                } else if buf[i + 2] == 0 && i + 3 < len && buf[i + 3] == 1 {
                    // 00 00 00 01
                    return Some((from + i, 4));
                }
            }
        }

        None
    }

    /// Try to extract the next complete NAL unit (including start code)
    fn next_nal(&mut self) -> Option<Vec<u8>> {
        // Find first start code
        let (start_pos, start_len) = self.find_start_code(self.pos)?;

        // Find next start code (or end of buffer)
        let nal_start = start_pos;
        let nal_data_start = start_pos + start_len;

        if let Some((next_pos, _)) = self.find_start_code(nal_data_start) {
            // Complete NAL found
            let nal = self.buffer[nal_start..next_pos].to_vec();
            self.pos = next_pos;

            // Compact buffer periodically
            if self.pos > 1024 * 1024 {
                self.buffer.drain(..self.pos);
                self.pos = 0;
            }

            Some(nal)
        } else {
            // Need more data - keep at least 4 bytes for potential split start code
            None
        }
    }

    /// Flush remaining data (for EOF)
    fn flush(&mut self) -> Option<Vec<u8>> {
        if let Some((start_pos, _)) = self.find_start_code(self.pos) {
            if start_pos < self.buffer.len() - 4 {
                let nal = self.buffer[start_pos..].to_vec();
                self.pos = self.buffer.len();
                return Some(nal);
            }
        }
        None
    }
}

/// Groups NALs into access units (complete video frames)
struct AccessUnitGrouper {
    current_au: Vec<u8>,
    has_vcl: bool,
}

impl AccessUnitGrouper {
    fn new() -> Self {
        Self {
            current_au: Vec::with_capacity(256 * 1024),
            has_vcl: false,
        }
    }

    /// Add a NAL unit, returns complete access unit if ready
    fn push(&mut self, nal: &[u8]) -> Option<Vec<u8>> {
        if nal.len() < 4 {
            return None;
        }

        // Get NAL type (first byte after start code)
        let nal_header_pos = if nal[2] == 1 { 3 } else { 4 };
        if nal_header_pos >= nal.len() {
            return None;
        }

        let nal_type = NalType::from(nal[nal_header_pos]);

        // Check if this NAL starts a new access unit
        let starts_new_au = if self.has_vcl {
            // After VCL, a new AU starts with AUD, SPS, PPS, SEI, or another VCL
            nal_type.starts_access_unit() || nal_type.is_vcl()
        } else {
            false
        };

        if starts_new_au && !self.current_au.is_empty() {
            // Return completed AU
            let au = std::mem::take(&mut self.current_au);
            self.has_vcl = false;

            // Start new AU with current NAL
            self.current_au.extend_from_slice(nal);
            if nal_type.is_vcl() {
                self.has_vcl = true;
            }

            return Some(au);
        }

        // Add to current AU
        self.current_au.extend_from_slice(nal);
        if nal_type.is_vcl() {
            self.has_vcl = true;
        }

        None
    }

    /// Flush remaining data
    fn flush(&mut self) -> Option<Vec<u8>> {
        if !self.current_au.is_empty() {
            Some(std::mem::take(&mut self.current_au))
        } else {
            None
        }
    }
}

// ============================================================================
// Configuration
// ============================================================================

const DEFAULT_SYMBOL_SIZE: usize = 1200;
const DEFAULT_REDUNDANCY: f64 = 1.3;
const MAX_NAL_SIZE: usize = 256 * 1024; // 256KB max NAL
const READ_CHUNK_SIZE: usize = 32 * 1024; // 32KB read chunks

// ============================================================================
// CLI
// ============================================================================

use std::env;

struct SendArgs {
    listen: SocketAddr,
    remote: SocketAddr,
    symbol_size: usize,
    redundancy: f64,
    enable_arq: bool,
    group_au: bool, // Group NALs into access units
    verbose: bool,
}

struct RecvArgs {
    listen: SocketAddr,
    enable_arq: bool,
    verbose: bool,
}

enum Command {
    Send(SendArgs),
    Recv(RecvArgs),
}

fn parse_args() -> Result<Command, String> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        return Err("Usage: h264_streaming <send|recv> [options]".to_string());
    }

    let mut verbose = false;
    for arg in &args {
        if arg == "-v" || arg == "--verbose" {
            verbose = true;
        }
    }

    match args[1].as_str() {
        "send" => {
            let mut listen: SocketAddr = "0.0.0.0:9000".parse().unwrap();
            let mut remote: SocketAddr = "127.0.0.1:9001".parse().unwrap();
            let mut symbol_size = DEFAULT_SYMBOL_SIZE;
            let mut redundancy = DEFAULT_REDUNDANCY;
            let mut enable_arq = true;
            let mut group_au = false;

            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "-l" | "--listen" => {
                        i += 1;
                        listen = args
                            .get(i)
                            .ok_or("Missing listen address")?
                            .parse()
                            .map_err(|e| format!("Invalid listen address: {}", e))?;
                    }
                    "-r" | "--remote" => {
                        i += 1;
                        remote = args
                            .get(i)
                            .ok_or("Missing remote address")?
                            .parse()
                            .map_err(|e| format!("Invalid remote address: {}", e))?;
                    }
                    "--symbol-size" => {
                        i += 1;
                        symbol_size = args
                            .get(i)
                            .ok_or("Missing symbol size")?
                            .parse()
                            .map_err(|e| format!("Invalid symbol size: {}", e))?;
                    }
                    "--redundancy" => {
                        i += 1;
                        redundancy = args
                            .get(i)
                            .ok_or("Missing redundancy")?
                            .parse()
                            .map_err(|e| format!("Invalid redundancy: {}", e))?;
                    }
                    "--no-arq" => {
                        enable_arq = false;
                    }
                    "--group-au" | "--access-unit" => {
                        group_au = true;
                    }
                    "-v" | "--verbose" => {}
                    _ => {}
                }
                i += 1;
            }

            Ok(Command::Send(SendArgs {
                listen,
                remote,
                symbol_size,
                redundancy,
                enable_arq,
                group_au,
                verbose,
            }))
        }
        "recv" => {
            let mut listen: SocketAddr = "0.0.0.0:9001".parse().unwrap();
            let mut enable_arq = true;

            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "-l" | "--listen" => {
                        i += 1;
                        listen = args
                            .get(i)
                            .ok_or("Missing listen address")?
                            .parse()
                            .map_err(|e| format!("Invalid listen address: {}", e))?;
                    }
                    "--no-arq" => {
                        enable_arq = false;
                    }
                    "-v" | "--verbose" => {}
                    _ => {}
                }
                i += 1;
            }

            Ok(Command::Recv(RecvArgs {
                listen,
                enable_arq,
                verbose,
            }))
        }
        _ => Err(format!("Unknown command: {}", args[1])),
    }
}

// ============================================================================
// Sender
// ============================================================================

fn run_sender(args: SendArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!(
        "H.264 Sender: {} -> {}, redundancy={:.0}%, ARQ={}, AU={}",
        args.listen,
        args.remote,
        (args.redundancy - 1.0) * 100.0,
        if args.enable_arq { "on" } else { "off" },
        if args.group_au { "grouped" } else { "per-NAL" }
    );

    // Bind UDP socket
    let socket = UdpSocket::bind(args.listen)?;
    socket.connect(args.remote)?;
    socket.set_nonblocking(true)?;

    // Calculate max symbols per frame based on MAX_NAL_SIZE
    let max_symbols = (MAX_NAL_SIZE + args.symbol_size - 1) / args.symbol_size;

    // Create encoder
    let mut params =
        ReliableParams::new(max_symbols, args.symbol_size).with_redundancy(args.redundancy);

    if !args.enable_arq {
        params = params.without_arq();
    }

    let mut encoder = ReliableEncoder::new(params, socket);

    // Create NAL parser
    let mut parser = AnnexBParser::new();
    let mut grouper = if args.group_au {
        Some(AccessUnitGrouper::new())
    } else {
        None
    };

    // Read from stdin
    let stdin = io::stdin();
    let mut handle = stdin.lock();
    let mut read_buf = vec![0u8; READ_CHUNK_SIZE];

    let mut nals_sent = 0u64;
    let mut bytes_sent = 0u64;
    let start = Instant::now();
    let mut last_stats = Instant::now();

    loop {
        // Read chunk from stdin
        match handle.read(&mut read_buf) {
            Ok(0) => {
                // EOF - flush remaining NALs
                if let Some(nal) = parser.flush() {
                    if let Some(ref mut g) = grouper {
                        if let Some(au) = g.push(&nal) {
                            send_unit(
                                &mut encoder,
                                &au,
                                &mut nals_sent,
                                &mut bytes_sent,
                                args.verbose,
                            )?;
                        }
                        if let Some(au) = g.flush() {
                            send_unit(
                                &mut encoder,
                                &au,
                                &mut nals_sent,
                                &mut bytes_sent,
                                args.verbose,
                            )?;
                        }
                    } else {
                        send_unit(
                            &mut encoder,
                            &nal,
                            &mut nals_sent,
                            &mut bytes_sent,
                            args.verbose,
                        )?;
                    }
                } else if let Some(ref mut g) = grouper {
                    if let Some(au) = g.flush() {
                        send_unit(
                            &mut encoder,
                            &au,
                            &mut nals_sent,
                            &mut bytes_sent,
                            args.verbose,
                        )?;
                    }
                }

                let elapsed = start.elapsed().as_secs_f64();
                eprintln!(
                    "Sender done: {} NALs, {:.1} MB in {:.1}s ({:.1} NALs/s)",
                    nals_sent,
                    bytes_sent as f64 / 1_000_000.0,
                    elapsed,
                    nals_sent as f64 / elapsed
                );
                return Ok(());
            }
            Ok(n) => {
                parser.push(&read_buf[..n]);

                // Process all complete NALs
                while let Some(nal) = parser.next_nal() {
                    if let Some(ref mut g) = grouper {
                        if let Some(au) = g.push(&nal) {
                            send_unit(
                                &mut encoder,
                                &au,
                                &mut nals_sent,
                                &mut bytes_sent,
                                args.verbose,
                            )?;
                        }
                    } else {
                        send_unit(
                            &mut encoder,
                            &nal,
                            &mut nals_sent,
                            &mut bytes_sent,
                            args.verbose,
                        )?;
                    }
                }
            }
            Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        }

        // Print stats periodically
        if last_stats.elapsed() > Duration::from_secs(2) {
            let stats = encoder.stats();
            let elapsed = start.elapsed().as_secs_f64();
            eprintln!(
                "TX: {} NALs, {:.1} MB in {:.1}s ({:.1} NALs/s), RTT={:.1}ms, loss={:.1}%",
                nals_sent,
                bytes_sent as f64 / 1_000_000.0,
                elapsed,
                nals_sent as f64 / elapsed,
                stats.rtt_ms(),
                stats.loss_rate() * 100.0
            );
            last_stats = Instant::now();
        }
    }
}

fn send_unit(
    encoder: &mut ReliableEncoder<UdpSocket>,
    data: &[u8],
    nals_sent: &mut u64,
    bytes_sent: &mut u64,
    verbose: bool,
) -> io::Result<()> {
    if data.len() > MAX_NAL_SIZE {
        eprintln!("Warning: NAL too large ({} bytes), skipping", data.len());
        return Ok(());
    }

    // Get NAL type for logging
    let nal_type = if data.len() > 4 {
        let pos = if data[2] == 1 { 3 } else { 4 };
        if pos < data.len() {
            NalType::from(data[pos])
        } else {
            NalType::Unknown
        }
    } else {
        NalType::Unknown
    };

    if verbose {
        eprintln!("NAL {}: {:?}, {} bytes", nals_sent, nal_type, data.len());
    }

    encoder.send_frame(data)?;
    *nals_sent += 1;
    *bytes_sent += data.len() as u64;

    Ok(())
}

// ============================================================================
// Receiver
// ============================================================================

fn run_receiver(args: RecvArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!(
        "H.264 Receiver on {}, ARQ={}",
        args.listen,
        if args.enable_arq { "on" } else { "off" }
    );

    // Bind UDP socket
    let socket = UdpSocket::bind(args.listen)?;
    socket.set_read_timeout(Some(Duration::from_millis(100)))?;

    // Wait for first packet to get sender address
    let mut first_buf = [0u8; 65536];
    let (_first_len, sender_addr) = loop {
        match socket.recv_from(&mut first_buf) {
            Ok((len, addr)) => break (len, addr),
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => continue,
            Err(ref e) if e.kind() == io::ErrorKind::TimedOut => continue,
            Err(e) => return Err(e.into()),
        }
    };

    eprintln!("Connected to {}", sender_addr);
    socket.connect(sender_addr)?;
    socket.set_nonblocking(true)?;

    // Calculate max symbols per frame
    let max_symbols = (MAX_NAL_SIZE + DEFAULT_SYMBOL_SIZE - 1) / DEFAULT_SYMBOL_SIZE;

    // Create decoder
    let mut params = ReliableParams::new(max_symbols, DEFAULT_SYMBOL_SIZE);
    if !args.enable_arq {
        params = params.without_arq();
    }

    let mut decoder = ReliableDecoder::new(params, socket);

    // Setup stdout writer
    let mut stdout = io::stdout().lock();

    let mut nals_received = 0u64;
    let mut bytes_received = 0u64;
    let mut nals_recovered = 0u64;
    let start = Instant::now();
    let mut last_stats = Instant::now();

    loop {
        // Receive and process packets
        match decoder.recv() {
            Ok(RecvResult::Frame {
                data, recovered, ..
            }) => {
                // Write NAL unit to stdout (includes start code)
                stdout.write_all(&data)?;
                stdout.flush()?;

                nals_received += 1;
                bytes_received += data.len() as u64;
                if recovered {
                    nals_recovered += 1;
                }

                if args.verbose {
                    let nal_type = if data.len() > 4 {
                        let pos = if data[2] == 1 { 3 } else { 4 };
                        if pos < data.len() {
                            NalType::from(data[pos])
                        } else {
                            NalType::Unknown
                        }
                    } else {
                        NalType::Unknown
                    };
                    eprintln!(
                        "NAL {}: {:?}, {} bytes{}",
                        nals_received,
                        nal_type,
                        data.len(),
                        if recovered { " (FEC recovered)" } else { "" }
                    );
                }
            }
            Ok(RecvResult::Pending) | Ok(RecvResult::Control) => {
                // Continue processing
            }
            Ok(RecvResult::Timeout { frame_id }) => {
                if args.verbose {
                    eprintln!("NAL {} timed out (dropped)", frame_id);
                }
            }
            Ok(RecvResult::WouldBlock) => {
                std::thread::sleep(Duration::from_micros(100));
            }
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_micros(100));
            }
            Err(ref e) if e.kind() == io::ErrorKind::TimedOut => {
                std::thread::sleep(Duration::from_micros(100));
            }
            Err(e) => {
                eprintln!("Receive error: {}", e);
            }
        }

        // Print stats periodically
        if last_stats.elapsed() > Duration::from_secs(2) {
            let elapsed = start.elapsed().as_secs_f64();
            eprintln!(
                "RX: {} NALs ({} recovered), {:.1} MB in {:.1}s ({:.1} NALs/s)",
                nals_received,
                nals_recovered,
                bytes_received as f64 / 1_000_000.0,
                elapsed,
                nals_received as f64 / elapsed
            );
            last_stats = Instant::now();
        }
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match parse_args() {
        Ok(Command::Send(args)) => run_sender(args)?,
        Ok(Command::Recv(args)) => run_receiver(args)?,
        Err(e) => {
            eprintln!("Error: {}", e);
            eprintln!();
            eprintln!("Usage:");
            eprintln!("  h264_streaming send [options]");
            eprintln!("  h264_streaming recv [options]");
            eprintln!();
            eprintln!("Send options:");
            eprintln!("  -l, --listen <addr>      Listen address (default: 0.0.0.0:9000)");
            eprintln!("  -r, --remote <addr>      Remote address (default: 127.0.0.1:9001)");
            eprintln!("  --symbol-size <size>     Symbol size in bytes (default: 1200)");
            eprintln!("  --redundancy <factor>    Redundancy factor (default: 1.3)");
            eprintln!("  --no-arq                 Disable ARQ retransmission");
            eprintln!("  --group-au               Group NALs into access units");
            eprintln!("  -v, --verbose            Verbose output");
            eprintln!();
            eprintln!("Recv options:");
            eprintln!("  -l, --listen <addr>      Listen address (default: 0.0.0.0:9001)");
            eprintln!("  --no-arq                 Disable ARQ");
            eprintln!("  -v, --verbose            Verbose output");
            eprintln!();
            eprintln!("Example:");
            eprintln!("  # Terminal 1 (receiver):");
            eprintln!("  cargo run --release --example h264_streaming -- recv | ffplay -f h264 -probesize 32 -");
            eprintln!();
            eprintln!("  # Terminal 2 (sender):");
            eprintln!("  ffmpeg -re -i input.mp4 -c:v libx264 -tune zerolatency -f h264 - 2>/dev/null | \\");
            eprintln!("      cargo run --release --example h264_streaming -- send");
            std::process::exit(1);
        }
    }

    Ok(())
}
