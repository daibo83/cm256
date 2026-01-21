//! Hybrid FEC+ARQ streaming example using the reliable module.
//!
//! This example demonstrates the hybrid reliability approach ported from
//! Dynamic-FEC: adaptive FEC based on packet loss rate, combined with
//! ARQ retransmission as a fallback.
//!
//! # Features
//!
//! - **Adaptive FEC**: Parity count adjusts dynamically based on measured loss rate
//! - **ARQ Fallback**: Retransmits missing packets if FEC cannot recover
//! - **Network Stats**: EWMA-based RTT and loss rate tracking
//! - **Frame Dropping**: Old frames are dropped to maintain low latency
//!
//! # Usage
//!
//! ## Receiver (start first):
//! ```bash
//! cargo run --release --example hybrid_streaming -- recv | ffplay -f h264 -
//! ```
//!
//! ## Sender:
//! ```bash
//! ffmpeg -re -i input.mp4 -c:v libx264 -tune zerolatency -f h264 - | \
//!     cargo run --release --example hybrid_streaming -- send
//! ```
//!
//! ## With ARQ disabled (pure FEC):
//! ```bash
//! cargo run --release --example hybrid_streaming -- send --no-arq
//! ```

use cm256::reliable::{
    calculate_parity_count, RecvResult, ReliableDecoder, ReliableEncoder, ReliableParams,
};
use std::io::{self, Read, Write};
use std::net::{SocketAddr, UdpSocket};
use std::time::{Duration, Instant};

// ============================================================================
// Configuration
// ============================================================================

const DEFAULT_SYMBOL_SIZE: usize = 1200;
const DEFAULT_SYMBOLS_PER_FRAME: usize = 8;
const DEFAULT_REDUNDANCY: f64 = 1.3;

// ============================================================================
// CLI
// ============================================================================

use std::env;

struct SendArgs {
    listen: SocketAddr,
    remote: SocketAddr,
    symbol_size: usize,
    symbols_per_frame: usize,
    redundancy: f64,
    enable_arq: bool,
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
        return Err("Usage: hybrid_streaming <send|recv> [options]".to_string());
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
            let mut symbols_per_frame = DEFAULT_SYMBOLS_PER_FRAME;
            let mut redundancy = DEFAULT_REDUNDANCY;
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
                    "--symbols" => {
                        i += 1;
                        symbols_per_frame = args
                            .get(i)
                            .ok_or("Missing symbols per frame")?
                            .parse()
                            .map_err(|e| format!("Invalid symbols per frame: {}", e))?;
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
                    "-v" | "--verbose" => {}
                    _ => {}
                }
                i += 1;
            }

            Ok(Command::Send(SendArgs {
                listen,
                remote,
                symbol_size,
                symbols_per_frame,
                redundancy,
                enable_arq,
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
        "Hybrid Sender: {} -> {}, redundancy={:.0}%, ARQ={}",
        args.listen,
        args.remote,
        (args.redundancy - 1.0) * 100.0,
        if args.enable_arq { "on" } else { "off" }
    );

    // Bind UDP socket
    let socket = UdpSocket::bind(args.listen)?;
    socket.connect(args.remote)?;
    socket.set_nonblocking(true)?;

    // Create encoder
    let mut params = ReliableParams::new(args.symbols_per_frame, args.symbol_size)
        .with_redundancy(args.redundancy);

    if !args.enable_arq {
        params = params.without_arq();
    }

    let mut encoder = ReliableEncoder::new(params, socket);

    // Read from stdin
    let stdin = io::stdin();
    let mut handle = stdin.lock();

    let frame_size = args.symbols_per_frame * args.symbol_size;
    let mut buffer = vec![0u8; frame_size];

    let mut frames_sent = 0u64;
    let start = Instant::now();
    let mut last_stats = Instant::now();

    loop {
        // Read a full frame from stdin
        let mut total = 0;
        while total < frame_size {
            match handle.read(&mut buffer[total..]) {
                Ok(0) => {
                    // EOF - send remaining data
                    if total > 0 {
                        if let Err(e) = encoder.send_frame(&buffer[..total]) {
                            eprintln!("Send error: {}", e);
                        }
                        frames_sent += 1;
                    }
                    let elapsed = start.elapsed().as_secs_f64();
                    eprintln!(
                        "Sender done: {} frames in {:.1}s ({:.1} fps)",
                        frames_sent,
                        elapsed,
                        frames_sent as f64 / elapsed
                    );
                    return Ok(());
                }
                Ok(n) => total += n,
                Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        }

        // Send the frame
        if let Err(e) = encoder.send_frame(&buffer) {
            eprintln!("Send error: {}", e);
        }
        frames_sent += 1;

        // Print stats periodically
        if last_stats.elapsed() > Duration::from_secs(2) {
            let stats = encoder.stats();
            let elapsed = start.elapsed().as_secs_f64();
            eprintln!(
                "TX: {} frames in {:.1}s ({:.1} fps), RTT={:.1}ms, loss={:.1}%",
                frames_sent,
                elapsed,
                frames_sent as f64 / elapsed,
                stats.rtt_ms(),
                stats.loss_rate() * 100.0
            );
            last_stats = Instant::now();
        }

        if args.verbose && frames_sent % 100 == 0 {
            let stats = encoder.stats();
            eprintln!(
                "Frame {}: parity={}, RTT={:.1}ms, loss={:.1}%",
                frames_sent,
                calculate_parity_count(args.symbols_per_frame, stats.loss_rate()),
                stats.rtt_ms(),
                stats.loss_rate() * 100.0
            );
        }
    }
}

// ============================================================================
// Receiver
// ============================================================================

fn run_receiver(args: RecvArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!(
        "Hybrid Receiver on {}, ARQ={}",
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

    // Create decoder
    let mut params = ReliableParams::new(DEFAULT_SYMBOLS_PER_FRAME, DEFAULT_SYMBOL_SIZE);
    if !args.enable_arq {
        params = params.without_arq();
    }

    let mut decoder = ReliableDecoder::new(params, socket);

    // Setup stdout writer
    let mut stdout = io::stdout().lock();

    let mut frames_received = 0u64;
    let start = Instant::now();
    let mut last_stats = Instant::now();

    loop {
        // Receive and process packets
        match decoder.recv() {
            Ok(RecvResult::Frame {
                data, recovered, ..
            }) => {
                stdout.write_all(&data)?;
                stdout.flush()?;
                frames_received += 1;

                if args.verbose {
                    eprintln!(
                        "Frame {}: {} bytes{}",
                        frames_received,
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
                    eprintln!("Frame {} timed out", frame_id);
                }
            }
            Ok(RecvResult::WouldBlock) => {
                // No data available, sleep briefly
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
                "RX: {} frames in {:.1}s ({:.1} fps)",
                frames_received,
                elapsed,
                frames_received as f64 / elapsed
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
            eprintln!("  hybrid_streaming send [options]");
            eprintln!("  hybrid_streaming recv [options]");
            eprintln!();
            eprintln!("Send options:");
            eprintln!("  -l, --listen <addr>      Listen address (default: 0.0.0.0:9000)");
            eprintln!("  -r, --remote <addr>      Remote address (default: 127.0.0.1:9001)");
            eprintln!("  --symbol-size <size>     Symbol size in bytes (default: 1200)");
            eprintln!("  --symbols <count>        Symbols per frame (default: 8)");
            eprintln!("  --redundancy <factor>    Redundancy factor (default: 1.3)");
            eprintln!("  --no-arq                 Disable ARQ retransmission");
            eprintln!("  -v, --verbose            Verbose output");
            eprintln!();
            eprintln!("Recv options:");
            eprintln!("  -l, --listen <addr>      Listen address (default: 0.0.0.0:9001)");
            eprintln!("  --no-arq                 Disable ARQ");
            eprintln!("  -v, --verbose            Verbose output");
            std::process::exit(1);
        }
    }

    Ok(())
}
