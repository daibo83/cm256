//! Simple H.264 streaming with CM256 block FEC and adaptive redundancy.
//!
//! This example uses the same block-based approach as nyxpsi/examples/reliable_ffmpeg.rs
//! but replaces RaptorQ with CM256 (Cauchy Reed-Solomon) erasure coding.
//!
//! Key differences from RaptorQ:
//! - CM256 is systematic: source packets are sent unmodified
//! - CM256 requires exactly K source packets to decode (no overhead tolerance)
//! - CM256 is faster for small block counts (< 256 blocks)
//!
//! # Usage
//!
//! ## Receiver (start first):
//! ```bash
//! cargo run --release --bin reliable_ffmpeg -- recv | ffplay -f h264 -
//! ```
//!
//! ## Sender:
//! ```bash
//! ffmpeg -re -i input.mp4 -c:v libx264 -tune zerolatency -f h264 - | \
//!     cargo run --release --bin reliable_ffmpeg -- send
//! ```
//!
//! ## With custom redundancy (e.g., 50% overhead):
//! ```bash
//! ffmpeg ... | cargo run --release --bin reliable_ffmpeg -- send --redundancy 1.5
//! ```

use cm256::{Block, BlockMut, Params};
use std::collections::BTreeMap;
use std::io::{self, Read, Write};
use std::net::SocketAddr;
use std::time::{Duration, Instant};
use tokio::net::UdpSocket;

// ============================================================================
// Configuration
// ============================================================================

const DEFAULT_SYMBOL_SIZE: usize = 1200;
const SYMBOLS_PER_BLOCK: usize = 8; // 8 source symbols per block
const NETWORK_QUALITY_WINDOW: usize = 20;
const DEFAULT_REDUNDANCY: f64 = 1.3; // 30% overhead by default (2-3 parity symbols)

// ============================================================================
// Network Statistics (from nyxpsi)
// ============================================================================

struct NetworkStats {
    packet_loss_rate: f64,
    latencies: Vec<u32>,
}

impl NetworkStats {
    fn new() -> Self {
        NetworkStats {
            packet_loss_rate: 0.0,
            latencies: Vec::with_capacity(NETWORK_QUALITY_WINDOW),
        }
    }

    fn update(&mut self, packet_received: bool, latency: Option<u128>) {
        self.packet_loss_rate = 0.9 * self.packet_loss_rate + 0.1 * (!packet_received as u8 as f64);
        if let Some(lat) = latency {
            if self.latencies.len() >= NETWORK_QUALITY_WINDOW {
                self.latencies.remove(0);
            }
            self.latencies.push(lat as u32);
        }
    }

    fn get_network_quality(&self) -> f64 {
        if self.latencies.is_empty() {
            return 0.5;
        }
        let avg_latency = self.latencies.iter().sum::<u32>() as f64 / self.latencies.len() as f64;
        let normalized_latency = 1.0 / (1.0 + avg_latency / 1000.0);
        let packet_success_rate = 1.0 - self.packet_loss_rate;
        (normalized_latency + packet_success_rate) / 2.0
    }

    fn get_loss_rate(&self) -> f64 {
        self.packet_loss_rate
    }
}

fn calculate_parity_count(loss_rate: f64, base_redundancy: f64, source_count: usize) -> usize {
    // Calculate number of parity symbols based on loss rate and redundancy
    let redundancy = (base_redundancy + loss_rate * 2.0).clamp(1.1, 3.0);
    let total = (source_count as f64 * redundancy).ceil() as usize;
    let parity = total.saturating_sub(source_count);
    // CM256 requires parity_count >= 1 and total <= 256
    parity.clamp(1, 256 - source_count)
}

// ============================================================================
// Simple Pacer
// ============================================================================

struct Pacer {
    interval: Duration,
    last_send: Instant,
    tokens: f64,
    max_burst: f64,
}

impl Pacer {
    fn new(packets_per_second: u32) -> Self {
        let interval = if packets_per_second > 0 {
            Duration::from_secs_f64(1.0 / packets_per_second as f64)
        } else {
            Duration::ZERO
        };
        Self {
            interval,
            last_send: Instant::now(),
            tokens: 1.0,
            max_burst: 4.0,
        }
    }

    fn unlimited() -> Self {
        Self {
            interval: Duration::ZERO,
            last_send: Instant::now(),
            tokens: f64::MAX,
            max_burst: f64::MAX,
        }
    }

    async fn wait(&mut self) {
        if self.interval.is_zero() {
            return;
        }

        let elapsed = self.last_send.elapsed();
        if !self.interval.is_zero() {
            self.tokens += elapsed.as_secs_f64() / self.interval.as_secs_f64();
            self.tokens = self.tokens.min(self.max_burst);
        }
        self.last_send = Instant::now();

        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            return;
        }

        let wait_time = self.interval.mul_f64(1.0 - self.tokens);
        tokio::time::sleep(wait_time).await;
        self.tokens = 0.0;
        self.last_send = Instant::now();
    }
}

// ============================================================================
// Packet Header Format
// ============================================================================

/// Packet header format:
/// - block_seq: u32 (4 bytes) - block sequence number
/// - symbol_index: u8 (1 byte) - symbol index within block (0..source_count for source, source_count.. for parity)
/// - source_count: u8 (1 byte) - number of source symbols in this block
/// - parity_count: u8 (1 byte) - number of parity symbols in this block
/// - symbol_size: u16 (2 bytes) - size of each symbol in bytes
/// - payload: symbol data
const HEADER_SIZE: usize = 9;

#[derive(Clone, Copy, Debug)]
struct PacketHeader {
    block_seq: u32,
    symbol_index: u8,
    source_count: u8,
    parity_count: u8,
    symbol_size: u16,
}

impl PacketHeader {
    fn encode(&self, buf: &mut [u8]) {
        buf[0..4].copy_from_slice(&self.block_seq.to_le_bytes());
        buf[4] = self.symbol_index;
        buf[5] = self.source_count;
        buf[6] = self.parity_count;
        buf[7..9].copy_from_slice(&self.symbol_size.to_le_bytes());
    }

    fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() < HEADER_SIZE {
            return None;
        }
        Some(Self {
            block_seq: u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
            symbol_index: buf[4],
            source_count: buf[5],
            parity_count: buf[6],
            symbol_size: u16::from_le_bytes([buf[7], buf[8]]),
        })
    }
}

// ============================================================================
// Block Encoder
// ============================================================================

/// Encodes a block of data into source + parity packets using CM256
fn encode_block(data: &[u8], symbol_size: usize, parity_count: usize) -> (Vec<Vec<u8>>, usize) {
    // Pad data to be a multiple of symbol_size
    let source_count = (data.len() + symbol_size - 1) / symbol_size;
    let padded_len = source_count * symbol_size;

    let mut padded_data = data.to_vec();
    padded_data.resize(padded_len, 0);

    // Create params
    let params = match Params::new(source_count, parity_count, symbol_size) {
        Ok(p) => p,
        Err(_) => {
            // Fallback: just send source symbols without parity
            let mut packets = Vec::new();
            for i in 0..source_count {
                let start = i * symbol_size;
                let end = start + symbol_size;
                packets.push(padded_data[start..end].to_vec());
            }
            return (packets, source_count);
        }
    };

    // Create source blocks
    let source_blocks: Vec<Block> = (0..source_count)
        .map(|i| {
            let start = i * symbol_size;
            let end = start + symbol_size;
            Block::new(i as u8, &padded_data[start..end])
        })
        .collect();

    // Encode parity blocks
    let mut parity_data = vec![0u8; parity_count * symbol_size];
    if cm256::encode(&params, &source_blocks, &mut parity_data).is_err() {
        // Fallback: just send source symbols
        let mut packets = Vec::new();
        for i in 0..source_count {
            let start = i * symbol_size;
            let end = start + symbol_size;
            packets.push(padded_data[start..end].to_vec());
        }
        return (packets, source_count);
    }

    // Build packets: source symbols + parity symbols
    let mut packets = Vec::with_capacity(source_count + parity_count);

    // Source symbols
    for i in 0..source_count {
        let start = i * symbol_size;
        let end = start + symbol_size;
        packets.push(padded_data[start..end].to_vec());
    }

    // Parity symbols
    for i in 0..parity_count {
        let start = i * symbol_size;
        let end = start + symbol_size;
        packets.push(parity_data[start..end].to_vec());
    }

    (packets, source_count)
}

// ============================================================================
// Block Decoder
// ============================================================================

/// Block decoder that accumulates packets until data can be recovered
struct BlockDecoder {
    source_count: usize,
    parity_count: usize,
    symbol_size: usize,
    received: BTreeMap<u8, Vec<u8>>, // index -> data
    decoded: bool,
}

impl BlockDecoder {
    fn new(source_count: u8, parity_count: u8, symbol_size: u16) -> Self {
        Self {
            source_count: source_count as usize,
            parity_count: parity_count as usize,
            symbol_size: symbol_size as usize,
            received: BTreeMap::new(),
            decoded: false,
        }
    }

    fn add_packet(&mut self, index: u8, data: &[u8]) -> Option<Vec<u8>> {
        if self.decoded {
            return None;
        }

        // Store the packet
        if data.len() >= self.symbol_size {
            self.received
                .insert(index, data[..self.symbol_size].to_vec());
        }

        // Check if we have enough packets to decode
        if self.received.len() >= self.source_count {
            return self.try_decode();
        }

        None
    }

    fn try_decode(&mut self) -> Option<Vec<u8>> {
        if self.received.len() < self.source_count {
            return None;
        }

        let params = Params::new(self.source_count, self.parity_count, self.symbol_size).ok()?;

        // Check if we have all source symbols (no decoding needed)
        let have_all_sources = (0..self.source_count as u8).all(|i| self.received.contains_key(&i));

        if have_all_sources {
            // Just concatenate source symbols
            let mut result = Vec::with_capacity(self.source_count * self.symbol_size);
            for i in 0..self.source_count as u8 {
                result.extend_from_slice(&self.received[&i]);
            }
            self.decoded = true;
            return Some(result);
        }

        // Need to decode using parity
        // Build BlockMut array for decode()
        let mut block_data: Vec<Vec<u8>> = Vec::with_capacity(self.source_count);
        let mut block_indices: Vec<u8> = Vec::with_capacity(self.source_count);

        // Collect source_count blocks from received (prefer source blocks)
        for (&idx, data) in &self.received {
            if block_data.len() >= self.source_count {
                break;
            }
            block_data.push(data.clone());
            block_indices.push(idx);
        }

        if block_data.len() < self.source_count {
            return None;
        }

        // Create mutable blocks for decoding
        let mut blocks_mut: Vec<BlockMut> = block_data
            .iter_mut()
            .zip(block_indices.iter())
            .map(|(data, &idx)| BlockMut::new(idx, data))
            .collect();

        // Decode
        match cm256::decode(&params, &mut blocks_mut) {
            Ok(recovered_indices) => {
                // Build result from decoded data
                // After decode, recovery blocks contain recovered source data
                // We need to reconstruct the original order
                let mut result_blocks: Vec<Option<Vec<u8>>> = vec![None; self.source_count];

                for (block, &idx) in blocks_mut.iter().zip(block_indices.iter()) {
                    if (idx as usize) < self.source_count {
                        // This was already a source block
                        result_blocks[idx as usize] = Some(block.data.to_vec());
                    }
                }

                // Fill in recovered blocks
                for &recovered_idx in recovered_indices.iter() {
                    // Find which block in our array was used to recover this
                    for (j, &idx) in block_indices.iter().enumerate() {
                        if idx >= self.source_count as u8 {
                            // This was a parity block used for recovery
                            if result_blocks[recovered_idx as usize].is_none() {
                                result_blocks[recovered_idx as usize] = Some(block_data[j].clone());
                                break;
                            }
                        }
                    }
                }

                // If still missing, try alternative approach
                for i in 0..self.source_count {
                    if result_blocks[i].is_none() {
                        // Check if we have this source directly
                        if let Some(data) = self.received.get(&(i as u8)) {
                            result_blocks[i] = Some(data.clone());
                        }
                    }
                }

                // Concatenate all blocks
                let mut result = Vec::with_capacity(self.source_count * self.symbol_size);
                for i in 0..self.source_count {
                    if let Some(data) = &result_blocks[i] {
                        result.extend_from_slice(data);
                    } else {
                        // Missing block - fill with zeros (shouldn't happen)
                        result.extend(vec![0u8; self.symbol_size]);
                    }
                }

                self.decoded = true;
                Some(result)
            }
            Err(_) => None,
        }
    }

    fn packets_received(&self) -> usize {
        self.received.len()
    }
}

// ============================================================================
// CLI
// ============================================================================

use std::env;

struct SendArgs {
    listen: SocketAddr,
    remote: SocketAddr,
    symbol_size: usize,
    redundancy: f64,
    pps: u32,
    mbps: Option<f64>,
    verbose: bool,
}

struct RecvArgs {
    listen: SocketAddr,
    verbose: bool,
}

enum Command {
    Send(SendArgs),
    Recv(RecvArgs),
}

fn parse_args() -> Result<Command, String> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        return Err("Usage: reliable_ffmpeg <send|recv> [options]".to_string());
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
            let mut pps = 0u32;
            let mut mbps = None;

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
                    "--pps" => {
                        i += 1;
                        pps = args
                            .get(i)
                            .ok_or("Missing pps")?
                            .parse()
                            .map_err(|e| format!("Invalid pps: {}", e))?;
                    }
                    "--mbps" => {
                        i += 1;
                        mbps = Some(
                            args.get(i)
                                .ok_or("Missing mbps")?
                                .parse()
                                .map_err(|e| format!("Invalid mbps: {}", e))?,
                        );
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
                pps,
                mbps,
                verbose,
            }))
        }
        "recv" => {
            let mut listen: SocketAddr = "0.0.0.0:9001".parse().unwrap();

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
                    "-v" | "--verbose" => {}
                    _ => {}
                }
                i += 1;
            }

            Ok(Command::Recv(RecvArgs { listen, verbose }))
        }
        _ => Err(format!("Unknown command: {}", args[1])),
    }
}

// ============================================================================
// Sender
// ============================================================================

async fn run_sender(args: SendArgs) -> Result<(), Box<dyn std::error::Error>> {
    let effective_pps = if let Some(rate_mbps) = args.mbps {
        let bits_per_packet = args.symbol_size * 8;
        let target_bps = rate_mbps * 1_000_000.0;
        (target_bps / bits_per_packet as f64) as u32
    } else {
        args.pps
    };

    let mut pacer = if effective_pps > 0 {
        Pacer::new(effective_pps)
    } else {
        Pacer::unlimited()
    };

    let rate_str = if effective_pps > 0 {
        format!(
            "{} pps ({:.1} Mbps)",
            effective_pps,
            (effective_pps as f64 * args.symbol_size as f64 * 8.0) / 1_000_000.0
        )
    } else {
        "unlimited".to_string()
    };

    eprintln!(
        "CM256 Sender: {} -> {}, redundancy={:.0}%, rate={}",
        args.listen,
        args.remote,
        (args.redundancy - 1.0) * 100.0,
        rate_str
    );

    let socket = UdpSocket::bind(args.listen).await?;
    socket.connect(args.remote).await?;

    // Channel for stdin
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<u8>>(256);

    // Stdin reader thread
    let sym_size = args.symbol_size;
    std::thread::spawn(move || {
        let stdin = io::stdin();
        let mut handle = stdin.lock();
        let mut buffer = vec![0u8; sym_size];

        loop {
            let mut total = 0;
            while total < sym_size {
                match handle.read(&mut buffer[total..]) {
                    Ok(0) => {
                        if total > 0 {
                            let _ = tx.blocking_send(buffer[..total].to_vec());
                        }
                        return;
                    }
                    Ok(n) => total += n,
                    Err(_) => return,
                }
            }
            if tx.blocking_send(buffer.clone()).is_err() {
                return;
            }
        }
    });

    let mut network_stats = NetworkStats::new();
    let mut count = 0u64;
    let mut bytes_sent = 0u64;
    let start = Instant::now();
    let mut block_seq = 0u32;

    // Buffer to accumulate data for a block
    let block_size = args.symbol_size * SYMBOLS_PER_BLOCK;
    let mut block_buffer = Vec::with_capacity(block_size);

    while let Some(data) = rx.recv().await {
        block_buffer.extend_from_slice(&data);

        // When we have enough data, encode and send a block
        if block_buffer.len() >= block_size {
            let block_data: Vec<u8> = block_buffer.drain(..block_size).collect();

            // Adapt parity count based on network quality
            let parity_count = calculate_parity_count(
                network_stats.get_loss_rate(),
                args.redundancy,
                SYMBOLS_PER_BLOCK,
            );

            // Encode the block
            let (packets, source_count) = encode_block(&block_data, args.symbol_size, parity_count);

            if args.verbose {
                eprintln!(
                    "Block {}: {} bytes -> {} source + {} parity packets",
                    block_seq,
                    block_data.len(),
                    source_count,
                    packets.len() - source_count
                );
            }

            // Send all packets for this block
            for (idx, packet_data) in packets.iter().enumerate() {
                pacer.wait().await;

                // Build packet with header
                let header = PacketHeader {
                    block_seq,
                    symbol_index: idx as u8,
                    source_count: source_count as u8,
                    parity_count: (packets.len() - source_count) as u8,
                    symbol_size: args.symbol_size as u16,
                };

                let mut msg = Vec::with_capacity(HEADER_SIZE + packet_data.len());
                msg.resize(HEADER_SIZE, 0);
                header.encode(&mut msg[..HEADER_SIZE]);
                msg.extend_from_slice(packet_data);

                match socket.send(&msg).await {
                    Ok(_) => {
                        count += 1;
                        bytes_sent += msg.len() as u64;
                        network_stats.update(true, Some(1));
                    }
                    Err(e) => {
                        eprintln!("Send error: {}", e);
                        network_stats.update(false, None);
                    }
                }
            }

            block_seq = block_seq.wrapping_add(1);

            if block_seq % 100 == 0 {
                let elapsed = start.elapsed().as_secs_f64();
                let actual_mbps = (bytes_sent as f64 * 8.0) / elapsed / 1_000_000.0;
                eprintln!(
                    "TX: {} blocks, {} pkts, {:.1} MB in {:.1}s ({:.1} Mbps, quality={:.2})",
                    block_seq,
                    count,
                    bytes_sent as f64 / 1_000_000.0,
                    elapsed,
                    actual_mbps,
                    network_stats.get_network_quality()
                );
            }
        }
    }

    // Send remaining data in buffer
    if !block_buffer.is_empty() {
        let parity_count = calculate_parity_count(
            network_stats.get_loss_rate(),
            args.redundancy,
            (block_buffer.len() + args.symbol_size - 1) / args.symbol_size,
        );
        let (packets, source_count) = encode_block(&block_buffer, args.symbol_size, parity_count);

        for (idx, packet_data) in packets.iter().enumerate() {
            pacer.wait().await;
            let header = PacketHeader {
                block_seq,
                symbol_index: idx as u8,
                source_count: source_count as u8,
                parity_count: (packets.len() - source_count) as u8,
                symbol_size: args.symbol_size as u16,
            };
            let mut msg = Vec::with_capacity(HEADER_SIZE + packet_data.len());
            msg.resize(HEADER_SIZE, 0);
            header.encode(&mut msg[..HEADER_SIZE]);
            msg.extend_from_slice(packet_data);
            let _ = socket.send(&msg).await;
            count += 1;
            bytes_sent += msg.len() as u64;
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let actual_mbps = (bytes_sent as f64 * 8.0) / elapsed / 1_000_000.0;
    eprintln!(
        "Sender done: {} blocks, {} packets, {:.1} MB in {:.1}s ({:.1} Mbps)",
        block_seq,
        count,
        bytes_sent as f64 / 1_000_000.0,
        elapsed,
        actual_mbps
    );

    Ok(())
}

// ============================================================================
// Receiver
// ============================================================================

async fn run_receiver(args: RecvArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("CM256 Receiver on {}", args.listen);

    let socket = UdpSocket::bind(args.listen).await?;
    eprintln!("Waiting for connection...");

    // Wait for first packet
    let mut first_buf = vec![0u8; 65536];
    let (first_len, sender_addr) = socket.recv_from(&mut first_buf).await?;
    socket.connect(sender_addr).await?;
    eprintln!("Connected to {}", sender_addr);

    // Stdout writer
    let (write_tx, write_rx) = std::sync::mpsc::channel::<Vec<u8>>();
    std::thread::spawn(move || {
        let mut stdout = io::stdout().lock();
        while let Ok(data) = write_rx.recv() {
            let _ = stdout.write_all(&data);
            let _ = stdout.flush();
        }
    });

    let mut network_stats = NetworkStats::new();
    let mut count = 0u64;
    let mut blocks_decoded = 0u64;
    let start = Instant::now();

    // Block tracking
    let mut block_decoders: BTreeMap<u32, BlockDecoder> = BTreeMap::new();
    let mut next_output_block: Option<u32> = None;
    let mut decoded_blocks: BTreeMap<u32, Vec<u8>> = BTreeMap::new();
    let max_pending_blocks = 32;

    // Helper to output blocks in order
    let output_blocks = |decoded: &mut BTreeMap<u32, Vec<u8>>,
                         next_block: &mut Option<u32>,
                         write_tx: &std::sync::mpsc::Sender<Vec<u8>>,
                         blocks_decoded: &mut u64| {
        loop {
            let block_seq = match next_block {
                Some(s) => *s,
                None => return,
            };
            if let Some(data) = decoded.remove(&block_seq) {
                let _ = write_tx.send(data);
                *next_block = Some(block_seq.wrapping_add(1));
                *blocks_decoded += 1;
            } else {
                break;
            }
        }
    };

    // Process a packet
    let process_packet = |buf: &[u8],
                          len: usize,
                          block_decoders: &mut BTreeMap<u32, BlockDecoder>,
                          decoded_blocks: &mut BTreeMap<u32, Vec<u8>>,
                          next_output_block: &mut Option<u32>,
                          verbose: bool|
     -> bool {
        let header = match PacketHeader::decode(&buf[..len]) {
            Some(h) => h,
            None => return false,
        };

        let payload = &buf[HEADER_SIZE..len];

        // Initialize next_output_block if not set
        if next_output_block.is_none() {
            *next_output_block = Some(header.block_seq);
        }

        // Skip if block is too old
        if let Some(next) = *next_output_block {
            let diff = header.block_seq.wrapping_sub(next);
            if diff > 0x80000000 {
                return false;
            }
        }

        // Skip if already decoded
        if decoded_blocks.contains_key(&header.block_seq) {
            return false;
        }

        // Get or create decoder for this block
        let decoder = block_decoders.entry(header.block_seq).or_insert_with(|| {
            BlockDecoder::new(header.source_count, header.parity_count, header.symbol_size)
        });

        if let Some(data) = decoder.add_packet(header.symbol_index, payload) {
            if verbose {
                eprintln!(
                    "Block {} decoded after {} packets ({} bytes)",
                    header.block_seq,
                    decoder.packets_received(),
                    data.len()
                );
            }
            decoded_blocks.insert(header.block_seq, data);
            block_decoders.remove(&header.block_seq);
            return true;
        }

        false
    };

    // Process first packet
    if process_packet(
        &first_buf,
        first_len,
        &mut block_decoders,
        &mut decoded_blocks,
        &mut next_output_block,
        args.verbose,
    ) {
        output_blocks(
            &mut decoded_blocks,
            &mut next_output_block,
            &write_tx,
            &mut blocks_decoded,
        );
    }
    count += 1;
    network_stats.update(true, Some(1));

    let mut buf = vec![0u8; 65536];

    loop {
        match socket.recv(&mut buf).await {
            Ok(len) => {
                count += 1;
                network_stats.update(true, Some(1));

                if process_packet(
                    &buf,
                    len,
                    &mut block_decoders,
                    &mut decoded_blocks,
                    &mut next_output_block,
                    args.verbose,
                ) {
                    output_blocks(
                        &mut decoded_blocks,
                        &mut next_output_block,
                        &write_tx,
                        &mut blocks_decoded,
                    );
                }

                // Clean up old decoders
                if block_decoders.len() > max_pending_blocks {
                    if let Some(&oldest) = block_decoders.keys().next() {
                        if let Some(next) = next_output_block {
                            let diff = oldest.wrapping_sub(next);
                            if diff > 0x80000000 || diff > max_pending_blocks as u32 {
                                block_decoders.remove(&oldest);
                                eprintln!("Skipping block {} (timeout)", oldest);
                            }
                        }
                    }
                }

                // Skip ahead if decoded buffer too large
                if decoded_blocks.len() > max_pending_blocks {
                    if let Some(&first_block) = decoded_blocks.keys().next() {
                        if let Some(next) = next_output_block {
                            if first_block != next {
                                let gap = first_block.wrapping_sub(next);
                                if gap < 0x80000000 {
                                    eprintln!("Skipping {} missing blocks", gap);
                                    next_output_block = Some(first_block);
                                    output_blocks(
                                        &mut decoded_blocks,
                                        &mut next_output_block,
                                        &write_tx,
                                        &mut blocks_decoded,
                                    );
                                }
                            }
                        }
                    }
                }

                if count % 5000 == 0 {
                    let elapsed = start.elapsed().as_secs_f64();
                    eprintln!(
                        "RX: {} pkts, {} blocks decoded in {:.1}s (pending: {} decoders, {} decoded, quality={:.2})",
                        count,
                        blocks_decoded,
                        elapsed,
                        block_decoders.len(),
                        decoded_blocks.len(),
                        network_stats.get_network_quality()
                    );
                }
            }
            Err(e) => {
                eprintln!("Recv error: {}", e);
                network_stats.update(false, None);
            }
        }
    }
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    match parse_args() {
        Ok(Command::Send(args)) => run_sender(args).await?,
        Ok(Command::Recv(args)) => run_receiver(args).await?,
        Err(e) => {
            eprintln!("Error: {}", e);
            eprintln!();
            eprintln!("Usage:");
            eprintln!("  reliable_ffmpeg send [options]");
            eprintln!("  reliable_ffmpeg recv [options]");
            eprintln!();
            eprintln!("Send options:");
            eprintln!("  -l, --listen <addr>      Listen address (default: 0.0.0.0:9000)");
            eprintln!("  -r, --remote <addr>      Remote address (default: 127.0.0.1:9001)");
            eprintln!("  --symbol-size <size>     Symbol size in bytes (default: 1200)");
            eprintln!("  --redundancy <factor>    Redundancy factor (default: 1.3 = 30% overhead)");
            eprintln!("  --pps <rate>             Packets per second limit (0 = unlimited)");
            eprintln!("  --mbps <rate>            Target bitrate in Mbps");
            eprintln!("  -v, --verbose            Verbose output");
            eprintln!();
            eprintln!("Recv options:");
            eprintln!("  -l, --listen <addr>      Listen address (default: 0.0.0.0:9001)");
            eprintln!("  -v, --verbose            Verbose output");
            std::process::exit(1);
        }
    }

    Ok(())
}
