//! FFmpeg H.264 streaming with Diagonal Interleaving FEC
//!
//! This example uses the streaming FEC module for low-latency video streaming.
//! Unlike block FEC (ffmpeg_pipe.rs), streaming FEC provides continuous protection
//! with lower latency - each packet is protected as soon as it's sent.
//!
//! # Usage
//!
//! Receiver (start first):
//! ```bash
//! cargo run --release --example ffmpeg_streaming -- recv | ffplay -f h264 -
//! ```
//!
//! Sender:
//! ```bash
//! ffmpeg -re -i input.mp4 -c:v libx264 -tune zerolatency -f h264 - | \
//!     cargo run --release --example ffmpeg_streaming -- send
//! ```
//!
//! # Key Differences from Block FEC (ffmpeg_pipe)
//!
//! - **Lower latency**: Parities are generated per-packet, not per-message
//! - **Better burst recovery**: Overlapping windows catch consecutive losses
//! - **Continuous protection**: Every packet is covered by multiple parity windows

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use cm256::streaming::{StreamingDecoder, StreamingEncoder, StreamingParams};
use std::collections::BTreeMap;
use std::io::{self, Read, Write};
use std::net::SocketAddr;
use tokio::net::UdpSocket;
use tracing::{debug, info, warn, Level};

// ============================================================================
// Packet Format
// ============================================================================

/// Packet header for streaming FEC
///
/// Format:
/// - bytes 0-1: sequence number (u16, little-endian)
/// - byte 2: packet type (0 = source, 1 = parity)
/// - byte 3: parity index (for parity packets) or flags (for source)
/// - bytes 4-5: parity end_seq (for parity packets) or data length (for source)
/// - bytes 6-7: reserved
#[derive(Debug, Clone, Copy)]
struct PacketHeader {
    seq: u16,
    is_parity: bool,
    parity_index: u8,
    parity_end_seq: u16,
    data_len: u16,
}

impl PacketHeader {
    const SIZE: usize = 8;

    fn source(seq: u16, data_len: u16) -> Self {
        Self {
            seq,
            is_parity: false,
            parity_index: 0,
            parity_end_seq: 0,
            data_len,
        }
    }

    fn parity(seq: u16, parity_index: u8, end_seq: u16) -> Self {
        Self {
            seq,
            is_parity: true,
            parity_index,
            parity_end_seq: end_seq,
            data_len: 0,
        }
    }

    fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..2].copy_from_slice(&self.seq.to_le_bytes());
        buf[2] = if self.is_parity { 1 } else { 0 };
        buf[3] = self.parity_index;
        if self.is_parity {
            buf[4..6].copy_from_slice(&self.parity_end_seq.to_le_bytes());
        } else {
            buf[4..6].copy_from_slice(&self.data_len.to_le_bytes());
        }
        buf
    }

    fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < Self::SIZE {
            return None;
        }
        let seq = u16::from_le_bytes([buf[0], buf[1]]);
        let is_parity = buf[2] != 0;
        let parity_index = buf[3];
        let field = u16::from_le_bytes([buf[4], buf[5]]);

        Some(Self {
            seq,
            is_parity,
            parity_index,
            parity_end_seq: if is_parity { field } else { 0 },
            data_len: if is_parity { 0 } else { field },
        })
    }
}

// ============================================================================
// CLI
// ============================================================================

#[derive(Parser, Debug)]
#[command(name = "ffmpeg-streaming")]
#[command(about = "Stream H.264 video with diagonal interleaving FEC")]
struct Args {
    #[command(subcommand)]
    command: Command,

    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Send H.264 from stdin over network with streaming FEC
    Send {
        /// Local address to bind to
        #[arg(short, long, default_value = "0.0.0.0:9000")]
        listen: SocketAddr,

        /// Remote address to send to
        #[arg(short, long, default_value = "127.0.0.1:9001")]
        remote: SocketAddr,

        /// FEC delay (window size in packets)
        #[arg(long, default_value = "8")]
        delay: u8,

        /// Number of parity packets per window
        #[arg(long, default_value = "2")]
        parities: u8,

        /// Packet size in bytes
        #[arg(long, default_value = "1200")]
        packet_size: usize,
    },
    /// Receive stream and output H.264 to stdout
    Recv {
        /// Local address to bind to
        #[arg(short, long, default_value = "0.0.0.0:9001")]
        listen: SocketAddr,

        /// FEC delay (must match sender)
        #[arg(long, default_value = "8")]
        delay: u8,

        /// Number of parity packets per window (must match sender)
        #[arg(long, default_value = "2")]
        parities: u8,

        /// Packet size in bytes (must match sender)
        #[arg(long, default_value = "1200")]
        packet_size: usize,
    },
}

// ============================================================================
// Sender
// ============================================================================

async fn run_sender(
    addr: SocketAddr,
    remote: SocketAddr,
    delay: u8,
    parities: u8,
    packet_size: usize,
) -> Result<()> {
    let params = StreamingParams::new(delay, parities, packet_size)
        .context("Invalid streaming parameters")?;

    info!(
        "Streaming Sender: {} -> {}, delay={}, parities={}, packet_size={}",
        addr, remote, delay, parities, packet_size
    );
    info!(
        "  Max burst recovery: {} packets, overhead: {:.1}%",
        params.max_burst(),
        params.overhead() * 100.0
    );

    let socket = UdpSocket::bind(addr)
        .await
        .with_context(|| format!("Failed to bind to {}", addr))?;

    let mut encoder = StreamingEncoder::new(params);
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<u8>>(256);

    // Stdin reader thread - reads fixed-size chunks
    let pkt_size = packet_size;
    std::thread::spawn(move || {
        let stdin = io::stdin();
        let mut handle = stdin.lock();
        let mut buffer = vec![0u8; pkt_size];

        loop {
            // Read exactly packet_size bytes (or less at EOF)
            let mut total_read = 0;
            while total_read < pkt_size {
                match handle.read(&mut buffer[total_read..]) {
                    Ok(0) => {
                        // EOF
                        if total_read > 0 {
                            // Send partial final packet
                            let _ = tx.blocking_send(buffer[..total_read].to_vec());
                        }
                        return;
                    }
                    Ok(n) => {
                        total_read += n;
                    }
                    Err(e) => {
                        eprintln!("stdin read error: {}", e);
                        return;
                    }
                }
            }
            if tx.blocking_send(buffer.clone()).is_err() {
                return;
            }
        }
    });

    let mut source_count = 0u64;
    let mut parity_count = 0u64;
    let mut total_bytes = 0u64;

    while let Some(data) = rx.recv().await {
        let actual_len = data.len();

        // Pad to packet_size for FEC (streaming FEC needs fixed-size symbols)
        let mut padded = data;
        if padded.len() < packet_size {
            padded.resize(packet_size, 0);
        }

        // Encode with streaming FEC
        let result = encoder.add_source(&padded);

        // Send source packet
        let header = PacketHeader::source(result.source_seq, actual_len as u16);
        let mut packet = Vec::with_capacity(PacketHeader::SIZE + packet_size);
        packet.extend_from_slice(&header.to_bytes());
        packet.extend_from_slice(&padded);

        socket.send_to(&packet, remote).await?;
        source_count += 1;
        total_bytes += actual_len as u64;

        // Send parity packets
        for parity in &result.parities {
            let header = PacketHeader::parity(
                result.source_seq, // Use source seq for ordering
                parity.parity_index,
                parity.end_seq,
            );
            let mut packet = Vec::with_capacity(PacketHeader::SIZE + packet_size);
            packet.extend_from_slice(&header.to_bytes());
            packet.extend_from_slice(&parity.data);

            socket.send_to(&packet, remote).await?;
            parity_count += 1;
        }

        // Small delay to prevent overwhelming the network
        if source_count % 4 == 0 {
            tokio::time::sleep(tokio::time::Duration::from_micros(50)).await;
        }

        if source_count % 1000 == 0 {
            info!(
                "Sent {} sources, {} parities, {} bytes",
                source_count, parity_count, total_bytes
            );
        }
    }

    info!(
        "Sender done: {} sources, {} parities, {} bytes",
        source_count, parity_count, total_bytes
    );
    Ok(())
}

// ============================================================================
// Receiver
// ============================================================================

async fn run_receiver(addr: SocketAddr, delay: u8, parities: u8, packet_size: usize) -> Result<()> {
    let params = StreamingParams::new(delay, parities, packet_size)
        .context("Invalid streaming parameters")?;

    info!(
        "Streaming Receiver on {}, delay={}, parities={}, packet_size={}",
        addr, delay, parities, packet_size
    );

    let socket = UdpSocket::bind(addr)
        .await
        .with_context(|| format!("Failed to bind to {}", addr))?;

    let mut decoder = StreamingDecoder::new(params);
    // Keep more history for recovery
    decoder.set_history_windows(16);

    let mut recv_buffer = vec![0u8; 65536];

    // Channels for output ordering
    let (output_tx, mut output_rx) = tokio::sync::mpsc::channel::<(u16, Vec<u8>)>(1000);
    let (write_tx, write_rx) = std::sync::mpsc::channel::<Vec<u8>>();

    // Stdout writer thread
    std::thread::spawn(move || {
        let mut stdout = io::stdout().lock();
        let mut total_bytes = 0u64;
        let mut msg_count = 0u64;

        while let Ok(data) = write_rx.recv() {
            if stdout.write_all(&data).is_err() {
                break;
            }
            total_bytes += data.len() as u64;
            msg_count += 1;

            // Batch writes
            while let Ok(more_data) = write_rx.try_recv() {
                if stdout.write_all(&more_data).is_err() {
                    return;
                }
                total_bytes += more_data.len() as u64;
                msg_count += 1;
            }
            let _ = stdout.flush();

            if msg_count % 1000 == 0 {
                eprintln!("INFO: Wrote {} packets, {} bytes", msg_count, total_bytes);
            }
        }
    });

    // Ordering task - reorders packets by sequence number
    tokio::task::spawn(async move {
        let mut output_queue: BTreeMap<u16, Vec<u8>> = BTreeMap::new();
        let mut next_seq: u16 = 0;
        let mut first_packet = true;

        while let Some((seq, data)) = output_rx.recv().await {
            if first_packet {
                next_seq = seq;
                first_packet = false;
            }

            output_queue.insert(seq, data);

            // Output in-order packets
            while let Some(data) = output_queue.remove(&next_seq) {
                if write_tx.send(data).is_err() {
                    return;
                }
                next_seq = next_seq.wrapping_add(1);
            }

            // Handle queue overflow (skip ahead if too far behind)
            if output_queue.len() > 200 {
                if let Some(&first_seq) = output_queue.keys().next() {
                    let gap = first_seq.wrapping_sub(next_seq);
                    if gap < 32768 {
                        // first_seq is ahead
                        eprintln!(
                            "WARN: Queue overflow, skipping {} -> {}",
                            next_seq, first_seq
                        );
                        next_seq = first_seq;
                        while let Some(data) = output_queue.remove(&next_seq) {
                            if write_tx.send(data).is_err() {
                                return;
                            }
                            next_seq = next_seq.wrapping_add(1);
                        }
                    }
                }
            }
        }
    });

    let mut packet_count = 0u64;
    let mut source_count = 0u64;
    let mut parity_count = 0u64;
    let mut recovered_count = 0u64;

    // Track data lengths for each sequence (source packets include actual length)
    let mut data_lengths: std::collections::HashMap<u16, u16> = std::collections::HashMap::new();

    loop {
        let n = socket.recv(&mut recv_buffer).await?;
        packet_count += 1;

        if n < PacketHeader::SIZE {
            continue;
        }

        let Some(header) = PacketHeader::from_bytes(&recv_buffer[..PacketHeader::SIZE]) else {
            continue;
        };

        let payload = &recv_buffer[PacketHeader::SIZE..n];
        if payload.len() != packet_size {
            // Pad if needed (shouldn't happen normally)
            continue;
        }

        if header.is_parity {
            // Parity packet
            decoder.add_parity(header.parity_end_seq, header.parity_index, payload);
            parity_count += 1;

            debug!(
                "RX parity: end_seq={}, index={}",
                header.parity_end_seq, header.parity_index
            );
        } else {
            // Source packet
            decoder.add_source(header.seq, payload);
            data_lengths.insert(header.seq, header.data_len);
            source_count += 1;

            debug!("RX source: seq={}, len={}", header.seq, header.data_len);

            // Output immediately if we have this source
            let actual_len = header.data_len as usize;
            let data = payload[..actual_len.min(packet_size)].to_vec();
            if output_tx.try_send((header.seq, data)).is_err() {
                warn!("Output buffer full, dropping packet {}", header.seq);
            }
        }

        // Try recovery periodically
        if packet_count % 4 == 0 {
            let recovered = decoder.try_recover();
            for (seq, data) in recovered {
                recovered_count += 1;

                // Get actual data length (default to full packet if unknown)
                let actual_len = data_lengths
                    .get(&seq)
                    .copied()
                    .unwrap_or(packet_size as u16) as usize;
                let trimmed = data[..actual_len.min(packet_size)].to_vec();

                debug!("Recovered seq={}, len={}", seq, actual_len);

                if output_tx.try_send((seq, trimmed)).is_err() {
                    warn!("Output buffer full, dropping recovered packet {}", seq);
                }
            }
        }

        // Cleanup old length tracking
        if packet_count % 1000 == 0 {
            let cutoff = decoder
                .get_source(0)
                .map(|_| 0u16)
                .unwrap_or(0)
                .wrapping_sub(500);
            data_lengths.retain(|&seq, _| seq.wrapping_sub(cutoff) < 32768);
        }

        if packet_count % 5000 == 0 {
            info!(
                "Received {} packets ({} source, {} parity), recovered {}",
                packet_count, source_count, parity_count, recovered_count
            );
        }
    }
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let level = if args.verbose {
        Level::DEBUG
    } else {
        Level::INFO
    };

    tracing_subscriber::fmt()
        .with_max_level(level)
        .with_writer(std::io::stderr)
        .init();

    match args.command {
        Command::Send {
            listen,
            remote,
            delay,
            parities,
            packet_size,
        } => run_sender(listen, remote, delay, parities, packet_size).await?,
        Command::Recv {
            listen,
            delay,
            parities,
            packet_size,
        } => run_receiver(listen, delay, parities, packet_size).await?,
    }

    Ok(())
}
