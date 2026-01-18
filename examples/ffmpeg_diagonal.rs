//! FFmpeg H.264 streaming with True Diagonal Interleaving FEC
//!
//! This example uses the diagonal interleaving FEC module for low-latency video streaming.
//! Each parity is a simple XOR of two source packets separated by τ (tau) positions.
//!
//! # Usage
//!
//! Receiver (start first):
//! ```bash
//! cargo run --release --example ffmpeg_diagonal -- recv | ffplay -f h264 -
//! ```
//!
//! Sender:
//! ```bash
//! ffmpeg -re -i input.mp4 -c:v libx264 -tune zerolatency -f h264 - | \
//!     cargo run --release --example ffmpeg_diagonal -- send
//! ```
//!
//! # How Diagonal Interleaving Works
//!
//! ```text
//! P_t = S_t ⊕ S_{t-τ} ⊕ S_{t-2τ} ⊕ ... ⊕ S_{t-(span-1)τ}
//! ```
//!
//! Each parity XORs `span` source packets separated by τ positions.
//! - span=2: 100% overhead (original behavior, one parity per source)
//! - span=6: ~20% overhead (one parity per 5 sources)
//!
//! This creates diagonal "stripes" of protection that can recover burst losses.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use cm256::diagonal::{DiagonalDecoder, DiagonalEncoder, DiagonalParams, ReceivedDiagonalParity};
use std::collections::BTreeMap;
use std::io::{self, Read, Write};
use std::net::SocketAddr;
use tokio::net::UdpSocket;
use tracing::{debug, info, warn, Level};

// ============================================================================
// Packet Format
// ============================================================================

/// Packet header for diagonal FEC
///
/// Format:
/// - bytes 0-1: sequence number (u16 LE)
/// - byte 2: packet type (0 = source, 1 = parity)
/// - byte 3: depth index (for parity) or reserved (for source)
/// - bytes 4-5: depth τ (for parity) or data length (for source)
/// - byte 6: span (for parity) or reserved (for source)
/// - byte 7: reserved
#[derive(Debug, Clone, Copy)]
struct PacketHeader {
    seq: u16,
    is_parity: bool,
    depth_index: u8,
    depth: u16,
    span: u8,
    data_len: u16,
}

impl PacketHeader {
    const SIZE: usize = 8;

    fn source(seq: u16, data_len: u16) -> Self {
        Self {
            seq,
            is_parity: false,
            depth_index: 0,
            depth: 0,
            span: 0,
            data_len,
        }
    }

    fn parity(seq: u16, depth_index: u8, depth: u16, span: u8) -> Self {
        Self {
            seq,
            is_parity: true,
            depth_index,
            depth,
            span,
            data_len: 0,
        }
    }

    fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..2].copy_from_slice(&self.seq.to_le_bytes());
        buf[2] = if self.is_parity { 1 } else { 0 };
        buf[3] = self.depth_index;
        if self.is_parity {
            buf[4..6].copy_from_slice(&self.depth.to_le_bytes());
            buf[6] = self.span;
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
        let depth_index = buf[3];
        let field = u16::from_le_bytes([buf[4], buf[5]]);
        let span = buf[6];

        Some(Self {
            seq,
            is_parity,
            depth_index,
            depth: if is_parity { field } else { 0 },
            span: if is_parity { span } else { 0 },
            data_len: if is_parity { 0 } else { field },
        })
    }
}

// ============================================================================
// CLI
// ============================================================================

#[derive(Parser, Debug)]
#[command(name = "ffmpeg-diagonal")]
#[command(about = "Stream H.264 video with diagonal interleaving FEC")]
struct Args {
    #[command(subcommand)]
    command: Command,

    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Send H.264 from stdin over network with diagonal FEC
    Send {
        /// Local address to bind to
        #[arg(short, long, default_value = "0.0.0.0:9000")]
        listen: SocketAddr,

        /// Remote address to send to
        #[arg(short, long, default_value = "127.0.0.1:9001")]
        remote: SocketAddr,

        /// Interleaving depths (comma-separated). Each creates one parity stream.
        /// Example: "8" for simple, "4,8,16" for extended
        #[arg(long, default_value = "8", value_delimiter = ',')]
        depths: Vec<u16>,

        /// Span: number of sources XORed per parity. Controls overhead.
        /// span=2: 100% overhead, span=6: 20% overhead
        #[arg(long, default_value = "2")]
        span: u8,

        /// Packet size in bytes
        #[arg(long, default_value = "1200")]
        packet_size: usize,
    },
    /// Receive stream and output H.264 to stdout
    Recv {
        /// Local address to bind to
        #[arg(short, long, default_value = "0.0.0.0:9001")]
        listen: SocketAddr,

        /// Interleaving depths (must match sender)
        #[arg(long, default_value = "8", value_delimiter = ',')]
        depths: Vec<u16>,

        /// Span (must match sender)
        #[arg(long, default_value = "2")]
        span: u8,

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
    depths: Vec<u16>,
    span: u8,
    packet_size: usize,
) -> Result<()> {
    let params = if depths.len() == 1 {
        DiagonalParams::with_span(depths[0], span, packet_size)
    } else {
        DiagonalParams::extended_with_span(&depths, span, packet_size)
    }
    .context("Invalid diagonal parameters")?;

    info!(
        "Diagonal Sender: {} -> {}, depths={:?}, span={}, packet_size={}",
        addr, remote, depths, span, packet_size
    );
    info!(
        "  Max burst recovery: {} packets, overhead: {:.0}%",
        params.max_burst(),
        params.overhead() * 100.0
    );

    let socket = UdpSocket::bind(addr)
        .await
        .with_context(|| format!("Failed to bind to {}", addr))?;

    let mut encoder = DiagonalEncoder::new(params);
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<u8>>(256);

    // Stdin reader thread
    let pkt_size = packet_size;
    std::thread::spawn(move || {
        let stdin = io::stdin();
        let mut handle = stdin.lock();
        let mut buffer = vec![0u8; pkt_size];

        loop {
            let mut total_read = 0;
            while total_read < pkt_size {
                match handle.read(&mut buffer[total_read..]) {
                    Ok(0) => {
                        if total_read > 0 {
                            let _ = tx.blocking_send(buffer[..total_read].to_vec());
                        }
                        return;
                    }
                    Ok(n) => total_read += n,
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

        // Pad to packet_size
        let mut padded = data;
        if padded.len() < packet_size {
            padded.resize(packet_size, 0);
        }

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
            let header =
                PacketHeader::parity(parity.seq, parity.depth_index, parity.depth, parity.span);
            let mut packet = Vec::with_capacity(PacketHeader::SIZE + packet_size);
            packet.extend_from_slice(&header.to_bytes());
            packet.extend_from_slice(&parity.data);
            socket.send_to(&packet, remote).await?;
            parity_count += 1;
        }

        if source_count % 4 == 0 {
            tokio::time::sleep(tokio::time::Duration::from_micros(50)).await;
        }

        if source_count % 1000 == 0 {
            info!(
                "Sent {} sources, {} parities ({:.0}% overhead), {} bytes",
                source_count,
                parity_count,
                (parity_count as f64 / source_count as f64) * 100.0,
                total_bytes
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

async fn run_receiver(
    addr: SocketAddr,
    depths: Vec<u16>,
    span: u8,
    packet_size: usize,
) -> Result<()> {
    let params = if depths.len() == 1 {
        DiagonalParams::with_span(depths[0], span, packet_size)
    } else {
        DiagonalParams::extended_with_span(&depths, span, packet_size)
    }
    .context("Invalid diagonal parameters")?;

    info!(
        "Diagonal Receiver on {}, depths={:?}, span={}, packet_size={}",
        addr, depths, span, packet_size
    );

    let socket = UdpSocket::bind(addr)
        .await
        .with_context(|| format!("Failed to bind to {}", addr))?;

    let mut decoder = DiagonalDecoder::new(params);
    decoder.set_history_size(1000);

    let mut recv_buffer = vec![0u8; 65536];

    // Output channels
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

    // Ordering task
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

            while let Some(data) = output_queue.remove(&next_seq) {
                if write_tx.send(data).is_err() {
                    return;
                }
                next_seq = next_seq.wrapping_add(1);
            }

            if output_queue.len() > 200 {
                if let Some(&first_seq) = output_queue.keys().next() {
                    let gap = first_seq.wrapping_sub(next_seq);
                    if gap < 32768 {
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
            continue;
        }

        if header.is_parity {
            decoder.add_parity(ReceivedDiagonalParity {
                seq: header.seq,
                depth_index: header.depth_index,
                depth: header.depth,
                span: header.span,
                data: payload.to_vec(),
            });
            parity_count += 1;
            debug!("RX parity: seq={}, depth={}", header.seq, header.depth);
        } else {
            decoder.add_source(header.seq, payload);
            data_lengths.insert(header.seq, header.data_len);
            source_count += 1;
            debug!("RX source: seq={}, len={}", header.seq, header.data_len);

            // Output immediately
            let actual_len = header.data_len as usize;
            let data = payload[..actual_len.min(packet_size)].to_vec();
            if output_tx.try_send((header.seq, data)).is_err() {
                warn!("Output buffer full, dropping packet {}", header.seq);
            }
        }

        // Try recovery
        if packet_count % 2 == 0 {
            let recovered = decoder.try_recover();
            for (seq, data) in recovered {
                recovered_count += 1;
                let actual_len = data_lengths
                    .get(&seq)
                    .copied()
                    .unwrap_or(packet_size as u16) as usize;
                let trimmed = data[..actual_len.min(packet_size)].to_vec();
                debug!("Recovered seq={}", seq);
                if output_tx.try_send((seq, trimmed)).is_err() {
                    warn!("Output buffer full, dropping recovered {}", seq);
                }
            }
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
            depths,
            span,
            packet_size,
        } => run_sender(listen, remote, depths, span, packet_size).await?,
        Command::Recv {
            listen,
            depths,
            span,
            packet_size,
        } => run_receiver(listen, depths, span, packet_size).await?,
    }

    Ok(())
}
