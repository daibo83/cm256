//! FFmpeg H.264 streaming with Reliable Transport (FEC + ARQ)
//!
//! This example demonstrates the reliable transport module which combines:
//! - Streaming FEC for instant recovery (~95% of losses)
//! - ARQ retransmission for guaranteed delivery
//! - BBR congestion control
//! - Adaptive FEC tuning
//!
//! Target latency: 50ms + RTT
//!
//! # Usage
//!
//! ## Receiver (start first):
//! ```bash
//! cargo run --release --example reliable_ffmpeg -- recv | ffplay -f h264 -
//! ```
//!
//! ## Sender:
//! ```bash
//! ffmpeg -re -i input.mp4 -c:v libx264 -tune zerolatency -f h264 - | \
//!     cargo run --release --example reliable_ffmpeg -- send
//! ```
//!
//! ## With simulated packet loss (for testing):
//! ```bash
//! # Receiver with 10% simulated loss
//! cargo run --release --example reliable_ffmpeg -- recv --loss 10 | ffplay -f h264 -
//! ```
//!
//! # Key Features
//!
//! - **Guaranteed delivery**: Unlike pure FEC, ARQ ensures 100% delivery
//! - **Low latency**: FEC recovers most losses instantly, ARQ only for edge cases
//! - **Adaptive**: FEC overhead adjusts to network conditions
//! - **Stats**: Real-time network statistics (RTT, loss rate, bandwidth)

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use cm256::reliable::{AsyncReliableDecoder, AsyncReliableEncoder, RecvResult, ReliableConfig};
use cm256::transport::{AsyncDatagramRecvMut, AsyncDatagramSendMut};
use std::collections::BTreeMap;
use std::future::Future;
use std::io::{self, Read, Write};
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::net::UdpSocket;
use tracing::{debug, info, warn, Level};

// ============================================================================
// Transport Wrapper (UDP with optional simulated loss)
// ============================================================================

/// UDP socket wrapper with optional packet loss simulation
struct UdpTransport {
    socket: UdpSocket,
    /// Simulated loss percentage (0-100)
    loss_percent: u8,
    /// Packet counter for deterministic loss
    send_counter: AtomicU64,
    recv_counter: AtomicU64,
}

impl UdpTransport {
    fn new(socket: UdpSocket, loss_percent: u8) -> Self {
        Self {
            socket,
            loss_percent,
            send_counter: AtomicU64::new(0),
            recv_counter: AtomicU64::new(0),
        }
    }

    fn should_drop(&self, counter: u64) -> bool {
        if self.loss_percent == 0 {
            return false;
        }
        // Use deterministic hash for reproducible loss pattern
        let hash = counter.wrapping_mul(0x9E3779B97F4A7C15);
        let roll = ((hash >> 56) as u8) % 100;
        roll < self.loss_percent
    }
}

impl AsyncDatagramSendMut for UdpTransport {
    fn send_datagram_async<'a>(
        &'a mut self,
        data: &'a [u8],
    ) -> Pin<Box<dyn Future<Output = io::Result<usize>> + Send + 'a>> {
        Box::pin(async move {
            let count = self.send_counter.fetch_add(1, Ordering::Relaxed);
            if self.should_drop(count) {
                // Simulate loss: pretend we sent it
                return Ok(data.len());
            }
            self.socket.send(data).await
        })
    }
}

impl AsyncDatagramRecvMut for UdpTransport {
    fn recv_datagram_async<'a>(
        &'a mut self,
        buf: &'a mut [u8],
    ) -> Pin<Box<dyn Future<Output = io::Result<usize>> + Send + 'a>> {
        Box::pin(async move {
            loop {
                let n = self.socket.recv(buf).await?;
                let count = self.recv_counter.fetch_add(1, Ordering::Relaxed);
                if !self.should_drop(count) {
                    return Ok(n);
                }
                // Dropped - receive next packet
            }
        })
    }
}

// ============================================================================
// CLI
// ============================================================================

#[derive(Parser, Debug)]
#[command(name = "reliable-ffmpeg")]
#[command(about = "Stream H.264 video with reliable FEC+ARQ transport")]
struct Args {
    #[command(subcommand)]
    command: Command,

    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Send H.264 from stdin over network with reliable transport
    Send {
        /// Local address to bind to
        #[arg(short, long, default_value = "0.0.0.0:9000")]
        listen: SocketAddr,

        /// Remote address to send to
        #[arg(short, long, default_value = "127.0.0.1:9001")]
        remote: SocketAddr,

        /// Simulated packet loss percentage (0-100)
        #[arg(long, default_value = "0")]
        loss: u8,

        /// Config preset: default, low-latency, high-throughput, high-rtt
        #[arg(long, default_value = "default")]
        preset: String,

        /// Symbol/packet size in bytes
        #[arg(long, default_value = "1200")]
        packet_size: usize,
    },
    /// Receive stream and output H.264 to stdout
    Recv {
        /// Local address to bind to
        #[arg(short, long, default_value = "0.0.0.0:9001")]
        listen: SocketAddr,

        /// Simulated packet loss percentage (0-100)
        #[arg(long, default_value = "0")]
        loss: u8,

        /// Config preset: default, low-latency, high-throughput, high-rtt
        #[arg(long, default_value = "default")]
        preset: String,

        /// Symbol/packet size in bytes (must match sender)
        #[arg(long, default_value = "1200")]
        packet_size: usize,
    },
}

fn get_config(preset: &str, packet_size: usize) -> Result<ReliableConfig> {
    let mut config = match preset {
        "default" => ReliableConfig::default(),
        "low-latency" => ReliableConfig::low_latency(),
        "high-throughput" => ReliableConfig::high_throughput(),
        "high-rtt" => ReliableConfig::high_rtt(),
        _ => anyhow::bail!(
            "Unknown preset: {}. Use: default, low-latency, high-throughput, high-rtt",
            preset
        ),
    };
    config.symbol_bytes = packet_size;
    Ok(config)
}

// ============================================================================
// Sender
// ============================================================================

async fn run_sender(
    listen: SocketAddr,
    remote: SocketAddr,
    loss_percent: u8,
    preset: String,
    packet_size: usize,
) -> Result<()> {
    let config = get_config(&preset, packet_size)?;

    info!(
        "Reliable Sender: {} -> {}, preset={}, packet_size={}, simulated_loss={}%",
        listen, remote, preset, packet_size, loss_percent
    );
    info!(
        "  FEC: delay={}, parities={}, step={}",
        config.fec_delay, config.fec_parities, config.fec_step_size
    );
    info!(
        "  ARQ: window={}, buffer={}, max_retries={}",
        config.window_size, config.send_buffer_size, config.max_retries
    );
    info!(
        "  ACK: every {} pkts or {}ms, min interval {}ms",
        config.ack_every_n_packets, config.ack_interval_ms, config.min_ack_interval_ms
    );

    // Create UDP socket
    let socket = UdpSocket::bind(listen)
        .await
        .with_context(|| format!("Failed to bind to {}", listen))?;
    socket
        .connect(remote)
        .await
        .with_context(|| format!("Failed to connect to {}", remote))?;

    info!("UDP socket bound to {}, connected to {}", listen, remote);

    // Wrap with loss simulation
    let transport = UdpTransport::new(socket, loss_percent);

    // Create reliable encoder
    let mut encoder = AsyncReliableEncoder::new(config.clone(), transport)
        .context("Failed to create reliable encoder")?;

    // Channel for stdin data
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
    let mut total_bytes = 0u64;
    let start_time = std::time::Instant::now();

    while let Some(data) = rx.recv().await {
        let actual_len = data.len();

        // Pad to packet_size
        let mut padded = data;
        if padded.len() < packet_size {
            padded.resize(packet_size, 0);
        }

        // Send with FEC protection
        let seq = encoder.send(&padded).await?;
        source_count += 1;
        total_bytes += actual_len as u64;

        debug!("Sent seq={}, len={}", seq, actual_len);

        // Pace sending slightly
        if source_count % 4 == 0 {
            tokio::time::sleep(tokio::time::Duration::from_micros(50)).await;
        }

        // Print stats periodically
        if source_count % 1000 == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let stats = encoder.stats();
            let bbr = encoder.bbr();

            info!(
                "TX: {} pkts, {:.1} MB, {:.1} Mbps | Loss: {:.1}% | RTT: {:.0}ms | BW: {:.1} Mbps | CWND: {}",
                source_count,
                total_bytes as f64 / 1_000_000.0,
                (total_bytes as f64 * 8.0) / elapsed / 1_000_000.0,
                stats.loss_rate() * 100.0,
                stats.avg_rtt_ms(),
                bbr.btl_bw() as f64 * 8.0 / 1_000_000.0,
                bbr.cwnd(),
            );
        }
    }

    // Flush remaining parities
    encoder.flush().await?;

    let elapsed = start_time.elapsed().as_secs_f64();
    info!(
        "Sender done: {} packets, {} bytes in {:.1}s ({:.1} Mbps)",
        source_count,
        total_bytes,
        elapsed,
        (total_bytes as f64 * 8.0) / elapsed / 1_000_000.0
    );

    Ok(())
}

// ============================================================================
// Receiver
// ============================================================================

async fn run_receiver(
    listen: SocketAddr,
    loss_percent: u8,
    preset: String,
    packet_size: usize,
) -> Result<()> {
    let config = get_config(&preset, packet_size)?;

    info!(
        "Reliable Receiver on {}, preset={}, packet_size={}, simulated_loss={}%",
        listen, preset, packet_size, loss_percent
    );

    // Create UDP socket
    let socket = UdpSocket::bind(listen)
        .await
        .with_context(|| format!("Failed to bind to {}", listen))?;

    info!("Waiting for connection...");

    // Wait for first packet to get sender address AND save the packet
    let mut first_buf = vec![0u8; 65536];
    let (first_len, sender_addr) = socket.recv_from(&mut first_buf).await?;
    socket.connect(sender_addr).await?;

    info!("Connected to sender at {}", sender_addr);

    // Wrap with loss simulation
    let transport = UdpTransport::new(socket, loss_percent);

    // Create reliable decoder
    let mut decoder = AsyncReliableDecoder::new(config.clone(), transport)
        .context("Failed to create reliable decoder")?;

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

    // Ordering task - ensures packets are output in sequence order
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

            // Output packets in order
            while let Some(data) = output_queue.remove(&next_seq) {
                if write_tx.send(data).is_err() {
                    return;
                }
                next_seq = next_seq.wrapping_add(1);
            }

            // Handle queue overflow (skip gaps if too many buffered)
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
    let mut recovered_count = 0u64;
    let start_time = std::time::Instant::now();

    // Process the first packet that we already received
    // Parse the reliable transport header (8 bytes) and extract payload
    if first_len >= 8 {
        // Header: seq (2) + type (1) + reserved (5) = 8 bytes
        let seq = u16::from_le_bytes([first_buf[0], first_buf[1]]);
        let pkt_type = first_buf[2];

        // Type 0 = Source packet
        if pkt_type == 0 {
            let payload = first_buf[8..first_len].to_vec();
            debug!("Injected first packet: seq={}, len={}", seq, payload.len());
            packet_count += 1;
            source_count += 1;

            // Also add to FEC decoder for recovery purposes
            let mut padded = payload.clone();
            if padded.len() < packet_size {
                padded.resize(packet_size, 0);
            }
            decoder.fec_mut().add_source(seq, &padded);
            decoder.arq_mut().on_receive(seq);

            if output_tx.try_send((seq, payload)).is_err() {
                warn!("Output buffer full, dropping first packet");
            }
        }
    }

    loop {
        let result = decoder.recv().await;

        match result {
            Ok(RecvResult::Source {
                seq,
                data,
                recovered,
            }) => {
                packet_count += 1;
                source_count += 1;
                if recovered {
                    recovered_count += 1;
                    debug!("Recovered seq={}", seq);
                } else {
                    debug!("Received seq={}, len={}", seq, data.len());
                }

                if output_tx.try_send((seq, data)).is_err() {
                    warn!("Output buffer full, dropping packet {}", seq);
                }
            }
            Ok(RecvResult::Parity) => {
                packet_count += 1;
                debug!("Received parity packet");
            }
            Ok(RecvResult::Ack) => {
                // Shouldn't happen on receiver side
                debug!("Received ACK (unexpected)");
            }
            Ok(RecvResult::WouldBlock) => {
                // Non-blocking mode returned no packet
                tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
            }
            Err(e) => {
                warn!("Receive error: {}", e);
                continue;
            }
        }

        // Print stats periodically
        if packet_count % 5000 == 0 && packet_count > 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let stats = decoder.stats();

            info!(
                "RX: {} pkts ({} src, {} recovered) | {:.1}s | Loss: {:.1}% | RTT: {:.0}ms",
                packet_count,
                source_count,
                recovered_count,
                elapsed,
                stats.loss_rate() * 100.0,
                stats.avg_rtt_ms(),
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
            loss,
            preset,
            packet_size,
        } => {
            run_sender(listen, remote, loss, preset, packet_size).await?;
        }
        Command::Recv {
            listen,
            loss,
            preset,
            packet_size,
        } => {
            run_receiver(listen, loss, preset, packet_size).await?;
        }
    }

    Ok(())
}
