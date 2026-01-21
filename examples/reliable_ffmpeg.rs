//! Simple H.264 streaming with FEC + NACK-based ARQ.
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

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use cm256::reliable::{RecvResult, ReliableConfig, SimpleSession};
use cm256::transport::{AsyncDatagramRecvMut, AsyncDatagramSendMut};
use rand::Rng;
use std::collections::BTreeMap;
use std::future::Future;
use std::io::{self, Read, Write};
use std::net::SocketAddr;
use std::pin::Pin;

use tokio::net::UdpSocket;
use tracing::{debug, info, warn, Level};

// ============================================================================
// Transport Wrapper (UDP with optional simulated loss)
// ============================================================================

struct UdpTransport {
    socket: UdpSocket,
    loss_percent: f64,
}

impl UdpTransport {
    fn new(socket: UdpSocket, loss_percent: f64) -> Self {
        Self {
            socket,
            loss_percent,
        }
    }

    fn should_drop(&self) -> bool {
        if self.loss_percent <= 0.0 {
            return false;
        }
        rand::thread_rng().gen_range(0.0..100.0) < self.loss_percent
    }
}

impl AsyncDatagramSendMut for UdpTransport {
    fn send_datagram_async<'a>(
        &'a mut self,
        data: &'a [u8],
    ) -> Pin<Box<dyn Future<Output = io::Result<usize>> + Send + 'a>> {
        Box::pin(async move {
            if self.should_drop() {
                return Ok(data.len()); // Simulate loss
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
                if !self.should_drop() {
                    return Ok(n);
                }
            }
        })
    }
}

// ============================================================================
// CLI
// ============================================================================

#[derive(Parser, Debug)]
#[command(name = "reliable-ffmpeg")]
#[command(about = "Stream H.264 with FEC + NACK-based ARQ")]
struct Args {
    #[command(subcommand)]
    command: Command,

    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand, Debug)]
enum Command {
    Send {
        #[arg(short, long, default_value = "0.0.0.0:9000")]
        listen: SocketAddr,

        #[arg(short, long, default_value = "127.0.0.1:9001")]
        remote: SocketAddr,

        #[arg(long, default_value = "0.0")]
        loss: f64,

        #[arg(long, default_value = "1200")]
        packet_size: usize,

        #[arg(long)]
        fec_delay: Option<u8>,

        #[arg(long)]
        fec_parities: Option<u8>,
    },
    Recv {
        #[arg(short, long, default_value = "0.0.0.0:9001")]
        listen: SocketAddr,

        #[arg(long, default_value = "0.0")]
        loss: f64,

        #[arg(long, default_value = "1200")]
        packet_size: usize,

        #[arg(long)]
        fec_delay: Option<u8>,

        #[arg(long)]
        fec_parities: Option<u8>,
    },
}

fn get_config(
    packet_size: usize,
    fec_delay: Option<u8>,
    fec_parities: Option<u8>,
) -> ReliableConfig {
    let mut config = ReliableConfig::default();
    config.symbol_bytes = packet_size;
    if let Some(d) = fec_delay {
        config.fec_delay = d;
    }
    if let Some(p) = fec_parities {
        config.fec_parities = p;
    }
    config
}

// ============================================================================
// Sender
// ============================================================================

async fn run_sender(
    listen: SocketAddr,
    remote: SocketAddr,
    loss_percent: f64,
    packet_size: usize,
    fec_delay: Option<u8>,
    fec_parities: Option<u8>,
) -> Result<()> {
    let config = get_config(packet_size, fec_delay, fec_parities);

    info!(
        "Sender: {} -> {}, loss={}%, FEC: delay={}, parities={} ({:.0}% overhead)",
        listen,
        remote,
        loss_percent,
        config.fec_delay,
        config.fec_parities,
        config.fec_parities as f64 / config.fec_step_size as f64 * 100.0
    );

    let socket = UdpSocket::bind(listen).await.context("bind failed")?;
    socket.connect(remote).await.context("connect failed")?;

    let transport = UdpTransport::new(socket, loss_percent);
    let mut session = SimpleSession::new(config.clone(), transport)?;

    // Channel for stdin
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<u8>>(256);

    // Stdin reader thread
    let pkt_size = packet_size;
    std::thread::spawn(move || {
        let stdin = io::stdin();
        let mut handle = stdin.lock();
        let mut buffer = vec![0u8; pkt_size];

        loop {
            let mut total = 0;
            while total < pkt_size {
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

    let mut count = 0u64;
    let start = std::time::Instant::now();

    loop {
        tokio::select! {
            biased;

            // Check for NACKs (non-blocking)
            nack_result = tokio::time::timeout(
                tokio::time::Duration::from_millis(1),
                session.recv()
            ) => {
                if let Ok(Ok(RecvResult::Nack)) = nack_result {
                    debug!("Processed NACK, retransmitted packets");
                }
            }

            // Send data
            data_opt = rx.recv() => {
                match data_opt {
                    Some(data) => {
                        let mut padded = data;
                        if padded.len() < packet_size {
                            padded.resize(packet_size, 0);
                        }

                        let seq = session.send(&padded).await?;
                        count += 1;
                        debug!("Sent seq={}", seq);

                        if count % 1000 == 0 {
                            let elapsed = start.elapsed().as_secs_f64();
                            info!("TX: {} pkts in {:.1}s ({:.1} pps)",
                                count, elapsed, count as f64 / elapsed);
                        }
                    }
                    None => break, // stdin closed
                }
            }
        }
    }

    session.flush().await?;
    info!("Sender done: {} packets", count);
    Ok(())
}

// ============================================================================
// Receiver with simple ordering
// ============================================================================

async fn run_receiver(
    listen: SocketAddr,
    loss_percent: f64,
    packet_size: usize,
    fec_delay: Option<u8>,
    fec_parities: Option<u8>,
) -> Result<()> {
    let config = get_config(packet_size, fec_delay, fec_parities);

    info!(
        "Receiver on {}, loss={}%, FEC: delay={}, parities={}",
        listen, loss_percent, config.fec_delay, config.fec_parities
    );

    let socket = UdpSocket::bind(listen).await.context("bind failed")?;
    info!("Waiting for connection...");

    // Wait for first packet
    let mut first_buf = vec![0u8; 65536];
    let (first_len, sender_addr) = socket.recv_from(&mut first_buf).await?;
    socket.connect(sender_addr).await?;
    info!("Connected to {}", sender_addr);

    let transport = UdpTransport::new(socket, loss_percent);
    let mut session = SimpleSession::new(config.clone(), transport)?;

    // Stdout writer
    let (write_tx, write_rx) = std::sync::mpsc::channel::<Vec<u8>>();
    std::thread::spawn(move || {
        let mut stdout = io::stdout().lock();
        while let Ok(data) = write_rx.recv() {
            let _ = stdout.write_all(&data);
            let _ = stdout.flush();
        }
    });

    let mut count = 0u64;
    let mut recovered = 0u64;
    let mut output_count = 0u64;
    let start = std::time::Instant::now();

    // Simple ordering: buffer out-of-order packets
    let mut reorder_buffer: BTreeMap<u16, Vec<u8>> = BTreeMap::new();
    let mut next_output_seq: Option<u16> = None;
    let max_reorder_buffer = 64; // Max packets to buffer before giving up on gaps

    // Helper to output in-order packets
    let mut output_packets = |reorder_buffer: &mut BTreeMap<u16, Vec<u8>>,
                              next_seq: &mut Option<u16>,
                              write_tx: &std::sync::mpsc::Sender<Vec<u8>>,
                              output_count: &mut u64| {
        loop {
            let seq = match next_seq {
                Some(s) => *s,
                None => return,
            };
            if let Some(data) = reorder_buffer.remove(&seq) {
                let _ = write_tx.send(data);
                *next_seq = Some(seq.wrapping_add(1));
                *output_count += 1;
            } else {
                break;
            }
        }
    };

    // Process first packet
    if first_len >= 8 && first_buf[2] == 0 {
        let seq = u16::from_le_bytes([first_buf[0], first_buf[1]]);
        let payload = first_buf[8..first_len].to_vec();
        let mut data = payload.clone();
        if data.len() < packet_size {
            data.resize(packet_size, 0);
        }
        session.decoder_mut().add_source(seq, &data);
        session.receiver_arq_mut().on_receive(seq);

        next_output_seq = Some(seq);
        reorder_buffer.insert(seq, payload);
        output_packets(
            &mut reorder_buffer,
            &mut next_output_seq,
            &write_tx,
            &mut output_count,
        );
        count += 1;
    }

    loop {
        match session.recv().await {
            Ok(RecvResult::Source {
                seq,
                data,
                recovered: is_recovered,
            }) => {
                count += 1;
                if is_recovered {
                    recovered += 1;
                    debug!("Recovered seq={}", seq);
                }

                // Initialize next_output_seq if not set
                if next_output_seq.is_none() {
                    next_output_seq = Some(seq);
                }

                // Add to reorder buffer
                reorder_buffer.insert(seq, data);

                // Output in-order packets
                output_packets(
                    &mut reorder_buffer,
                    &mut next_output_seq,
                    &write_tx,
                    &mut output_count,
                );

                // If buffer too large, skip ahead
                if reorder_buffer.len() > max_reorder_buffer {
                    if let Some(&first_seq) = reorder_buffer.keys().next() {
                        let gap = first_seq.wrapping_sub(next_output_seq.unwrap_or(0));
                        if gap > 0 && gap < 32768 {
                            warn!("Skipping {} missing packets (buffer full)", gap);
                            next_output_seq = Some(first_seq);
                            output_packets(
                                &mut reorder_buffer,
                                &mut next_output_seq,
                                &write_tx,
                                &mut output_count,
                            );
                        }
                    }
                }

                // Send NACK if we have gaps
                if session.should_nack() {
                    if let Err(e) = session.send_nack().await {
                        warn!("NACK failed: {}", e);
                    }
                }

                if count % 5000 == 0 {
                    let elapsed = start.elapsed().as_secs_f64();
                    info!(
                        "RX: {} pkts, {} recovered, {} output in {:.1}s (buffer={})",
                        count,
                        recovered,
                        output_count,
                        elapsed,
                        reorder_buffer.len()
                    );
                }
            }
            Ok(RecvResult::Parity) => {
                // Send NACK if we have gaps
                if session.should_nack() {
                    let _ = session.send_nack().await;
                }
            }
            Ok(RecvResult::Nack) => {
                debug!("Received NACK (unexpected on receiver)");
            }
            Ok(RecvResult::WouldBlock) => {
                tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
            }
            Err(e) => {
                warn!("Recv error: {}", e);
            }
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
            packet_size,
            fec_delay,
            fec_parities,
        } => {
            run_sender(listen, remote, loss, packet_size, fec_delay, fec_parities).await?;
        }
        Command::Recv {
            listen,
            loss,
            packet_size,
            fec_delay,
            fec_parities,
        } => {
            run_receiver(listen, loss, packet_size, fec_delay, fec_parities).await?;
        }
    }

    Ok(())
}
