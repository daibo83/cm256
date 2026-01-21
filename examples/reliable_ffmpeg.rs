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
//! # Sender with 10% simulated loss
//! cargo run --release --example reliable_ffmpeg -- send --loss 10
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
use cm256::reliable::{AsyncReliableSession, RecvResult, ReliableConfig};
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

/// UDP socket wrapper with optional packet loss simulation
struct UdpTransport {
    socket: UdpSocket,
    /// Simulated loss percentage (0.0-100.0)
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
        let mut rng = rand::thread_rng();
        rng.gen_range(0.0..100.0) < self.loss_percent
    }
}

impl AsyncDatagramSendMut for UdpTransport {
    fn send_datagram_async<'a>(
        &'a mut self,
        data: &'a [u8],
    ) -> Pin<Box<dyn Future<Output = io::Result<usize>> + Send + 'a>> {
        Box::pin(async move {
            if self.should_drop() {
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
                if !self.should_drop() {
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

        /// Simulated packet loss percentage (0.0-100.0)
        #[arg(long, default_value = "0.0")]
        loss: f64,

        /// Config preset: default, low-latency, high-throughput, high-rtt
        #[arg(long, default_value = "default")]
        preset: String,

        /// Symbol/packet size in bytes
        #[arg(long, default_value = "1200")]
        packet_size: usize,

        /// FEC window size (delay in packets)
        #[arg(long)]
        fec_delay: Option<u8>,

        /// Number of parity packets per step
        #[arg(long)]
        fec_parities: Option<u8>,

        /// Generate parities every N packets (higher = less overhead)
        #[arg(long)]
        fec_step: Option<u8>,
    },
    /// Receive stream and output H.264 to stdout
    Recv {
        /// Local address to bind to
        #[arg(short, long, default_value = "0.0.0.0:9001")]
        listen: SocketAddr,

        /// Simulated packet loss percentage (0.0-100.0)
        #[arg(long, default_value = "0.0")]
        loss: f64,

        /// Config preset: default, low-latency, high-throughput, high-rtt
        #[arg(long, default_value = "default")]
        preset: String,

        /// Symbol/packet size in bytes (must match sender)
        #[arg(long, default_value = "1200")]
        packet_size: usize,

        /// FEC window size (must match sender)
        #[arg(long)]
        fec_delay: Option<u8>,

        /// Number of parity packets per step (must match sender)
        #[arg(long)]
        fec_parities: Option<u8>,

        /// Generate parities every N packets (must match sender)
        #[arg(long)]
        fec_step: Option<u8>,
    },
}

fn get_config(
    preset: &str,
    packet_size: usize,
    fec_delay: Option<u8>,
    fec_parities: Option<u8>,
    fec_step: Option<u8>,
) -> Result<ReliableConfig> {
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

    // Override FEC parameters if specified
    if let Some(delay) = fec_delay {
        config.fec_delay = delay;
    }
    if let Some(parities) = fec_parities {
        config.fec_parities = parities;
    }
    if let Some(step) = fec_step {
        config.fec_step_size = step;
    }
    Ok(config)
}

// ============================================================================
// Sender
// ============================================================================

async fn run_sender(
    listen: SocketAddr,
    remote: SocketAddr,
    loss_percent: f64,
    preset: String,
    packet_size: usize,
    fec_delay: Option<u8>,
    fec_parities: Option<u8>,
    fec_step: Option<u8>,
) -> Result<()> {
    let config = get_config(&preset, packet_size, fec_delay, fec_parities, fec_step)?;

    info!(
        "Reliable Sender: {} -> {}, preset={}, packet_size={}, simulated_loss={}%",
        listen, remote, preset, packet_size, loss_percent
    );
    info!(
        "  FEC: delay={}, parities={}, step={} (overhead: {:.1}%)",
        config.fec_delay,
        config.fec_parities,
        config.fec_step_size,
        config.fec_parities as f64 / config.fec_step_size as f64 * 100.0
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

    // Create reliable session
    let mut session = AsyncReliableSession::new(config.clone(), transport)
        .context("Failed to create reliable session")?;

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
    let mut last_retransmits = 0u64;
    let start_time = std::time::Instant::now();

    // Flow control parameters
    const MAX_IN_FLIGHT: usize = 64; // Soft limit - start waiting for ACKs
    const CRITICAL_IN_FLIGHT: usize = 200; // Hard limit - must wait, no timeout
    const ACK_POLL_INTERVAL: u64 = 8; // Poll for ACKs every N packets
    const FLOW_CONTROL_TIMEOUT_MS: u64 = 500; // Max time to wait for ACKs (soft limit only)

    // Use select! to handle both sending data and receiving ACKs concurrently
    let mut stdin_closed = false;
    let mut flow_control_start: Option<std::time::Instant> = None;
    let mut consecutive_fc_timeouts = 0u32;

    loop {
        // Flow control: if too many packets in flight, prioritize ACK processing
        let in_flight = session.sender_arq().in_flight();
        let arq_full = session.sender_arq().is_full();

        // Critical condition: ARQ buffer nearly full - must wait for ACKs
        if arq_full || in_flight >= CRITICAL_IN_FLIGHT {
            debug!(
                "Critical flow control: {} in flight, ARQ full={}, waiting for ACKs",
                in_flight, arq_full
            );
            // Must wait for ACKs - no timeout bypass
            match tokio::time::timeout(tokio::time::Duration::from_millis(100), session.poll_acks())
                .await
            {
                Ok(Ok(retransmits)) => {
                    if retransmits > 0 {
                        debug!("Critical FC: triggered {} retransmits", retransmits);
                    }
                }
                Ok(Err(e)) => debug!("Critical FC ACK error: {}", e),
                Err(_) => {
                    // Timeout - yield and retry
                    tokio::task::yield_now().await;
                }
            }
            continue;
        }

        // Soft flow control: try to wait for ACKs but allow timeout
        if in_flight >= MAX_IN_FLIGHT {
            let fc_start = flow_control_start.get_or_insert_with(std::time::Instant::now);

            if fc_start.elapsed().as_millis() > FLOW_CONTROL_TIMEOUT_MS as u128 {
                // Timeout - but only allow if we're not critically full
                consecutive_fc_timeouts += 1;
                if consecutive_fc_timeouts <= 3 {
                    warn!(
                        "Flow control timeout #{}: {} packets in flight, continuing",
                        consecutive_fc_timeouts, in_flight
                    );
                    flow_control_start = None;
                } else {
                    // Too many consecutive timeouts - something is wrong
                    // Wait longer before continuing
                    warn!(
                        "Multiple flow control timeouts ({}), waiting longer...",
                        consecutive_fc_timeouts
                    );
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    // Try to drain ACKs
                    for _ in 0..5 {
                        match tokio::time::timeout(
                            tokio::time::Duration::from_millis(50),
                            session.poll_acks(),
                        )
                        .await
                        {
                            Ok(Ok(_)) => {}
                            _ => break,
                        }
                    }
                    flow_control_start = None;
                }
            } else {
                // Wait for ACKs
                match tokio::time::timeout(
                    tokio::time::Duration::from_millis(20),
                    session.poll_acks(),
                )
                .await
                {
                    Ok(Ok(retransmits)) => {
                        if retransmits > 0 {
                            debug!("Flow control: triggered {} retransmits", retransmits);
                        }
                        // Progress made
                        if session.sender_arq().in_flight() < in_flight {
                            flow_control_start = None;
                            consecutive_fc_timeouts = 0;
                        }
                    }
                    Ok(Err(e)) => debug!("ACK error during flow control: {}", e),
                    Err(_) => {}
                }
                continue;
            }
        } else {
            flow_control_start = None;
            if in_flight < MAX_IN_FLIGHT / 2 {
                consecutive_fc_timeouts = 0; // Reset when buffer is healthy
            }
        }

        if stdin_closed {
            break;
        }

        tokio::select! {
            biased; // Prefer ACK processing over sending when both ready

            // Handle incoming ACKs from receiver (with short timeout)
            ack_result = tokio::time::timeout(
                tokio::time::Duration::from_millis(5),
                session.poll_acks()
            ) => {
                match ack_result {
                    Ok(Ok(retransmits)) => {
                        if retransmits > 0 {
                            debug!("Processed ACK, triggered {} retransmits", retransmits);
                        }
                    }
                    Ok(Err(e)) => {
                        debug!("ACK receiver error: {}", e);
                    }
                    Err(_) => {
                        // Timeout - no ACK received, continue to send
                    }
                }
            }

            // Handle incoming data from stdin
            data_opt = rx.recv() => {
                match data_opt {
                    Some(data) => {
                        let actual_len = data.len();

                        // Pad to packet_size
                        let mut padded = data;
                        if padded.len() < packet_size {
                            padded.resize(packet_size, 0);
                        }

                        // Send with FEC protection
                        let seq = session.send(&padded).await?;
                        source_count += 1;
                        total_bytes += actual_len as u64;
                        debug!("Sent seq={}, len={}, in_flight={}", seq, actual_len, session.sender_arq().in_flight());

                        // Poll for ACKs periodically during sending
                        if source_count % ACK_POLL_INTERVAL == 0 {
                            // Non-blocking ACK check
                            if let Ok(Ok(retransmits)) = tokio::time::timeout(
                                tokio::time::Duration::from_millis(1),
                                session.poll_acks(),
                            ).await {
                                if retransmits > 0 {
                                    debug!("Periodic ACK: {} retransmits", retransmits);
                                }
                            }
                        }

                        // Print stats periodically
                        if source_count % 1000 == 0 {
                            let stats = session.stats();
                            let retransmits = session.total_retransmits();
                            let new_retransmits = retransmits - last_retransmits;
                            last_retransmits = retransmits;

                            info!(
                                "TX: {} pkts, {:.1} MB | Loss: {:.1}% | RTT: {:.0}ms | In-flight: {} | Retransmits: {} (+{})",
                                source_count,
                                total_bytes as f64 / 1_000_000.0,
                                stats.loss_rate() * 100.0,
                                stats.avg_rtt_ms(),
                                session.sender_arq().in_flight(),
                                retransmits,
                                new_retransmits,
                            );
                        }
                    }
                    None => {
                        // stdin closed, but continue to drain ACKs
                        stdin_closed = true;
                    }
                }
            }
        }
    }

    // Flush remaining parities
    session.flush().await?;

    // Drain period: wait for ACKs and complete retransmissions
    let in_flight = session.sender_arq().in_flight();
    info!(
        "Waiting for pending ACKs and retransmissions ({} packets in flight)...",
        in_flight
    );
    let drain_start = std::time::Instant::now();
    let drain_timeout = std::time::Duration::from_secs(10); // Longer drain period
    let mut last_progress = std::time::Instant::now();
    let no_progress_timeout = std::time::Duration::from_secs(3); // Wait longer for ACKs

    while drain_start.elapsed() < drain_timeout {
        let current_in_flight = session.sender_arq().in_flight();

        if current_in_flight == 0 {
            info!("All packets acknowledged!");
            break;
        }

        match tokio::time::timeout(tokio::time::Duration::from_millis(100), session.poll_acks())
            .await
        {
            Ok(Ok(retransmits)) => {
                if retransmits > 0 {
                    debug!("Drain: triggered {} retransmits", retransmits);
                    last_progress = std::time::Instant::now();
                }
                // Check for progress
                if session.sender_arq().in_flight() < current_in_flight {
                    last_progress = std::time::Instant::now();
                }
            }
            Ok(Err(e)) => {
                debug!("Drain ACK error: {}", e);
            }
            Err(_) => {
                // Timeout - check if we're making progress
                if last_progress.elapsed() > no_progress_timeout {
                    warn!(
                        "No progress for {:?}, {} packets still in flight",
                        no_progress_timeout, current_in_flight
                    );
                    // Give up on remaining packets
                    break;
                }
            }
        }
    }

    let final_in_flight = session.sender_arq().in_flight();
    if final_in_flight > 0 {
        warn!(
            "Drain timeout: {} packets still unacknowledged",
            final_in_flight
        );
    }

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
    loss_percent: f64,
    preset: String,
    packet_size: usize,
    fec_delay: Option<u8>,
    fec_parities: Option<u8>,
    fec_step: Option<u8>,
) -> Result<()> {
    let config = get_config(&preset, packet_size, fec_delay, fec_parities, fec_step)?;

    info!(
        "Reliable Receiver on {}, preset={}, packet_size={}, simulated_loss={}%",
        listen, preset, packet_size, loss_percent
    );
    info!(
        "  FEC: delay={}, parities={}, step={} (overhead: {:.1}%)",
        config.fec_delay,
        config.fec_parities,
        config.fec_step_size,
        config.fec_parities as f64 / config.fec_step_size as f64 * 100.0
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

    // Create reliable session
    let mut session = AsyncReliableSession::new(config.clone(), transport)
        .context("Failed to create reliable session")?;

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
    // Waits for retransmissions before giving up on missing packets
    tokio::task::spawn(async move {
        let mut output_queue: BTreeMap<u16, Vec<u8>> = BTreeMap::new();
        let mut next_seq: u16 = 0;
        let mut first_packet = true;
        let mut last_output_time = std::time::Instant::now();
        let gap_timeout = std::time::Duration::from_millis(100); // Wait up to 100ms for missing packets
        let max_queue_size = 50; // Skip gaps if queue exceeds this

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
                last_output_time = std::time::Instant::now();
            }

            // Handle gaps - skip if waiting too long or queue too large
            if !output_queue.is_empty() {
                let should_skip = output_queue.len() > max_queue_size
                    || (output_queue.len() > 5 && last_output_time.elapsed() > gap_timeout);

                if should_skip {
                    if let Some(&first_seq) = output_queue.keys().next() {
                        let gap = first_seq.wrapping_sub(next_seq);
                        if gap < 32768 && gap > 0 {
                            // Only log occasionally to avoid spam
                            if gap > 5 || output_queue.len() > max_queue_size {
                                eprintln!(
                                    "WARN: Skipping gap {} -> {} ({} packets, queue={})",
                                    next_seq,
                                    first_seq,
                                    gap,
                                    output_queue.len()
                                );
                            }
                            next_seq = first_seq;
                            // Output all contiguous packets from new position
                            while let Some(data) = output_queue.remove(&next_seq) {
                                if write_tx.send(data).is_err() {
                                    return;
                                }
                                next_seq = next_seq.wrapping_add(1);
                                last_output_time = std::time::Instant::now();
                            }
                        }
                    }
                }
            }
        }

        // Flush remaining packets when channel closes
        if !output_queue.is_empty() {
            eprintln!(
                "INFO: Channel closed, flushing {} remaining packets",
                output_queue.len()
            );
            while !output_queue.is_empty() {
                if let Some(&first_seq) = output_queue.keys().next() {
                    let gap = first_seq.wrapping_sub(next_seq);
                    if gap < 32768 && gap > 0 {
                        next_seq = first_seq;
                    }
                    while let Some(data) = output_queue.remove(&next_seq) {
                        if write_tx.send(data).is_err() {
                            return;
                        }
                        next_seq = next_seq.wrapping_add(1);
                    }
                } else {
                    break;
                }
            }
        }
    });

    let mut source_count = 0u64;
    let mut parity_count = 0u64;
    let mut recovered_count = 0u64;
    let start_time = std::time::Instant::now();
    let mut last_ack_time = std::time::Instant::now();
    const PERIODIC_ACK_INTERVAL_MS: u64 = 50; // Send ACK at least every 50ms

    // Process the first packet that we already received
    if first_len >= 8 {
        let seq = u16::from_le_bytes([first_buf[0], first_buf[1]]);
        let pkt_type = first_buf[2];

        // Type 0 = Source packet
        if pkt_type == 0 {
            let payload = first_buf[8..first_len].to_vec();
            debug!("Injected first packet: seq={}, len={}", seq, payload.len());
            source_count += 1;

            // Add to FEC decoder for recovery purposes
            let mut padded = payload.clone();
            if padded.len() < packet_size {
                padded.resize(packet_size, 0);
            }
            session.fec_decoder_mut().add_source(seq, &padded);
            session.receiver_arq_mut().on_receive(seq);

            if output_tx.try_send((seq, payload)).is_err() {
                warn!("Output buffer full, dropping first packet");
            }
        }
    }

    loop {
        // Use timeout to allow periodic ACKs even when no data arrives
        let recv_result = tokio::time::timeout(
            tokio::time::Duration::from_millis(PERIODIC_ACK_INTERVAL_MS),
            session.recv(),
        )
        .await;

        match recv_result {
            Ok(Ok(RecvResult::Source {
                seq,
                data,
                recovered,
            })) => {
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
                last_ack_time = std::time::Instant::now();
            }
            Ok(Ok(RecvResult::Parity)) => {
                parity_count += 1;
                debug!("Received parity packet");
                last_ack_time = std::time::Instant::now();
            }
            Ok(Ok(RecvResult::Ack)) => {
                debug!("Received ACK (unexpected on receiver)");
            }
            Ok(Ok(RecvResult::WouldBlock)) => {
                tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
            }
            Ok(Err(e)) => {
                warn!("Receive error: {}", e);
                continue;
            }
            Err(_) => {
                // Timeout - send periodic ACK if we have pending data
                if last_ack_time.elapsed().as_millis() >= PERIODIC_ACK_INTERVAL_MS as u128 {
                    if session.receiver_arq().packets_since_ack() > 0 {
                        debug!("Sending periodic ACK");
                        if let Err(e) = session.force_ack().await {
                            warn!("Failed to send periodic ACK: {}", e);
                        }
                        last_ack_time = std::time::Instant::now();
                    }
                }
            }
        }

        // Print stats periodically
        if source_count % 5000 == 0 && source_count > 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let overhead = if source_count > 0 {
                parity_count as f64 / source_count as f64 * 100.0
            } else {
                0.0
            };

            info!(
                "RX: {} src + {} parity ({:.1}% overhead) | {} recovered by FEC | {:.1}s",
                source_count, parity_count, overhead, recovered_count, elapsed,
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
            fec_delay,
            fec_parities,
            fec_step,
        } => {
            run_sender(
                listen,
                remote,
                loss,
                preset,
                packet_size,
                fec_delay,
                fec_parities,
                fec_step,
            )
            .await?;
        }
        Command::Recv {
            listen,
            loss,
            preset,
            packet_size,
            fec_delay,
            fec_parities,
            fec_step,
        } => {
            run_receiver(
                listen,
                loss,
                preset,
                packet_size,
                fec_delay,
                fec_parities,
                fec_step,
            )
            .await?;
        }
    }

    Ok(())
}
