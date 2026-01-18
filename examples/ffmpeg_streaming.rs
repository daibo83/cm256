//! FFmpeg H.264 streaming with Diagonal Interleaving FEC
//!
//! This example uses the streaming FEC module for low-latency video streaming.
//! Unlike block FEC (ffmpeg_pipe.rs), streaming FEC provides continuous protection
//! with lower latency - each packet is protected as soon as it's sent.
//!
//! # Usage
//!
//! ## UDP Transport (default)
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
//! ## QUIC Transport
//!
//! Receiver (start first, acts as QUIC server):
//! ```bash
//! cargo run --release --example ffmpeg_streaming -- recv --transport quic | ffplay -f h264 -
//! ```
//!
//! Sender (QUIC client):
//! ```bash
//! ffmpeg -re -i input.mp4 -c:v libx264 -tune zerolatency -f h264 - | \
//!     cargo run --release --example ffmpeg_streaming -- send --transport quic
//! ```
//!
//! # Key Differences from Block FEC (ffmpeg_pipe)
//!
//! - **Lower latency**: Parities are generated per-packet, not per-message
//! - **Better burst recovery**: Overlapping windows catch consecutive losses
//! - **Continuous protection**: Every packet is covered by multiple parity windows

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use cm256::streaming::{StreamingDecoder, StreamingEncoder, StreamingParams};
use cm256::transport::{AsyncDatagramRecv, AsyncDatagramSend};
use std::collections::BTreeMap;
use std::future::Future;
use std::io::{self, Read, Write};
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use tokio::net::UdpSocket;
use tracing::{debug, info, warn, Level};

// ============================================================================
// Transport Abstraction
// ============================================================================

/// Transport type selection
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum TransportType {
    /// UDP transport (default)
    #[default]
    Udp,
    /// QUIC transport (uses unreliable datagrams)
    Quic,
}

/// Unified transport enum that implements async datagram traits
enum Transport {
    Udp(UdpSocket),
    Quic(quinn::Connection),
}

impl AsyncDatagramSend for Transport {
    fn send_datagram_async<'a>(
        &'a self,
        data: &'a [u8],
    ) -> Pin<Box<dyn Future<Output = io::Result<usize>> + Send + 'a>> {
        match self {
            Transport::Udp(socket) => Box::pin(async move { socket.send(data).await }),
            Transport::Quic(conn) => Box::pin(async move {
                let bytes = bytes::Bytes::copy_from_slice(data);
                conn.send_datagram(bytes)
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                Ok(data.len())
            }),
        }
    }
}

impl AsyncDatagramRecv for Transport {
    fn recv_datagram_async<'a>(
        &'a self,
        buf: &'a mut [u8],
    ) -> Pin<Box<dyn Future<Output = io::Result<usize>> + Send + 'a>> {
        match self {
            Transport::Udp(socket) => Box::pin(async move { socket.recv(buf).await }),
            Transport::Quic(conn) => Box::pin(async move {
                let datagram = conn
                    .read_datagram()
                    .await
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                let len = datagram.len().min(buf.len());
                buf[..len].copy_from_slice(&datagram[..len]);
                Ok(len)
            }),
        }
    }
}

// ============================================================================
// QUIC Setup Helpers
// ============================================================================

/// Generate self-signed certificate for QUIC
fn generate_self_signed_cert() -> Result<(
    rustls::pki_types::CertificateDer<'static>,
    rustls::pki_types::PrivateKeyDer<'static>,
)> {
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
        .context("Failed to generate self-signed certificate")?;
    let key = rustls::pki_types::PrivateKeyDer::Pkcs8(cert.key_pair.serialize_der().into());
    let cert = cert.cert.into();
    Ok((cert, key))
}

/// Create QUIC server config with self-signed cert
fn make_server_config() -> Result<quinn::ServerConfig> {
    let (cert, key) = generate_self_signed_cert()?;

    let mut server_config = quinn::ServerConfig::with_single_cert(vec![cert], key)
        .context("Failed to create server config")?;

    // Enable datagrams
    let transport = Arc::get_mut(&mut server_config.transport).unwrap();
    transport.max_idle_timeout(Some(std::time::Duration::from_secs(30).try_into().unwrap()));
    transport.datagram_receive_buffer_size(Some(65536));

    Ok(server_config)
}

/// Create QUIC client config (skip server verification for demo)
fn make_client_config() -> Result<quinn::ClientConfig> {
    let mut client_config = quinn::ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(
            rustls::ClientConfig::builder()
                .dangerous()
                .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
                .with_no_client_auth(),
        )
        .unwrap(),
    ));

    // Enable datagrams
    let mut transport = quinn::TransportConfig::default();
    transport.max_idle_timeout(Some(std::time::Duration::from_secs(30).try_into().unwrap()));
    transport.datagram_receive_buffer_size(Some(65536));
    client_config.transport_config(Arc::new(transport));

    Ok(client_config)
}

/// Skip server certificate verification (for demo purposes only)
#[derive(Debug)]
struct SkipServerVerification;

impl rustls::client::danger::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::RSA_PKCS1_SHA256,
            rustls::SignatureScheme::RSA_PKCS1_SHA384,
            rustls::SignatureScheme::RSA_PKCS1_SHA512,
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::ECDSA_NISTP384_SHA384,
            rustls::SignatureScheme::ECDSA_NISTP521_SHA512,
            rustls::SignatureScheme::RSA_PSS_SHA256,
            rustls::SignatureScheme::RSA_PSS_SHA384,
            rustls::SignatureScheme::RSA_PSS_SHA512,
            rustls::SignatureScheme::ED25519,
        ]
    }
}

// ============================================================================
// Packet Header (for manual encoding when using raw transport)
// ============================================================================

/// Packet type discriminator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum PacketType {
    Source = 0,
    Parity = 1,
}

/// Packet header for streaming FEC
#[derive(Debug, Clone, Copy)]
struct PacketHeader {
    seq: u16,
    packet_type: PacketType,
    parity_index: u8,
    aux_field: u16, // data_len for source, end_seq for parity
}

impl PacketHeader {
    const SIZE: usize = 8;

    fn source(seq: u16, data_len: u16) -> Self {
        Self {
            seq,
            packet_type: PacketType::Source,
            parity_index: 0,
            aux_field: data_len,
        }
    }

    fn parity(seq: u16, parity_index: u8, end_seq: u16) -> Self {
        Self {
            seq,
            packet_type: PacketType::Parity,
            parity_index,
            aux_field: end_seq,
        }
    }

    fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..2].copy_from_slice(&self.seq.to_le_bytes());
        buf[2] = self.packet_type as u8;
        buf[3] = self.parity_index;
        buf[4..6].copy_from_slice(&self.aux_field.to_le_bytes());
        buf
    }

    fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < Self::SIZE {
            return None;
        }
        let seq = u16::from_le_bytes([buf[0], buf[1]]);
        let packet_type = match buf[2] {
            0 => PacketType::Source,
            1 => PacketType::Parity,
            _ => return None,
        };
        let parity_index = buf[3];
        let aux_field = u16::from_le_bytes([buf[4], buf[5]]);

        Some(Self {
            seq,
            packet_type,
            parity_index,
            aux_field,
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

        /// Transport type (udp or quic)
        #[arg(long, default_value = "udp")]
        transport: TransportType,

        /// FEC delay (window size in packets)
        #[arg(long, default_value = "15")]
        delay: u8,

        /// Number of parity packets per window
        #[arg(long, default_value = "1")]
        parities: u8,

        /// Step size (generate parities every N packets)
        #[arg(long, default_value = "5")]
        step_size: u8,

        /// Packet size in bytes
        #[arg(long, default_value = "1200")]
        packet_size: usize,
    },
    /// Receive stream and output H.264 to stdout
    Recv {
        /// Local address to bind to
        #[arg(short, long, default_value = "0.0.0.0:9001")]
        listen: SocketAddr,

        /// Transport type (udp or quic)
        #[arg(long, default_value = "udp")]
        transport: TransportType,

        /// FEC delay (must match sender)
        #[arg(long, default_value = "15")]
        delay: u8,

        /// Number of parity packets per window (must match sender)
        #[arg(long, default_value = "1")]
        parities: u8,

        /// Step size (must match sender)
        #[arg(long, default_value = "5")]
        step_size: u8,

        /// Packet size (ignored - auto-detected from sender)
        #[arg(long, default_value = "1200", hide = true)]
        packet_size: usize,
    },
}

// ============================================================================
// Sender
// ============================================================================

async fn run_sender(
    addr: SocketAddr,
    remote: SocketAddr,
    transport_type: TransportType,
    delay: u8,
    parities: u8,
    step_size: u8,
    packet_size: usize,
) -> Result<()> {
    let params = StreamingParams::with_step_size(delay, parities, step_size, packet_size)
        .context("Invalid streaming parameters")?;

    info!(
        "Streaming Sender: {} -> {}, transport={:?}, delay={}, parities={}, step_size={}, packet_size={}",
        addr, remote, transport_type, delay, parities, step_size, packet_size
    );
    info!(
        "  Max burst recovery: {} packets, overhead: {:.1}%",
        params.max_burst(),
        params.overhead() * 100.0
    );

    // Create transport based on type
    let transport = match transport_type {
        TransportType::Udp => {
            let socket = UdpSocket::bind(addr)
                .await
                .with_context(|| format!("Failed to bind to {}", addr))?;
            socket
                .connect(remote)
                .await
                .with_context(|| format!("Failed to connect to {}", remote))?;
            info!("UDP socket connected to {}", remote);
            Transport::Udp(socket)
        }
        TransportType::Quic => {
            let client_config = make_client_config()?;
            let mut endpoint = quinn::Endpoint::client(addr)
                .with_context(|| format!("Failed to bind QUIC endpoint to {}", addr))?;
            endpoint.set_default_client_config(client_config);

            info!("Connecting to QUIC server at {}...", remote);
            let connection = endpoint
                .connect(remote, "localhost")
                .context("Failed to start QUIC connection")?
                .await
                .context("Failed to establish QUIC connection")?;
            info!("QUIC connection established, waiting for receiver ready signal...");

            // Wait for receiver to signal it's ready
            let ready = connection
                .read_datagram()
                .await
                .context("Failed to receive ready signal")?;
            if &ready[..] != b"READY" {
                anyhow::bail!("Unexpected ready signal: {:?}", ready);
            }
            info!("Receiver ready, starting stream");

            Transport::Quic(connection)
        }
    };

    let mut encoder = StreamingEncoder::new(params);
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

    let mut send_buffer = Vec::with_capacity(PacketHeader::SIZE + packet_size);
    let mut source_count = 0u64;
    let mut parity_count = 0u64;
    let mut total_bytes = 0u64;

    while let Some(data) = rx.recv().await {
        let actual_len = data.len();

        // Pad to packet_size for FEC
        let mut padded = data;
        if padded.len() < packet_size {
            padded.resize(packet_size, 0);
        }

        let result = encoder.add_source(&padded);

        // Send source packet
        let header = PacketHeader::source(result.source_seq, actual_len as u16);
        send_buffer.clear();
        send_buffer.extend_from_slice(&header.to_bytes());
        send_buffer.extend_from_slice(&padded);
        transport.send_datagram_async(&send_buffer).await?;
        source_count += 1;
        total_bytes += actual_len as u64;

        // Send parity packets
        for parity in &result.parities {
            let header =
                PacketHeader::parity(result.source_seq, parity.parity_index, parity.end_seq);
            send_buffer.clear();
            send_buffer.extend_from_slice(&header.to_bytes());
            send_buffer.extend_from_slice(&parity.data);
            transport.send_datagram_async(&send_buffer).await?;
            parity_count += 1;
        }

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

    // Flush remaining parities
    let flush_parities = encoder.flush();
    let seq = encoder.next_seq().wrapping_sub(1);
    for parity in &flush_parities {
        let header = PacketHeader::parity(seq, parity.parity_index, parity.end_seq);
        send_buffer.clear();
        send_buffer.extend_from_slice(&header.to_bytes());
        send_buffer.extend_from_slice(&parity.data);
        transport.send_datagram_async(&send_buffer).await?;
        parity_count += 1;
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
    transport_type: TransportType,
    delay: u8,
    parities: u8,
    step_size: u8,
    _packet_size: usize, // Ignored - auto-detected from first packet
) -> Result<()> {
    info!(
        "Streaming Receiver on {}, transport={:?}, delay={}, parities={}, step_size={} (packet_size auto-detected)",
        addr, transport_type, delay, parities, step_size
    );

    let mut recv_buffer = vec![0u8; 65536];

    // Create transport based on type, capturing first packet for UDP
    let (transport, first_packet_len) = match transport_type {
        TransportType::Udp => {
            let socket = UdpSocket::bind(addr)
                .await
                .with_context(|| format!("Failed to bind to {}", addr))?;

            // Wait for first packet to get sender address, then connect
            info!("Waiting for UDP connection...");
            let (n, sender_addr) = socket.recv_from(&mut recv_buffer).await?;
            info!("Connected to sender at {}", sender_addr);
            socket.connect(sender_addr).await?;
            (Transport::Udp(socket), Some(n))
        }
        TransportType::Quic => {
            let server_config = make_server_config()?;
            let endpoint = quinn::Endpoint::server(server_config, addr)
                .with_context(|| format!("Failed to bind QUIC endpoint to {}", addr))?;

            info!("Waiting for QUIC connection...");
            let incoming = endpoint
                .accept()
                .await
                .context("No incoming QUIC connection")?;
            let connection = incoming.await.context("Failed to accept QUIC connection")?;
            info!(
                "QUIC connection established from {}",
                connection.remote_address()
            );
            (Transport::Quic(connection), None)
        }
    };

    // For QUIC, send ready signal before receiving first packet
    if matches!(transport_type, TransportType::Quic) {
        info!("Sending ready signal to sender...");
        transport
            .send_datagram_async(b"READY")
            .await
            .context("Failed to send ready signal")?;
    }

    // Get first packet to determine packet size
    let first_packet_len = match first_packet_len {
        Some(n) => n, // UDP: already have first packet
        None => {
            // QUIC: need to receive first packet
            transport
                .recv_datagram_async(&mut recv_buffer)
                .await
                .context("Failed to receive first packet")?
        }
    };

    // Auto-detect packet size from first datagram
    if first_packet_len < PacketHeader::SIZE {
        anyhow::bail!(
            "First packet too small ({} bytes), expected at least {} bytes",
            first_packet_len,
            PacketHeader::SIZE
        );
    }
    let packet_size = first_packet_len - PacketHeader::SIZE;
    info!(
        "Auto-detected packet_size={} from first datagram",
        packet_size
    );

    // Now create params and decoder with correct packet size
    let params = StreamingParams::with_step_size(delay, parities, step_size, packet_size)
        .context("Invalid streaming parameters")?;
    info!(
        "  Max burst recovery: {} packets, overhead: {:.1}%",
        params.max_burst(),
        params.overhead() * 100.0
    );

    let mut decoder = StreamingDecoder::new(params);
    decoder.set_history_windows(16);

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

    // Helper closure to process a single packet
    let process_packet = |n: usize,
                          recv_buffer: &[u8],
                          decoder: &mut StreamingDecoder,
                          data_lengths: &mut std::collections::HashMap<u16, u16>,
                          output_tx: &tokio::sync::mpsc::Sender<(u16, Vec<u8>)>,
                          source_count: &mut u64,
                          parity_count: &mut u64| {
        if n < PacketHeader::SIZE {
            return;
        }

        let Some(header) = PacketHeader::from_bytes(&recv_buffer[..PacketHeader::SIZE]) else {
            return;
        };

        let payload = &recv_buffer[PacketHeader::SIZE..n];
        if payload.len() < packet_size {
            // Shouldn't happen since we auto-detected packet_size from first packet
            warn!(
                "Unexpected packet size: {} < {}, skipping",
                payload.len(),
                packet_size
            );
            return;
        }

        match header.packet_type {
            PacketType::Source => {
                decoder.add_source(header.seq, &payload[..packet_size]);
                data_lengths.insert(header.seq, header.aux_field);
                *source_count += 1;

                debug!("RX source: seq={}, len={}", header.seq, header.aux_field);

                let actual_len = header.aux_field as usize;
                let data = payload[..actual_len.min(packet_size)].to_vec();
                if output_tx.try_send((header.seq, data)).is_err() {
                    warn!("Output buffer full, dropping packet {}", header.seq);
                }
            }
            PacketType::Parity => {
                decoder.add_parity(
                    header.aux_field,
                    header.parity_index,
                    &payload[..packet_size],
                );
                *parity_count += 1;
                debug!(
                    "RX parity: end_seq={}, index={}",
                    header.aux_field, header.parity_index
                );
            }
        }
    };

    // Process the first packet (already in recv_buffer)
    process_packet(
        first_packet_len,
        &recv_buffer,
        &mut decoder,
        &mut data_lengths,
        &output_tx,
        &mut source_count,
        &mut parity_count,
    );
    packet_count += 1;

    loop {
        let n = match transport.recv_datagram_async(&mut recv_buffer).await {
            Ok(n) => n,
            Err(e) => {
                warn!("Receive error: {}", e);
                continue;
            }
        };

        packet_count += 1;

        process_packet(
            n,
            &recv_buffer,
            &mut decoder,
            &mut data_lengths,
            &output_tx,
            &mut source_count,
            &mut parity_count,
        );

        // Try recovery periodically
        if packet_count % 4 == 0 {
            let recovered = decoder.try_recover();
            for (seq, data) in recovered {
                recovered_count += 1;
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
            transport,
            delay,
            parities,
            step_size,
            packet_size,
        } => {
            run_sender(
                listen,
                remote,
                transport,
                delay,
                parities,
                step_size,
                packet_size,
            )
            .await?
        }
        Command::Recv {
            listen,
            transport,
            delay,
            parities,
            step_size,
            packet_size,
        } => run_receiver(listen, transport, delay, parities, step_size, packet_size).await?,
    }

    Ok(())
}
