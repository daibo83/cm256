//! # Reliable Transport
//!
//! A reliable transport layer combining streaming FEC with ARQ for guaranteed delivery.
//!
//! ## Overview
//!
//! This module provides reliable data transport over unreliable datagrams (UDP/QUIC) by combining:
//!
//! - **Streaming FEC** (cm256) - Instant recovery of ~95% of packet losses
//! - **Sliding Window ARQ** - Retransmission for losses exceeding FEC capacity
//! - **Adaptive FEC** - Dynamic redundancy adjustment based on network conditions
//! - **BBR Congestion Control** - Bandwidth estimation and pacing
//!
//! ## Target Latency
//!
//! Designed for **50ms + RTT** end-to-end latency:
//! - FEC recovers most losses in ~10-20ms
//! - ARQ fallback adds ~15-35ms + RTT
//!
//! ## Architecture
//!
//! See `architecture.md` for detailed design documentation.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │              ReliableTransport                      │
//! ├─────────────────────────────────────────────────────┤
//! │  StreamingFEC ──→ AdaptiveFEC ──→ BBR ──→ ARQ      │
//! │       │                                    │        │
//! │       └────────────────────────────────────┘        │
//! │                        │                            │
//! │                   Transport                         │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use cm256::reliable::{ReliableEncoder, ReliableDecoder, ReliableConfig};
//! use std::net::UdpSocket;
//!
//! // Sender
//! let socket = UdpSocket::bind("0.0.0.0:0")?;
//! socket.connect("127.0.0.1:9000")?;
//! let mut encoder = ReliableEncoder::new(ReliableConfig::default(), socket);
//! encoder.send(&data)?;
//!
//! // Receiver
//! let socket = UdpSocket::bind("0.0.0.0:9000")?;
//! let mut decoder = ReliableDecoder::new(ReliableConfig::default(), socket);
//! let received = decoder.recv()?;
//! ```

mod adaptive;
mod arq;
mod async_decoder;
mod async_encoder;
mod bbr;
mod decoder;
mod encoder;
mod protocol;
mod session;
mod stats;

// Re-exports
pub use adaptive::AdaptiveFec;
pub use arq::{ReceiverArq, SenderArq};
pub use async_decoder::AsyncReliableDecoder;
pub use async_encoder::AsyncReliableEncoder;
pub use bbr::{BbrMode, BbrState};
pub use decoder::{RecvResult, ReliableDecoder};
pub use encoder::ReliableEncoder;
pub use protocol::{AckPacket, PacketType};
pub use session::{AsyncReliableSession, SessionRole};
pub use stats::NetworkStats;

/// Configuration for reliable transport.
#[derive(Debug, Clone)]
pub struct ReliableConfig {
    // === ACK parameters ===
    /// Send ACK after receiving this many packets.
    pub ack_every_n_packets: u16,

    /// Maximum time between ACKs in milliseconds.
    pub ack_interval_ms: u16,

    /// Minimum time between ACKs in milliseconds (rate limiting).
    pub min_ack_interval_ms: u16,

    /// Number of symbols tracked in ACK bitmap.
    pub window_size: u16,

    // === FEC parameters ===
    /// FEC window size (delay).
    pub fec_delay: u8,

    /// Initial number of parity symbols per window.
    pub fec_parities: u8,

    /// FEC step size (parity generation frequency).
    pub fec_step_size: u8,

    /// Symbol size in bytes.
    pub symbol_bytes: usize,

    // === ARQ parameters ===
    /// Size of send buffer for retransmission.
    pub send_buffer_size: u16,

    /// Maximum retransmission attempts per packet.
    pub max_retries: u8,

    // === BBR parameters ===
    /// Initial congestion window in packets.
    pub initial_cwnd: u32,

    /// Interval for probing minimum RTT in milliseconds.
    pub min_rtt_probe_interval_ms: u32,
}

impl Default for ReliableConfig {
    fn default() -> Self {
        Self {
            // ACK: optimized for 50ms + RTT
            ack_every_n_packets: 8,
            ack_interval_ms: 20,
            min_ack_interval_ms: 5,
            window_size: 64,

            // FEC: balanced for typical networks (~12.5% overhead)
            // Can recover 1 loss per 8 packets
            fec_delay: 8,
            fec_parities: 1,
            fec_step_size: 8,
            symbol_bytes: 1200,

            // ARQ
            send_buffer_size: 256,
            max_retries: 5,

            // BBR
            initial_cwnd: 10,
            min_rtt_probe_interval_ms: 10000,
        }
    }
}

impl ReliableConfig {
    /// Configuration optimized for ultra-low latency (VoIP, gaming).
    /// Higher FEC overhead (~25%) for faster recovery, minimal ARQ.
    pub fn low_latency() -> Self {
        Self {
            ack_every_n_packets: 4,
            ack_interval_ms: 10,
            min_ack_interval_ms: 3,
            window_size: 64,
            fec_delay: 4,
            fec_parities: 1,
            fec_step_size: 4,
            symbol_bytes: 1200,
            send_buffer_size: 64,
            max_retries: 1,
            initial_cwnd: 10,
            min_rtt_probe_interval_ms: 10000,
        }
    }

    /// Configuration optimized for high throughput (file transfer).
    /// Lower FEC overhead (~6.25%), relies more on ARQ.
    pub fn high_throughput() -> Self {
        Self {
            ack_every_n_packets: 32,
            ack_interval_ms: 100,
            min_ack_interval_ms: 20,
            window_size: 64,
            fec_delay: 16,
            fec_parities: 1,
            fec_step_size: 16,
            symbol_bytes: 1400,
            send_buffer_size: 256,
            max_retries: 3,
            initial_cwnd: 32,
            min_rtt_probe_interval_ms: 10000,
        }
    }

    /// Configuration for high-RTT networks (satellite, intercontinental).
    /// More aggressive FEC (~25%) to avoid costly retransmissions.
    pub fn high_rtt() -> Self {
        Self {
            ack_every_n_packets: 64,
            ack_interval_ms: 200,
            min_ack_interval_ms: 50,
            window_size: 64,
            fec_delay: 8,
            fec_parities: 2,
            fec_step_size: 8,
            symbol_bytes: 1200,
            send_buffer_size: 512,
            max_retries: 4,
            initial_cwnd: 16,
            min_rtt_probe_interval_ms: 30000,
        }
    }
}

#[cfg(test)]
mod tests;
