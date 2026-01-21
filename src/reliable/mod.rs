//! # Reliable Transport
//!
//! Simple reliable transport: Streaming FEC + NACK-based ARQ.
//!
//! ## Design Philosophy
//!
//! - **FEC first**: Streaming FEC recovers most losses instantly
//! - **Simple ARQ**: NACK-based retransmission for remaining gaps
//! - **No complexity**: No BBR, no adaptive FEC, no flow control
//!
//! ## Example
//!
//! ```rust,ignore
//! use cm256::reliable::{SimpleSession, ReliableConfig};
//! use tokio::net::UdpSocket;
//!
//! // Sender
//! let socket = UdpSocket::bind("0.0.0.0:9000").await?;
//! socket.connect("127.0.0.1:9001").await?;
//! let mut session = SimpleSession::new(ReliableConfig::default(), socket)?;
//! session.send(&data).await?;
//!
//! // Receiver  
//! let socket = UdpSocket::bind("0.0.0.0:9001").await?;
//! let mut session = SimpleSession::new(ReliableConfig::default(), socket)?;
//! let result = session.recv().await?;
//! if session.should_nack() {
//!     session.send_nack().await?;
//! }
//! ```

mod arq;
mod protocol;
mod session;

// Re-exports
pub use arq::{ReceiverArq, SenderArq};
pub use protocol::{NackPacket, PacketType};
pub use session::{RecvResult, SimpleSession};

/// Configuration for reliable transport.
#[derive(Debug, Clone)]
pub struct ReliableConfig {
    // === FEC parameters ===
    /// FEC window size (delay in packets).
    pub fec_delay: u8,

    /// Number of parity symbols per step.
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

    // === NACK parameters ===
    /// Send NACK after receiving this many packets with gaps.
    pub ack_every_n_packets: u16,

    /// Maximum time between NACKs in milliseconds.
    pub ack_interval_ms: u16,
}

impl Default for ReliableConfig {
    fn default() -> Self {
        Self {
            // FEC: 25% overhead (2 parities per 8 packets)
            fec_delay: 8,
            fec_parities: 2,
            fec_step_size: 8,
            symbol_bytes: 1200,

            // ARQ: reasonable buffer
            send_buffer_size: 256,
            max_retries: 3,

            // NACK: responsive
            ack_every_n_packets: 8,
            ack_interval_ms: 20,
        }
    }
}

impl ReliableConfig {
    /// Low latency config: smaller FEC window, faster NACKs.
    pub fn low_latency() -> Self {
        Self {
            fec_delay: 4,
            fec_parities: 1,
            fec_step_size: 4,
            symbol_bytes: 1200,
            send_buffer_size: 128,
            max_retries: 2,
            ack_every_n_packets: 4,
            ack_interval_ms: 10,
        }
    }

    /// High redundancy config: more FEC overhead for lossy networks.
    pub fn high_redundancy() -> Self {
        Self {
            fec_delay: 8,
            fec_parities: 4, // 50% overhead
            fec_step_size: 8,
            symbol_bytes: 1200,
            send_buffer_size: 256,
            max_retries: 3,
            ack_every_n_packets: 8,
            ack_interval_ms: 20,
        }
    }
}

#[cfg(test)]
mod tests;
