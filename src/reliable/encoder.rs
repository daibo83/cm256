//! Reliable encoder (sender side).
//!
//! Combines streaming FEC, ARQ, adaptive FEC, and BBR congestion control.

use std::io;
use std::time::Instant;

use crate::streaming::{StreamingEncoder, StreamingParams};
use crate::transport::DatagramSendMut;

use super::adaptive::AdaptiveFec;
use super::arq::SenderArq;
use super::bbr::BbrState;
use super::protocol::{AckPacket, PacketType};
use super::stats::NetworkStats;
use super::ReliableConfig;

/// Reliable encoder with FEC, ARQ, and congestion control.
#[derive(Debug)]
pub struct ReliableEncoder<T> {
    /// Streaming FEC encoder.
    fec: StreamingEncoder,

    /// ARQ state for retransmission.
    arq: SenderArq,

    /// BBR congestion control.
    bbr: BbrState,

    /// Adaptive FEC tuner.
    adaptive: AdaptiveFec,

    /// Network statistics.
    stats: NetworkStats,

    /// Underlying transport.
    transport: T,

    /// Configuration.
    config: ReliableConfig,

    /// Send buffer for packet construction.
    send_buffer: Vec<u8>,
}

impl<T: DatagramSendMut> ReliableEncoder<T> {
    /// Create a new reliable encoder.
    pub fn new(config: ReliableConfig, transport: T) -> io::Result<Self> {
        let fec_params = StreamingParams::with_step_size(
            config.fec_delay,
            config.fec_parities,
            config.fec_step_size,
            config.symbol_bytes,
        )
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, format!("{:?}", e)))?;

        let fec = StreamingEncoder::new(fec_params);
        let arq = SenderArq::new(config.send_buffer_size, config.max_retries);
        let bbr = BbrState::with_config(
            config.initial_cwnd,
            config.symbol_bytes as u32,
            std::time::Duration::from_millis(config.min_rtt_probe_interval_ms as u64),
        );
        let adaptive = AdaptiveFec::new(
            config.fec_delay,
            config.fec_parities,
            config.fec_step_size,
            config.symbol_bytes,
        );

        let send_buffer_capacity = config.symbol_bytes + 16;

        Ok(Self {
            fec,
            arq,
            bbr,
            adaptive,
            stats: NetworkStats::new(),
            transport,
            config,
            send_buffer: Vec::with_capacity(send_buffer_capacity),
        })
    }

    /// Get the configuration.
    pub fn config(&self) -> &ReliableConfig {
        &self.config
    }

    /// Get the underlying transport.
    pub fn transport(&self) -> &T {
        &self.transport
    }

    /// Get mutable access to the underlying transport.
    pub fn transport_mut(&mut self) -> &mut T {
        &mut self.transport
    }

    /// Get the FEC encoder.
    pub fn fec(&self) -> &StreamingEncoder {
        &self.fec
    }

    /// Get the ARQ state.
    pub fn arq(&self) -> &SenderArq {
        &self.arq
    }

    /// Get the BBR state.
    pub fn bbr(&self) -> &BbrState {
        &self.bbr
    }

    /// Get the network statistics.
    pub fn stats(&self) -> &NetworkStats {
        &self.stats
    }

    /// Check if we can send (BBR allows and ARQ buffer not full).
    pub fn can_send(&self) -> bool {
        self.bbr.can_send() && !self.arq.is_full()
    }

    /// Send data with FEC protection.
    ///
    /// Returns the sequence number assigned to this packet.
    pub fn send(&mut self, data: &[u8]) -> io::Result<u16> {
        assert_eq!(
            data.len(),
            self.config.symbol_bytes,
            "Data must be exactly {} bytes",
            self.config.symbol_bytes
        );

        // Add to FEC encoder
        let result = self.fec.add_source(data);
        let seq = result.source_seq;

        // Buffer for ARQ retransmission
        self.arq.on_send(data);

        // Build and send source packet
        self.send_source_packet(seq, data)?;

        // Send any parity packets
        for parity in &result.parities {
            self.send_parity_packet(parity)?;
        }

        // Update BBR
        self.bbr.on_send();
        self.stats.on_packet_sent();

        Ok(seq)
    }

    /// Process a received ACK packet.
    pub fn on_ack(&mut self, ack: &AckPacket) -> io::Result<()> {
        let now = Instant::now();

        // Update stats
        let received = ack.received_count();
        let total = AckPacket::BITMAP_BITS as u32;
        let bytes_acked = received as u64 * self.config.symbol_bytes as u64;

        self.stats
            .on_ack(received, total, ack.rtt_millis(), bytes_acked);

        // Update BBR
        self.bbr.on_ack(
            received,
            bytes_acked,
            std::time::Duration::from_millis(ack.rtt_millis() as u64),
            now,
        );

        // Update adaptive FEC
        self.adaptive.update(&self.stats);

        // Process ARQ - get sequences that need retransmission
        let retransmits = self.arq.on_ack(ack);

        // Retransmit missing packets
        for seq in retransmits {
            self.retransmit(seq)?;
        }

        Ok(())
    }

    /// Process a raw received packet (parses ACK if applicable).
    pub fn process_incoming(&mut self, data: &[u8]) -> io::Result<bool> {
        if data.len() < 3 {
            return Ok(false);
        }

        let packet_type = PacketType::from_u8(data[2]);

        match packet_type {
            Some(PacketType::Ack) => {
                if let Some(ack) = AckPacket::from_bytes(data) {
                    self.on_ack(&ack)?;
                    return Ok(true);
                }
            }
            _ => {}
        }

        Ok(false)
    }

    /// Retransmit a specific sequence.
    fn retransmit(&mut self, seq: u16) -> io::Result<()> {
        if let Some(data) = self.arq.get_retransmit_data(seq) {
            self.send_source_packet(seq, &data)?;
            self.bbr.on_send();
        }
        Ok(())
    }

    /// Send a source packet.
    fn send_source_packet(&mut self, seq: u16, data: &[u8]) -> io::Result<()> {
        self.send_buffer.clear();

        // Header: seq (2) + type (1) + reserved (5) = 8 bytes
        self.send_buffer.extend_from_slice(&seq.to_le_bytes());
        self.send_buffer.push(PacketType::Source as u8);
        self.send_buffer.extend_from_slice(&[0u8; 5]); // reserved

        // Payload
        self.send_buffer.extend_from_slice(data);

        self.transport.send_datagram(&self.send_buffer)?;
        Ok(())
    }

    /// Send a parity packet.
    fn send_parity_packet(&mut self, parity: &crate::streaming::ParityOutput) -> io::Result<()> {
        self.send_buffer.clear();

        // Header: end_seq (2) + type (1) + parity_index (1) + reserved (4) = 8 bytes
        self.send_buffer
            .extend_from_slice(&parity.end_seq.to_le_bytes());
        self.send_buffer.push(PacketType::Parity as u8);
        self.send_buffer.push(parity.parity_index);
        self.send_buffer.extend_from_slice(&[0u8; 4]); // reserved

        // Payload
        self.send_buffer.extend_from_slice(&parity.data);

        self.transport.send_datagram(&self.send_buffer)?;
        self.bbr.on_send();

        Ok(())
    }

    /// Flush the FEC encoder.
    pub fn flush(&mut self) -> io::Result<()> {
        let parities = self.fec.flush();
        for parity in &parities {
            self.send_parity_packet(parity)?;
        }
        Ok(())
    }

    /// Reset the encoder state.
    pub fn reset(&mut self) {
        self.fec.reset();
        self.arq.reset();
        self.bbr.reset();
        self.stats.reset();
    }

    /// Consume the encoder and return the transport.
    pub fn into_transport(self) -> T {
        self.transport
    }
}
