//! Reliable decoder (receiver side).
//!
//! Combines streaming FEC decoder with ARQ for guaranteed delivery.

use std::io;
use std::time::Instant;

use crate::streaming::{StreamingDecoder, StreamingParams};
use crate::transport::{DatagramRecvMut, DatagramSendMut};

use super::arq::ReceiverArq;
use super::protocol::PacketType;
use super::stats::NetworkStats;
use super::ReliableConfig;

/// Result of receiving a packet.
#[derive(Debug, Clone)]
pub enum RecvResult {
    /// A source packet was received or recovered.
    Source {
        /// Sequence number.
        seq: u16,
        /// The data.
        data: Vec<u8>,
        /// Whether this was recovered via FEC.
        recovered: bool,
    },
    /// A parity packet was received (internal use).
    Parity,
    /// An ACK packet was received (shouldn't happen on receiver).
    Ack,
    /// No packet available (non-blocking).
    WouldBlock,
}

/// Reliable decoder with FEC and ARQ.
#[derive(Debug)]
pub struct ReliableDecoder<T> {
    /// Streaming FEC decoder.
    fec: StreamingDecoder,

    /// ARQ state for tracking received packets.
    arq: ReceiverArq,

    /// Network statistics.
    stats: NetworkStats,

    /// Underlying transport.
    transport: T,

    /// Configuration.
    config: ReliableConfig,

    /// Receive buffer.
    recv_buffer: Vec<u8>,

    /// Last received packet time (for RTT calculation).
    last_recv_time: Option<Instant>,
}

impl<T> ReliableDecoder<T> {
    /// Create a new reliable decoder.
    pub fn new(config: ReliableConfig, transport: T) -> io::Result<Self> {
        let fec_params = StreamingParams::with_step_size(
            config.fec_delay,
            config.fec_parities,
            config.fec_step_size,
            config.symbol_bytes,
        )
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, format!("{:?}", e)))?;

        let fec = StreamingDecoder::new(fec_params);
        let arq = ReceiverArq::new();

        let recv_buffer_size = config.symbol_bytes + 16;

        Ok(Self {
            fec,
            arq,
            stats: NetworkStats::new(),
            transport,
            config,
            recv_buffer: vec![0u8; recv_buffer_size],
            last_recv_time: None,
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

    /// Get the FEC decoder.
    pub fn fec(&self) -> &StreamingDecoder {
        &self.fec
    }

    /// Get the ARQ state.
    pub fn arq(&self) -> &ReceiverArq {
        &self.arq
    }

    /// Get the network statistics.
    pub fn stats(&self) -> &NetworkStats {
        &self.stats
    }

    /// Check if a source packet is available.
    pub fn has_source(&self, seq: u16) -> bool {
        self.fec.has_source(seq)
    }

    /// Get a source packet if available.
    pub fn get_source(&self, seq: u16) -> Option<Vec<u8>> {
        self.fec.get_source(seq).map(|s| s.to_vec())
    }
}

impl<T: DatagramRecvMut + DatagramSendMut> ReliableDecoder<T> {
    /// Receive a packet (blocking).
    pub fn recv(&mut self) -> io::Result<RecvResult> {
        let n = self.transport.recv_datagram(&mut self.recv_buffer)?;
        self.process_packet(n)
    }

    /// Try to receive a packet (non-blocking).
    pub fn try_recv(&mut self) -> io::Result<RecvResult> {
        match self.transport.try_recv_datagram(&mut self.recv_buffer)? {
            Some(n) => self.process_packet(n),
            None => Ok(RecvResult::WouldBlock),
        }
    }

    /// Process a received packet.
    fn process_packet(&mut self, n: usize) -> io::Result<RecvResult> {
        let now = Instant::now();
        self.last_recv_time = Some(now);

        if n < 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "packet too small",
            ));
        }

        let packet_type = PacketType::from_u8(self.recv_buffer[2]);

        match packet_type {
            Some(PacketType::Source) => {
                let seq = u16::from_le_bytes([self.recv_buffer[0], self.recv_buffer[1]]);
                let payload = self.recv_buffer[8..n].to_vec();

                // Pad if needed
                let mut data = payload.clone();
                if data.len() < self.config.symbol_bytes {
                    data.resize(self.config.symbol_bytes, 0);
                }

                // Add to FEC decoder
                self.fec.add_source(seq, &data);

                // Update ARQ
                self.arq.on_receive(seq);

                // Try FEC recovery
                let _recovered = self.fec.try_recover();

                // Maybe send ACK
                self.maybe_send_ack()?;

                Ok(RecvResult::Source {
                    seq,
                    data: payload,
                    recovered: false,
                })
            }
            Some(PacketType::Parity) => {
                let end_seq = u16::from_le_bytes([self.recv_buffer[0], self.recv_buffer[1]]);
                let parity_index = self.recv_buffer[3];
                let payload = &self.recv_buffer[8..n];

                // Pad if needed
                let mut data = payload.to_vec();
                if data.len() < self.config.symbol_bytes {
                    data.resize(self.config.symbol_bytes, 0);
                }

                // Add to FEC decoder
                self.fec.add_parity(end_seq, parity_index, &data);

                // Try FEC recovery
                let recovered = self.fec.try_recover();

                // Mark recovered packets in ARQ
                for (seq, _) in &recovered {
                    self.arq.on_receive(*seq);
                }

                // Maybe send ACK
                self.maybe_send_ack()?;

                // Return first recovered packet if any
                if let Some((seq, data)) = recovered.into_iter().next() {
                    return Ok(RecvResult::Source {
                        seq,
                        data,
                        recovered: true,
                    });
                }

                Ok(RecvResult::Parity)
            }
            Some(PacketType::Ack) => {
                // Shouldn't receive ACKs on receiver side, but handle gracefully
                Ok(RecvResult::Ack)
            }
            None => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unknown packet type",
            )),
        }
    }

    /// Check if we should send an ACK and send it.
    fn maybe_send_ack(&mut self) -> io::Result<()> {
        let should_ack = self
            .arq
            .should_ack_by_count(self.config.ack_every_n_packets)
            || self
                .arq
                .should_ack_by_time(self.config.ack_interval_ms as u64);

        if should_ack
            && self
                .arq
                .can_send_ack(self.config.min_ack_interval_ms as u64)
        {
            self.send_ack()?;
        }

        Ok(())
    }

    /// Send an ACK packet.
    pub fn send_ack(&mut self) -> io::Result<()> {
        let ack = self.arq.build_ack(&self.stats);
        let bytes = ack.to_bytes();
        self.transport.send_datagram(&bytes)?;
        Ok(())
    }

    /// Force send an ACK immediately.
    pub fn force_ack(&mut self) -> io::Result<()> {
        self.send_ack()
    }

    /// Reset the decoder state.
    pub fn reset(&mut self) {
        self.fec.reset();
        self.arq.reset();
        self.stats.reset();
    }

    /// Consume the decoder and return the transport.
    pub fn into_transport(self) -> T {
        self.transport
    }
}

impl<T: DatagramRecvMut> ReliableDecoder<T> {
    /// Receive without sending ACKs (for unidirectional receive).
    pub fn recv_no_ack(&mut self) -> io::Result<RecvResult> {
        let n = self.transport.recv_datagram(&mut self.recv_buffer)?;
        self.process_packet_no_ack(n)
    }

    /// Process packet without ACK logic.
    fn process_packet_no_ack(&mut self, n: usize) -> io::Result<RecvResult> {
        self.last_recv_time = Some(Instant::now());

        if n < 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "packet too small",
            ));
        }

        let packet_type = PacketType::from_u8(self.recv_buffer[2]);

        match packet_type {
            Some(PacketType::Source) => {
                let seq = u16::from_le_bytes([self.recv_buffer[0], self.recv_buffer[1]]);
                let payload = &self.recv_buffer[8..n];

                let mut data = payload.to_vec();
                if data.len() < self.config.symbol_bytes {
                    data.resize(self.config.symbol_bytes, 0);
                }

                self.fec.add_source(seq, &data);
                self.arq.on_receive(seq);
                let _recovered = self.fec.try_recover();

                Ok(RecvResult::Source {
                    seq,
                    data: payload.to_vec(),
                    recovered: false,
                })
            }
            Some(PacketType::Parity) => {
                let end_seq = u16::from_le_bytes([self.recv_buffer[0], self.recv_buffer[1]]);
                let parity_index = self.recv_buffer[3];
                let payload = &self.recv_buffer[8..n];

                let mut data = payload.to_vec();
                if data.len() < self.config.symbol_bytes {
                    data.resize(self.config.symbol_bytes, 0);
                }

                self.fec.add_parity(end_seq, parity_index, &data);

                let recovered = self.fec.try_recover();
                for (seq, _) in &recovered {
                    self.arq.on_receive(*seq);
                }

                if let Some((seq, data)) = recovered.into_iter().next() {
                    return Ok(RecvResult::Source {
                        seq,
                        data,
                        recovered: true,
                    });
                }

                Ok(RecvResult::Parity)
            }
            Some(PacketType::Ack) => Ok(RecvResult::Ack),
            None => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unknown packet type",
            )),
        }
    }
}
