//! Async reliable decoder (receiver side).
//!
//! Asynchronous version of [`ReliableDecoder`] for use with async runtimes.

use std::io;
use std::time::Instant;

use crate::streaming::{StreamingDecoder, StreamingParams};
use crate::transport::{AsyncDatagramRecvMut, AsyncDatagramSendMut};

use super::arq::ReceiverArq;
use super::protocol::PacketType;
use super::stats::NetworkStats;
use super::RecvResult;
use super::ReliableConfig;

/// Async reliable decoder with FEC and ARQ.
///
/// This is the asynchronous version of [`super::ReliableDecoder`], designed for
/// use with async runtimes like Tokio.
///
/// # Example
///
/// ```rust,ignore
/// use cm256::reliable::{AsyncReliableDecoder, ReliableConfig};
/// use tokio::net::UdpSocket;
///
/// #[tokio::main]
/// async fn main() -> std::io::Result<()> {
///     let socket = UdpSocket::bind("0.0.0.0:9000").await?;
///
///     let mut decoder = AsyncReliableDecoder::new(ReliableConfig::default(), socket)?;
///     let result = decoder.recv().await?;
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct AsyncReliableDecoder<T> {
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

impl<T> AsyncReliableDecoder<T> {
    /// Create a new async reliable decoder.
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

    /// Get mutable access to the FEC decoder.
    pub fn fec_mut(&mut self) -> &mut StreamingDecoder {
        &mut self.fec
    }

    /// Get the ARQ state.
    pub fn arq(&self) -> &ReceiverArq {
        &self.arq
    }

    /// Get mutable access to the ARQ state.
    pub fn arq_mut(&mut self) -> &mut ReceiverArq {
        &mut self.arq
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

impl<T: AsyncDatagramRecvMut + AsyncDatagramSendMut + Unpin> AsyncReliableDecoder<T> {
    /// Receive a packet asynchronously (with ACK support).
    pub async fn recv(&mut self) -> io::Result<RecvResult> {
        let n = self
            .transport
            .recv_datagram_async(&mut self.recv_buffer)
            .await?;
        self.process_packet(n).await
    }

    /// Process a received packet asynchronously.
    async fn process_packet(&mut self, n: usize) -> io::Result<RecvResult> {
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
                self.maybe_send_ack().await?;

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
                self.maybe_send_ack().await?;

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

    /// Check if we should send an ACK and send it asynchronously.
    async fn maybe_send_ack(&mut self) -> io::Result<()> {
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
            self.send_ack().await?;
        }

        Ok(())
    }

    /// Send an ACK packet asynchronously.
    pub async fn send_ack(&mut self) -> io::Result<()> {
        let ack = self.arq.build_ack(&self.stats);
        let bytes = ack.to_bytes();
        self.transport.send_datagram_async(&bytes).await?;
        Ok(())
    }

    /// Force send an ACK immediately asynchronously.
    pub async fn force_ack(&mut self) -> io::Result<()> {
        self.send_ack().await
    }
}

impl<T: AsyncDatagramRecvMut + Unpin> AsyncReliableDecoder<T> {
    /// Receive without sending ACKs asynchronously (for unidirectional receive).
    pub async fn recv_no_ack(&mut self) -> io::Result<RecvResult> {
        let n = self
            .transport
            .recv_datagram_async(&mut self.recv_buffer)
            .await?;
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
