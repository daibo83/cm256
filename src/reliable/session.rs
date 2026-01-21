//! Simple reliable session: FEC + NACK-based ARQ.
//!
//! No BBR, no adaptive FEC, no flow control. Just:
//! - Streaming FEC for primary recovery
//! - Simple NACK-based retransmission for gaps

use std::collections::VecDeque;
use std::io;

use crate::streaming::{StreamingDecoder, StreamingEncoder, StreamingParams};
use crate::transport::{AsyncDatagramRecvMut, AsyncDatagramSendMut};

use super::arq::{ReceiverArq, SenderArq};
use super::protocol::{NackPacket, PacketType};
use super::ReliableConfig;

/// Result of receiving a packet.
#[derive(Debug)]
pub enum RecvResult {
    /// Source data received (or recovered).
    Source {
        seq: u16,
        data: Vec<u8>,
        recovered: bool,
    },
    /// Parity packet received (no data to output).
    Parity,
    /// NACK received (sender side).
    Nack,
    /// Would block (non-blocking mode).
    WouldBlock,
}

/// Simple reliable session: FEC + NACK-based ARQ.
#[derive(Debug)]
pub struct SimpleSession<T> {
    /// FEC encoder.
    encoder: StreamingEncoder,
    /// FEC decoder.
    decoder: StreamingDecoder,
    /// Sender ARQ (retransmit buffer).
    sender_arq: SenderArq,
    /// Receiver ARQ (gap tracking).
    receiver_arq: ReceiverArq,
    /// Transport.
    transport: T,
    /// Config.
    config: ReliableConfig,
    /// Send buffer.
    send_buf: Vec<u8>,
    /// Receive buffer.
    recv_buf: Vec<u8>,
    /// Queue of recovered packets to return.
    recovered_queue: VecDeque<(u16, Vec<u8>)>,
}

impl<T> SimpleSession<T> {
    /// Create a new session.
    pub fn new(config: ReliableConfig, transport: T) -> io::Result<Self> {
        let params = StreamingParams::with_step_size(
            config.fec_delay,
            config.fec_parities,
            config.fec_step_size,
            config.symbol_bytes,
        )
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, format!("{:?}", e)))?;

        let encoder = StreamingEncoder::new(params.clone());
        let decoder = StreamingDecoder::new(params);
        let sender_arq = SenderArq::new(config.send_buffer_size, config.max_retries);
        let receiver_arq = ReceiverArq::new();

        let buf_size = config.symbol_bytes + 16;

        Ok(Self {
            encoder,
            decoder,
            sender_arq,
            receiver_arq,
            transport,
            config,
            send_buf: Vec::with_capacity(buf_size),
            recv_buf: vec![0u8; buf_size],
            recovered_queue: VecDeque::new(),
        })
    }

    /// Get config.
    pub fn config(&self) -> &ReliableConfig {
        &self.config
    }

    /// Get sender ARQ.
    pub fn sender_arq(&self) -> &SenderArq {
        &self.sender_arq
    }

    /// Get receiver ARQ.
    pub fn receiver_arq(&self) -> &ReceiverArq {
        &self.receiver_arq
    }

    /// Get FEC decoder.
    pub fn decoder(&self) -> &StreamingDecoder {
        &self.decoder
    }

    /// Get mutable FEC decoder.
    pub fn decoder_mut(&mut self) -> &mut StreamingDecoder {
        &mut self.decoder
    }

    /// Get mutable receiver ARQ.
    pub fn receiver_arq_mut(&mut self) -> &mut ReceiverArq {
        &mut self.receiver_arq
    }

    /// Reset session.
    pub fn reset(&mut self) {
        self.encoder.reset();
        self.decoder.reset();
        self.sender_arq.reset();
        self.receiver_arq.reset();
        self.recovered_queue.clear();
    }

    /// Get transport.
    pub fn transport(&self) -> &T {
        &self.transport
    }

    /// Get mutable transport.
    pub fn transport_mut(&mut self) -> &mut T {
        &mut self.transport
    }

    /// Queue recovered packets and mark them as received.
    fn queue_recovered(&mut self, recovered: Vec<(u16, Vec<u8>)>) {
        for (seq, data) in recovered {
            self.receiver_arq.on_receive(seq);
            self.recovered_queue.push_back((seq, data));
        }
    }
}

impl<T: AsyncDatagramSendMut + AsyncDatagramRecvMut + Unpin> SimpleSession<T> {
    /// Send data with FEC. Returns sequence number.
    pub async fn send(&mut self, data: &[u8]) -> io::Result<u16> {
        assert_eq!(data.len(), self.config.symbol_bytes);

        // Add to FEC encoder
        let result = self.encoder.add_source(data);
        let seq = result.source_seq;

        // Buffer for ARQ
        self.sender_arq.on_send(data);

        // Send source packet
        self.send_source(seq, data).await?;

        // Send parities
        for parity in &result.parities {
            self.send_parity(parity).await?;
        }

        Ok(seq)
    }

    /// Send source packet.
    async fn send_source(&mut self, seq: u16, data: &[u8]) -> io::Result<()> {
        self.send_buf.clear();
        self.send_buf.extend_from_slice(&seq.to_le_bytes());
        self.send_buf.push(PacketType::Source as u8);
        self.send_buf.extend_from_slice(&[0u8; 5]); // reserved
        self.send_buf.extend_from_slice(data);
        self.transport.send_datagram_async(&self.send_buf).await?;
        Ok(())
    }

    /// Send parity packet.
    async fn send_parity(&mut self, parity: &crate::streaming::ParityOutput) -> io::Result<()> {
        self.send_buf.clear();
        self.send_buf
            .extend_from_slice(&parity.end_seq.to_le_bytes());
        self.send_buf.push(PacketType::Parity as u8);
        self.send_buf.push(parity.parity_index);
        self.send_buf.extend_from_slice(&[0u8; 4]); // reserved
        self.send_buf.extend_from_slice(&parity.data);
        self.transport.send_datagram_async(&self.send_buf).await?;
        Ok(())
    }

    /// Flush FEC encoder.
    pub async fn flush(&mut self) -> io::Result<()> {
        let parities = self.encoder.flush();
        for parity in &parities {
            self.send_parity(parity).await?;
        }
        Ok(())
    }

    /// Receive a packet.
    pub async fn recv(&mut self) -> io::Result<RecvResult> {
        // First, return any queued recovered packets
        if let Some((seq, data)) = self.recovered_queue.pop_front() {
            return Ok(RecvResult::Source {
                seq,
                data,
                recovered: true,
            });
        }

        let n = self
            .transport
            .recv_datagram_async(&mut self.recv_buf)
            .await?;
        self.process_packet(n).await
    }

    /// Process received packet.
    async fn process_packet(&mut self, n: usize) -> io::Result<RecvResult> {
        // Source/Parity format: [seq (2), type (1), ... (5 reserved), data]
        // NACK format: [type (1), count (1), seqs...]

        if n < 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "packet too small",
            ));
        }

        // Check for Source/Parity first (type at offset 2)
        if n >= 8 {
            let pkt_type = PacketType::from_u8(self.recv_buf[2]);
            match pkt_type {
                Some(PacketType::Source) => {
                    let seq = u16::from_le_bytes([self.recv_buf[0], self.recv_buf[1]]);
                    let payload = self.recv_buf[8..n].to_vec();

                    let mut data = payload.clone();
                    if data.len() < self.config.symbol_bytes {
                        data.resize(self.config.symbol_bytes, 0);
                    }

                    self.decoder.add_source(seq, &data);
                    self.receiver_arq.on_receive(seq);

                    // Try recovery and queue any recovered packets
                    let recovered = self.decoder.try_recover();
                    self.queue_recovered(recovered);

                    return Ok(RecvResult::Source {
                        seq,
                        data: payload,
                        recovered: false,
                    });
                }
                Some(PacketType::Parity) => {
                    let end_seq = u16::from_le_bytes([self.recv_buf[0], self.recv_buf[1]]);
                    let parity_index = self.recv_buf[3];
                    let payload = &self.recv_buf[8..n];

                    let mut data = payload.to_vec();
                    if data.len() < self.config.symbol_bytes {
                        data.resize(self.config.symbol_bytes, 0);
                    }

                    self.decoder.add_parity(end_seq, parity_index, &data);

                    // Try recovery and queue all recovered packets
                    let recovered = self.decoder.try_recover();
                    self.queue_recovered(recovered);

                    // Return first recovered packet if any, otherwise Parity
                    if let Some((seq, data)) = self.recovered_queue.pop_front() {
                        return Ok(RecvResult::Source {
                            seq,
                            data,
                            recovered: true,
                        });
                    }

                    return Ok(RecvResult::Parity);
                }
                _ => {}
            }
        }

        // Check for NACK (type at offset 0)
        if self.recv_buf[0] == PacketType::Nack as u8 {
            if let Some(nack) = NackPacket::from_bytes(&self.recv_buf[..n]) {
                for seq in &nack.sequences {
                    if let Some(data) = self.sender_arq.get_retransmit(*seq) {
                        self.send_source(*seq, &data).await?;
                    }
                }
            }
            return Ok(RecvResult::Nack);
        }

        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unknown packet type",
        ))
    }

    /// Send NACK for missing sequences.
    pub async fn send_nack(&mut self) -> io::Result<()> {
        if !self.receiver_arq.has_gaps() {
            return Ok(());
        }
        let nack = self.receiver_arq.build_nack();
        if nack.is_empty() {
            return Ok(());
        }
        let bytes = nack.to_bytes();
        self.transport.send_datagram_async(&bytes).await?;
        Ok(())
    }

    /// Check if should send NACK.
    pub fn should_nack(&self) -> bool {
        self.receiver_arq.has_gaps()
            && (self
                .receiver_arq
                .should_nack_by_count(self.config.ack_every_n_packets)
                || self
                    .receiver_arq
                    .should_nack_by_time(self.config.ack_interval_ms as u64))
    }
}
