//! Async reliable session (bidirectional sender/receiver).
//!
//! Provides a unified session that handles both sending and receiving with
//! proper ARQ retransmission. This is the recommended API for reliable transport.

use std::io;
use std::time::Instant;

use crate::streaming::{StreamingDecoder, StreamingEncoder, StreamingParams};
use crate::transport::{AsyncDatagramRecvMut, AsyncDatagramSendMut};

use super::adaptive::AdaptiveFec;
use super::arq::{ReceiverArq, SenderArq};
use super::bbr::BbrState;
use super::protocol::{AckPacket, PacketType};
use super::stats::NetworkStats;
use super::RecvResult;
use super::ReliableConfig;

/// Async reliable session combining encoder and decoder.
///
/// This struct handles bidirectional reliable transport, properly processing
/// ACKs and triggering retransmissions. Use this instead of separate
/// `AsyncReliableEncoder` and `AsyncReliableDecoder` when you need full
/// ARQ reliability.
///
/// # Architecture
///
/// The session maintains both sender and receiver state:
/// - **Sender side**: FEC encoder, sender ARQ, BBR congestion control
/// - **Receiver side**: FEC decoder, receiver ARQ
///
/// When sending data, any incoming ACKs are also processed to trigger
/// retransmissions. When receiving data, ACKs are sent as needed.
///
/// # Example
///
/// ```rust,ignore
/// use cm256::reliable::{AsyncReliableSession, ReliableConfig};
/// use tokio::net::UdpSocket;
///
/// #[tokio::main]
/// async fn main() -> std::io::Result<()> {
///     let socket = UdpSocket::bind("0.0.0.0:9000").await?;
///     socket.connect("127.0.0.1:9001").await?;
///
///     let mut session = AsyncReliableSession::new(ReliableConfig::default(), socket)?;
///     
///     // Send data - ACKs are processed automatically
///     session.send(&[0x42; 1200]).await?;
///     
///     // Receive data - ACKs are sent automatically
///     let result = session.recv().await?;
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct AsyncReliableSession<T> {
    // === Sender state ===
    /// Streaming FEC encoder.
    fec_encoder: StreamingEncoder,

    /// ARQ state for retransmission.
    sender_arq: SenderArq,

    /// BBR congestion control.
    bbr: BbrState,

    /// Adaptive FEC tuner.
    adaptive: AdaptiveFec,

    // === Receiver state ===
    /// Streaming FEC decoder.
    fec_decoder: StreamingDecoder,

    /// ARQ state for tracking received packets.
    receiver_arq: ReceiverArq,

    // === Shared state ===
    /// Network statistics.
    stats: NetworkStats,

    /// Underlying transport (bidirectional).
    transport: T,

    /// Configuration.
    config: ReliableConfig,

    /// Buffer for sending packets.
    send_buffer: Vec<u8>,

    /// Buffer for receiving packets.
    recv_buffer: Vec<u8>,

    /// Last received packet time.
    last_recv_time: Option<Instant>,
}

impl<T> AsyncReliableSession<T> {
    /// Create a new async reliable session.
    pub fn new(config: ReliableConfig, transport: T) -> io::Result<Self> {
        let fec_params = StreamingParams::with_step_size(
            config.fec_delay,
            config.fec_parities,
            config.fec_step_size,
            config.symbol_bytes,
        )
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, format!("{:?}", e)))?;

        let fec_encoder = StreamingEncoder::new(fec_params.clone());
        let fec_decoder = StreamingDecoder::new(fec_params);
        let sender_arq = SenderArq::new(config.send_buffer_size, config.max_retries);
        let receiver_arq = ReceiverArq::new();
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
        let recv_buffer_size = config.symbol_bytes + 16;

        Ok(Self {
            fec_encoder,
            sender_arq,
            bbr,
            adaptive,
            fec_decoder,
            receiver_arq,
            stats: NetworkStats::new(),
            transport,
            config,
            send_buffer: Vec::with_capacity(send_buffer_capacity),
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

    /// Get the FEC encoder.
    pub fn fec_encoder(&self) -> &StreamingEncoder {
        &self.fec_encoder
    }

    /// Get the FEC decoder.
    pub fn fec_decoder(&self) -> &StreamingDecoder {
        &self.fec_decoder
    }

    /// Get mutable access to the FEC decoder.
    pub fn fec_decoder_mut(&mut self) -> &mut StreamingDecoder {
        &mut self.fec_decoder
    }

    /// Get the sender ARQ state.
    pub fn sender_arq(&self) -> &SenderArq {
        &self.sender_arq
    }

    /// Get mutable access to the sender ARQ state.
    pub fn sender_arq_mut(&mut self) -> &mut SenderArq {
        &mut self.sender_arq
    }

    /// Get the receiver ARQ state.
    pub fn receiver_arq(&self) -> &ReceiverArq {
        &self.receiver_arq
    }

    /// Get mutable access to the receiver ARQ state.
    pub fn receiver_arq_mut(&mut self) -> &mut ReceiverArq {
        &mut self.receiver_arq
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
        self.bbr.can_send() && !self.sender_arq.is_full()
    }

    /// Get total number of retransmissions.
    pub fn total_retransmits(&self) -> u64 {
        self.sender_arq.total_retransmits()
    }

    /// Reset the session state.
    pub fn reset(&mut self) {
        self.fec_encoder.reset();
        self.fec_decoder.reset();
        self.sender_arq.reset();
        self.receiver_arq.reset();
        self.bbr.reset();
        self.stats.reset();
    }

    /// Consume the session and return the transport.
    pub fn into_transport(self) -> T {
        self.transport
    }
}

impl<T: AsyncDatagramSendMut + AsyncDatagramRecvMut + Unpin> AsyncReliableSession<T> {
    /// Send data with FEC protection.
    ///
    /// Returns the sequence number assigned to this packet.
    ///
    /// Note: This method only sends data. To process incoming ACKs and
    /// trigger retransmissions, use [`poll_acks`] or [`send_with_ack_poll`].
    pub async fn send(&mut self, data: &[u8]) -> io::Result<u16> {
        assert_eq!(
            data.len(),
            self.config.symbol_bytes,
            "Data must be exactly {} bytes",
            self.config.symbol_bytes
        );

        // Add to FEC encoder
        let result = self.fec_encoder.add_source(data);
        let seq = result.source_seq;

        // Buffer for ARQ retransmission
        self.sender_arq.on_send(data);

        // Build and send source packet
        self.send_source_packet(seq, data).await?;

        // Send any parity packets
        for parity in &result.parities {
            self.send_parity_packet(parity).await?;
        }

        // Update BBR
        self.bbr.on_send();
        self.stats.on_packet_sent();

        Ok(seq)
    }

    /// Send data and poll for incoming ACKs.
    ///
    /// This combines `send` with `try_poll_acks` to handle both directions
    /// in a single call. Use this when you're primarily sending but want
    /// to ensure ACKs are processed for retransmissions.
    pub async fn send_with_ack_poll(&mut self, data: &[u8]) -> io::Result<u16> {
        let seq = self.send(data).await?;

        // Try to process any pending ACKs (non-blocking check)
        // We ignore WouldBlock errors since we're just opportunistically checking
        let _ = self.try_poll_acks().await;

        Ok(seq)
    }

    /// Poll for and process incoming ACKs.
    ///
    /// This blocks until an ACK is received and processed. Use this in a
    /// separate task or with `tokio::select!` to handle ACKs while sending.
    ///
    /// Returns the number of retransmissions triggered by the ACK.
    pub async fn poll_acks(&mut self) -> io::Result<usize> {
        let n = self
            .transport
            .recv_datagram_async(&mut self.recv_buffer)
            .await?;

        if n < 3 {
            return Ok(0);
        }

        let packet_type = PacketType::from_u8(self.recv_buffer[2]);

        match packet_type {
            Some(PacketType::Ack) => {
                if let Some(ack) = AckPacket::from_bytes(&self.recv_buffer[..n]) {
                    let retransmits = self.process_ack(&ack).await?;
                    return Ok(retransmits);
                }
            }
            _ => {
                // Not an ACK - could be data if this is a bidirectional session
                // For now, we ignore non-ACK packets in this method
            }
        }

        Ok(0)
    }

    /// Try to poll for ACKs without blocking.
    ///
    /// This attempts to receive and process ACKs in a non-blocking manner.
    /// On most platforms, this will still block briefly on the socket.
    ///
    /// For truly non-blocking operation, use `poll_acks` in a separate task.
    pub async fn try_poll_acks(&mut self) -> io::Result<usize> {
        // For now, this is the same as poll_acks
        // In the future, we could add non-blocking socket support
        self.poll_acks().await
    }

    /// Process an ACK packet and trigger retransmissions.
    ///
    /// Returns the number of packets retransmitted.
    async fn process_ack(&mut self, ack: &AckPacket) -> io::Result<usize> {
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
        let retransmits = self.sender_arq.on_ack(ack);
        let retransmit_count = retransmits.len();

        // Retransmit missing packets
        for seq in retransmits {
            self.retransmit(seq).await?;
        }

        Ok(retransmit_count)
    }

    /// Retransmit a specific sequence.
    async fn retransmit(&mut self, seq: u16) -> io::Result<()> {
        if let Some(data) = self.sender_arq.get_retransmit_data(seq) {
            self.send_source_packet(seq, &data).await?;
            self.bbr.on_send();
        }
        Ok(())
    }

    /// Send a source packet.
    async fn send_source_packet(&mut self, seq: u16, data: &[u8]) -> io::Result<()> {
        self.send_buffer.clear();

        // Header: seq (2) + type (1) + reserved (5) = 8 bytes
        self.send_buffer.extend_from_slice(&seq.to_le_bytes());
        self.send_buffer.push(PacketType::Source as u8);
        self.send_buffer.extend_from_slice(&[0u8; 5]); // reserved

        // Payload
        self.send_buffer.extend_from_slice(data);

        self.transport
            .send_datagram_async(&self.send_buffer)
            .await?;
        Ok(())
    }

    /// Send a parity packet.
    async fn send_parity_packet(
        &mut self,
        parity: &crate::streaming::ParityOutput,
    ) -> io::Result<()> {
        self.send_buffer.clear();

        // Header: end_seq (2) + type (1) + parity_index (1) + reserved (4) = 8 bytes
        self.send_buffer
            .extend_from_slice(&parity.end_seq.to_le_bytes());
        self.send_buffer.push(PacketType::Parity as u8);
        self.send_buffer.push(parity.parity_index);
        self.send_buffer.extend_from_slice(&[0u8; 4]); // reserved

        // Payload
        self.send_buffer.extend_from_slice(&parity.data);

        self.transport
            .send_datagram_async(&self.send_buffer)
            .await?;
        self.bbr.on_send();

        Ok(())
    }

    /// Flush the FEC encoder.
    pub async fn flush(&mut self) -> io::Result<()> {
        let parities = self.fec_encoder.flush();
        for parity in &parities {
            self.send_parity_packet(parity).await?;
        }
        Ok(())
    }

    /// Receive a packet.
    ///
    /// This processes incoming source/parity packets and sends ACKs as needed.
    /// Use this on the receiver side of the session.
    pub async fn recv(&mut self) -> io::Result<RecvResult> {
        let n = self
            .transport
            .recv_datagram_async(&mut self.recv_buffer)
            .await?;
        self.process_incoming_packet(n).await
    }

    /// Process an incoming packet (source, parity, or ACK).
    async fn process_incoming_packet(&mut self, n: usize) -> io::Result<RecvResult> {
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
                self.fec_decoder.add_source(seq, &data);

                // Update ARQ
                self.receiver_arq.on_receive(seq);

                // Try FEC recovery
                let _recovered = self.fec_decoder.try_recover();

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
                self.fec_decoder.add_parity(end_seq, parity_index, &data);

                // Try FEC recovery
                let recovered = self.fec_decoder.try_recover();

                // Mark recovered packets in ARQ
                for (seq, _) in &recovered {
                    self.receiver_arq.on_receive(*seq);
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
                // Process ACK on sender side
                if let Some(ack) = AckPacket::from_bytes(&self.recv_buffer[..n]) {
                    let _ = self.process_ack(&ack).await?;
                }
                Ok(RecvResult::Ack)
            }
            None => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unknown packet type",
            )),
        }
    }

    /// Check if we should send an ACK and send it.
    async fn maybe_send_ack(&mut self) -> io::Result<()> {
        let should_ack = self
            .receiver_arq
            .should_ack_by_count(self.config.ack_every_n_packets)
            || self
                .receiver_arq
                .should_ack_by_time(self.config.ack_interval_ms as u64);

        if should_ack
            && self
                .receiver_arq
                .can_send_ack(self.config.min_ack_interval_ms as u64)
        {
            self.send_ack().await?;
        }

        Ok(())
    }

    /// Send an ACK packet.
    pub async fn send_ack(&mut self) -> io::Result<()> {
        let ack = self.receiver_arq.build_ack(&self.stats);
        let bytes = ack.to_bytes();
        self.transport.send_datagram_async(&bytes).await?;
        Ok(())
    }

    /// Force send an ACK immediately.
    pub async fn force_ack(&mut self) -> io::Result<()> {
        self.send_ack().await
    }
}

/// Role of the session (sender or receiver).
///
/// This is used to configure the session for optimal behavior based on
/// its primary role in the communication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionRole {
    /// Primarily sends data, receives ACKs.
    Sender,
    /// Primarily receives data, sends ACKs.
    Receiver,
    /// Bidirectional - both sends and receives data.
    Bidirectional,
}
