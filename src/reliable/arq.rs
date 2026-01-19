//! ARQ (Automatic Repeat reQuest) state machines for reliable transport.
//!
//! Provides sender and receiver state tracking for sliding window ARQ
//! with 64-symbol bitmap acknowledgments.

use std::collections::VecDeque;
use std::time::Instant;

use super::protocol::AckPacket;
use super::stats::NetworkStats;

/// A packet stored in the send buffer for potential retransmission.
#[derive(Debug, Clone)]
pub struct SentPacket {
    /// Sequence number.
    pub seq: u16,

    /// Packet data (source symbol only, not parity).
    pub data: Vec<u8>,

    /// When the packet was originally sent.
    pub send_time: Instant,

    /// Number of times this packet has been retransmitted.
    pub retransmit_count: u8,
}

/// Sender-side ARQ state machine.
///
/// Maintains a buffer of sent packets for retransmission and processes
/// ACK bitmaps to determine which packets need to be retransmitted.
#[derive(Debug)]
pub struct SenderArq {
    /// Ring buffer of sent packets.
    send_buffer: VecDeque<SentPacket>,

    /// Maximum buffer size.
    max_buffer_size: usize,

    /// Base sequence (oldest unacked).
    base_seq: u16,

    /// Next sequence number to assign.
    next_seq: u16,

    /// Maximum retransmit attempts per packet.
    max_retries: u8,

    /// Total number of retransmissions.
    total_retransmits: u64,
}

impl SenderArq {
    /// Create a new sender ARQ state machine.
    pub fn new(buffer_size: u16, max_retries: u8) -> Self {
        Self {
            send_buffer: VecDeque::with_capacity(buffer_size as usize),
            max_buffer_size: buffer_size as usize,
            base_seq: 0,
            next_seq: 0,
            max_retries,
            total_retransmits: 0,
        }
    }

    /// Get the next sequence number (for sending).
    pub fn next_seq(&self) -> u16 {
        self.next_seq
    }

    /// Get the base sequence (oldest unacked).
    pub fn base_seq(&self) -> u16 {
        self.base_seq
    }

    /// Get total number of retransmissions.
    pub fn total_retransmits(&self) -> u64 {
        self.total_retransmits
    }

    /// Get the number of packets in flight (sent but not acked).
    pub fn in_flight(&self) -> usize {
        self.send_buffer.len()
    }

    /// Check if the send buffer is full.
    pub fn is_full(&self) -> bool {
        self.send_buffer.len() >= self.max_buffer_size
    }

    /// Record a packet being sent.
    ///
    /// Returns the assigned sequence number, or None if buffer is full.
    pub fn on_send(&mut self, data: &[u8]) -> Option<u16> {
        if self.is_full() {
            return None;
        }

        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);

        self.send_buffer.push_back(SentPacket {
            seq,
            data: data.to_vec(),
            send_time: Instant::now(),
            retransmit_count: 0,
        });

        Some(seq)
    }

    /// Process an ACK packet and return sequences that need retransmission.
    ///
    /// Also advances the base_seq and frees acknowledged packets from the buffer.
    pub fn on_ack(&mut self, ack: &AckPacket) -> Vec<u16> {
        // Advance base_seq if ACK indicates progress
        self.advance_base(ack.base_seq);

        // Find packets that need retransmission
        let mut retransmits = Vec::new();

        for missing_seq in ack.missing_seqs() {
            // Check if this seq is in our buffer and eligible for retransmit
            if let Some(pkt) = self.find_packet_mut(missing_seq) {
                if pkt.retransmit_count < self.max_retries {
                    retransmits.push(missing_seq);
                }
            }
        }

        retransmits
    }

    /// Get data for retransmission and mark as retransmitted.
    ///
    /// Returns None if the packet is not in the buffer or max retries exceeded.
    pub fn get_retransmit_data(&mut self, seq: u16) -> Option<Vec<u8>> {
        let max_retries = self.max_retries;

        {
            let pkt = self.find_packet_mut(seq)?;

            if pkt.retransmit_count >= max_retries {
                return None;
            }

            pkt.retransmit_count += 1;
            pkt.send_time = Instant::now();
        }

        self.total_retransmits += 1;

        let pkt = self.find_packet_mut(seq)?;
        Some(pkt.data.clone())
    }

    /// Peek at packet data without marking as retransmitted.
    pub fn peek_packet(&self, seq: u16) -> Option<&[u8]> {
        self.find_packet(seq).map(|p| p.data.as_slice())
    }

    /// Advance the base sequence, removing acknowledged packets.
    fn advance_base(&mut self, new_base: u16) {
        // Only advance if new_base is ahead of current base
        // Handle wraparound: new_base is ahead if (new_base - base) < 32768
        let diff = new_base.wrapping_sub(self.base_seq);
        if diff == 0 || diff >= 32768 {
            return; // No advancement or wrapped backwards
        }

        // Remove packets from front of buffer until we reach new_base
        while !self.send_buffer.is_empty() {
            if let Some(front) = self.send_buffer.front() {
                let front_diff = new_base.wrapping_sub(front.seq);
                if front_diff > 0 && front_diff < 32768 {
                    // front.seq is before new_base, remove it
                    self.send_buffer.pop_front();
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        self.base_seq = new_base;
    }

    /// Find a packet in the buffer by sequence number.
    fn find_packet(&self, seq: u16) -> Option<&SentPacket> {
        self.send_buffer.iter().find(|p| p.seq == seq)
    }

    /// Find a packet in the buffer by sequence number (mutable).
    fn find_packet_mut(&mut self, seq: u16) -> Option<&mut SentPacket> {
        self.send_buffer.iter_mut().find(|p| p.seq == seq)
    }

    /// Reset the ARQ state.
    pub fn reset(&mut self) {
        self.send_buffer.clear();
        self.base_seq = 0;
        self.next_seq = 0;
        self.total_retransmits = 0;
    }
}

/// Receiver-side ARQ state machine.
///
/// Tracks received packets using a 64-symbol bitmap and generates
/// ACK packets for the sender.
#[derive(Debug)]
pub struct ReceiverArq {
    /// Base sequence (left edge of window).
    base_seq: u16,

    /// Bitmap of received symbols (64 bits = 8 bytes).
    received_bitmap: [u8; 8],

    /// Highest sequence number seen.
    max_seen: u16,

    /// Packets received since last ACK.
    packets_since_ack: u16,

    /// Last ACK send time.
    last_ack_time: Option<Instant>,
}

impl ReceiverArq {
    /// Create a new receiver ARQ state machine.
    pub fn new() -> Self {
        Self {
            base_seq: 0,
            received_bitmap: [0u8; 8],
            max_seen: 0,
            packets_since_ack: 0,
            last_ack_time: None,
        }
    }

    /// Get the base sequence.
    pub fn base_seq(&self) -> u16 {
        self.base_seq
    }

    /// Get the maximum seen sequence.
    pub fn max_seen(&self) -> u16 {
        self.max_seen
    }

    /// Get packets received since last ACK.
    pub fn packets_since_ack(&self) -> u16 {
        self.packets_since_ack
    }

    /// Record receipt of a packet.
    ///
    /// Returns true if this is a new packet (not a duplicate).
    pub fn on_receive(&mut self, seq: u16) -> bool {
        // Update max_seen
        let max_diff = seq.wrapping_sub(self.max_seen);
        if max_diff > 0 && max_diff < 32768 {
            self.max_seen = seq;
        }

        // Check if seq is within our window
        let offset = seq.wrapping_sub(self.base_seq);

        if offset >= 64 {
            if offset < 32768 {
                // Seq is ahead of our window, need to advance
                self.advance_window(seq);
                // Recalculate offset after advancing
                let new_offset = seq.wrapping_sub(self.base_seq);
                if new_offset < 64 {
                    self.set_bit(new_offset as usize);
                    self.packets_since_ack += 1;
                    return true;
                }
            }
            // Seq is behind our window (old duplicate)
            return false;
        }

        // Check if already received (duplicate)
        let byte_idx = (offset / 8) as usize;
        let bit_idx = offset % 8;
        let already_received = (self.received_bitmap[byte_idx] >> bit_idx) & 1 == 1;

        if already_received {
            return false;
        }

        // Mark as received
        self.set_bit(offset as usize);
        self.packets_since_ack += 1;

        // Try to advance window if we have contiguous receipts from base
        self.try_advance_window();

        true
    }

    /// Build an ACK packet with current state.
    pub fn build_ack(&mut self, stats: &NetworkStats) -> AckPacket {
        self.packets_since_ack = 0;
        self.last_ack_time = Some(Instant::now());

        AckPacket::with_stats(
            self.base_seq,
            self.max_seen,
            self.received_bitmap,
            stats.loss_rate(),
            stats.avg_rtt_ms(),
        )
    }

    /// Build an ACK packet without stats.
    pub fn build_ack_simple(&mut self) -> AckPacket {
        self.packets_since_ack = 0;
        self.last_ack_time = Some(Instant::now());

        AckPacket::new(self.base_seq, self.max_seen, self.received_bitmap)
    }

    /// Check if we should send an ACK based on packet count.
    pub fn should_ack_by_count(&self, threshold: u16) -> bool {
        self.packets_since_ack >= threshold
    }

    /// Check if we should send an ACK based on time.
    pub fn should_ack_by_time(&self, interval_ms: u64) -> bool {
        match self.last_ack_time {
            None => true,
            Some(t) => t.elapsed().as_millis() >= interval_ms as u128,
        }
    }

    /// Check if enough time has passed since last ACK (for rate limiting).
    pub fn can_send_ack(&self, min_interval_ms: u64) -> bool {
        match self.last_ack_time {
            None => true,
            Some(t) => t.elapsed().as_millis() >= min_interval_ms as u128,
        }
    }

    /// Set a bit in the bitmap.
    fn set_bit(&mut self, offset: usize) {
        if offset < 64 {
            let byte_idx = offset / 8;
            let bit_idx = offset % 8;
            self.received_bitmap[byte_idx] |= 1 << bit_idx;
        }
    }

    /// Clear a bit in the bitmap.
    #[allow(dead_code)]
    fn clear_bit(&mut self, offset: usize) {
        if offset < 64 {
            let byte_idx = offset / 8;
            let bit_idx = offset % 8;
            self.received_bitmap[byte_idx] &= !(1 << bit_idx);
        }
    }

    /// Try to advance the window if we have contiguous receipts from base.
    fn try_advance_window(&mut self) {
        // Count contiguous received packets from the start
        let mut advance_by = 0u16;

        for i in 0..64 {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            if (self.received_bitmap[byte_idx] >> bit_idx) & 1 == 1 {
                advance_by += 1;
            } else {
                break;
            }
        }

        if advance_by > 0 {
            self.shift_window(advance_by);
        }
    }

    /// Advance window to include a specific sequence.
    fn advance_window(&mut self, target_seq: u16) {
        let needed_advance = target_seq.wrapping_sub(self.base_seq).saturating_sub(63);
        if needed_advance > 0 && needed_advance < 32768 {
            self.shift_window(needed_advance);
        }
    }

    /// Shift the window forward by n positions.
    fn shift_window(&mut self, n: u16) {
        if n == 0 {
            return;
        }

        if n >= 64 {
            // Complete reset
            self.base_seq = self.base_seq.wrapping_add(n);
            self.received_bitmap = [0u8; 8];
            return;
        }

        let n = n as usize;

        // Shift bitmap left by n bits
        // This is a bit fiddly because we're shifting across byte boundaries
        let mut new_bitmap = [0u8; 8];

        for i in 0..64 {
            let src_pos = i + n;
            if src_pos < 64 {
                let src_byte = src_pos / 8;
                let src_bit = src_pos % 8;
                let bit = (self.received_bitmap[src_byte] >> src_bit) & 1;

                let dst_byte = i / 8;
                let dst_bit = i % 8;
                new_bitmap[dst_byte] |= bit << dst_bit;
            }
        }

        self.received_bitmap = new_bitmap;
        self.base_seq = self.base_seq.wrapping_add(n as u16);
    }

    /// Reset the receiver state.
    pub fn reset(&mut self) {
        self.base_seq = 0;
        self.received_bitmap = [0u8; 8];
        self.max_seen = 0;
        self.packets_since_ack = 0;
        self.last_ack_time = None;
    }
}

impl Default for ReceiverArq {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sender_basic() {
        let mut sender = SenderArq::new(128, 2);

        assert_eq!(sender.next_seq(), 0);
        assert_eq!(sender.in_flight(), 0);

        // Send some packets
        let seq0 = sender.on_send(b"packet0").unwrap();
        let seq1 = sender.on_send(b"packet1").unwrap();
        let seq2 = sender.on_send(b"packet2").unwrap();

        assert_eq!(seq0, 0);
        assert_eq!(seq1, 1);
        assert_eq!(seq2, 2);
        assert_eq!(sender.in_flight(), 3);
    }

    #[test]
    fn test_sender_ack_processing() {
        let mut sender = SenderArq::new(128, 2);

        // Send packets 0-3
        for i in 0..4 {
            sender.on_send(&[i as u8; 100]);
        }

        // ACK with base=2, meaning 0,1 are fully acked
        // Bitmap shows 2,3 received
        let bitmap = [0b00000011, 0, 0, 0, 0, 0, 0, 0];
        let ack = AckPacket::new(2, 3, bitmap);

        let retransmits = sender.on_ack(&ack);

        // Packets 0,1 should be removed, no retransmits needed
        assert!(retransmits.is_empty());
        assert_eq!(sender.base_seq(), 2);
        assert_eq!(sender.in_flight(), 2); // packets 2,3 still in buffer
    }

    #[test]
    fn test_sender_retransmit() {
        let mut sender = SenderArq::new(128, 2);

        // Send packets 0-3
        for i in 0..4 {
            sender.on_send(&[i as u8; 100]);
        }

        // ACK with bitmap showing packet 1 missing
        // base=0, bitmap: 0=recv, 1=missing, 2=recv, 3=recv
        let bitmap = [0b00001101, 0, 0, 0, 0, 0, 0, 0];
        let ack = AckPacket::new(0, 3, bitmap);

        let retransmits = sender.on_ack(&ack);

        assert_eq!(retransmits, vec![1]);

        // Get data for retransmit
        let data = sender.get_retransmit_data(1).unwrap();
        assert_eq!(data, &[1u8; 100]);
    }

    #[test]
    fn test_receiver_basic() {
        let mut receiver = ReceiverArq::new();

        assert!(receiver.on_receive(0));
        assert!(receiver.on_receive(1));
        assert!(receiver.on_receive(2));

        // Duplicate should return false
        assert!(!receiver.on_receive(1));

        assert_eq!(receiver.max_seen(), 2);
        assert_eq!(receiver.packets_since_ack(), 3);
    }

    #[test]
    fn test_receiver_out_of_order() {
        let mut receiver = ReceiverArq::new();

        // Receive out of order: 0, 2, 1, 3
        assert!(receiver.on_receive(0));
        assert!(receiver.on_receive(2));
        assert!(receiver.on_receive(1));
        assert!(receiver.on_receive(3));

        // Window should have advanced past contiguous packets
        assert!(receiver.base_seq() >= 4);
    }

    #[test]
    fn test_receiver_gap() {
        let mut receiver = ReceiverArq::new();

        // Receive 0, 1, skip 2, receive 3
        receiver.on_receive(0);
        receiver.on_receive(1);
        receiver.on_receive(3);

        // Base should be 2 (stopped at gap)
        assert_eq!(receiver.base_seq(), 2);

        // Build ACK and check bitmap
        let stats = NetworkStats::new();
        let ack = receiver.build_ack(&stats);

        // Bit 0 (seq 2) should be 0 (missing)
        // Bit 1 (seq 3) should be 1 (received)
        assert_eq!(ack.is_received(2), Some(false));
        assert_eq!(ack.is_received(3), Some(true));
    }

    #[test]
    fn test_receiver_window_advance() {
        let mut receiver = ReceiverArq::new();

        // Receive packet far ahead, forcing window advance
        receiver.on_receive(100);

        // Window should have advanced to include 100
        assert!(receiver.base_seq() <= 100);
        assert!(100 - receiver.base_seq() < 64);
    }

    #[test]
    fn test_receiver_build_ack() {
        let mut receiver = ReceiverArq::new();

        receiver.on_receive(0);
        receiver.on_receive(1);
        // Skip 2
        receiver.on_receive(3);
        receiver.on_receive(4);

        let stats = NetworkStats::new();
        let ack = receiver.build_ack(&stats);

        // Check the missing sequence
        let missing: Vec<u16> = ack.missing_seqs().collect();
        assert!(missing.contains(&2));

        // packets_since_ack should reset
        assert_eq!(receiver.packets_since_ack(), 0);
    }

    #[test]
    fn test_sequence_wraparound() {
        let mut sender = SenderArq::new(128, 2);

        // Start near wraparound
        sender.base_seq = 65530;
        sender.next_seq = 65530;

        // Send packets across wraparound
        for _ in 0..10 {
            sender.on_send(b"data");
        }

        assert_eq!(sender.in_flight(), 10);

        // ACK that advances past wraparound
        let bitmap = [0xFF; 8];
        let ack = AckPacket::new(4, 5, bitmap); // base=4 (wrapped)

        sender.on_ack(&ack);

        assert_eq!(sender.base_seq(), 4);
    }
}
