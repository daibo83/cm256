//! Simple NACK-based ARQ for symbol-level retransmission.
//!
//! - Sender buffers sent packets for retransmission
//! - Receiver tracks gaps and generates NACK packets
//! - No flow control, no congestion control - just retransmit on demand

use std::collections::VecDeque;
use std::time::Instant;

use super::protocol::NackPacket;

/// A packet stored for potential retransmission.
#[derive(Debug, Clone)]
pub struct SentPacket {
    /// Sequence number.
    pub seq: u16,
    /// Packet data.
    pub data: Vec<u8>,
    /// When originally sent.
    pub send_time: Instant,
    /// Number of retransmissions.
    pub retransmit_count: u8,
}

/// Sender-side ARQ: buffers packets for retransmission.
#[derive(Debug)]
pub struct SenderArq {
    /// Ring buffer of sent packets.
    buffer: VecDeque<SentPacket>,
    /// Maximum buffer size.
    max_size: usize,
    /// Next sequence number.
    next_seq: u16,
    /// Maximum retransmit attempts.
    max_retries: u8,
    /// Total retransmissions.
    total_retransmits: u64,
}

impl SenderArq {
    /// Create a new sender ARQ.
    pub fn new(buffer_size: u16, max_retries: u8) -> Self {
        Self {
            buffer: VecDeque::with_capacity(buffer_size as usize),
            max_size: buffer_size as usize,
            next_seq: 0,
            max_retries,
            total_retransmits: 0,
        }
    }

    /// Get next sequence number.
    pub fn next_seq(&self) -> u16 {
        self.next_seq
    }

    /// Get total retransmissions.
    pub fn total_retransmits(&self) -> u64 {
        self.total_retransmits
    }

    /// Record a packet being sent. Returns the assigned sequence number.
    pub fn on_send(&mut self, data: &[u8]) -> u16 {
        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);

        // Remove oldest if buffer full
        if self.buffer.len() >= self.max_size {
            self.buffer.pop_front();
        }

        self.buffer.push_back(SentPacket {
            seq,
            data: data.to_vec(),
            send_time: Instant::now(),
            retransmit_count: 0,
        });

        seq
    }

    /// Get packet data for retransmission. Returns None if not found or max retries exceeded.
    pub fn get_retransmit(&mut self, seq: u16) -> Option<Vec<u8>> {
        let pkt = self.buffer.iter_mut().find(|p| p.seq == seq)?;

        if pkt.retransmit_count >= self.max_retries {
            return None;
        }

        pkt.retransmit_count += 1;
        pkt.send_time = Instant::now();
        self.total_retransmits += 1;

        Some(pkt.data.clone())
    }

    /// Process a NACK and return sequences to retransmit.
    pub fn on_nack(&mut self, nack: &NackPacket) -> Vec<u16> {
        let mut to_retransmit = Vec::new();

        for &seq in &nack.sequences {
            if let Some(pkt) = self.buffer.iter().find(|p| p.seq == seq) {
                if pkt.retransmit_count < self.max_retries {
                    to_retransmit.push(seq);
                }
            }
        }

        to_retransmit
    }

    /// Advance base sequence, removing old packets.
    pub fn advance(&mut self, new_base: u16) {
        while let Some(front) = self.buffer.front() {
            let diff = new_base.wrapping_sub(front.seq);
            if diff > 0 && diff < 32768 {
                self.buffer.pop_front();
            } else {
                break;
            }
        }
    }

    /// Reset state.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.next_seq = 0;
        self.total_retransmits = 0;
    }
}

/// Receiver-side ARQ: tracks received sequences and generates NACKs.
#[derive(Debug)]
pub struct ReceiverArq {
    /// Expected next sequence.
    next_expected: u16,
    /// Highest sequence seen.
    max_seen: u16,
    /// Received sequence bitmap (simple tracking for NACK generation).
    /// Tracks sequences from next_expected to max_seen.
    received: Vec<bool>,
    /// Maximum gap to track.
    max_gap: usize,
    /// Packets received since last NACK.
    packets_since_nack: u16,
    /// Last NACK time.
    last_nack_time: Option<Instant>,
    /// Whether first packet received.
    initialized: bool,
}

impl ReceiverArq {
    /// Create a new receiver ARQ.
    pub fn new() -> Self {
        Self {
            next_expected: 0,
            max_seen: 0,
            received: Vec::new(),
            max_gap: 256, // Track up to 256 sequences
            packets_since_nack: 0,
            last_nack_time: None,
            initialized: false,
        }
    }

    /// Get next expected sequence.
    pub fn next_expected(&self) -> u16 {
        self.next_expected
    }

    /// Get max seen sequence.
    pub fn max_seen(&self) -> u16 {
        self.max_seen
    }

    /// Record receipt of a packet. Returns true if new (not duplicate).
    pub fn on_receive(&mut self, seq: u16) -> bool {
        if !self.initialized {
            // First packet - initialize and advance past it
            self.next_expected = seq.wrapping_add(1);
            self.max_seen = seq;
            self.initialized = true;
            self.packets_since_nack += 1;
            return true;
        }

        // Check if duplicate or old (seq < next_expected)
        let diff = seq.wrapping_sub(self.next_expected);
        if diff >= 32768 {
            // seq is before next_expected (old/duplicate)
            return false;
        }

        // seq >= next_expected
        let offset = diff as usize;

        // Update max_seen if needed
        let max_diff = seq.wrapping_sub(self.max_seen);
        if max_diff > 0 && max_diff < 32768 {
            self.max_seen = seq;
        }

        // Ensure received vector is large enough
        if offset >= self.received.len() {
            if offset < self.max_gap {
                self.received.resize(offset + 1, false);
            } else {
                // Too far ahead, truncate tracking
                return true;
            }
        }

        // Check for duplicate
        if self.received[offset] {
            return false;
        }

        // Mark as received
        self.received[offset] = true;
        self.packets_since_nack += 1;

        // Advance next_expected if we have contiguous packets from start
        while !self.received.is_empty() && self.received[0] {
            self.received.remove(0);
            self.next_expected = self.next_expected.wrapping_add(1);
        }

        true
    }

    /// Build a NACK packet for missing sequences.
    pub fn build_nack(&mut self) -> NackPacket {
        let mut missing = Vec::new();

        for (i, &recv) in self.received.iter().enumerate() {
            if !recv {
                let seq = self.next_expected.wrapping_add(i as u16);
                missing.push(seq);
                if missing.len() >= NackPacket::MAX_SEQUENCES {
                    break;
                }
            }
        }

        self.packets_since_nack = 0;
        self.last_nack_time = Some(Instant::now());

        NackPacket::new(missing)
    }

    /// Check if we should send a NACK (have gaps).
    pub fn has_gaps(&self) -> bool {
        self.received.iter().any(|&r| !r)
    }

    /// Check if should NACK by packet count.
    pub fn should_nack_by_count(&self, threshold: u16) -> bool {
        self.has_gaps() && self.packets_since_nack >= threshold
    }

    /// Check if should NACK by time.
    pub fn should_nack_by_time(&self, interval_ms: u64) -> bool {
        if !self.has_gaps() {
            return false;
        }
        match self.last_nack_time {
            None => true,
            Some(t) => t.elapsed().as_millis() >= interval_ms as u128,
        }
    }

    /// Reset state.
    pub fn reset(&mut self) {
        self.next_expected = 0;
        self.max_seen = 0;
        self.received.clear();
        self.packets_since_nack = 0;
        self.last_nack_time = None;
        self.initialized = false;
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
        let mut sender = SenderArq::new(128, 3);

        let seq0 = sender.on_send(b"packet0");
        let seq1 = sender.on_send(b"packet1");
        let seq2 = sender.on_send(b"packet2");

        assert_eq!(seq0, 0);
        assert_eq!(seq1, 1);
        assert_eq!(seq2, 2);
        assert_eq!(sender.next_seq(), 3);
    }

    #[test]
    fn test_sender_retransmit() {
        let mut sender = SenderArq::new(128, 3);

        sender.on_send(b"packet0");
        sender.on_send(b"packet1");
        sender.on_send(b"packet2");

        let nack = NackPacket::new(vec![1]);
        let retransmits = sender.on_nack(&nack);
        assert_eq!(retransmits, vec![1]);

        let data = sender.get_retransmit(1).unwrap();
        assert_eq!(data, b"packet1");
    }

    #[test]
    fn test_sender_max_retries() {
        let mut sender = SenderArq::new(128, 2);
        sender.on_send(b"packet0");

        // First two retransmits succeed
        assert!(sender.get_retransmit(0).is_some());
        assert!(sender.get_retransmit(0).is_some());

        // Third fails (max_retries = 2)
        assert!(sender.get_retransmit(0).is_none());
    }

    #[test]
    fn test_receiver_in_order() {
        let mut receiver = ReceiverArq::new();

        assert!(receiver.on_receive(0));
        assert!(receiver.on_receive(1));
        assert!(receiver.on_receive(2));

        assert_eq!(receiver.next_expected(), 3);
        assert!(!receiver.has_gaps());
    }

    #[test]
    fn test_receiver_gap() {
        let mut receiver = ReceiverArq::new();

        receiver.on_receive(0);
        receiver.on_receive(1);
        // Skip 2
        receiver.on_receive(3);
        receiver.on_receive(4);

        assert_eq!(receiver.next_expected(), 2);
        assert!(receiver.has_gaps());

        let nack = receiver.build_nack();
        assert_eq!(nack.sequences, vec![2]);
    }

    #[test]
    fn test_receiver_fill_gap() {
        let mut receiver = ReceiverArq::new();

        receiver.on_receive(0);
        receiver.on_receive(2);
        receiver.on_receive(3);

        assert_eq!(receiver.next_expected(), 1);

        // Fill the gap
        receiver.on_receive(1);

        assert_eq!(receiver.next_expected(), 4);
        assert!(!receiver.has_gaps());
    }

    #[test]
    fn test_receiver_duplicate() {
        let mut receiver = ReceiverArq::new();

        assert!(receiver.on_receive(0));
        assert!(!receiver.on_receive(0)); // Duplicate
    }

    #[test]
    fn test_receiver_multiple_gaps() {
        let mut receiver = ReceiverArq::new();

        receiver.on_receive(0);
        // Skip 1, 2
        receiver.on_receive(3);
        // Skip 4
        receiver.on_receive(5);

        let nack = receiver.build_nack();
        assert!(nack.sequences.contains(&1));
        assert!(nack.sequences.contains(&2));
        assert!(nack.sequences.contains(&4));
    }
}
