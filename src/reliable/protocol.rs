//! Protocol definitions for reliable transport.
//!
//! Defines packet types and the ACK packet format used for the ARQ layer.

/// Packet type discriminator for wire protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PacketType {
    /// Source data symbol.
    Source = 0,
    /// FEC parity symbol.
    Parity = 1,
    /// ACK with bitmap of received symbols.
    Ack = 2,
}

impl PacketType {
    /// Parse packet type from byte.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Source),
            1 => Some(Self::Parity),
            2 => Some(Self::Ack),
            _ => None,
        }
    }
}

/// ACK packet with 64-symbol bitmap.
///
/// Total size: 16 bytes (8 byte header + 8 byte bitmap)
///
/// # Wire Format
///
/// ```text
/// ┌─────────┬─────────┬─────────┬─────────┬─────────┬───────────┐
/// │ base_seq│ type=2  │ max_seen│ loss_%  │ rtt_ms  │ reserved  │
/// │ (2B)    │ (1B)    │ (2B)    │ (1B)    │ (1B)    │ (1B)      │
/// ├─────────┴─────────┴─────────┴─────────┴─────────┴───────────┤
/// │              Bitmap (8 bytes = 64 bits)                     │
/// │         bit i = 1 if (base_seq + i) received                │
/// └─────────────────────────────────────────────────────────────┘
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AckPacket {
    /// Cumulative ACK: all sequences < base_seq are received.
    pub base_seq: u16,

    /// Highest sequence number seen by receiver.
    pub max_seen: u16,

    /// Observed loss rate (0-255 maps to 0-100%).
    pub loss_rate: u8,

    /// RTT sample in milliseconds (0-255).
    pub rtt_ms: u8,

    /// Bitmap of received symbols: bit i = 1 if (base_seq + i) received.
    pub bitmap: [u8; 8],
}

impl AckPacket {
    /// Packet size in bytes.
    pub const SIZE: usize = 16;

    /// Number of symbols tracked by bitmap.
    pub const BITMAP_BITS: usize = 64;

    /// Create a new ACK packet.
    pub fn new(base_seq: u16, max_seen: u16, bitmap: [u8; 8]) -> Self {
        Self {
            base_seq,
            max_seen,
            loss_rate: 0,
            rtt_ms: 0,
            bitmap,
        }
    }

    /// Create a new ACK packet with network stats.
    pub fn with_stats(
        base_seq: u16,
        max_seen: u16,
        bitmap: [u8; 8],
        loss_rate_pct: f64,
        rtt_ms: u32,
    ) -> Self {
        Self {
            base_seq,
            max_seen,
            loss_rate: (loss_rate_pct.clamp(0.0, 1.0) * 255.0) as u8,
            rtt_ms: rtt_ms.min(255) as u8,
            bitmap,
        }
    }

    /// Serialize the ACK packet to bytes.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];

        // Header (8 bytes)
        buf[0..2].copy_from_slice(&self.base_seq.to_le_bytes());
        buf[2] = PacketType::Ack as u8;
        buf[3..5].copy_from_slice(&self.max_seen.to_le_bytes());
        buf[5] = self.loss_rate;
        buf[6] = self.rtt_ms;
        buf[7] = 0; // reserved

        // Bitmap (8 bytes)
        buf[8..16].copy_from_slice(&self.bitmap);

        buf
    }

    /// Deserialize an ACK packet from bytes.
    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < Self::SIZE {
            return None;
        }

        // Verify packet type
        if buf[2] != PacketType::Ack as u8 {
            return None;
        }

        let base_seq = u16::from_le_bytes([buf[0], buf[1]]);
        let max_seen = u16::from_le_bytes([buf[3], buf[4]]);
        let loss_rate = buf[5];
        let rtt_ms = buf[6];

        let mut bitmap = [0u8; 8];
        bitmap.copy_from_slice(&buf[8..16]);

        Some(Self {
            base_seq,
            max_seen,
            loss_rate,
            rtt_ms,
            bitmap,
        })
    }

    /// Check if a specific sequence was received.
    ///
    /// Returns `None` if seq is outside the bitmap range.
    pub fn is_received(&self, seq: u16) -> Option<bool> {
        let offset = seq.wrapping_sub(self.base_seq);
        if offset >= Self::BITMAP_BITS as u16 {
            return None;
        }

        let byte_idx = (offset / 8) as usize;
        let bit_idx = offset % 8;

        Some((self.bitmap[byte_idx] >> bit_idx) & 1 == 1)
    }

    /// Get an iterator over missing sequence numbers.
    pub fn missing_seqs(&self) -> impl Iterator<Item = u16> + '_ {
        (0..Self::BITMAP_BITS as u16).filter_map(move |i| {
            let seq = self.base_seq.wrapping_add(i);
            let byte_idx = (i / 8) as usize;
            let bit_idx = i % 8;

            let received = (self.bitmap[byte_idx] >> bit_idx) & 1 == 1;
            if received {
                None
            } else {
                Some(seq)
            }
        })
    }

    /// Get an iterator over received sequence numbers.
    pub fn received_seqs(&self) -> impl Iterator<Item = u16> + '_ {
        (0..Self::BITMAP_BITS as u16).filter_map(move |i| {
            let seq = self.base_seq.wrapping_add(i);
            let byte_idx = (i / 8) as usize;
            let bit_idx = i % 8;

            let received = (self.bitmap[byte_idx] >> bit_idx) & 1 == 1;
            if received {
                Some(seq)
            } else {
                None
            }
        })
    }

    /// Count the number of received symbols in the bitmap.
    pub fn received_count(&self) -> u32 {
        self.bitmap.iter().map(|b| b.count_ones()).sum()
    }

    /// Count the number of missing symbols in the bitmap.
    pub fn missing_count(&self) -> u32 {
        Self::BITMAP_BITS as u32 - self.received_count()
    }

    /// Get the loss rate as a percentage (0.0 - 1.0).
    pub fn loss_rate_pct(&self) -> f64 {
        self.loss_rate as f64 / 255.0
    }

    /// Get the RTT in milliseconds.
    pub fn rtt_millis(&self) -> u32 {
        self.rtt_ms as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_type_roundtrip() {
        assert_eq!(PacketType::from_u8(0), Some(PacketType::Source));
        assert_eq!(PacketType::from_u8(1), Some(PacketType::Parity));
        assert_eq!(PacketType::from_u8(2), Some(PacketType::Ack));
        assert_eq!(PacketType::from_u8(3), None);
    }

    #[test]
    fn test_ack_packet_roundtrip() {
        let bitmap = [0b10110001, 0b11111111, 0, 0, 0, 0, 0, 0];
        let ack = AckPacket::with_stats(100, 163, bitmap, 0.15, 50);

        let bytes = ack.to_bytes();
        assert_eq!(bytes.len(), AckPacket::SIZE);

        let parsed = AckPacket::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.base_seq, 100);
        assert_eq!(parsed.max_seen, 163);
        assert_eq!(parsed.bitmap, bitmap);
        assert_eq!(parsed.rtt_ms, 50);
    }

    #[test]
    fn test_is_received() {
        // bitmap[0] = 0b10110001 means:
        // bit 0 (seq 100): 1 = received
        // bit 1 (seq 101): 0 = missing
        // bit 2 (seq 102): 0 = missing
        // bit 3 (seq 103): 0 = missing
        // bit 4 (seq 104): 1 = received
        // bit 5 (seq 105): 1 = received
        // bit 6 (seq 106): 0 = missing
        // bit 7 (seq 107): 1 = received
        let bitmap = [0b10110001, 0, 0, 0, 0, 0, 0, 0];
        let ack = AckPacket::new(100, 107, bitmap);

        assert_eq!(ack.is_received(100), Some(true));
        assert_eq!(ack.is_received(101), Some(false));
        assert_eq!(ack.is_received(102), Some(false));
        assert_eq!(ack.is_received(103), Some(false));
        assert_eq!(ack.is_received(104), Some(true));
        assert_eq!(ack.is_received(105), Some(true));
        assert_eq!(ack.is_received(106), Some(false));
        assert_eq!(ack.is_received(107), Some(true));

        // Out of range
        assert_eq!(ack.is_received(99), None);
        assert_eq!(ack.is_received(164), None);
    }

    #[test]
    fn test_missing_seqs() {
        let bitmap = [0b10110001, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        let ack = AckPacket::new(100, 163, bitmap);

        let missing: Vec<u16> = ack.missing_seqs().collect();
        assert_eq!(missing, vec![101, 102, 103, 106]);
    }

    #[test]
    fn test_counts() {
        let bitmap = [0b10110001, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        let ack = AckPacket::new(100, 163, bitmap);

        // 4 received in first byte + 8*7 = 60 in remaining bytes
        assert_eq!(ack.received_count(), 4 + 56);
        assert_eq!(ack.missing_count(), 4);
    }

    #[test]
    fn test_sequence_wraparound() {
        // Test near u16::MAX
        let bitmap = [0xFF; 8];
        let ack = AckPacket::new(65530, 65535, bitmap);

        // Should handle wraparound
        assert_eq!(ack.is_received(65530), Some(true));
        assert_eq!(ack.is_received(65535), Some(true));
        // 65530 + 10 = 65540 wraps to 4
        assert_eq!(ack.is_received(4), Some(true));
    }
}
