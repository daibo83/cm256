//! Protocol definitions for reliable transport.
//!
//! Simple packet types: Source, Parity, and NACK for symbol-level ARQ.

/// Packet type discriminator for wire protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PacketType {
    /// Source data symbol.
    Source = 0,
    /// FEC parity symbol.
    Parity = 1,
    /// NACK requesting retransmission of specific symbols.
    Nack = 2,
}

impl PacketType {
    /// Parse packet type from byte.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Source),
            1 => Some(Self::Parity),
            2 => Some(Self::Nack),
            _ => None,
        }
    }
}

/// NACK packet requesting retransmission of specific symbols.
///
/// # Wire Format
///
/// ```text
/// ┌─────────┬─────────┬─────────────────────────────────┐
/// │ type=2  │ count   │ seq[0], seq[1], ... seq[count-1]│
/// │ (1B)    │ (1B)    │ (2B each, little-endian)        │
/// └─────────┴─────────┴─────────────────────────────────┘
/// ```
///
/// Maximum 127 sequences per NACK packet (256 bytes max).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NackPacket {
    /// Sequence numbers being requested.
    pub sequences: Vec<u16>,
}

impl NackPacket {
    /// Header size in bytes (type + count).
    pub const HEADER_SIZE: usize = 2;

    /// Maximum sequences per NACK.
    pub const MAX_SEQUENCES: usize = 127;

    /// Create a new NACK packet.
    pub fn new(sequences: Vec<u16>) -> Self {
        debug_assert!(sequences.len() <= Self::MAX_SEQUENCES);
        Self { sequences }
    }

    /// Serialize the NACK packet to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let count = self.sequences.len().min(Self::MAX_SEQUENCES);
        let mut buf = Vec::with_capacity(Self::HEADER_SIZE + count * 2);

        buf.push(PacketType::Nack as u8);
        buf.push(count as u8);

        for &seq in self.sequences.iter().take(count) {
            buf.extend_from_slice(&seq.to_le_bytes());
        }

        buf
    }

    /// Deserialize a NACK packet from bytes.
    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < Self::HEADER_SIZE {
            return None;
        }

        if buf[0] != PacketType::Nack as u8 {
            return None;
        }

        let count = buf[1] as usize;
        let expected_len = Self::HEADER_SIZE + count * 2;

        if buf.len() < expected_len {
            return None;
        }

        let mut sequences = Vec::with_capacity(count);
        for i in 0..count {
            let offset = Self::HEADER_SIZE + i * 2;
            let seq = u16::from_le_bytes([buf[offset], buf[offset + 1]]);
            sequences.push(seq);
        }

        Some(Self { sequences })
    }

    /// Check if a specific sequence is in this NACK.
    pub fn contains(&self, seq: u16) -> bool {
        self.sequences.contains(&seq)
    }

    /// Number of sequences in this NACK.
    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_type_roundtrip() {
        assert_eq!(PacketType::from_u8(0), Some(PacketType::Source));
        assert_eq!(PacketType::from_u8(1), Some(PacketType::Parity));
        assert_eq!(PacketType::from_u8(2), Some(PacketType::Nack));
        assert_eq!(PacketType::from_u8(3), None);
    }

    #[test]
    fn test_nack_packet_empty() {
        let nack = NackPacket::new(vec![]);
        let bytes = nack.to_bytes();
        assert_eq!(bytes.len(), 2);
        assert_eq!(bytes[0], PacketType::Nack as u8);
        assert_eq!(bytes[1], 0);

        let parsed = NackPacket::from_bytes(&bytes).unwrap();
        assert!(parsed.is_empty());
    }

    #[test]
    fn test_nack_packet_roundtrip() {
        let seqs = vec![100, 105, 200, 65535, 0];
        let nack = NackPacket::new(seqs.clone());

        let bytes = nack.to_bytes();
        assert_eq!(bytes.len(), 2 + 5 * 2);

        let parsed = NackPacket::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.sequences, seqs);
        assert!(parsed.contains(100));
        assert!(parsed.contains(65535));
        assert!(!parsed.contains(101));
    }

    #[test]
    fn test_nack_packet_max() {
        let seqs: Vec<u16> = (0..127).collect();
        let nack = NackPacket::new(seqs.clone());

        let bytes = nack.to_bytes();
        let parsed = NackPacket::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.len(), 127);
    }
}
