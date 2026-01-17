//! # Diagonal Interleaving FEC
//!
//! True diagonal interleaving as described in "Erasure Coding for Real-Time Streaming"
//! (Leong & Ho, 2012).
//!
//! ## How It Works
//!
//! Each parity symbol is a simple XOR of source symbols at specific offsets:
//!
//! ```text
//! P_t = S_t ⊕ S_{t-τ}
//! ```
//!
//! Where τ (tau) is the interleaving depth. This creates diagonal "stripes" of protection:
//!
//! ```text
//! Time:    0   1   2   3   4   5   6   7   8   ...
//! Source:  S₀  S₁  S₂  S₃  S₄  S₅  S₆  S₇  S₈  ...
//!          ╲   ╲   ╲   ╲   ╲   ╲   ╲   ╲   ╲
//!           ╲   ╲   ╲   ╲   ╲   ╲   ╲   ╲   ╲
//! Parity:      P₁  P₂  P₃  P₄  P₅  P₆  P₇  P₈  ...
//!              ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
//!            S₁⊕S₀ S₂⊕S₁ ...
//! ```
//!
//! ## Burst Tolerance
//!
//! With interleaving depth τ:
//! - A burst of up to τ consecutive losses can be recovered
//! - Each lost source S_i can be recovered from P_{i+τ} ⊕ S_{i+τ}
//!
//! ## Extended Diagonal (Multiple Parities)
//!
//! For higher burst tolerance, use multiple interleaving depths:
//! - P_t,0 = S_t ⊕ S_{t-τ₀}
//! - P_t,1 = S_t ⊕ S_{t-τ₁}
//! - etc.
//!
//! ## Overhead
//!
//! - Simple diagonal: 1 parity per source = 100% overhead
//! - Extended (n depths): n parities per source = n×100% overhead

use std::collections::{HashMap, VecDeque};

use crate::Error;

// =============================================================================
// Configuration
// =============================================================================

/// Parameters for diagonal interleaving FEC.
///
/// # Example
///
/// ```rust
/// use cm256::diagonal::DiagonalParams;
///
/// // Simple diagonal with τ=8 (can recover burst of 8)
/// let params = DiagonalParams::simple(8, 1200).unwrap();
///
/// // Extended diagonal with τ=[4, 8, 12] (better random loss recovery)
/// let params = DiagonalParams::extended(&[4, 8, 12], 1200).unwrap();
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagonalParams {
    /// Interleaving depths (τ values).
    /// Each depth creates one parity stream.
    depths: Vec<u16>,

    /// Maximum depth (for buffer sizing).
    max_depth: u16,

    /// Size of each symbol in bytes.
    symbol_bytes: usize,
}

impl DiagonalParams {
    /// Create simple diagonal interleaving with a single depth.
    ///
    /// - `depth`: Interleaving depth τ (burst tolerance)
    /// - `symbol_bytes`: Size of each symbol in bytes
    ///
    /// Overhead: 100% (1 parity per source)
    pub fn simple(depth: u16, symbol_bytes: usize) -> Result<Self, Error> {
        if depth == 0 || symbol_bytes == 0 {
            return Err(Error::InvalidParams);
        }
        Ok(Self {
            depths: vec![depth],
            max_depth: depth,
            symbol_bytes,
        })
    }

    /// Create extended diagonal interleaving with multiple depths.
    ///
    /// - `depths`: Array of interleaving depths [τ₀, τ₁, ...]
    /// - `symbol_bytes`: Size of each symbol in bytes
    ///
    /// Overhead: n×100% where n = number of depths
    ///
    /// # Example Configurations
    ///
    /// - `[4, 8]`: 200% overhead, good for bursts up to 8
    /// - `[4, 8, 16]`: 300% overhead, handles longer bursts
    /// - `[2, 4, 8, 16]`: 400% overhead, robust against varied loss patterns
    pub fn extended(depths: &[u16], symbol_bytes: usize) -> Result<Self, Error> {
        if depths.is_empty() || symbol_bytes == 0 {
            return Err(Error::InvalidParams);
        }
        if depths.iter().any(|&d| d == 0) {
            return Err(Error::InvalidParams);
        }
        let max_depth = *depths.iter().max().unwrap();
        Ok(Self {
            depths: depths.to_vec(),
            max_depth,
            symbol_bytes,
        })
    }

    /// Get the interleaving depths.
    pub fn depths(&self) -> &[u16] {
        &self.depths
    }

    /// Get the maximum depth.
    pub fn max_depth(&self) -> u16 {
        self.max_depth
    }

    /// Get the symbol size in bytes.
    pub fn symbol_bytes(&self) -> usize {
        self.symbol_bytes
    }

    /// Get the number of parity streams.
    pub fn num_parities(&self) -> usize {
        self.depths.len()
    }

    /// Overhead ratio (parity bytes / source bytes).
    pub fn overhead(&self) -> f32 {
        self.depths.len() as f32
    }

    /// Maximum burst that can be recovered.
    pub fn max_burst(&self) -> u16 {
        self.max_depth
    }
}

// =============================================================================
// Encoder
// =============================================================================

/// Result of adding a source symbol to the encoder.
#[derive(Debug, Clone)]
pub struct DiagonalEncodeResult {
    /// Sequence number assigned to the source symbol.
    pub source_seq: u16,

    /// Generated parity symbols (one per depth, after warmup).
    pub parities: Vec<DiagonalParity>,
}

/// A generated diagonal parity symbol.
#[derive(Debug, Clone)]
pub struct DiagonalParity {
    /// Sequence number of this parity (matches source_seq).
    pub seq: u16,

    /// Which depth/stream this parity belongs to (index into depths array).
    pub depth_index: u8,

    /// The interleaving depth τ for this parity.
    pub depth: u16,

    /// The parity data: S_seq ⊕ S_{seq-depth}
    pub data: Vec<u8>,
}

/// Diagonal interleaving encoder.
///
/// Generates one parity per source per depth, where each parity is:
/// P_t = S_t ⊕ S_{t-τ}
#[derive(Debug)]
pub struct DiagonalEncoder {
    params: DiagonalParams,

    /// Ring buffer of recent source symbols.
    /// Size = max_depth + 1
    buffer: VecDeque<Vec<u8>>,

    /// Sequence number of the oldest symbol in buffer.
    buffer_start_seq: u16,

    /// Next sequence number to assign.
    next_seq: u16,
}

impl DiagonalEncoder {
    /// Create a new diagonal encoder.
    pub fn new(params: DiagonalParams) -> Self {
        let buffer_size = params.max_depth as usize + 1;
        Self {
            params,
            buffer: VecDeque::with_capacity(buffer_size),
            buffer_start_seq: 0,
            next_seq: 0,
        }
    }

    /// Get the parameters.
    pub fn params(&self) -> &DiagonalParams {
        &self.params
    }

    /// Get the next sequence number.
    pub fn next_seq(&self) -> u16 {
        self.next_seq
    }

    /// Add a source symbol and generate parities.
    ///
    /// Returns the assigned sequence number and parity symbols.
    /// During warmup (first max_depth packets), fewer parities are generated.
    pub fn add_source(&mut self, data: &[u8]) -> DiagonalEncodeResult {
        assert_eq!(
            data.len(),
            self.params.symbol_bytes,
            "Source data must be exactly {} bytes",
            self.params.symbol_bytes
        );

        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);

        // Add to buffer
        self.buffer.push_back(data.to_vec());

        // Generate parities for each depth
        let mut parities = Vec::with_capacity(self.params.depths.len());

        for (depth_idx, &depth) in self.params.depths.iter().enumerate() {
            // Check if we have enough history for this depth
            if self.buffer.len() > depth as usize {
                // P_t = S_t ⊕ S_{t-depth}
                let current_idx = self.buffer.len() - 1;
                let delayed_idx = current_idx - depth as usize;

                let current = &self.buffer[current_idx];
                let delayed = &self.buffer[delayed_idx];

                // XOR the two sources
                let mut parity_data = vec![0u8; self.params.symbol_bytes];
                for i in 0..self.params.symbol_bytes {
                    parity_data[i] = current[i] ^ delayed[i];
                }

                parities.push(DiagonalParity {
                    seq,
                    depth_index: depth_idx as u8,
                    depth,
                    data: parity_data,
                });
            }
        }

        // Trim buffer if it's too large
        let max_buffer_size = self.params.max_depth as usize + 1;
        while self.buffer.len() > max_buffer_size {
            self.buffer.pop_front();
            self.buffer_start_seq = self.buffer_start_seq.wrapping_add(1);
        }

        DiagonalEncodeResult {
            source_seq: seq,
            parities,
        }
    }

    /// Reset the encoder state.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.buffer_start_seq = 0;
        self.next_seq = 0;
    }
}

// =============================================================================
// Decoder
// =============================================================================

/// Received parity for decoding.
#[derive(Debug, Clone)]
pub struct ReceivedDiagonalParity {
    /// Sequence number of this parity.
    pub seq: u16,

    /// Depth index (which parity stream).
    pub depth_index: u8,

    /// The interleaving depth τ.
    pub depth: u16,

    /// The parity data.
    pub data: Vec<u8>,
}

/// Diagonal interleaving decoder.
///
/// Recovery is simple: if S_t is lost but we have P_t and S_{t-τ}:
/// S_t = P_t ⊕ S_{t-τ}
///
/// Or if S_{t-τ} is lost but we have P_t and S_t:
/// S_{t-τ} = P_t ⊕ S_t
#[derive(Debug)]
pub struct DiagonalDecoder {
    params: DiagonalParams,

    /// Received source symbols: seq -> data
    sources: HashMap<u16, Vec<u8>>,

    /// Received parity symbols: (seq, depth_index) -> parity
    parities: HashMap<(u16, u8), ReceivedDiagonalParity>,

    /// Highest sequence seen (for cleanup).
    max_seen_seq: Option<u16>,

    /// History window size (in symbols).
    history_size: u16,
}

impl DiagonalDecoder {
    /// Create a new diagonal decoder.
    pub fn new(params: DiagonalParams) -> Self {
        Self {
            history_size: params.max_depth * 4,
            params,
            sources: HashMap::new(),
            parities: HashMap::new(),
            max_seen_seq: None,
        }
    }

    /// Get the parameters.
    pub fn params(&self) -> &DiagonalParams {
        &self.params
    }

    /// Set the history window size.
    pub fn set_history_size(&mut self, size: u16) {
        self.history_size = size;
    }

    /// Add a received source symbol.
    pub fn add_source(&mut self, seq: u16, data: &[u8]) {
        debug_assert_eq!(data.len(), self.params.symbol_bytes);
        self.sources.insert(seq, data.to_vec());
        self.update_max_seq(seq);
    }

    /// Add a received parity symbol.
    pub fn add_parity(&mut self, parity: ReceivedDiagonalParity) {
        debug_assert_eq!(parity.data.len(), self.params.symbol_bytes);
        self.parities
            .insert((parity.seq, parity.depth_index), parity);
    }

    /// Check if a source is available.
    pub fn has_source(&self, seq: u16) -> bool {
        self.sources.contains_key(&seq)
    }

    /// Get a source if available.
    pub fn get_source(&self, seq: u16) -> Option<&[u8]> {
        self.sources.get(&seq).map(|v| v.as_slice())
    }

    /// Attempt to recover missing symbols.
    ///
    /// Returns newly recovered (seq, data) pairs.
    pub fn try_recover(&mut self) -> Vec<(u16, Vec<u8>)> {
        let mut recovered = Vec::new();
        let mut made_progress = true;

        // Keep iterating until no more progress
        while made_progress {
            made_progress = false;

            // Try each parity
            let parity_keys: Vec<_> = self.parities.keys().cloned().collect();

            for (parity_seq, depth_idx) in parity_keys {
                let Some(parity) = self.parities.get(&(parity_seq, depth_idx)) else {
                    continue;
                };

                let depth = parity.depth;
                let delayed_seq = parity_seq.wrapping_sub(depth);

                let have_current = self.sources.contains_key(&parity_seq);
                let have_delayed = self.sources.contains_key(&delayed_seq);

                if have_current && have_delayed {
                    // Both sources present, parity not needed
                    continue;
                }

                if !have_current && !have_delayed {
                    // Can't recover - need at least one source
                    continue;
                }

                // Can recover!
                let (missing_seq, known_seq) = if have_current {
                    (delayed_seq, parity_seq)
                } else {
                    (parity_seq, delayed_seq)
                };

                let known_data = self.sources.get(&known_seq).unwrap();
                let parity_data = &parity.data;

                // Recover: missing = parity ⊕ known
                let mut recovered_data = vec![0u8; self.params.symbol_bytes];
                for i in 0..self.params.symbol_bytes {
                    recovered_data[i] = parity_data[i] ^ known_data[i];
                }

                self.sources.insert(missing_seq, recovered_data.clone());
                recovered.push((missing_seq, recovered_data));
                made_progress = true;
            }
        }

        // Cleanup old data
        self.cleanup();

        recovered
    }

    /// Reset the decoder state.
    pub fn reset(&mut self) {
        self.sources.clear();
        self.parities.clear();
        self.max_seen_seq = None;
    }

    fn update_max_seq(&mut self, seq: u16) {
        match self.max_seen_seq {
            None => self.max_seen_seq = Some(seq),
            Some(max) if seq.wrapping_sub(max) < 32768 => self.max_seen_seq = Some(seq),
            _ => {}
        }
    }

    fn cleanup(&mut self) {
        let Some(max_seq) = self.max_seen_seq else {
            return;
        };

        // Remove old sources
        self.sources.retain(|&seq, _| {
            let age = max_seq.wrapping_sub(seq);
            age < self.history_size
        });

        // Remove old parities
        self.parities.retain(|(seq, _), _| {
            let age = max_seq.wrapping_sub(*seq);
            age < self.history_size
        });
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_params() {
        let params = DiagonalParams::simple(8, 1200).unwrap();
        assert_eq!(params.depths(), &[8]);
        assert_eq!(params.max_depth(), 8);
        assert_eq!(params.symbol_bytes(), 1200);
        assert_eq!(params.num_parities(), 1);
        assert!((params.overhead() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_extended_params() {
        let params = DiagonalParams::extended(&[4, 8, 16], 1200).unwrap();
        assert_eq!(params.depths(), &[4, 8, 16]);
        assert_eq!(params.max_depth(), 16);
        assert_eq!(params.num_parities(), 3);
        assert!((params.overhead() - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_invalid_params() {
        assert!(DiagonalParams::simple(0, 1200).is_err());
        assert!(DiagonalParams::simple(8, 0).is_err());
        assert!(DiagonalParams::extended(&[], 1200).is_err());
        assert!(DiagonalParams::extended(&[0, 8], 1200).is_err());
    }

    #[test]
    fn test_encoder_warmup() {
        let params = DiagonalParams::simple(4, 16).unwrap();
        let mut encoder = DiagonalEncoder::new(params);

        // First 4 packets: no parities (warmup)
        for i in 0..4 {
            let data = vec![i as u8; 16];
            let result = encoder.add_source(&data);
            assert_eq!(result.source_seq, i);
            assert!(
                result.parities.is_empty(),
                "Packet {} should have no parity",
                i
            );
        }

        // 5th packet: first parity
        let data = vec![4u8; 16];
        let result = encoder.add_source(&data);
        assert_eq!(result.source_seq, 4);
        assert_eq!(result.parities.len(), 1);
        assert_eq!(result.parities[0].depth, 4);
    }

    #[test]
    fn test_parity_is_xor() {
        let params = DiagonalParams::simple(2, 4).unwrap();
        let mut encoder = DiagonalEncoder::new(params);

        let s0 = vec![0x11, 0x22, 0x33, 0x44];
        let s1 = vec![0x55, 0x66, 0x77, 0x88];
        let s2 = vec![0x99, 0xAA, 0xBB, 0xCC];

        encoder.add_source(&s0);
        encoder.add_source(&s1);
        let result = encoder.add_source(&s2);

        // P_2 = S_2 ⊕ S_0
        let expected: Vec<u8> = s2.iter().zip(s0.iter()).map(|(a, b)| a ^ b).collect();
        assert_eq!(result.parities.len(), 1);
        assert_eq!(result.parities[0].data, expected);
    }

    #[test]
    fn test_recover_single_loss() {
        let params = DiagonalParams::simple(2, 16).unwrap();
        let mut encoder = DiagonalEncoder::new(params.clone());
        let mut decoder = DiagonalDecoder::new(params);

        let originals: Vec<Vec<u8>> = (0..5).map(|i| vec![(i * 10) as u8; 16]).collect();

        for data in &originals {
            let result = encoder.add_source(data);

            // "Lose" packet 2
            if result.source_seq != 2 {
                decoder.add_source(result.source_seq, data);
            }

            // Receive all parities
            for parity in result.parities {
                decoder.add_parity(ReceivedDiagonalParity {
                    seq: parity.seq,
                    depth_index: parity.depth_index,
                    depth: parity.depth,
                    data: parity.data,
                });
            }
        }

        // Recover
        let recovered = decoder.try_recover();
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].0, 2);
        assert_eq!(recovered[0].1, originals[2]);
    }

    #[test]
    fn test_recover_burst() {
        // With depth=4, should recover burst of up to 4
        let params = DiagonalParams::simple(4, 16).unwrap();
        let mut encoder = DiagonalEncoder::new(params.clone());
        let mut decoder = DiagonalDecoder::new(params);

        let originals: Vec<Vec<u8>> = (0..12).map(|i| vec![(i * 10) as u8; 16]).collect();

        for (i, data) in originals.iter().enumerate() {
            let result = encoder.add_source(data);

            // "Lose" packets 4, 5, 6, 7 (burst of 4)
            if !(4..=7).contains(&i) {
                decoder.add_source(result.source_seq, data);
            }

            for parity in result.parities {
                decoder.add_parity(ReceivedDiagonalParity {
                    seq: parity.seq,
                    depth_index: parity.depth_index,
                    depth: parity.depth,
                    data: parity.data,
                });
            }
        }

        // Recover
        let recovered = decoder.try_recover();

        // Should recover all 4 lost packets
        let recovered_seqs: Vec<u16> = recovered.iter().map(|(seq, _)| *seq).collect();
        assert!(recovered_seqs.contains(&4));
        assert!(recovered_seqs.contains(&5));
        assert!(recovered_seqs.contains(&6));
        assert!(recovered_seqs.contains(&7));

        // Verify data
        for (seq, data) in recovered {
            assert_eq!(data, originals[seq as usize]);
        }
    }

    #[test]
    fn test_extended_diagonal() {
        // Multiple depths for better coverage
        let params = DiagonalParams::extended(&[2, 4], 16).unwrap();
        let mut encoder = DiagonalEncoder::new(params.clone());
        let mut decoder = DiagonalDecoder::new(params);

        let originals: Vec<Vec<u8>> = (0..10).map(|i| vec![(i * 10) as u8; 16]).collect();

        for (i, data) in originals.iter().enumerate() {
            let result = encoder.add_source(data);

            // Lose packets 3 and 5 (non-consecutive)
            if i != 3 && i != 5 {
                decoder.add_source(result.source_seq, data);
            }

            for parity in result.parities {
                decoder.add_parity(ReceivedDiagonalParity {
                    seq: parity.seq,
                    depth_index: parity.depth_index,
                    depth: parity.depth,
                    data: parity.data,
                });
            }
        }

        let recovered = decoder.try_recover();

        // Should recover both
        let recovered_seqs: Vec<u16> = recovered.iter().map(|(seq, _)| *seq).collect();
        assert!(recovered_seqs.contains(&3), "Should recover packet 3");
        assert!(recovered_seqs.contains(&5), "Should recover packet 5");
    }

    #[test]
    fn test_no_loss() {
        let params = DiagonalParams::simple(4, 16).unwrap();
        let mut encoder = DiagonalEncoder::new(params.clone());
        let mut decoder = DiagonalDecoder::new(params);

        for i in 0..10 {
            let data = vec![i as u8; 16];
            let result = encoder.add_source(&data);
            decoder.add_source(result.source_seq, &data);
            for parity in result.parities {
                decoder.add_parity(ReceivedDiagonalParity {
                    seq: parity.seq,
                    depth_index: parity.depth_index,
                    depth: parity.depth,
                    data: parity.data,
                });
            }
        }

        // No recovery needed
        let recovered = decoder.try_recover();
        assert!(recovered.is_empty());
    }
}
