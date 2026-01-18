//! # Diagonal Interleaving FEC
//!
//! True diagonal interleaving as described in "Erasure Coding for Real-Time Streaming"
//! (Leong & Ho, 2012).
//!
//! ## How It Works
//!
//! Each parity symbol is an XOR of source symbols at specific offsets:
//!
//! ```text
//! P_t = S_t ⊕ S_{t-τ} ⊕ S_{t-2τ} ⊕ ... ⊕ S_{t-(span-1)τ}
//! ```
//!
//! Where:
//! - τ (tau) is the interleaving depth (controls burst tolerance)
//! - span is the number of sources XOR'd per parity (controls overhead)
//!
//! ## Overhead Control
//!
//! The `span` parameter controls overhead:
//! - span=2: 1 parity per 1 source = 100% overhead (default, original behavior)
//! - span=3: 1 parity per 2 sources = 50% overhead
//! - span=5: 1 parity per 4 sources = 25% overhead
//! - span=6: 1 parity per 5 sources = 20% overhead
//!
//! Formula: overhead = 1/(span-1) or equivalently overhead = 1/data_sources_per_parity
//!
//! ## Burst Tolerance
//!
//! With interleaving depth τ and span s:
//! - A burst of up to τ consecutive losses can be recovered
//! - Recovery requires (span-1) of the span sources to be available
//! - Each lost source S_i can be recovered from P_{i+(span-1)τ} ⊕ other sources
//!
//! ## Extended Diagonal (Multiple Parities)
//!
//! For higher burst tolerance, use multiple interleaving depths:
//! - P_t,0 = S_t ⊕ S_{t-τ₀} ⊕ ...
//! - P_t,1 = S_t ⊕ S_{t-τ₁} ⊕ ...
//! - etc.
//!
//! ## Example Configurations
//!
//! | Config | Overhead | Burst Tolerance | Use Case |
//! |--------|----------|-----------------|----------|
//! | span=2, τ=4 | 100% | 4 packets | Maximum burst protection |
//! | span=5, τ=4 | 25% | 4 packets | Balanced |
//! | span=6, τ=8 | 20% | 8 packets | Low overhead, good burst |

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
/// // Simple diagonal with τ=8 (can recover burst of 8), 100% overhead
/// let params = DiagonalParams::simple(8, 1200).unwrap();
///
/// // Low overhead: span=6 gives 20% overhead, τ=4 gives burst tolerance of 4
/// let params = DiagonalParams::with_span(4, 6, 1200).unwrap();
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

    /// Number of sources XOR'd per parity (controls overhead).
    /// span=2 → 100% overhead, span=5 → 25% overhead, span=6 → 20% overhead
    span: u8,

    /// Size of each symbol in bytes.
    symbol_bytes: usize,
}

impl DiagonalParams {
    /// Create simple diagonal interleaving with a single depth.
    ///
    /// - `depth`: Interleaving depth τ (burst tolerance)
    /// - `symbol_bytes`: Size of each symbol in bytes
    ///
    /// Overhead: 100% (1 parity per source, span=2)
    pub fn simple(depth: u16, symbol_bytes: usize) -> Result<Self, Error> {
        if depth == 0 || symbol_bytes == 0 {
            return Err(Error::InvalidParams);
        }
        Ok(Self {
            depths: vec![depth],
            max_depth: depth,
            span: 2,
            symbol_bytes,
        })
    }

    /// Create diagonal interleaving with configurable span (overhead control).
    ///
    /// - `depth`: Interleaving depth τ (burst tolerance)
    /// - `span`: Number of sources XOR'd per parity (2-255)
    ///   - span=2: 100% overhead (1 parity per 1 source)
    ///   - span=3: 50% overhead (1 parity per 2 sources)
    ///   - span=5: 25% overhead (1 parity per 4 sources)
    ///   - span=6: 20% overhead (1 parity per 5 sources)
    /// - `symbol_bytes`: Size of each symbol in bytes
    ///
    /// # Recovery Requirements
    ///
    /// With span=s, recovery requires (s-1) of the s sources to be available.
    /// Burst tolerance remains τ, but sources must be spaced by τ apart.
    pub fn with_span(depth: u16, span: u8, symbol_bytes: usize) -> Result<Self, Error> {
        if depth == 0 || symbol_bytes == 0 || span < 2 {
            return Err(Error::InvalidParams);
        }
        Ok(Self {
            depths: vec![depth],
            max_depth: depth,
            span,
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
            span: 2,
            symbol_bytes,
        })
    }

    /// Create extended diagonal with configurable span.
    ///
    /// - `depths`: Array of interleaving depths [τ₀, τ₁, ...]
    /// - `span`: Number of sources XOR'd per parity (2-255)
    /// - `symbol_bytes`: Size of each symbol in bytes
    pub fn extended_with_span(
        depths: &[u16],
        span: u8,
        symbol_bytes: usize,
    ) -> Result<Self, Error> {
        if depths.is_empty() || symbol_bytes == 0 || span < 2 {
            return Err(Error::InvalidParams);
        }
        if depths.iter().any(|&d| d == 0) {
            return Err(Error::InvalidParams);
        }
        let max_depth = *depths.iter().max().unwrap();
        Ok(Self {
            depths: depths.to_vec(),
            max_depth,
            span,
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

    /// Get the span (number of sources per parity).
    pub fn span(&self) -> u8 {
        self.span
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
    ///
    /// With span=s: overhead = num_depths / (s - 1)
    /// - span=2: 1.0 per depth (100%)
    /// - span=5: 0.25 per depth (25%)
    /// - span=6: 0.2 per depth (20%)
    pub fn overhead(&self) -> f32 {
        self.depths.len() as f32 / (self.span - 1) as f32
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

    /// Number of sources XOR'd in this parity.
    pub span: u8,

    /// The parity data: S_seq ⊕ S_{seq-depth} ⊕ S_{seq-2*depth} ⊕ ...
    pub data: Vec<u8>,
}

/// Diagonal interleaving encoder.
///
/// Generates one parity per source per depth, where each parity is:
/// P_t = S_t ⊕ S_{t-τ} ⊕ S_{t-2τ} ⊕ ... ⊕ S_{t-(span-1)τ}
#[derive(Debug)]
pub struct DiagonalEncoder {
    params: DiagonalParams,

    /// Ring buffer of recent source symbols.
    /// Size = (span-1) * max_depth + 1
    buffer: VecDeque<Vec<u8>>,

    /// Sequence number of the oldest symbol in buffer.
    buffer_start_seq: u16,

    /// Next sequence number to assign.
    next_seq: u16,
}

impl DiagonalEncoder {
    /// Create a new diagonal encoder.
    pub fn new(params: DiagonalParams) -> Self {
        // Need buffer for (span-1) * max_depth + 1 symbols
        let buffer_size = (params.span as usize - 1) * params.max_depth as usize + 1;
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
    /// During warmup (first (span-1)*max_depth packets), fewer parities are generated.
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
        let span = self.params.span as usize;

        for (depth_idx, &depth) in self.params.depths.iter().enumerate() {
            // Check if we have enough history for this depth and span
            // Need (span-1) * depth symbols of history
            let required_history = (span - 1) * depth as usize;
            if self.buffer.len() > required_history {
                // Only emit parity every (span-1) sources
                // This gives overhead = 1/(span-1) per depth
                if seq as usize % (span - 1) != 0 {
                    continue;
                }

                // P_t = S_t ⊕ S_{t-depth} ⊕ S_{t-2*depth} ⊕ ... ⊕ S_{t-(span-1)*depth}
                let current_idx = self.buffer.len() - 1;
                // XOR all span sources
                let mut parity_data = vec![0u8; self.params.symbol_bytes];
                for s in 0..span {
                    let src_idx = current_idx - s * depth as usize;
                    let src = &self.buffer[src_idx];
                    for i in 0..self.params.symbol_bytes {
                        parity_data[i] ^= src[i];
                    }
                }

                parities.push(DiagonalParity {
                    seq,
                    depth_index: depth_idx as u8,
                    depth,
                    span: self.params.span,
                    data: parity_data,
                });
            }
        }

        // Trim buffer if it's too large
        let max_buffer_size = (span - 1) * self.params.max_depth as usize + 1;
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

    /// Number of sources XOR'd in this parity.
    pub span: u8,

    /// The parity data.
    pub data: Vec<u8>,
}

/// Diagonal interleaving decoder.
///
/// Recovery with span=s: if exactly 1 of the s sources is lost but we have P_t
/// and the other (s-1) sources, we can recover the missing one:
///
/// S_missing = P_t ⊕ S_known1 ⊕ S_known2 ⊕ ... ⊕ S_known{s-1}
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
        let span = params.span as u16;
        Self {
            history_size: params.max_depth * span * 2,
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
        let mut used_parities = Vec::new();

        // Keep iterating until no more progress
        while made_progress {
            made_progress = false;

            // Try each parity
            let parity_keys: Vec<_> = self.parities.keys().cloned().collect();

            for (parity_seq, depth_idx) in parity_keys {
                if used_parities.contains(&(parity_seq, depth_idx)) {
                    continue;
                }
                let Some(parity) = self.parities.get(&(parity_seq, depth_idx)) else {
                    continue;
                };

                let depth = parity.depth;
                let span = parity.span as usize;

                // Collect source sequence numbers for this parity
                // P_t = S_t ⊕ S_{t-depth} ⊕ S_{t-2*depth} ⊕ ... ⊕ S_{t-(span-1)*depth}
                let mut source_seqs = Vec::with_capacity(span);
                for s in 0..span {
                    let src_seq = parity_seq.wrapping_sub((s as u16) * depth);
                    source_seqs.push(src_seq);
                }

                // Count available and missing sources
                let mut missing_seq = None;
                let mut missing_count = 0;
                for &src_seq in &source_seqs {
                    if !self.sources.contains_key(&src_seq) {
                        missing_count += 1;
                        missing_seq = Some(src_seq);
                    }
                }

                // Can only recover if exactly 1 source is missing
                if missing_count != 1 {
                    continue;
                }

                let missing_seq = missing_seq.unwrap();

                // Recover: missing = parity ⊕ all_known_sources
                let mut recovered_data = parity.data.clone();
                for &src_seq in &source_seqs {
                    if src_seq != missing_seq {
                        let known_data = self.sources.get(&src_seq).unwrap();
                        for i in 0..self.params.symbol_bytes {
                            recovered_data[i] ^= known_data[i];
                        }
                    }
                }

                self.sources.insert(missing_seq, recovered_data.clone());
                recovered.push((missing_seq, recovered_data));
                used_parities.push((parity_seq, depth_idx));
                made_progress = true;
            }
        }

        // Remove used parities
        for key in used_parities {
            self.parities.remove(&key);
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
                    span: parity.span,
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
                    span: parity.span,
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
                    span: parity.span,
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
                    span: parity.span,
                    data: parity.data,
                });
            }
        }

        // No recovery needed
        let recovered = decoder.try_recover();
        assert!(recovered.is_empty());
    }

    #[test]
    fn test_span_overhead() {
        // span=2 → 100% overhead
        let p2 = DiagonalParams::simple(4, 16).unwrap();
        assert!((p2.overhead() - 1.0).abs() < 0.001);
        assert_eq!(p2.span(), 2);

        // span=5 → 25% overhead (1 parity per 4 sources)
        let p5 = DiagonalParams::with_span(4, 5, 16).unwrap();
        assert!((p5.overhead() - 0.25).abs() < 0.001);
        assert_eq!(p5.span(), 5);

        // span=6 → 20% overhead (1 parity per 5 sources)
        let p6 = DiagonalParams::with_span(4, 6, 16).unwrap();
        assert!((p6.overhead() - 0.2).abs() < 0.001);
        assert_eq!(p6.span(), 6);
    }

    #[test]
    fn test_span5_parity_xor() {
        // span=5: P_t = S_t ⊕ S_{t-τ} ⊕ S_{t-2τ} ⊕ S_{t-3τ} ⊕ S_{t-4τ}
        // With span=5, emit parity every 4 sources (after warmup)
        let params = DiagonalParams::with_span(1, 5, 4).unwrap(); // depth=1 for easy testing
        let mut encoder = DiagonalEncoder::new(params);

        let sources: Vec<Vec<u8>> = (0..5).map(|i| vec![(i * 0x11) as u8; 4]).collect();

        // Add first 4 sources - no parity yet (warmup)
        for src in &sources[..4] {
            let result = encoder.add_source(src);
            assert!(result.parities.is_empty());
        }

        // 5th source should trigger parity (index 4, source_count=4, 4%4=0)
        let result = encoder.add_source(&sources[4]);
        assert_eq!(result.parities.len(), 1);

        // P_4 = S_4 ⊕ S_3 ⊕ S_2 ⊕ S_1 ⊕ S_0
        let mut expected = vec![0u8; 4];
        for src in &sources {
            for j in 0..4 {
                expected[j] ^= src[j];
            }
        }
        assert_eq!(result.parities[0].data, expected);
        assert_eq!(result.parities[0].span, 5);
    }

    #[test]
    fn test_span5_recover_single() {
        // With span=5, can recover if exactly 1 of 5 sources is lost
        let params = DiagonalParams::with_span(2, 5, 16).unwrap();
        let mut encoder = DiagonalEncoder::new(params.clone());
        let mut decoder = DiagonalDecoder::new(params);

        let originals: Vec<Vec<u8>> = (0..20).map(|i| vec![(i * 7) as u8; 16]).collect();

        for (i, data) in originals.iter().enumerate() {
            let result = encoder.add_source(data);

            // "Lose" packet 10
            if i != 10 {
                decoder.add_source(result.source_seq, data);
            }

            for parity in result.parities {
                decoder.add_parity(ReceivedDiagonalParity {
                    seq: parity.seq,
                    depth_index: parity.depth_index,
                    depth: parity.depth,
                    span: parity.span,
                    data: parity.data,
                });
            }
        }

        // Recover
        let recovered = decoder.try_recover();
        let recovered_seqs: Vec<u16> = recovered.iter().map(|(seq, _)| *seq).collect();
        assert!(recovered_seqs.contains(&10), "Should recover packet 10");

        // Verify data
        for (seq, data) in &recovered {
            if *seq == 10 {
                assert_eq!(data, &originals[10]);
            }
        }
    }

    #[test]
    fn test_span6_actual_overhead() {
        // Verify that span=6 actually produces ~20% overhead
        let params = DiagonalParams::with_span(8, 6, 16).unwrap();
        let mut encoder = DiagonalEncoder::new(params);

        let mut source_count = 0u32;
        let mut parity_count = 0u32;

        // Send 1000 sources
        for i in 0..1000u32 {
            let data: Vec<u8> = (0..16).map(|j| ((i * 17 + j) & 0xFF) as u8).collect();
            let result = encoder.add_source(&data);
            source_count += 1;
            parity_count += result.parities.len() as u32;
        }

        let actual_overhead = parity_count as f64 / source_count as f64;
        // Should be close to 20% (0.2)
        assert!(
            (actual_overhead - 0.2).abs() < 0.05,
            "Expected ~20% overhead, got {:.1}%",
            actual_overhead * 100.0
        );
    }
}
