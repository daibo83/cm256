//! # Streaming Codes (Diagonal Interleaving)
//!
//! Low-latency FEC for real-time streaming applications using diagonal interleaving.
//!
//! ## Overview
//!
//! Unlike block FEC where parity protects a fixed block of symbols, diagonal interleaving
//! creates parity symbols that span diagonally across time. Each parity symbol covers
//! a sliding window of `delay` source symbols, providing continuous protection.
//!
//! ```text
//! Time:     0   1   2   3   4   5   6   7   8   ...
//! Source:   S₀  S₁  S₂  S₃  S₄  S₅  S₆  S₇  S₈  ...
//!           ╲   ╲   ╲   ╲   ╲
//! Parity:       P₀  P₁  P₂  P₃  P₄  ...
//!               │   │   │   │   │
//!               └───┴───┴───┴───┴── Each Pᵢ covers [Sᵢ₋ₐ₊₁, Sᵢ]
//! ```
//!
//! ## When to Use
//!
//! - Video/audio streaming where latency < 100ms is critical
//! - Networks with bursty packet loss
//! - When you need continuous protection (not block-based)
//!
//! ## Example
//!
//! ```rust
//! use cm256::streaming::{StreamingEncoder, StreamingDecoder, StreamingParams};
//!
//! // delay=8 symbols, 2 parities per window, 1200 bytes per symbol
//! let params = StreamingParams::new(8, 2, 1200).unwrap();
//!
//! let mut encoder = StreamingEncoder::new(params);
//! let mut decoder = StreamingDecoder::new(params);
//!
//! // Encode some data
//! let data = vec![0x42u8; 1200];
//! let result = encoder.add_source(&data);
//!
//! // result.parities contains generated parity symbols (if window is full)
//! ```

use std::collections::{HashMap, VecDeque};

use crate::{gf256_mul_mem_inplace, gf256_muladd_mem, Error, Gf256};

// =============================================================================
// Configuration
// =============================================================================

/// Parameters for streaming FEC encoding/decoding.
///
/// The streaming code uses a sliding window of `delay` source symbols.
/// Parity symbols are generated every `step_size` source symbols.
///
/// # Properties
///
/// - **Burst tolerance**: Can recover from loss of up to `num_parities` consecutive symbols
/// - **Overhead**: `num_parities / step_size` (e.g., 3/5 = 60%)
/// - **Latency**: Decoding requires waiting for the full window
/// - **Overlap**: Each packet is covered by `delay / step_size` parity groups
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamingParams {
    /// Number of source symbols covered by each parity (span/delay).
    /// Range: 2..=255 (limited by GF(256) Cauchy matrix).
    delay: u8,

    /// Number of parity symbols generated per window.
    /// Range: 1..=(256 - delay).
    num_parities: u8,

    /// How often to generate parity (every step_size source symbols).
    /// Range: 1..=delay. Lower = more overlap, higher overhead.
    step_size: u8,

    /// Size of each symbol in bytes.
    symbol_bytes: usize,
}

impl StreamingParams {
    /// Create new streaming parameters.
    ///
    /// # Arguments
    ///
    /// * `delay` - Number of source symbols each parity covers (2..=255)
    /// * `num_parities` - Number of parity symbols per window (1..=(256-delay))
    /// * `symbol_bytes` - Size of each symbol in bytes (> 0)
    ///
    /// Uses `step_size = delay` by default (block-like behavior, lowest overhead).
    /// Use `with_step_size()` for more frequent parity generation.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidParams` if parameters are out of range.
    /// Returns `Error::TooManyBlocks` if delay + num_parities > 256.
    pub fn new(delay: u8, num_parities: u8, symbol_bytes: usize) -> Result<Self, Error> {
        Self::with_step_size(delay, num_parities, delay, symbol_bytes)
    }

    /// Create new streaming parameters with custom step size.
    ///
    /// # Arguments
    ///
    /// * `delay` - Number of source symbols each parity covers (2..=255)
    /// * `num_parities` - Number of parity symbols per window (1..=(256-delay))
    /// * `step_size` - Generate parities every N source symbols (1..=delay)
    /// * `symbol_bytes` - Size of each symbol in bytes (> 0)
    ///
    /// # Overhead Calculation
    ///
    /// - `step_size = delay`: Overhead = `num_parities / delay` (e.g., 3/15 = 20%)
    /// - `step_size = 1`: Overhead = `num_parities` per source (e.g., 3 = 300%)
    /// - `step_size = 5`: Overhead = `num_parities / 5` (e.g., 3/5 = 60%)
    pub fn with_step_size(
        delay: u8,
        num_parities: u8,
        step_size: u8,
        symbol_bytes: usize,
    ) -> Result<Self, Error> {
        if delay < 2 || num_parities == 0 || symbol_bytes == 0 {
            return Err(Error::InvalidParams);
        }
        if step_size == 0 || step_size > delay {
            return Err(Error::InvalidParams);
        }
        if (delay as usize) + (num_parities as usize) > 256 {
            return Err(Error::TooManyBlocks);
        }
        Ok(Self {
            delay,
            num_parities,
            step_size,
            symbol_bytes,
        })
    }

    /// Get the delay (window size).
    #[inline]
    pub fn delay(&self) -> u8 {
        self.delay
    }

    /// Get the number of parities per window.
    #[inline]
    pub fn num_parities(&self) -> u8 {
        self.num_parities
    }

    /// Get the step size.
    #[inline]
    pub fn step_size(&self) -> u8 {
        self.step_size
    }

    /// Get the symbol size in bytes.
    #[inline]
    pub fn symbol_bytes(&self) -> usize {
        self.symbol_bytes
    }

    /// Maximum burst loss that can be fully recovered.
    #[inline]
    pub fn max_burst(&self) -> u8 {
        self.num_parities
    }

    /// Overhead ratio (parity bytes / source bytes).
    #[inline]
    pub fn overhead(&self) -> f32 {
        self.num_parities as f32 / self.step_size as f32
    }
}

// =============================================================================
// Sequence Number Utilities
// =============================================================================

/// Check if sequence `a` is before `b` in wrapping u16 space.
/// Returns true if (b - a) < 32768 (half the sequence space).
#[inline]
fn seq_before(a: u16, b: u16) -> bool {
    b.wrapping_sub(a) < 32768
}

/// Check if a sequence is within a range [start, start + len).
#[inline]
#[allow(dead_code)]
fn seq_in_range(seq: u16, start: u16, len: u8) -> bool {
    seq.wrapping_sub(start) < len as u16
}

// =============================================================================
// Encoder
// =============================================================================

/// Result of adding a source symbol to the encoder.
#[derive(Debug, Clone)]
pub struct AddSourceResult {
    /// Sequence number assigned to the source symbol.
    pub source_seq: u16,

    /// Generated parity symbols (empty if window not yet full).
    pub parities: Vec<ParityOutput>,
}

/// A generated parity symbol.
#[derive(Debug, Clone)]
pub struct ParityOutput {
    /// Sequence of the newest source this parity covers.
    /// The parity covers sources [end_seq - delay + 1, end_seq].
    pub end_seq: u16,

    /// Index of this parity within the window (0..num_parities).
    pub parity_index: u8,

    /// The parity data.
    pub data: Vec<u8>,
}

/// Streaming FEC encoder using diagonal interleaving.
///
/// The encoder maintains a sliding window of recent source symbols and
/// generates parity symbols every `step_size` packets as the window advances.
#[derive(Debug)]
pub struct StreamingEncoder {
    params: StreamingParams,

    /// Ring buffer of recent source symbols.
    buffer: VecDeque<Vec<u8>>,

    /// Next sequence number to assign.
    next_seq: u16,

    /// Counter for step_size tracking.
    packets_since_parity: u8,
}

impl StreamingEncoder {
    /// Create a new streaming encoder.
    pub fn new(params: StreamingParams) -> Self {
        Self {
            params,
            buffer: VecDeque::with_capacity(params.delay as usize),
            next_seq: 0,
            packets_since_parity: 0,
        }
    }

    /// Get the current parameters.
    pub fn params(&self) -> StreamingParams {
        self.params
    }

    /// Get the next sequence number that will be assigned.
    pub fn next_seq(&self) -> u16 {
        self.next_seq
    }

    /// Add a source symbol to the encoder.
    ///
    /// Returns the assigned sequence number and any generated parity symbols.
    /// Parity symbols are generated every `step_size` packets when the window is full.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != params.symbol_bytes()`.
    pub fn add_source(&mut self, data: &[u8]) -> AddSourceResult {
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
        self.packets_since_parity += 1;

        // Check if buffer is full and it's time to generate parities
        if self.buffer.len() < self.params.delay as usize {
            return AddSourceResult {
                source_seq: seq,
                parities: vec![],
            };
        }

        // Only generate parities every step_size packets
        if self.packets_since_parity < self.params.step_size {
            // Slide window but don't generate parities yet
            self.buffer.pop_front();
            return AddSourceResult {
                source_seq: seq,
                parities: vec![],
            };
        }

        // Generate parities for the current window
        let parities = self.generate_parities(seq);

        // Slide window by step_size
        for _ in 0..self.params.step_size {
            if !self.buffer.is_empty() {
                self.buffer.pop_front();
            }
        }
        self.packets_since_parity = 0;

        AddSourceResult {
            source_seq: seq,
            parities,
        }
    }

    /// Flush the encoder, generating final parities for remaining data.
    ///
    /// Call this at end-of-stream to ensure all data is protected.
    /// Returns parity symbols for the final (possibly partial) window.
    pub fn flush(&mut self) -> Vec<ParityOutput> {
        if self.buffer.len() < 2 {
            // Need at least 2 symbols for meaningful parity
            self.buffer.clear();
            return vec![];
        }

        // Generate parities for remaining buffer
        let end_seq = self.next_seq.wrapping_sub(1);
        let parities = self.generate_parities_for_buffer(end_seq);

        self.buffer.clear();
        parities
    }

    /// Reset the encoder state.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.next_seq = 0;
        self.packets_since_parity = 0;
    }

    /// Generate parity symbols for the current full window.
    fn generate_parities(&self, end_seq: u16) -> Vec<ParityOutput> {
        debug_assert_eq!(self.buffer.len(), self.params.delay as usize);
        self.generate_parities_for_buffer(end_seq)
    }

    /// Generate parity symbols for whatever is in the buffer.
    fn generate_parities_for_buffer(&self, end_seq: u16) -> Vec<ParityOutput> {
        let window_len = self.buffer.len() as u8;
        let mut parities = Vec::with_capacity(self.params.num_parities as usize);

        for parity_idx in 0..self.params.num_parities {
            let mut parity_data = vec![0u8; self.params.symbol_bytes];

            for (pos, source) in self.buffer.iter().enumerate() {
                let coeff = self.coefficient(parity_idx, pos as u8, window_len);
                gf256_muladd_mem(&mut parity_data, source, coeff);
            }

            parities.push(ParityOutput {
                end_seq,
                parity_index: parity_idx,
                data: parity_data,
            });
        }

        parities
    }

    /// Get encoding coefficient for (parity_index, source_position).
    ///
    /// Uses Cauchy matrix structure for MDS property (same as cm256).
    /// The matrix is: a_ij = (y_j + x_0) / (x_i + y_j)
    /// where x_i = window_len + parity_idx (recovery row)
    ///       x_0 = window_len (first recovery row, for normalization)
    ///       y_j = source_pos (column)
    #[inline]
    fn coefficient(&self, parity_idx: u8, source_pos: u8, window_len: u8) -> Gf256 {
        // Recovery row index
        let x_i = window_len + parity_idx;
        // First recovery row index (for normalization)
        let x_0 = window_len;
        // Source column
        let y_j = source_pos;

        // Cauchy element: (y_j + x_0) / (x_i + y_j)
        let numerator = Gf256(y_j) + Gf256(x_0);
        let denominator = Gf256(x_i) + Gf256(y_j);
        numerator / denominator
    }
}

// =============================================================================
// Decoder
// =============================================================================

/// A received parity symbol for decoding.
#[derive(Debug, Clone)]
pub struct ReceivedParity {
    /// Sequence of the newest source this parity covers.
    pub end_seq: u16,

    /// Index of this parity within the window (0..num_parities).
    pub parity_index: u8,

    /// The parity data.
    pub data: Vec<u8>,
}

/// Streaming FEC decoder using diagonal interleaving.
///
/// The decoder collects source and parity symbols and attempts to recover
/// missing sources when enough parity is available.
#[derive(Debug)]
pub struct StreamingDecoder {
    params: StreamingParams,

    /// Received source symbols: seq -> data.
    sources: HashMap<u16, Vec<u8>>,

    /// Received parity symbols.
    parities: Vec<ReceivedParity>,

    /// Highest source sequence number seen.
    max_seen_seq: Option<u16>,

    /// Number of windows to keep for recovery attempts.
    /// Older data is garbage collected.
    history_windows: u16,
}

impl StreamingDecoder {
    /// Create a new streaming decoder.
    pub fn new(params: StreamingParams) -> Self {
        Self {
            params,
            sources: HashMap::new(),
            parities: Vec::new(),
            max_seen_seq: None,
            history_windows: 4, // Keep 4 windows of history by default
        }
    }

    /// Get the current parameters.
    pub fn params(&self) -> StreamingParams {
        self.params
    }

    /// Set the number of historical windows to keep for recovery.
    ///
    /// Higher values use more memory but allow recovery of older losses.
    pub fn set_history_windows(&mut self, windows: u16) {
        self.history_windows = windows;
    }

    /// Add a received source symbol.
    pub fn add_source(&mut self, seq: u16, data: &[u8]) {
        debug_assert_eq!(data.len(), self.params.symbol_bytes);

        self.sources.insert(seq, data.to_vec());
        self.update_max_seq(seq);
    }

    /// Add a received parity symbol.
    pub fn add_parity(&mut self, end_seq: u16, parity_index: u8, data: &[u8]) {
        debug_assert_eq!(data.len(), self.params.symbol_bytes);
        debug_assert!(parity_index < self.params.num_parities);

        self.parities.push(ReceivedParity {
            end_seq,
            parity_index,
            data: data.to_vec(),
        });
        self.update_max_seq(end_seq);
    }

    /// Check if a source symbol is available (received or recovered).
    pub fn has_source(&self, seq: u16) -> bool {
        self.sources.contains_key(&seq)
    }

    /// Get a source symbol if available.
    pub fn get_source(&self, seq: u16) -> Option<&[u8]> {
        self.sources.get(&seq).map(|v| v.as_slice())
    }

    /// Attempt to recover missing symbols.
    ///
    /// Returns a list of (sequence, data) for newly recovered symbols.
    /// Recovered symbols are also stored internally and can be retrieved
    /// via `get_source()`.
    pub fn try_recover(&mut self) -> Vec<(u16, Vec<u8>)> {
        let mut all_recovered = Vec::new();

        // Group parities by the window they cover
        let mut windows: HashMap<u16, Vec<usize>> = HashMap::new();
        for (idx, parity) in self.parities.iter().enumerate() {
            windows.entry(parity.end_seq).or_default().push(idx);
        }

        // Try recovery for each window
        for (end_seq, parity_indices) in windows {
            let start_seq = end_seq.wrapping_sub(self.params.delay as u16 - 1);

            // Find missing and present sources in this window
            let mut missing: Vec<(u8, u16)> = Vec::new();
            let mut present: Vec<(u8, u16)> = Vec::new();

            for i in 0..self.params.delay {
                let seq = start_seq.wrapping_add(i as u16);
                if self.sources.contains_key(&seq) {
                    present.push((i, seq));
                } else {
                    missing.push((i, seq));
                }
            }

            if missing.is_empty() {
                continue; // Nothing to recover
            }

            // Check if we have enough parities
            if parity_indices.len() < missing.len() {
                continue; // Not enough parities
            }

            // Collect the relevant parities
            let relevant_parities: Vec<_> = parity_indices
                .iter()
                .take(missing.len())
                .map(|&idx| &self.parities[idx])
                .collect();

            // Attempt to solve
            if let Some(solutions) = self.solve_window(&present, &missing, &relevant_parities) {
                for (seq, data) in solutions {
                    self.sources.insert(seq, data.clone());
                    all_recovered.push((seq, data));
                }
            }
        }

        // Cleanup old data
        self.cleanup();

        all_recovered
    }

    /// Reset the decoder state.
    pub fn reset(&mut self) {
        self.sources.clear();
        self.parities.clear();
        self.max_seen_seq = None;
    }

    /// Update the maximum seen sequence number.
    fn update_max_seq(&mut self, seq: u16) {
        match self.max_seen_seq {
            None => self.max_seen_seq = Some(seq),
            Some(max) if seq_before(max, seq) => self.max_seen_seq = Some(seq),
            _ => {}
        }
    }

    /// Solve for missing symbols using Gaussian elimination.
    fn solve_window(
        &self,
        present: &[(u8, u16)],
        missing: &[(u8, u16)],
        parities: &[&ReceivedParity],
    ) -> Option<Vec<(u16, Vec<u8>)>> {
        let m = missing.len();
        if m == 0 || parities.len() < m {
            return None;
        }

        let window_len = self.params.delay;

        // Build m×m coefficient matrix and RHS vectors
        let mut matrix: Vec<Vec<Gf256>> = vec![vec![Gf256(0); m]; m];
        let mut rhs: Vec<Vec<u8>> = vec![vec![0u8; self.params.symbol_bytes]; m];

        for (row, parity) in parities.iter().enumerate() {
            // Start with parity data
            rhs[row].copy_from_slice(&parity.data);

            // Subtract contribution of known sources
            for &(pos, seq) in present {
                let coeff = self.coefficient(parity.parity_index, pos, window_len);
                if let Some(source) = self.sources.get(&seq) {
                    gf256_muladd_mem(&mut rhs[row], source, coeff);
                }
            }

            // Build coefficient matrix for unknowns
            for (col, &(pos, _seq)) in missing.iter().enumerate() {
                matrix[row][col] = self.coefficient(parity.parity_index, pos, window_len);
            }
        }

        // Gaussian elimination with back-substitution
        self.gaussian_eliminate(&mut matrix, &mut rhs)?;

        // Build result
        Some(
            missing
                .iter()
                .zip(rhs.into_iter())
                .map(|(&(_pos, seq), data)| (seq, data))
                .collect(),
        )
    }

    /// Get encoding coefficient (same as encoder).
    #[inline]
    fn coefficient(&self, parity_idx: u8, source_pos: u8, window_len: u8) -> Gf256 {
        let x_i = window_len + parity_idx;
        let x_0 = window_len;
        let y_j = source_pos;

        let numerator = Gf256(y_j) + Gf256(x_0);
        let denominator = Gf256(x_i) + Gf256(y_j);
        numerator / denominator
    }

    /// Gaussian elimination with full pivoting and back-substitution.
    fn gaussian_eliminate(&self, matrix: &mut [Vec<Gf256>], rhs: &mut [Vec<u8>]) -> Option<()> {
        let n = matrix.len();
        if n == 0 {
            return Some(());
        }

        // Forward elimination
        for col in 0..n {
            // Find pivot (first non-zero in column)
            let pivot_row = (col..n).find(|&r| matrix[r][col].0 != 0)?;

            // Swap rows if needed
            if pivot_row != col {
                matrix.swap(col, pivot_row);
                rhs.swap(col, pivot_row);
            }

            let pivot = matrix[col][col];
            let pivot_inv = pivot.inv();

            // Scale pivot row to make pivot = 1
            for j in col..n {
                matrix[col][j] = matrix[col][j] * pivot_inv;
            }
            gf256_mul_mem_inplace(&mut rhs[col], pivot_inv);

            // Eliminate below pivot
            for row in (col + 1)..n {
                let factor = matrix[row][col];
                if factor.0 == 0 {
                    continue;
                }

                for j in col..n {
                    let pivot_val = matrix[col][j];
                    matrix[row][j] = matrix[row][j] + pivot_val * factor;
                }

                // rhs[row] += rhs[col] * factor
                let col_rhs = rhs[col].clone();
                gf256_muladd_mem(&mut rhs[row], &col_rhs, factor);
            }
        }

        // Back substitution
        for col in (1..n).rev() {
            for row in 0..col {
                let factor = matrix[row][col];
                if factor.0 == 0 {
                    continue;
                }

                // We don't need to update matrix since we only care about RHS
                let col_rhs = rhs[col].clone();
                gf256_muladd_mem(&mut rhs[row], &col_rhs, factor);
            }
        }

        Some(())
    }

    /// Cleanup old data that's outside the history window.
    fn cleanup(&mut self) {
        let Some(max_seq) = self.max_seen_seq else {
            return;
        };

        let history_len = self.history_windows * self.params.delay as u16;
        let min_keep = max_seq.wrapping_sub(history_len);

        // Remove old sources
        self.sources
            .retain(|&seq, _| !seq_before(seq, min_keep) || seq_before(max_seq, seq));

        // Remove old parities
        self.parities
            .retain(|p| !seq_before(p.end_seq, min_keep) || seq_before(max_seq, p.end_seq));
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params_valid() {
        let params = StreamingParams::new(8, 2, 1200).unwrap();
        assert_eq!(params.delay(), 8);
        assert_eq!(params.num_parities(), 2);
        assert_eq!(params.symbol_bytes(), 1200);
        assert_eq!(params.max_burst(), 2);
        assert!((params.overhead() - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_params_invalid() {
        // delay too small
        assert!(StreamingParams::new(1, 2, 1200).is_err());
        // num_parities zero
        assert!(StreamingParams::new(8, 0, 1200).is_err());
        // symbol_bytes zero
        assert!(StreamingParams::new(8, 2, 0).is_err());
        // exceeds 256
        assert!(StreamingParams::new(200, 100, 1200).is_err());
    }

    #[test]
    fn test_encoder_generates_parities_when_full() {
        let params = StreamingParams::new(4, 2, 16).unwrap();
        let mut encoder = StreamingEncoder::new(params);

        // Add 3 symbols - no parities yet
        for i in 0..3 {
            let data = vec![i as u8; 16];
            let result = encoder.add_source(&data);
            assert_eq!(result.source_seq, i);
            assert!(result.parities.is_empty());
        }

        // Add 4th symbol - should generate parities
        let data = vec![3u8; 16];
        let result = encoder.add_source(&data);
        assert_eq!(result.source_seq, 3);
        assert_eq!(result.parities.len(), 2);
        assert_eq!(result.parities[0].end_seq, 3);
        assert_eq!(result.parities[0].parity_index, 0);
        assert_eq!(result.parities[1].parity_index, 1);
    }

    #[test]
    fn test_encode_decode_no_loss() {
        let params = StreamingParams::new(4, 2, 32).unwrap();
        let mut encoder = StreamingEncoder::new(params);
        let mut decoder = StreamingDecoder::new(params);

        // Create and encode 8 symbols
        let originals: Vec<Vec<u8>> = (0..8).map(|i| vec![(i * 10) as u8; 32]).collect();

        for data in originals.iter() {
            let result = encoder.add_source(data);

            // "Receive" the source
            decoder.add_source(result.source_seq, data);

            // "Receive" parities
            for parity in &result.parities {
                decoder.add_parity(parity.end_seq, parity.parity_index, &parity.data);
            }
        }

        // No recovery needed - all sources present
        let recovered = decoder.try_recover();
        assert!(recovered.is_empty());

        // Verify all sources are accessible
        for i in 0..8 {
            assert!(decoder.has_source(i));
            assert_eq!(decoder.get_source(i).unwrap(), &originals[i as usize][..]);
        }
    }

    #[test]
    fn test_recover_single_loss() {
        let params = StreamingParams::new(4, 2, 32).unwrap();
        let mut encoder = StreamingEncoder::new(params);
        let mut decoder = StreamingDecoder::new(params);

        let originals: Vec<Vec<u8>> = (0..4).map(|i| vec![(i * 10) as u8; 32]).collect();

        for (i, data) in originals.iter().enumerate() {
            let result = encoder.add_source(data);

            // "Lose" packet 2
            if i != 2 {
                decoder.add_source(result.source_seq, data);
            }

            // Receive all parities
            for parity in &result.parities {
                decoder.add_parity(parity.end_seq, parity.parity_index, &parity.data);
            }
        }

        // Attempt recovery
        let recovered = decoder.try_recover();
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].0, 2);
        assert_eq!(recovered[0].1, originals[2]);

        // Verify recovered symbol is now accessible
        assert!(decoder.has_source(2));
        assert_eq!(decoder.get_source(2).unwrap(), &originals[2][..]);
    }

    #[test]
    fn test_recover_burst_at_limit() {
        // With 2 parities, should recover burst of 2
        let params = StreamingParams::new(4, 2, 32).unwrap();
        let mut encoder = StreamingEncoder::new(params);
        let mut decoder = StreamingDecoder::new(params);

        let originals: Vec<Vec<u8>> = (0..4).map(|i| vec![(i * 10) as u8; 32]).collect();

        for (i, data) in originals.iter().enumerate() {
            let result = encoder.add_source(data);

            // "Lose" packets 1 and 2 (burst of 2)
            if i != 1 && i != 2 {
                decoder.add_source(result.source_seq, data);
            }

            for parity in &result.parities {
                decoder.add_parity(parity.end_seq, parity.parity_index, &parity.data);
            }
        }

        let recovered = decoder.try_recover();
        assert_eq!(recovered.len(), 2);

        // Check both were recovered correctly
        let recovered_map: HashMap<u16, Vec<u8>> = recovered.into_iter().collect();
        assert_eq!(recovered_map.get(&1).unwrap(), &originals[1]);
        assert_eq!(recovered_map.get(&2).unwrap(), &originals[2]);
    }

    #[test]
    fn test_burst_exceeds_capacity() {
        // With 2 parities, cannot recover burst of 3
        let params = StreamingParams::new(4, 2, 32).unwrap();
        let mut encoder = StreamingEncoder::new(params);
        let mut decoder = StreamingDecoder::new(params);

        let originals: Vec<Vec<u8>> = (0..4).map(|i| vec![(i * 10) as u8; 32]).collect();

        for (i, data) in originals.iter().enumerate() {
            let result = encoder.add_source(data);

            // "Lose" packets 1, 2, and 3 (burst of 3)
            if i == 0 {
                decoder.add_source(result.source_seq, data);
            }

            for parity in &result.parities {
                decoder.add_parity(parity.end_seq, parity.parity_index, &parity.data);
            }
        }

        // Cannot recover - not enough parities
        let recovered = decoder.try_recover();
        assert!(recovered.is_empty());
    }

    #[test]
    fn test_sequence_wraparound() {
        let params = StreamingParams::new(4, 1, 16).unwrap();
        let mut encoder = StreamingEncoder::new(params);

        // Manually set near wraparound
        encoder.next_seq = 65534;

        let data = vec![0x42u8; 16];

        let r1 = encoder.add_source(&data);
        assert_eq!(r1.source_seq, 65534);

        let r2 = encoder.add_source(&data);
        assert_eq!(r2.source_seq, 65535);

        let r3 = encoder.add_source(&data);
        assert_eq!(r3.source_seq, 0); // Wrapped

        let r4 = encoder.add_source(&data);
        assert_eq!(r4.source_seq, 1);
        assert!(!r4.parities.is_empty()); // Should have parities now
    }

    #[test]
    fn test_seq_before() {
        assert!(seq_before(0, 1));
        assert!(seq_before(100, 200));
        assert!(!seq_before(200, 100));

        // Wraparound cases
        assert!(seq_before(65535, 0)); // 65535 is "before" 0
        assert!(seq_before(65535, 100));
        assert!(!seq_before(0, 65535)); // 0 is "after" 65535
    }

    #[test]
    fn test_seq_in_range() {
        assert!(seq_in_range(5, 5, 3)); // 5 in [5, 8)
        assert!(seq_in_range(6, 5, 3));
        assert!(seq_in_range(7, 5, 3));
        assert!(!seq_in_range(8, 5, 3));
        assert!(!seq_in_range(4, 5, 3));

        // Wraparound
        assert!(seq_in_range(0, 65534, 4)); // 0 in [65534, 65534+4) wrapping
        assert!(seq_in_range(65535, 65534, 4));
        assert!(seq_in_range(1, 65534, 4));
    }

    #[test]
    fn test_interleaved_loss() {
        // Lose non-consecutive packets
        let params = StreamingParams::new(4, 2, 32).unwrap();
        let mut encoder = StreamingEncoder::new(params);
        let mut decoder = StreamingDecoder::new(params);

        let originals: Vec<Vec<u8>> = (0..4).map(|i| vec![(i * 10) as u8; 32]).collect();

        for (i, data) in originals.iter().enumerate() {
            let result = encoder.add_source(data);

            // Lose packets 0 and 2 (interleaved)
            if i != 0 && i != 2 {
                decoder.add_source(result.source_seq, data);
            }

            for parity in &result.parities {
                decoder.add_parity(parity.end_seq, parity.parity_index, &parity.data);
            }
        }

        let recovered = decoder.try_recover();
        assert_eq!(recovered.len(), 2);

        let recovered_map: HashMap<u16, Vec<u8>> = recovered.into_iter().collect();
        assert_eq!(recovered_map.get(&0).unwrap(), &originals[0]);
        assert_eq!(recovered_map.get(&2).unwrap(), &originals[2]);
    }

    #[test]
    fn test_multiple_windows() {
        let params = StreamingParams::new(4, 2, 32).unwrap();
        let mut encoder = StreamingEncoder::new(params);
        let mut decoder = StreamingDecoder::new(params);

        let originals: Vec<Vec<u8>> = (0..12).map(|i| vec![(i * 10) as u8; 32]).collect();

        for (i, data) in originals.iter().enumerate() {
            let result = encoder.add_source(data);

            // Lose one packet per window: 2, 6, 10
            if i != 2 && i != 6 && i != 10 {
                decoder.add_source(result.source_seq, data);
            }

            for parity in &result.parities {
                decoder.add_parity(parity.end_seq, parity.parity_index, &parity.data);
            }
        }

        let recovered = decoder.try_recover();

        // Should recover packets from multiple windows
        // Note: packet 2 is in window ending at 3, packet 6 in window ending at 6-9, etc.
        let recovered_seqs: Vec<u16> = recovered.iter().map(|(seq, _)| *seq).collect();
        assert!(recovered_seqs.contains(&2));

        // Verify data integrity
        for (seq, data) in recovered {
            assert_eq!(data, originals[seq as usize]);
        }
    }

    #[test]
    fn test_encoder_flush() {
        let params = StreamingParams::new(4, 2, 16).unwrap();
        let mut encoder = StreamingEncoder::new(params);

        // Add only 2 symbols (less than window)
        encoder.add_source(&vec![1u8; 16]);
        encoder.add_source(&vec![2u8; 16]);

        // Flush should generate parities for partial window
        let parities = encoder.flush();
        assert_eq!(parities.len(), 2);
    }

    #[test]
    fn test_coefficient_consistency() {
        let params = StreamingParams::new(4, 2, 16).unwrap();
        let encoder = StreamingEncoder::new(params);
        let decoder = StreamingDecoder::new(params);

        // Encoder and decoder should compute same coefficients
        for parity_idx in 0..2 {
            for pos in 0..4 {
                let enc_coeff = encoder.coefficient(parity_idx, pos, 4);
                let dec_coeff = decoder.coefficient(parity_idx, pos, 4);
                assert_eq!(
                    enc_coeff.0, dec_coeff.0,
                    "Coefficient mismatch at parity_idx={}, pos={}",
                    parity_idx, pos
                );
            }
        }
    }
}
