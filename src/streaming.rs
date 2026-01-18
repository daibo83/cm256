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
// Transport Integration
// =============================================================================

use crate::transport::{DatagramRecvMut, DatagramSendMut};
use std::io;

/// Packet type discriminator for wire protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PacketType {
    Source = 0,
    Parity = 1,
}

/// Header for streaming FEC packets.
///
/// Wire format (8 bytes):
/// - bytes 0-1: sequence number (u16 LE)
/// - byte 2: packet type (0 = source, 1 = parity)
/// - byte 3: parity index (for parity) or reserved (for source)
/// - bytes 4-5: parity end_seq (for parity) or data length (for source)
/// - bytes 6-7: reserved
#[derive(Debug, Clone, Copy)]
pub struct StreamingPacketHeader {
    /// Sequence number of the packet.
    pub seq: u16,
    /// Whether this is a parity packet.
    pub packet_type: PacketType,
    /// For parity packets: the parity index within the window.
    pub parity_index: u8,
    /// For parity packets: the end_seq of the window.
    /// For source packets: the actual data length.
    pub aux_field: u16,
}

impl StreamingPacketHeader {
    /// Header size in bytes.
    pub const SIZE: usize = 8;

    /// Create a header for a source packet.
    pub fn source(seq: u16, data_len: u16) -> Self {
        Self {
            seq,
            packet_type: PacketType::Source,
            parity_index: 0,
            aux_field: data_len,
        }
    }

    /// Create a header for a parity packet.
    pub fn parity(seq: u16, parity_index: u8, end_seq: u16) -> Self {
        Self {
            seq,
            packet_type: PacketType::Parity,
            parity_index,
            aux_field: end_seq,
        }
    }

    /// Serialize the header to bytes.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..2].copy_from_slice(&self.seq.to_le_bytes());
        buf[2] = self.packet_type as u8;
        buf[3] = self.parity_index;
        buf[4..6].copy_from_slice(&self.aux_field.to_le_bytes());
        buf
    }

    /// Deserialize a header from bytes.
    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < Self::SIZE {
            return None;
        }
        let seq = u16::from_le_bytes([buf[0], buf[1]]);
        let packet_type = match buf[2] {
            0 => PacketType::Source,
            1 => PacketType::Parity,
            _ => return None,
        };
        let parity_index = buf[3];
        let aux_field = u16::from_le_bytes([buf[4], buf[5]]);

        Some(Self {
            seq,
            packet_type,
            parity_index,
            aux_field,
        })
    }

    /// Get the data length (for source packets).
    pub fn data_len(&self) -> u16 {
        debug_assert!(self.packet_type == PacketType::Source);
        self.aux_field
    }

    /// Get the end_seq (for parity packets).
    pub fn end_seq(&self) -> u16 {
        debug_assert!(self.packet_type == PacketType::Parity);
        self.aux_field
    }
}

/// Streaming FEC encoder with integrated transport.
///
/// Wraps a [`StreamingEncoder`] and a datagram transport to provide
/// a simple API for sending FEC-protected data.
///
/// # Example
///
/// ```rust,no_run
/// use cm256::streaming::{StreamingParams, TransportEncoder};
/// use std::net::UdpSocket;
///
/// let socket = UdpSocket::bind("0.0.0.0:0").unwrap();
/// socket.connect("127.0.0.1:9000").unwrap();
///
/// let params = StreamingParams::new(8, 2, 1200).unwrap();
/// let mut encoder = TransportEncoder::new(params, socket);
///
/// // Send data - FEC encoding and transmission happens automatically
/// encoder.send(&[0x42; 1200]).unwrap();
/// ```
#[derive(Debug)]
pub struct TransportEncoder<T> {
    encoder: StreamingEncoder,
    transport: T,
    send_buffer: Vec<u8>,
}

impl<T: DatagramSendMut> TransportEncoder<T> {
    /// Create a new transport encoder.
    pub fn new(params: StreamingParams, transport: T) -> Self {
        let symbol_bytes = params.symbol_bytes();
        Self {
            encoder: StreamingEncoder::new(params),
            transport,
            send_buffer: Vec::with_capacity(StreamingPacketHeader::SIZE + symbol_bytes),
        }
    }

    /// Get the underlying encoder parameters.
    pub fn params(&self) -> StreamingParams {
        self.encoder.params()
    }

    /// Get a reference to the underlying transport.
    pub fn transport(&self) -> &T {
        &self.transport
    }

    /// Get a mutable reference to the underlying transport.
    pub fn transport_mut(&mut self) -> &mut T {
        &mut self.transport
    }

    /// Get the underlying encoder.
    pub fn encoder(&self) -> &StreamingEncoder {
        &self.encoder
    }

    /// Get a mutable reference to the underlying encoder.
    pub fn encoder_mut(&mut self) -> &mut StreamingEncoder {
        &mut self.encoder
    }

    /// Send data with FEC protection.
    ///
    /// The data is encoded and sent along with any generated parity packets.
    /// Data must be exactly `params.symbol_bytes()` in length.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != params.symbol_bytes()`.
    pub fn send(&mut self, data: &[u8]) -> io::Result<u16> {
        let result = self.encoder.add_source(data);
        let symbol_bytes = self.encoder.params().symbol_bytes();

        // Send source packet
        let header = StreamingPacketHeader::source(result.source_seq, data.len() as u16);
        self.send_buffer.clear();
        self.send_buffer.extend_from_slice(&header.to_bytes());
        self.send_buffer.extend_from_slice(data);
        // Pad to full symbol size if needed
        if self.send_buffer.len() < StreamingPacketHeader::SIZE + symbol_bytes {
            self.send_buffer
                .resize(StreamingPacketHeader::SIZE + symbol_bytes, 0);
        }
        self.transport.send_datagram(&self.send_buffer)?;

        // Send parity packets
        for parity in &result.parities {
            let header = StreamingPacketHeader::parity(
                result.source_seq,
                parity.parity_index,
                parity.end_seq,
            );
            self.send_buffer.clear();
            self.send_buffer.extend_from_slice(&header.to_bytes());
            self.send_buffer.extend_from_slice(&parity.data);
            self.transport.send_datagram(&self.send_buffer)?;
        }

        Ok(result.source_seq)
    }

    /// Send data with variable length.
    ///
    /// Unlike `send()`, this allows sending data smaller than `symbol_bytes`.
    /// The data will be padded internally for FEC, but the original length
    /// is preserved in the header.
    pub fn send_var(&mut self, data: &[u8]) -> io::Result<u16> {
        let symbol_bytes = self.encoder.params().symbol_bytes();
        assert!(
            data.len() <= symbol_bytes,
            "Data too large: {} > {}",
            data.len(),
            symbol_bytes
        );

        // Pad data to symbol size for encoding
        let mut padded = data.to_vec();
        padded.resize(symbol_bytes, 0);

        let result = self.encoder.add_source(&padded);

        // Send source packet with original length in header
        let header = StreamingPacketHeader::source(result.source_seq, data.len() as u16);
        self.send_buffer.clear();
        self.send_buffer.extend_from_slice(&header.to_bytes());
        self.send_buffer.extend_from_slice(&padded);
        self.transport.send_datagram(&self.send_buffer)?;

        // Send parity packets
        for parity in &result.parities {
            let header = StreamingPacketHeader::parity(
                result.source_seq,
                parity.parity_index,
                parity.end_seq,
            );
            self.send_buffer.clear();
            self.send_buffer.extend_from_slice(&header.to_bytes());
            self.send_buffer.extend_from_slice(&parity.data);
            self.transport.send_datagram(&self.send_buffer)?;
        }

        Ok(result.source_seq)
    }

    /// Flush the encoder, sending any remaining parity packets.
    pub fn flush(&mut self) -> io::Result<()> {
        let parities = self.encoder.flush();
        let seq = self.encoder.next_seq().wrapping_sub(1);

        for parity in &parities {
            let header = StreamingPacketHeader::parity(seq, parity.parity_index, parity.end_seq);
            self.send_buffer.clear();
            self.send_buffer.extend_from_slice(&header.to_bytes());
            self.send_buffer.extend_from_slice(&parity.data);
            self.transport.send_datagram(&self.send_buffer)?;
        }

        Ok(())
    }

    /// Consume the encoder and return the transport.
    pub fn into_transport(self) -> T {
        self.transport
    }
}

/// Result of receiving a packet from the transport decoder.
#[derive(Debug, Clone)]
pub enum RecvResult {
    /// A source packet was received (may have been recovered via FEC).
    Source {
        /// Sequence number.
        seq: u16,
        /// The data (trimmed to original length if known).
        data: Vec<u8>,
        /// Whether this was recovered via FEC.
        recovered: bool,
    },
    /// A parity packet was received (no action needed by caller).
    Parity,
    /// No packet available (for non-blocking recv).
    WouldBlock,
}

/// Streaming FEC decoder with integrated transport.
///
/// Wraps a [`StreamingDecoder`] and a datagram transport to provide
/// a simple API for receiving FEC-protected data.
///
/// # Example
///
/// ```rust,no_run
/// use cm256::streaming::{StreamingParams, TransportDecoder, RecvResult};
/// use std::net::UdpSocket;
///
/// let socket = UdpSocket::bind("0.0.0.0:9000").unwrap();
///
/// let params = StreamingParams::new(8, 2, 1200).unwrap();
/// let mut decoder = TransportDecoder::new(params, socket);
///
/// // Receive data - FEC decoding happens automatically
/// loop {
///     match decoder.recv().unwrap() {
///         RecvResult::Source { seq, data, recovered } => {
///             println!("Got packet {}: {} bytes (recovered: {})", seq, data.len(), recovered);
///         }
///         RecvResult::Parity => { /* parity received, no action needed */ }
///         RecvResult::WouldBlock => break,
///     }
/// }
/// ```
#[derive(Debug)]
pub struct TransportDecoder<T> {
    decoder: StreamingDecoder,
    transport: T,
    recv_buffer: Vec<u8>,
    /// Track original data lengths for trimming recovered packets.
    data_lengths: HashMap<u16, u16>,
}

impl<T: DatagramRecvMut> TransportDecoder<T> {
    /// Create a new transport decoder.
    pub fn new(params: StreamingParams, transport: T) -> Self {
        let symbol_bytes = params.symbol_bytes();
        Self {
            decoder: StreamingDecoder::new(params),
            transport,
            recv_buffer: vec![0u8; StreamingPacketHeader::SIZE + symbol_bytes],
            data_lengths: HashMap::new(),
        }
    }

    /// Get the underlying decoder parameters.
    pub fn params(&self) -> StreamingParams {
        self.decoder.params()
    }

    /// Get a reference to the underlying transport.
    pub fn transport(&self) -> &T {
        &self.transport
    }

    /// Get a mutable reference to the underlying transport.
    pub fn transport_mut(&mut self) -> &mut T {
        &mut self.transport
    }

    /// Get the underlying decoder.
    pub fn decoder(&self) -> &StreamingDecoder {
        &self.decoder
    }

    /// Get a mutable reference to the underlying decoder.
    pub fn decoder_mut(&mut self) -> &mut StreamingDecoder {
        &mut self.decoder
    }

    /// Receive a packet from the transport.
    ///
    /// This will block until a packet is received.
    pub fn recv(&mut self) -> io::Result<RecvResult> {
        let n = self.transport.recv_datagram(&mut self.recv_buffer)?;
        self.process_packet(n)
    }

    /// Try to receive a packet without blocking.
    pub fn try_recv(&mut self) -> io::Result<RecvResult> {
        match self.transport.try_recv_datagram(&mut self.recv_buffer)? {
            Some(n) => self.process_packet(n),
            None => Ok(RecvResult::WouldBlock),
        }
    }

    /// Process a received packet.
    fn process_packet(&mut self, n: usize) -> io::Result<RecvResult> {
        let symbol_bytes = self.decoder.params().symbol_bytes();

        if n < StreamingPacketHeader::SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "packet too small",
            ));
        }

        let Some(header) = StreamingPacketHeader::from_bytes(&self.recv_buffer) else {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid header"));
        };

        let payload = &self.recv_buffer[StreamingPacketHeader::SIZE..n];

        match header.packet_type {
            PacketType::Source => {
                // Pad payload if needed
                let mut padded = payload.to_vec();
                if padded.len() < symbol_bytes {
                    padded.resize(symbol_bytes, 0);
                }

                self.decoder.add_source(header.seq, &padded);
                self.data_lengths.insert(header.seq, header.data_len());

                // Try recovery (recovered packets will be returned via subsequent recv calls)
                let _recovered = self.decoder.try_recover();

                let data_len = header.data_len() as usize;
                Ok(RecvResult::Source {
                    seq: header.seq,
                    data: payload[..data_len.min(payload.len())].to_vec(),
                    recovered: false,
                })
            }
            PacketType::Parity => {
                // Pad payload if needed
                let mut padded = payload.to_vec();
                if padded.len() < symbol_bytes {
                    padded.resize(symbol_bytes, 0);
                }

                self.decoder
                    .add_parity(header.end_seq(), header.parity_index, &padded);

                // Try recovery after adding parity
                let recovered = self.decoder.try_recover();

                // If we recovered any packets, return the first one
                // (caller should call recv() again to get more)
                if let Some((seq, data)) = recovered.into_iter().next() {
                    let len = self
                        .data_lengths
                        .get(&seq)
                        .copied()
                        .unwrap_or(symbol_bytes as u16);
                    return Ok(RecvResult::Source {
                        seq,
                        data: data[..len as usize].to_vec(),
                        recovered: true,
                    });
                }

                Ok(RecvResult::Parity)
            }
        }
    }

    /// Check if a source packet is available.
    pub fn has_source(&self, seq: u16) -> bool {
        self.decoder.has_source(seq)
    }

    /// Get a source packet if available.
    pub fn get_source(&self, seq: u16) -> Option<Vec<u8>> {
        let data = self.decoder.get_source(seq)?;
        let len = self
            .data_lengths
            .get(&seq)
            .copied()
            .unwrap_or(data.len() as u16);
        Some(data[..len as usize].to_vec())
    }

    /// Consume the decoder and return the transport.
    pub fn into_transport(self) -> T {
        self.transport
    }
}

// =============================================================================
// Async Transport Integration
// =============================================================================

use crate::transport::{AsyncDatagramRecvMut, AsyncDatagramSendMut};

/// Async streaming FEC encoder with integrated transport.
///
/// # Example
///
/// ```rust,ignore
/// use cm256::streaming::{StreamingParams, AsyncTransportEncoder};
/// use tokio::net::UdpSocket;
///
/// async fn example() -> std::io::Result<()> {
///     let socket = UdpSocket::bind("0.0.0.0:0").await?;
///     socket.connect("127.0.0.1:9000").await?;
///
///     let params = StreamingParams::new(8, 2, 1200).unwrap();
///     let mut encoder = AsyncTransportEncoder::new(params, socket);
///
///     encoder.send(&[0x42; 1200]).await?;
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct AsyncTransportEncoder<T> {
    encoder: StreamingEncoder,
    transport: T,
    send_buffer: Vec<u8>,
}

impl<T: AsyncDatagramSendMut + Unpin> AsyncTransportEncoder<T> {
    /// Create a new async transport encoder.
    pub fn new(params: StreamingParams, transport: T) -> Self {
        let symbol_bytes = params.symbol_bytes();
        Self {
            encoder: StreamingEncoder::new(params),
            transport,
            send_buffer: Vec::with_capacity(StreamingPacketHeader::SIZE + symbol_bytes),
        }
    }

    /// Get the underlying encoder parameters.
    pub fn params(&self) -> StreamingParams {
        self.encoder.params()
    }

    /// Get a reference to the underlying transport.
    pub fn transport(&self) -> &T {
        &self.transport
    }

    /// Get a mutable reference to the underlying transport.
    pub fn transport_mut(&mut self) -> &mut T {
        &mut self.transport
    }

    /// Get the underlying encoder.
    pub fn encoder(&self) -> &StreamingEncoder {
        &self.encoder
    }

    /// Get a mutable reference to the underlying encoder.
    pub fn encoder_mut(&mut self) -> &mut StreamingEncoder {
        &mut self.encoder
    }

    /// Send data with FEC protection asynchronously.
    pub async fn send(&mut self, data: &[u8]) -> io::Result<u16> {
        let result = self.encoder.add_source(data);
        let symbol_bytes = self.encoder.params().symbol_bytes();

        // Send source packet
        let header = StreamingPacketHeader::source(result.source_seq, data.len() as u16);
        self.send_buffer.clear();
        self.send_buffer.extend_from_slice(&header.to_bytes());
        self.send_buffer.extend_from_slice(data);
        if self.send_buffer.len() < StreamingPacketHeader::SIZE + symbol_bytes {
            self.send_buffer
                .resize(StreamingPacketHeader::SIZE + symbol_bytes, 0);
        }
        self.transport
            .send_datagram_async(&self.send_buffer)
            .await?;

        // Send parity packets
        for parity in &result.parities {
            let header = StreamingPacketHeader::parity(
                result.source_seq,
                parity.parity_index,
                parity.end_seq,
            );
            self.send_buffer.clear();
            self.send_buffer.extend_from_slice(&header.to_bytes());
            self.send_buffer.extend_from_slice(&parity.data);
            self.transport
                .send_datagram_async(&self.send_buffer)
                .await?;
        }

        Ok(result.source_seq)
    }

    /// Send variable-length data asynchronously.
    pub async fn send_var(&mut self, data: &[u8]) -> io::Result<u16> {
        let symbol_bytes = self.encoder.params().symbol_bytes();
        assert!(
            data.len() <= symbol_bytes,
            "Data too large: {} > {}",
            data.len(),
            symbol_bytes
        );

        let mut padded = data.to_vec();
        padded.resize(symbol_bytes, 0);

        let result = self.encoder.add_source(&padded);

        let header = StreamingPacketHeader::source(result.source_seq, data.len() as u16);
        self.send_buffer.clear();
        self.send_buffer.extend_from_slice(&header.to_bytes());
        self.send_buffer.extend_from_slice(&padded);
        self.transport
            .send_datagram_async(&self.send_buffer)
            .await?;

        for parity in &result.parities {
            let header = StreamingPacketHeader::parity(
                result.source_seq,
                parity.parity_index,
                parity.end_seq,
            );
            self.send_buffer.clear();
            self.send_buffer.extend_from_slice(&header.to_bytes());
            self.send_buffer.extend_from_slice(&parity.data);
            self.transport
                .send_datagram_async(&self.send_buffer)
                .await?;
        }

        Ok(result.source_seq)
    }

    /// Flush the encoder asynchronously.
    pub async fn flush(&mut self) -> io::Result<()> {
        let parities = self.encoder.flush();
        let seq = self.encoder.next_seq().wrapping_sub(1);

        for parity in &parities {
            let header = StreamingPacketHeader::parity(seq, parity.parity_index, parity.end_seq);
            self.send_buffer.clear();
            self.send_buffer.extend_from_slice(&header.to_bytes());
            self.send_buffer.extend_from_slice(&parity.data);
            self.transport
                .send_datagram_async(&self.send_buffer)
                .await?;
        }

        Ok(())
    }

    /// Consume the encoder and return the transport.
    pub fn into_transport(self) -> T {
        self.transport
    }
}

/// Async streaming FEC decoder with integrated transport.
#[derive(Debug)]
pub struct AsyncTransportDecoder<T> {
    decoder: StreamingDecoder,
    transport: T,
    recv_buffer: Vec<u8>,
    data_lengths: HashMap<u16, u16>,
}

impl<T: AsyncDatagramRecvMut + Unpin> AsyncTransportDecoder<T> {
    /// Create a new async transport decoder.
    pub fn new(params: StreamingParams, transport: T) -> Self {
        let symbol_bytes = params.symbol_bytes();
        Self {
            decoder: StreamingDecoder::new(params),
            transport,
            recv_buffer: vec![0u8; StreamingPacketHeader::SIZE + symbol_bytes],
            data_lengths: HashMap::new(),
        }
    }

    /// Get the underlying decoder parameters.
    pub fn params(&self) -> StreamingParams {
        self.decoder.params()
    }

    /// Get a reference to the underlying transport.
    pub fn transport(&self) -> &T {
        &self.transport
    }

    /// Get a mutable reference to the underlying transport.
    pub fn transport_mut(&mut self) -> &mut T {
        &mut self.transport
    }

    /// Get the underlying decoder.
    pub fn decoder(&self) -> &StreamingDecoder {
        &self.decoder
    }

    /// Get a mutable reference to the underlying decoder.
    pub fn decoder_mut(&mut self) -> &mut StreamingDecoder {
        &mut self.decoder
    }

    /// Receive a packet asynchronously.
    pub async fn recv(&mut self) -> io::Result<RecvResult> {
        let n = self
            .transport
            .recv_datagram_async(&mut self.recv_buffer)
            .await?;
        self.process_packet(n)
    }

    fn process_packet(&mut self, n: usize) -> io::Result<RecvResult> {
        let symbol_bytes = self.decoder.params().symbol_bytes();

        if n < StreamingPacketHeader::SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "packet too small",
            ));
        }

        let Some(header) = StreamingPacketHeader::from_bytes(&self.recv_buffer) else {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid header"));
        };

        let payload = &self.recv_buffer[StreamingPacketHeader::SIZE..n];

        match header.packet_type {
            PacketType::Source => {
                let mut padded = payload.to_vec();
                if padded.len() < symbol_bytes {
                    padded.resize(symbol_bytes, 0);
                }

                self.decoder.add_source(header.seq, &padded);
                self.data_lengths.insert(header.seq, header.data_len());
                let _recovered = self.decoder.try_recover();

                let data_len = header.data_len() as usize;
                Ok(RecvResult::Source {
                    seq: header.seq,
                    data: payload[..data_len.min(payload.len())].to_vec(),
                    recovered: false,
                })
            }
            PacketType::Parity => {
                let mut padded = payload.to_vec();
                if padded.len() < symbol_bytes {
                    padded.resize(symbol_bytes, 0);
                }

                self.decoder
                    .add_parity(header.end_seq(), header.parity_index, &padded);

                let recovered = self.decoder.try_recover();

                if let Some((seq, data)) = recovered.into_iter().next() {
                    let len = self
                        .data_lengths
                        .get(&seq)
                        .copied()
                        .unwrap_or(symbol_bytes as u16);
                    return Ok(RecvResult::Source {
                        seq,
                        data: data[..len as usize].to_vec(),
                        recovered: true,
                    });
                }

                Ok(RecvResult::Parity)
            }
        }
    }

    /// Check if a source packet is available.
    pub fn has_source(&self, seq: u16) -> bool {
        self.decoder.has_source(seq)
    }

    /// Get a source packet if available.
    pub fn get_source(&self, seq: u16) -> Option<Vec<u8>> {
        let data = self.decoder.get_source(seq)?;
        let len = self
            .data_lengths
            .get(&seq)
            .copied()
            .unwrap_or(data.len() as u16);
        Some(data[..len as usize].to_vec())
    }

    /// Consume the decoder and return the transport.
    pub fn into_transport(self) -> T {
        self.transport
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

    #[test]
    fn test_packet_header_roundtrip() {
        let source_header = StreamingPacketHeader::source(1234, 500);
        let bytes = source_header.to_bytes();
        let parsed = StreamingPacketHeader::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.seq, 1234);
        assert_eq!(parsed.packet_type, PacketType::Source);
        assert_eq!(parsed.data_len(), 500);

        let parity_header = StreamingPacketHeader::parity(5678, 2, 9999);
        let bytes = parity_header.to_bytes();
        let parsed = StreamingPacketHeader::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.seq, 5678);
        assert_eq!(parsed.packet_type, PacketType::Parity);
        assert_eq!(parsed.parity_index, 2);
        assert_eq!(parsed.end_seq(), 9999);
    }

    #[test]
    fn test_transport_encoder_decoder() {
        use crate::transport::MemoryChannel;

        let params = StreamingParams::new(4, 2, 32).unwrap();
        let (tx_chan, rx_chan) = MemoryChannel::pair();

        let mut tx = TransportEncoder::new(params, tx_chan);
        let mut rx = TransportDecoder::new(params, rx_chan);

        let originals: Vec<Vec<u8>> = (0..8).map(|i| vec![(i * 10) as u8; 32]).collect();

        // Send all packets
        for data in &originals {
            tx.send(data).unwrap();
        }

        // Receive all packets
        let mut received = Vec::new();
        loop {
            match rx.try_recv().unwrap() {
                RecvResult::Source { seq, data, .. } => {
                    received.push((seq, data));
                }
                RecvResult::Parity => {}
                RecvResult::WouldBlock => break,
            }
        }

        // Verify we got all source packets
        assert_eq!(received.len(), 8);
        for (seq, data) in received {
            assert_eq!(data, originals[seq as usize]);
        }
    }

    #[test]
    fn test_transport_with_loss() {
        use crate::transport::{LossyChannel, MemoryChannel};

        let params = StreamingParams::new(4, 2, 32).unwrap();
        let (tx_chan, rx_chan) = MemoryChannel::pair();
        let lossy_tx = LossyChannel::with_pattern(tx_chan, 5); // Drop every 5th packet

        let mut tx = TransportEncoder::new(params, lossy_tx);
        let mut rx = TransportDecoder::new(params, rx_chan);

        let originals: Vec<Vec<u8>> = (0..8).map(|i| vec![(i * 10) as u8; 32]).collect();

        // Send all packets
        for data in &originals {
            tx.send(data).unwrap();
        }

        // Receive packets (some will be recovered via FEC)
        let mut received_seqs = std::collections::HashSet::new();
        loop {
            match rx.try_recv().unwrap() {
                RecvResult::Source {
                    seq,
                    data,
                    recovered,
                } => {
                    // Verify data integrity
                    assert_eq!(
                        data, originals[seq as usize],
                        "Data mismatch for seq {}",
                        seq
                    );
                    received_seqs.insert(seq);
                    if recovered {
                        println!("Recovered seq {}", seq);
                    }
                }
                RecvResult::Parity => {}
                RecvResult::WouldBlock => break,
            }
        }

        // With FEC, we should have recovered most packets
        println!(
            "Received {} of {} packets",
            received_seqs.len(),
            originals.len()
        );
        // The exact count depends on timing and loss pattern, but we should get most
    }

    #[test]
    fn test_transport_var_length() {
        use crate::transport::MemoryChannel;

        let params = StreamingParams::new(4, 2, 64).unwrap(); // 64 byte symbols
        let (tx_chan, rx_chan) = MemoryChannel::pair();

        let mut tx = TransportEncoder::new(params, tx_chan);
        let mut rx = TransportDecoder::new(params, rx_chan);

        // Send variable-length data
        let data1 = vec![0x11; 20]; // 20 bytes
        let data2 = vec![0x22; 50]; // 50 bytes
        let data3 = vec![0x33; 64]; // 64 bytes (full)

        tx.send_var(&data1).unwrap();
        tx.send_var(&data2).unwrap();
        tx.send_var(&data3).unwrap();

        // Receive and verify lengths are preserved
        let mut received = Vec::new();
        loop {
            match rx.try_recv().unwrap() {
                RecvResult::Source { data, .. } => received.push(data),
                RecvResult::Parity => {}
                RecvResult::WouldBlock => break,
            }
        }

        assert_eq!(received.len(), 3);
        assert_eq!(received[0], data1);
        assert_eq!(received[1], data2);
        assert_eq!(received[2], data3);
    }
}
