//! # Reliable Streaming with Hybrid FEC + ARQ
//!
//! This module provides a complete reliability layer for real-time streaming,
//! combining Forward Error Correction (FEC) with Automatic Repeat reQuest (ARQ).
//!
//! Inspired by the Dynamic-FEC video streaming system, this implementation:
//! - Dynamically adjusts FEC redundancy based on measured packet loss
//! - Uses heartbeat probing to monitor network RTT and loss rate
//! - Falls back to ARQ retransmission when FEC cannot recover lost packets
//! - Supports frame-level timeouts for low-latency applications
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────┐     UDP (Data)      ┌─────────────┐
//! │   Sender    │ ──────────────────► │  Receiver   │
//! │             │  K data + M parity  │             │
//! │ Encoder     │ ◄────────────────── │  Decoder    │
//! │             │  ACK / NACK / Resend│             │
//! └──────┬──────┘                     └──────┬──────┘
//!        │       UDP (Heartbeat)             │
//!        │ ◄───────────────────────────────► │
//!        │   RTT/Loss probes (port+1)        │
//!        └───────────────────────────────────┘
//! ```
//!
//! ## Features
//!
//! - **Dynamic FEC**: Parity count adapts to loss rate (1-100% redundancy)
//! - **Heartbeat Monitoring**: Separate channel measures RTT/loss in real-time
//! - **ARQ Fallback**: Request retransmission when FEC fails
//! - **Frame Timeout**: Drop stale frames to maintain low latency
//! - **In-order Delivery**: Reassembles frames in sequence
//!
//! ## Example
//!
//! ```rust,ignore
//! use cm256::reliable::{ReliableEncoder, ReliableDecoder, ReliableParams};
//! use std::net::UdpSocket;
//!
//! // Sender
//! let params = ReliableParams::new(8, 1200);
//! let socket = UdpSocket::bind("0.0.0.0:5000")?;
//! socket.connect("receiver:5000")?;
//! let mut encoder = ReliableEncoder::new(params, socket);
//!
//! // Send video frames
//! encoder.send_frame(&video_frame_data)?;
//! ```

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io;
use std::time::{Duration, Instant};

use crate::transport::{DatagramRecvMut, DatagramSendMut};
use crate::{Block, BlockMut, Params};

// =============================================================================
// Configuration
// =============================================================================

/// Default symbol size in bytes (fits in MTU with headers).
pub const DEFAULT_SYMBOL_SIZE: usize = 1200;

/// Maximum number of source symbols per frame (limited by GF(256)).
pub const MAX_SYMBOLS_PER_FRAME: usize = 200;

/// Default heartbeat interval in milliseconds.
pub const HEARTBEAT_INTERVAL_MS: u64 = 200;

/// Default window size for network statistics.
pub const STATS_WINDOW_SIZE: usize = 20;

/// Maximum RTT before considering connection dead (microseconds).
pub const RTT_MAX_LIMIT: f64 = 10_000_000.0; // 10 seconds

/// Default frame timeout for dropping stale frames (milliseconds).
pub const DEFAULT_FRAME_TIMEOUT_MS: u64 = 300;

/// Default ARQ retry interval (milliseconds).
pub const DEFAULT_RETRY_INTERVAL_MS: u64 = 220;

// =============================================================================
// Network Statistics
// =============================================================================

/// Real-time network quality statistics.
///
/// Tracks RTT and packet loss rate using exponential moving average.
#[derive(Debug)]
pub struct NetworkStats {
    /// Exponentially weighted moving average of RTT in microseconds.
    rtt_us: f64,
    /// Exponentially weighted moving average of loss rate (0.0 - 1.0).
    loss_rate: f64,
    /// Recent RTT samples for variance calculation.
    rtt_samples: VecDeque<u64>,
    /// Smoothing factor for EWMA (higher = more responsive).
    alpha: f64,
    /// Number of packets sent for loss calculation.
    packets_sent: u64,
    /// Number of ACKs received.
    packets_acked: u64,
    /// Last update timestamp.
    last_update: Instant,
}

impl NetworkStats {
    /// Create new network statistics tracker.
    pub fn new() -> Self {
        Self {
            rtt_us: 50_000.0, // Start with 50ms assumption
            loss_rate: 0.0,
            rtt_samples: VecDeque::with_capacity(STATS_WINDOW_SIZE),
            alpha: 0.1,
            packets_sent: 0,
            packets_acked: 0,
            last_update: Instant::now(),
        }
    }

    /// Update statistics with a successful RTT measurement.
    pub fn update_rtt(&mut self, rtt_us: u64) {
        // EWMA update
        self.rtt_us = self.alpha * rtt_us as f64 + (1.0 - self.alpha) * self.rtt_us;

        // Keep samples for variance
        if self.rtt_samples.len() >= STATS_WINDOW_SIZE {
            self.rtt_samples.pop_front();
        }
        self.rtt_samples.push_back(rtt_us);

        self.packets_acked += 1;
        self.last_update = Instant::now();
    }

    /// Update statistics when a packet is sent.
    pub fn on_packet_sent(&mut self) {
        self.packets_sent += 1;
    }

    /// Update loss rate based on sent/acked ratio.
    pub fn update_loss(&mut self, lost: bool) {
        let sample = if lost { 1.0 } else { 0.0 };
        self.loss_rate = self.alpha * sample + (1.0 - self.alpha) * self.loss_rate;
    }

    /// Get current RTT estimate in microseconds.
    #[inline]
    pub fn rtt_us(&self) -> f64 {
        self.rtt_us
    }

    /// Get current RTT estimate in milliseconds.
    #[inline]
    pub fn rtt_ms(&self) -> f64 {
        self.rtt_us / 1000.0
    }

    /// Get current loss rate (0.0 - 1.0).
    #[inline]
    pub fn loss_rate(&self) -> f64 {
        self.loss_rate
    }

    /// Get loss rate as percentage.
    #[inline]
    pub fn loss_percent(&self) -> f64 {
        self.loss_rate * 100.0
    }

    /// Calculate RTT variance (jitter indicator).
    pub fn rtt_variance(&self) -> f64 {
        if self.rtt_samples.len() < 2 {
            return 0.0;
        }
        let mean = self.rtt_samples.iter().sum::<u64>() as f64 / self.rtt_samples.len() as f64;
        let variance: f64 = self
            .rtt_samples
            .iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>()
            / self.rtt_samples.len() as f64;
        variance.sqrt()
    }

    /// Get a normalized network quality score (0.0 = bad, 1.0 = good).
    pub fn quality_score(&self) -> f64 {
        // Combine latency and loss into a quality metric
        let latency_score = 1.0 / (1.0 + self.rtt_ms() / 100.0); // 100ms = 0.5
        let loss_score = 1.0 - self.loss_rate;
        (latency_score + loss_score) / 2.0
    }

    /// Check if the connection appears dead.
    pub fn is_dead(&self) -> bool {
        self.rtt_us >= RTT_MAX_LIMIT && self.loss_rate >= 0.99
    }
}

impl Default for NetworkStats {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Dynamic Parity Calculation
// =============================================================================

/// Calculate the number of parity symbols based on loss rate.
///
/// This implements the adaptive FEC strategy from the Dynamic-FEC system:
/// - <0.5% loss: minimal redundancy (K/50 + 1)
/// - <1% loss: light redundancy (K/20 + 1)
/// - <5% loss: moderate redundancy (K/7 + 1)
/// - <10% loss: heavy redundancy (K/4 + 1)
/// - <20% loss: very heavy redundancy (K/2 + 1)
/// - ≥20% loss: maximum redundancy (K)
///
/// # Arguments
/// * `source_count` - Number of source symbols (K)
/// * `loss_rate` - Current packet loss rate (0.0 - 1.0)
///
/// # Returns
/// Number of parity symbols (M)
pub fn calculate_parity_count(source_count: usize, loss_rate: f64) -> usize {
    let k = source_count as f64;
    let m = if loss_rate < 0.005 {
        k / 50.0 + 1.0
    } else if loss_rate < 0.01 {
        k / 20.0 + 1.0
    } else if loss_rate < 0.05 {
        k / 7.0 + 1.0
    } else if loss_rate < 0.10 {
        k / 4.0 + 1.0
    } else if loss_rate < 0.20 {
        k / 2.0 + 1.0
    } else {
        k
    };

    // Clamp to valid range
    let m = m.ceil() as usize;
    m.clamp(1, 256 - source_count)
}

/// Calculate parity count with a base redundancy factor.
///
/// Combines the adaptive loss-based calculation with a user-specified
/// minimum redundancy factor.
///
/// # Arguments
/// * `source_count` - Number of source symbols
/// * `loss_rate` - Current packet loss rate
/// * `base_redundancy` - Minimum redundancy factor (e.g., 1.3 = 30% overhead)
pub fn calculate_parity_with_redundancy(
    source_count: usize,
    loss_rate: f64,
    base_redundancy: f64,
) -> usize {
    // Loss-adaptive component
    let loss_parity = calculate_parity_count(source_count, loss_rate);

    // Base redundancy component
    let base_parity = ((source_count as f64 * (base_redundancy - 1.0)).ceil() as usize).max(1);

    // Take the maximum of both
    let parity = loss_parity.max(base_parity);
    parity.clamp(1, 256 - source_count)
}

// =============================================================================
// Packet Types
// =============================================================================

/// Packet type discriminator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PacketType {
    /// Source data packet.
    Source = 0,
    /// Parity (FEC) packet.
    Parity = 1,
    /// Acknowledgement packet.
    Ack = 2,
    /// Retransmission request (NACK).
    Nack = 3,
    /// Heartbeat probe.
    Heartbeat = 4,
    /// Heartbeat response.
    HeartbeatAck = 5,
}

impl PacketType {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Source),
            1 => Some(Self::Parity),
            2 => Some(Self::Ack),
            3 => Some(Self::Nack),
            4 => Some(Self::Heartbeat),
            5 => Some(Self::HeartbeatAck),
            _ => None,
        }
    }
}

// =============================================================================
// Wire Protocol
// =============================================================================

/// Header for reliable packets.
///
/// Wire format (16 bytes):
/// - byte 0: packet type
/// - byte 1: symbol index (within frame)
/// - bytes 2-3: source count (K)
/// - bytes 4-5: total count (N = K + M)
/// - bytes 6-9: frame ID (u32)
/// - bytes 10-13: timestamp (for RTT, truncated to u32)
/// - bytes 14-15: data length
#[derive(Debug, Clone, Copy)]
pub struct ReliableHeader {
    pub packet_type: PacketType,
    pub symbol_index: u8,
    pub source_count: u16,
    pub total_count: u16,
    pub frame_id: u32,
    pub timestamp: u32,
    pub data_len: u16,
}

impl ReliableHeader {
    pub const SIZE: usize = 16;

    /// Create a source packet header.
    pub fn source(frame_id: u32, symbol_index: u8, k: u16, n: u16, data_len: u16) -> Self {
        Self {
            packet_type: PacketType::Source,
            symbol_index,
            source_count: k,
            total_count: n,
            frame_id,
            timestamp: timestamp_now(),
            data_len,
        }
    }

    /// Create a parity packet header.
    pub fn parity(frame_id: u32, symbol_index: u8, k: u16, n: u16, data_len: u16) -> Self {
        Self {
            packet_type: PacketType::Parity,
            symbol_index,
            source_count: k,
            total_count: n,
            frame_id,
            timestamp: timestamp_now(),
            data_len,
        }
    }

    /// Create an ACK packet header.
    pub fn ack(frame_id: u32) -> Self {
        Self {
            packet_type: PacketType::Ack,
            symbol_index: 0,
            source_count: 0,
            total_count: 0,
            frame_id,
            timestamp: timestamp_now(),
            data_len: 0,
        }
    }

    /// Create a NACK (retransmit request) header.
    pub fn nack(frame_id: u32, start_symbol: u8, end_symbol: u8) -> Self {
        Self {
            packet_type: PacketType::Nack,
            symbol_index: start_symbol,
            source_count: end_symbol as u16,
            total_count: 0,
            frame_id,
            timestamp: timestamp_now(),
            data_len: 0,
        }
    }

    /// Create a heartbeat probe header.
    pub fn heartbeat(seq: u32) -> Self {
        Self {
            packet_type: PacketType::Heartbeat,
            symbol_index: 0,
            source_count: 0,
            total_count: 0,
            frame_id: seq,
            timestamp: timestamp_now(),
            data_len: 0,
        }
    }

    /// Create a heartbeat ACK header.
    pub fn heartbeat_ack(seq: u32, original_timestamp: u32) -> Self {
        Self {
            packet_type: PacketType::HeartbeatAck,
            symbol_index: 0,
            source_count: 0,
            total_count: 0,
            frame_id: seq,
            timestamp: original_timestamp,
            data_len: 0,
        }
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0] = self.packet_type as u8;
        buf[1] = self.symbol_index;
        buf[2..4].copy_from_slice(&self.source_count.to_le_bytes());
        buf[4..6].copy_from_slice(&self.total_count.to_le_bytes());
        buf[6..10].copy_from_slice(&self.frame_id.to_le_bytes());
        buf[10..14].copy_from_slice(&self.timestamp.to_le_bytes());
        buf[14..16].copy_from_slice(&self.data_len.to_le_bytes());
        buf
    }

    /// Deserialize from bytes.
    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < Self::SIZE {
            return None;
        }
        Some(Self {
            packet_type: PacketType::from_u8(buf[0])?,
            symbol_index: buf[1],
            source_count: u16::from_le_bytes([buf[2], buf[3]]),
            total_count: u16::from_le_bytes([buf[4], buf[5]]),
            frame_id: u32::from_le_bytes([buf[6], buf[7], buf[8], buf[9]]),
            timestamp: u32::from_le_bytes([buf[10], buf[11], buf[12], buf[13]]),
            data_len: u16::from_le_bytes([buf[14], buf[15]]),
        })
    }
}

/// Get a 32-bit timestamp (microseconds, wrapping).
fn timestamp_now() -> u32 {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    (now.as_micros() & 0xFFFFFFFF) as u32
}

/// Calculate elapsed time from a timestamp.
fn elapsed_from_timestamp(ts: u32) -> u32 {
    let now = timestamp_now();
    now.wrapping_sub(ts)
}

// =============================================================================
// Reliable Parameters
// =============================================================================

/// Parameters for reliable streaming.
#[derive(Debug, Clone, Copy)]
pub struct ReliableParams {
    /// Maximum source symbols per frame.
    pub max_symbols_per_frame: usize,
    /// Size of each symbol in bytes.
    pub symbol_bytes: usize,
    /// Base redundancy factor (1.0 = no extra, 1.3 = 30% overhead).
    pub base_redundancy: f64,
    /// Frame timeout in milliseconds (0 = no timeout).
    pub frame_timeout_ms: u64,
    /// ARQ retry interval in milliseconds.
    pub retry_interval_ms: u64,
    /// Maximum retransmission attempts per frame.
    pub max_retries: u8,
    /// Enable ARQ (retransmission requests).
    pub enable_arq: bool,
    /// Enable heartbeat probing.
    pub enable_heartbeat: bool,
    /// Heartbeat interval in milliseconds.
    pub heartbeat_interval_ms: u64,
}

impl ReliableParams {
    /// Create new parameters with defaults.
    pub fn new(max_symbols_per_frame: usize, symbol_bytes: usize) -> Self {
        Self {
            max_symbols_per_frame: max_symbols_per_frame.min(MAX_SYMBOLS_PER_FRAME),
            symbol_bytes,
            base_redundancy: 1.2, // 20% default overhead
            frame_timeout_ms: DEFAULT_FRAME_TIMEOUT_MS,
            retry_interval_ms: DEFAULT_RETRY_INTERVAL_MS,
            max_retries: 3,
            enable_arq: true,
            enable_heartbeat: true,
            heartbeat_interval_ms: HEARTBEAT_INTERVAL_MS,
        }
    }

    /// Set base redundancy factor.
    pub fn with_redundancy(mut self, factor: f64) -> Self {
        self.base_redundancy = factor.max(1.0);
        self
    }

    /// Set frame timeout.
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.frame_timeout_ms = timeout_ms;
        self
    }

    /// Disable ARQ (FEC-only mode).
    pub fn without_arq(mut self) -> Self {
        self.enable_arq = false;
        self
    }

    /// Disable heartbeat probing.
    pub fn without_heartbeat(mut self) -> Self {
        self.enable_heartbeat = false;
        self
    }
}

impl Default for ReliableParams {
    fn default() -> Self {
        Self::new(8, DEFAULT_SYMBOL_SIZE)
    }
}

// =============================================================================
// Frame Buffer (Receiver Side)
// =============================================================================

/// Buffer for receiving and reconstructing a single frame.
#[derive(Debug)]
struct FrameBuffer {
    #[allow(dead_code)]
    frame_id: u32,
    source_count: u16,
    parity_count: u16,
    symbol_bytes: usize,
    /// Received symbols: index -> data
    symbols: HashMap<u8, Vec<u8>>,
    /// Number of source symbols received
    sources_received: usize,
    /// Number of parity symbols received
    parities_received: usize,
    /// When this frame was first seen
    first_seen: Instant,
    /// When the last packet arrived
    last_packet: Instant,
    /// Whether this frame has been decoded
    decoded: bool,
    /// Retransmission attempts
    retry_count: u8,
    /// Original data length (before padding)
    data_len: usize,
}

impl FrameBuffer {
    fn new(
        frame_id: u32,
        source_count: u16,
        total_count: u16,
        symbol_bytes: usize,
        data_len: u16,
    ) -> Self {
        Self {
            frame_id,
            source_count,
            parity_count: total_count.saturating_sub(source_count),
            symbol_bytes,
            symbols: HashMap::new(),
            sources_received: 0,
            parities_received: 0,
            first_seen: Instant::now(),
            last_packet: Instant::now(),
            decoded: false,
            retry_count: 0,
            data_len: data_len as usize,
        }
    }

    /// Add a received symbol.
    fn add_symbol(&mut self, index: u8, is_parity: bool, data: &[u8], data_len: u16) -> bool {
        if self.decoded {
            return false;
        }

        let idx = index;
        if self.symbols.contains_key(&idx) {
            return false; // Duplicate
        }

        // Store the symbol (pad/truncate to symbol_bytes)
        let mut symbol = data.to_vec();
        symbol.resize(self.symbol_bytes, 0);
        self.symbols.insert(idx, symbol);

        if is_parity {
            self.parities_received += 1;
        } else {
            self.sources_received += 1;
            // Source packets carry the actual data length - update if more accurate
            if data_len > 0 && (self.data_len == 0 || (data_len as usize) < self.data_len) {
                self.data_len = data_len as usize;
            }
        }

        self.last_packet = Instant::now();
        true
    }

    /// Check if we can decode (have enough symbols).
    fn can_decode(&self) -> bool {
        !self.decoded && self.symbols.len() >= self.source_count as usize
    }

    /// Check if all source symbols are present (no FEC needed).
    fn all_sources_present(&self) -> bool {
        self.sources_received >= self.source_count as usize
    }

    /// Get missing symbol indices.
    fn missing_symbols(&self) -> Vec<u8> {
        let mut missing = Vec::new();
        for i in 0..self.source_count {
            if !self.symbols.contains_key(&(i as u8)) {
                missing.push(i as u8);
            }
        }
        missing
    }

    /// Attempt to decode the frame using FEC.
    fn try_decode(&mut self) -> Option<Vec<u8>> {
        if self.decoded {
            return None;
        }

        if !self.can_decode() {
            return None;
        }

        // If all sources present, just concatenate
        if self.all_sources_present() {
            let mut result = Vec::with_capacity(self.source_count as usize * self.symbol_bytes);
            for i in 0..self.source_count {
                if let Some(data) = self.symbols.get(&(i as u8)) {
                    result.extend_from_slice(data);
                }
            }
            self.decoded = true;
            // Trim to original data length
            if self.data_len > 0 && result.len() > self.data_len {
                result.truncate(self.data_len);
            }
            return Some(result);
        }

        // Need FEC decoding
        let k = self.source_count as usize;
        let m = self.parity_count as usize;

        let params = match Params::new(k, m, self.symbol_bytes) {
            Ok(p) => p,
            Err(_) => return None,
        };

        // Collect K symbols for decoding
        let mut block_data: Vec<Vec<u8>> = Vec::with_capacity(k);
        let mut block_indices: Vec<u8> = Vec::with_capacity(k);

        // Prefer source symbols
        for i in 0..self.source_count as u8 {
            if let Some(data) = self.symbols.get(&i) {
                block_data.push(data.clone());
                block_indices.push(i);
            }
            if block_data.len() >= k {
                break;
            }
        }

        // Add parity symbols if needed
        if block_data.len() < k {
            for i in self.source_count..(self.source_count + self.parity_count) {
                if let Some(data) = self.symbols.get(&(i as u8)) {
                    block_data.push(data.clone());
                    block_indices.push(i as u8);
                }
                if block_data.len() >= k {
                    break;
                }
            }
        }

        if block_data.len() < k {
            return None;
        }

        // Create mutable blocks for decoding
        let mut blocks_mut: Vec<BlockMut> = block_data
            .iter_mut()
            .zip(block_indices.iter())
            .map(|(data, &idx)| BlockMut::new(idx, data))
            .collect();

        // Decode
        match crate::decode(&params, &mut blocks_mut) {
            Ok(_recovered_indices) => {
                // Reconstruct frame from decoded blocks
                // After decode, blocks with index < K contain source data
                let mut result_blocks: Vec<Option<Vec<u8>>> = vec![None; k];

                for (block, &idx) in blocks_mut.iter().zip(block_indices.iter()) {
                    if (idx as usize) < k {
                        result_blocks[idx as usize] = Some(block.data.to_vec());
                    }
                }

                // Fill in from decoded parity blocks (they now contain recovered source data)
                for i in 0..k {
                    if result_blocks[i].is_none() {
                        // Find a parity block that was used to recover this
                        for (j, &idx) in block_indices.iter().enumerate() {
                            if idx as usize >= k && result_blocks[i].is_none() {
                                result_blocks[i] = Some(block_data[j].clone());
                                break;
                            }
                        }
                    }
                }

                // Concatenate result
                let mut result = Vec::with_capacity(k * self.symbol_bytes);
                for i in 0..k {
                    if let Some(data) = &result_blocks[i] {
                        result.extend_from_slice(data);
                    } else {
                        // Fill with zeros if still missing (shouldn't happen)
                        result.extend(vec![0u8; self.symbol_bytes]);
                    }
                }

                self.decoded = true;
                // Trim to original data length
                if self.data_len > 0 && result.len() > self.data_len {
                    result.truncate(self.data_len);
                }
                Some(result)
            }
            Err(_) => None,
        }
    }

    /// Check if frame has timed out.
    fn is_timed_out(&self, timeout_ms: u64) -> bool {
        if timeout_ms == 0 {
            return false;
        }
        self.first_seen.elapsed() > Duration::from_millis(timeout_ms)
    }

    /// Check if it's time to request retransmission.
    fn should_retry(&self, interval_ms: u64) -> bool {
        self.last_packet.elapsed() > Duration::from_millis(interval_ms)
    }
}

// =============================================================================
// Sender-side Frame Cache (for retransmission)
// =============================================================================

/// Cached frame data for retransmission.
#[derive(Debug)]
struct CachedFrame {
    frame_id: u32,
    source_symbols: Vec<Vec<u8>>,
    parity_symbols: Vec<Vec<u8>>,
    created: Instant,
}

impl CachedFrame {
    fn new(frame_id: u32, sources: Vec<Vec<u8>>, parities: Vec<Vec<u8>>) -> Self {
        Self {
            frame_id,
            source_symbols: sources,
            parity_symbols: parities,
            created: Instant::now(),
        }
    }

    fn get_symbol(&self, index: u8) -> Option<&[u8]> {
        let idx = index as usize;
        if idx < self.source_symbols.len() {
            Some(&self.source_symbols[idx])
        } else {
            let parity_idx = idx - self.source_symbols.len();
            self.parity_symbols.get(parity_idx).map(|v| v.as_slice())
        }
    }
}

// =============================================================================
// Reliable Encoder
// =============================================================================

/// Reliable streaming encoder with FEC and ARQ support.
///
/// Encodes frames into source + parity packets, handles ACKs/NACKs,
/// and retransmits lost packets on request.
#[derive(Debug)]
pub struct ReliableEncoder<T> {
    params: ReliableParams,
    transport: T,
    /// Network statistics
    stats: NetworkStats,
    /// Current frame ID
    next_frame_id: u32,
    /// Cached frames for retransmission
    frame_cache: VecDeque<CachedFrame>,
    /// Maximum cached frames
    max_cached_frames: usize,
    /// Send buffer
    send_buffer: Vec<u8>,
    /// Heartbeat sequence number
    heartbeat_seq: u32,
    /// Last heartbeat time
    last_heartbeat: Instant,
}

impl<T: DatagramSendMut + DatagramRecvMut> ReliableEncoder<T> {
    /// Create a new reliable encoder.
    pub fn new(params: ReliableParams, transport: T) -> Self {
        let max_packet_size = ReliableHeader::SIZE + params.symbol_bytes;
        Self {
            params,
            transport,
            stats: NetworkStats::new(),
            next_frame_id: 0,
            frame_cache: VecDeque::with_capacity(64),
            max_cached_frames: 64,
            send_buffer: Vec::with_capacity(max_packet_size),
            heartbeat_seq: 0,
            last_heartbeat: Instant::now(),
        }
    }

    /// Get current network statistics.
    pub fn stats(&self) -> &NetworkStats {
        &self.stats
    }

    /// Get a mutable reference to the transport.
    pub fn transport_mut(&mut self) -> &mut T {
        &mut self.transport
    }

    /// Send a frame with FEC protection.
    ///
    /// The frame is split into symbols, encoded with FEC parity,
    /// and transmitted. Returns the frame ID.
    pub fn send_frame(&mut self, data: &[u8]) -> io::Result<u32> {
        let frame_id = self.next_frame_id;
        self.next_frame_id = self.next_frame_id.wrapping_add(1);

        // Calculate symbol count
        let source_count = (data.len() + self.params.symbol_bytes - 1) / self.params.symbol_bytes;
        let source_count = source_count.min(self.params.max_symbols_per_frame);

        // Calculate parity count based on network conditions
        let parity_count = calculate_parity_with_redundancy(
            source_count,
            self.stats.loss_rate(),
            self.params.base_redundancy,
        );

        let total_count = source_count + parity_count;
        let k = source_count as u16;
        let n = total_count as u16;

        // Pad data to fit symbols
        let padded_len = source_count * self.params.symbol_bytes;
        let mut padded_data = data.to_vec();
        padded_data.resize(padded_len, 0);

        // Split into source symbols
        let mut source_symbols: Vec<Vec<u8>> = Vec::with_capacity(source_count);
        for i in 0..source_count {
            let start = i * self.params.symbol_bytes;
            let end = start + self.params.symbol_bytes;
            source_symbols.push(padded_data[start..end].to_vec());
        }

        // Encode parity symbols
        let parity_symbols = if parity_count > 0 {
            self.encode_parity(&source_symbols, source_count, parity_count)?
        } else {
            Vec::new()
        };

        // Send source packets
        for (i, symbol) in source_symbols.iter().enumerate() {
            let header = ReliableHeader::source(frame_id, i as u8, k, n, data.len() as u16);
            self.send_packet(&header, symbol)?;
        }

        // Send parity packets
        for (i, symbol) in parity_symbols.iter().enumerate() {
            let symbol_idx = source_count + i;
            let header = ReliableHeader::parity(
                frame_id,
                symbol_idx as u8,
                k,
                n,
                data.len() as u16, // Use actual data length, not symbol size
            );
            self.send_packet(&header, symbol)?;
        }

        // Cache for retransmission
        if self.params.enable_arq {
            self.cache_frame(frame_id, source_symbols, parity_symbols);
        }

        Ok(frame_id)
    }

    /// Encode parity symbols using CM256.
    fn encode_parity(
        &self,
        source_symbols: &[Vec<u8>],
        source_count: usize,
        parity_count: usize,
    ) -> io::Result<Vec<Vec<u8>>> {
        let params = Params::new(source_count, parity_count, self.params.symbol_bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;

        let blocks: Vec<Block> = source_symbols
            .iter()
            .enumerate()
            .map(|(i, data)| Block::new(i as u8, data))
            .collect();

        let mut parity_data = vec![0u8; parity_count * self.params.symbol_bytes];
        crate::encode(&params, &blocks, &mut parity_data)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        // Split parity data into symbols
        let mut parity_symbols = Vec::with_capacity(parity_count);
        for i in 0..parity_count {
            let start = i * self.params.symbol_bytes;
            let end = start + self.params.symbol_bytes;
            parity_symbols.push(parity_data[start..end].to_vec());
        }

        Ok(parity_symbols)
    }

    /// Send a packet.
    fn send_packet(&mut self, header: &ReliableHeader, payload: &[u8]) -> io::Result<()> {
        self.send_buffer.clear();
        self.send_buffer.extend_from_slice(&header.to_bytes());
        self.send_buffer.extend_from_slice(payload);
        self.transport.send_datagram(&self.send_buffer)?;
        self.stats.on_packet_sent();
        Ok(())
    }

    /// Cache a frame for retransmission.
    fn cache_frame(&mut self, frame_id: u32, sources: Vec<Vec<u8>>, parities: Vec<Vec<u8>>) {
        if self.frame_cache.len() >= self.max_cached_frames {
            self.frame_cache.pop_front();
        }
        self.frame_cache
            .push_back(CachedFrame::new(frame_id, sources, parities));
    }

    /// Handle a received control packet (ACK/NACK).
    pub fn handle_control(&mut self, buf: &[u8]) -> io::Result<()> {
        let header = match ReliableHeader::from_bytes(buf) {
            Some(h) => h,
            None => return Ok(()),
        };

        match header.packet_type {
            PacketType::Ack => {
                // Frame acknowledged - can remove from cache
                let rtt = elapsed_from_timestamp(header.timestamp);
                self.stats.update_rtt(rtt as u64);
                self.stats.update_loss(false);
                self.remove_cached_frame(header.frame_id);
            }
            PacketType::Nack => {
                // Retransmission request
                self.stats.update_loss(true);
                let start = header.symbol_index;
                let end = header.source_count as u8;
                self.retransmit(header.frame_id, start, end)?;
            }
            PacketType::HeartbeatAck => {
                // RTT measurement
                let rtt = elapsed_from_timestamp(header.timestamp);
                self.stats.update_rtt(rtt as u64);
            }
            _ => {}
        }

        Ok(())
    }

    /// Retransmit symbols for a frame.
    fn retransmit(&mut self, frame_id: u32, start: u8, end: u8) -> io::Result<()> {
        // First, collect the data we need to send
        let packets_to_send: Vec<(ReliableHeader, Vec<u8>)> = {
            let cached = match self.frame_cache.iter().find(|f| f.frame_id == frame_id) {
                Some(f) => f,
                None => return Ok(()), // Frame no longer cached
            };

            let k = cached.source_symbols.len() as u16;
            let n = (cached.source_symbols.len() + cached.parity_symbols.len()) as u16;

            let mut packets = Vec::new();
            for idx in start..=end {
                if let Some(symbol) = cached.get_symbol(idx) {
                    let is_parity = (idx as usize) >= cached.source_symbols.len();
                    let header = if is_parity {
                        ReliableHeader::parity(frame_id, idx, k, n, symbol.len() as u16)
                    } else {
                        ReliableHeader::source(frame_id, idx, k, n, symbol.len() as u16)
                    };
                    packets.push((header, symbol.to_vec()));
                }
            }
            packets
        };

        // Now send the packets
        for (header, symbol) in packets_to_send {
            self.send_packet(&header, &symbol)?;
        }

        Ok(())
    }

    /// Remove a frame from the cache.
    fn remove_cached_frame(&mut self, frame_id: u32) {
        self.frame_cache.retain(|f| f.frame_id != frame_id);
    }

    /// Send a heartbeat probe if it's time.
    pub fn maybe_send_heartbeat(&mut self) -> io::Result<bool> {
        if !self.params.enable_heartbeat {
            return Ok(false);
        }

        if self.last_heartbeat.elapsed() < Duration::from_millis(self.params.heartbeat_interval_ms)
        {
            return Ok(false);
        }

        let header = ReliableHeader::heartbeat(self.heartbeat_seq);
        self.heartbeat_seq = self.heartbeat_seq.wrapping_add(1);
        self.send_buffer.clear();
        self.send_buffer.extend_from_slice(&header.to_bytes());
        self.transport.send_datagram(&self.send_buffer)?;
        self.last_heartbeat = Instant::now();
        Ok(true)
    }

    /// Clean up old cached frames.
    pub fn cleanup(&mut self, max_age_ms: u64) {
        let max_age = Duration::from_millis(max_age_ms);
        self.frame_cache.retain(|f| f.created.elapsed() < max_age);
    }
}

// =============================================================================
// Reliable Decoder
// =============================================================================

/// Result of receiving a packet.
#[derive(Debug)]
pub enum RecvResult {
    /// A complete frame was decoded.
    Frame {
        frame_id: u32,
        data: Vec<u8>,
        recovered: bool,
    },
    /// Packet received but frame not yet complete.
    Pending,
    /// Control packet (ACK/NACK/Heartbeat) received.
    Control,
    /// Frame timed out and was dropped.
    Timeout { frame_id: u32 },
    /// No packet available (would block).
    WouldBlock,
}

/// Reliable streaming decoder with FEC recovery and ARQ support.
#[derive(Debug)]
pub struct ReliableDecoder<T> {
    params: ReliableParams,
    transport: T,
    /// Pending frame buffers
    frames: BTreeMap<u32, FrameBuffer>,
    /// Completed frames waiting for in-order delivery
    completed: BTreeMap<u32, Vec<u8>>,
    /// Next expected frame ID for in-order delivery
    next_frame_id: Option<u32>,
    /// Receive buffer
    recv_buffer: Vec<u8>,
    /// Send buffer (for ACKs/NACKs)
    send_buffer: Vec<u8>,
    /// Last retry check time
    last_retry_check: Instant,
    /// Network statistics (for informational purposes)
    stats: NetworkStats,
}

impl<T: DatagramSendMut + DatagramRecvMut> ReliableDecoder<T> {
    /// Create a new reliable decoder.
    pub fn new(params: ReliableParams, transport: T) -> Self {
        let max_packet_size = ReliableHeader::SIZE + params.symbol_bytes;
        Self {
            params,
            transport,
            frames: BTreeMap::new(),
            completed: BTreeMap::new(),
            next_frame_id: None,
            recv_buffer: vec![0u8; max_packet_size + 1024],
            send_buffer: Vec::with_capacity(ReliableHeader::SIZE),
            last_retry_check: Instant::now(),
            stats: NetworkStats::new(),
        }
    }

    /// Get network statistics.
    pub fn stats(&self) -> &NetworkStats {
        &self.stats
    }

    /// Get a mutable reference to the transport.
    pub fn transport_mut(&mut self) -> &mut T {
        &mut self.transport
    }

    /// Receive and process a packet.
    pub fn recv(&mut self) -> io::Result<RecvResult> {
        // First check for timeouts and retries
        self.check_timeouts_and_retries()?;

        // Try to receive a packet
        let n = match self.transport.try_recv_datagram(&mut self.recv_buffer)? {
            Some(n) => n,
            None => return Ok(RecvResult::WouldBlock),
        };

        self.process_packet(n)
    }

    /// Receive a packet (blocking).
    pub fn recv_blocking(&mut self) -> io::Result<RecvResult> {
        let n = self.transport.recv_datagram(&mut self.recv_buffer)?;
        self.process_packet(n)
    }

    /// Process a received packet.
    fn process_packet(&mut self, n: usize) -> io::Result<RecvResult> {
        if n < ReliableHeader::SIZE {
            return Ok(RecvResult::Pending);
        }

        let header = match ReliableHeader::from_bytes(&self.recv_buffer[..n]) {
            Some(h) => h,
            None => return Ok(RecvResult::Pending),
        };

        match header.packet_type {
            PacketType::Source | PacketType::Parity => self
                .handle_data_packet(&header, &self.recv_buffer[ReliableHeader::SIZE..n].to_vec()),
            PacketType::Heartbeat => {
                // Respond to heartbeat
                let ack = ReliableHeader::heartbeat_ack(header.frame_id, header.timestamp);
                self.send_buffer.clear();
                self.send_buffer.extend_from_slice(&ack.to_bytes());
                self.transport.send_datagram(&self.send_buffer)?;
                Ok(RecvResult::Control)
            }
            PacketType::HeartbeatAck => {
                // Update RTT stats
                let rtt = elapsed_from_timestamp(header.timestamp);
                self.stats.update_rtt(rtt as u64);
                Ok(RecvResult::Control)
            }
            _ => Ok(RecvResult::Control),
        }
    }

    /// Handle a data packet (source or parity).
    fn handle_data_packet(
        &mut self,
        header: &ReliableHeader,
        payload: &[u8],
    ) -> io::Result<RecvResult> {
        let frame_id = header.frame_id;
        let is_parity = header.packet_type == PacketType::Parity;

        // Initialize next_frame_id if not set
        if self.next_frame_id.is_none() {
            self.next_frame_id = Some(frame_id);
        }

        // Skip if frame is too old
        if let Some(next) = self.next_frame_id {
            let diff = frame_id.wrapping_sub(next);
            if diff > 0x80000000 {
                // frame_id < next (wrapped)
                return Ok(RecvResult::Pending);
            }
        }

        // Skip if already completed
        if self.completed.contains_key(&frame_id) {
            return Ok(RecvResult::Pending);
        }

        // Get or create frame buffer
        let frame = self.frames.entry(frame_id).or_insert_with(|| {
            FrameBuffer::new(
                frame_id,
                header.source_count,
                header.total_count,
                self.params.symbol_bytes,
                header.data_len,
            )
        });

        // Add the symbol
        frame.add_symbol(header.symbol_index, is_parity, payload, header.data_len);

        // Try to decode
        if let Some(data) = frame.try_decode() {
            let _recovered = !frame.all_sources_present();
            self.frames.remove(&frame_id);
            self.completed.insert(frame_id, data.clone());

            // Send ACK
            if self.params.enable_arq {
                self.send_ack(frame_id)?;
            }

            // Try to deliver in-order frames
            return self.try_deliver();
        }

        Ok(RecvResult::Pending)
    }

    /// Send an ACK for a frame.
    fn send_ack(&mut self, frame_id: u32) -> io::Result<()> {
        let header = ReliableHeader::ack(frame_id);
        self.send_buffer.clear();
        self.send_buffer.extend_from_slice(&header.to_bytes());
        self.transport.send_datagram(&self.send_buffer)?;
        Ok(())
    }

    /// Send a NACK (retransmit request).
    fn send_nack(&mut self, frame_id: u32, missing: &[u8]) -> io::Result<()> {
        if missing.is_empty() {
            return Ok(());
        }

        // Send ranges of missing symbols
        let mut start = missing[0];
        let mut end = missing[0];

        for &idx in &missing[1..] {
            if idx == end + 1 {
                end = idx;
            } else {
                // Send current range
                let header = ReliableHeader::nack(frame_id, start, end);
                self.send_buffer.clear();
                self.send_buffer.extend_from_slice(&header.to_bytes());
                self.transport.send_datagram(&self.send_buffer)?;
                start = idx;
                end = idx;
            }
        }

        // Send final range
        let header = ReliableHeader::nack(frame_id, start, end);
        self.send_buffer.clear();
        self.send_buffer.extend_from_slice(&header.to_bytes());
        self.transport.send_datagram(&self.send_buffer)?;

        Ok(())
    }

    /// Try to deliver completed frames in order.
    fn try_deliver(&mut self) -> io::Result<RecvResult> {
        loop {
            let next = match self.next_frame_id {
                Some(id) => id,
                None => return Ok(RecvResult::Pending),
            };

            if let Some(data) = self.completed.remove(&next) {
                self.next_frame_id = Some(next.wrapping_add(1));
                return Ok(RecvResult::Frame {
                    frame_id: next,
                    data,
                    recovered: false, // We don't track this for already-completed frames
                });
            } else {
                break;
            }
        }

        Ok(RecvResult::Pending)
    }

    /// Check for timeouts and send retransmission requests.
    fn check_timeouts_and_retries(&mut self) -> io::Result<()> {
        if self.last_retry_check.elapsed() < Duration::from_millis(50) {
            return Ok(());
        }
        self.last_retry_check = Instant::now();

        let timeout_ms = self.params.frame_timeout_ms;
        let retry_interval = self.params.retry_interval_ms;
        let max_retries = self.params.max_retries;

        // Collect frames to timeout
        let mut timed_out = Vec::new();
        let mut retry_frames = Vec::new();

        for (&frame_id, frame) in &mut self.frames {
            if frame.is_timed_out(timeout_ms) {
                timed_out.push(frame_id);
            } else if self.params.enable_arq
                && frame.should_retry(retry_interval)
                && frame.retry_count < max_retries
            {
                frame.retry_count += 1;
                let missing = frame.missing_symbols();
                if !missing.is_empty() {
                    retry_frames.push((frame_id, missing));
                }
            }
        }

        // Remove timed out frames
        for frame_id in timed_out {
            self.frames.remove(&frame_id);
            // Advance next_frame_id if this was the expected frame
            if self.next_frame_id == Some(frame_id) {
                self.next_frame_id = Some(frame_id.wrapping_add(1));
            }
        }

        // Send retransmission requests
        for (frame_id, missing) in retry_frames {
            self.send_nack(frame_id, &missing)?;
        }

        Ok(())
    }

    /// Get the next expected frame ID.
    pub fn next_frame_id(&self) -> Option<u32> {
        self.next_frame_id
    }

    /// Get the number of pending frames.
    pub fn pending_frames(&self) -> usize {
        self.frames.len()
    }

    /// Get the number of completed (waiting for delivery) frames.
    pub fn completed_frames(&self) -> usize {
        self.completed.len()
    }

    /// Clean up old state.
    pub fn cleanup(&mut self, max_pending: usize) {
        // Remove oldest pending frames if too many
        while self.frames.len() > max_pending {
            if let Some((&oldest, _)) = self.frames.iter().next() {
                self.frames.remove(&oldest);
            }
        }

        // Remove old completed frames that won't be delivered
        if let Some(next) = self.next_frame_id {
            self.completed.retain(|&id, _| {
                let diff = id.wrapping_sub(next);
                diff < 0x80000000 // Keep only frames >= next
            });
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::MemoryChannel;

    #[test]
    fn test_calculate_parity_count() {
        // Low loss: k/50 + 1 = 3
        assert_eq!(calculate_parity_count(100, 0.001), 3);
        // At 0.005: k/20 + 1 = 6
        assert_eq!(calculate_parity_count(100, 0.005), 6);

        // Medium loss: k/7 + 1 = 15.28 -> ceil = 16
        assert_eq!(calculate_parity_count(100, 0.03), 16);

        // High loss: k/2 + 1 = 51
        assert_eq!(calculate_parity_count(100, 0.15), 51);

        // Very high loss: k = 100
        assert_eq!(calculate_parity_count(100, 0.25), 100);
    }

    #[test]
    fn test_network_stats() {
        let mut stats = NetworkStats::new();

        // Initial values: starts at 50ms RTT
        assert!(stats.rtt_ms() > 0.0);
        assert_eq!(stats.loss_rate(), 0.0);

        // Update with RTT samples - EWMA converges slowly (alpha=0.1)
        // After 10 samples: 50 * 0.9^10 + 10 * (1 - 0.9^10) ≈ 23.9ms
        for _ in 0..10 {
            stats.update_rtt(10_000); // 10ms
        }
        assert!(stats.rtt_ms() < 30.0); // Should be converging toward 10ms
        assert!(stats.rtt_ms() > 15.0); // But not there yet with alpha=0.1

        // Update loss
        stats.update_loss(true);
        assert!(stats.loss_rate() > 0.0);
    }

    #[test]
    fn test_header_serialization() {
        let header = ReliableHeader::source(12345, 7, 10, 15, 1200);
        let bytes = header.to_bytes();
        let parsed = ReliableHeader::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.packet_type, PacketType::Source);
        assert_eq!(parsed.frame_id, 12345);
        assert_eq!(parsed.symbol_index, 7);
        assert_eq!(parsed.source_count, 10);
        assert_eq!(parsed.total_count, 15);
        assert_eq!(parsed.data_len, 1200);
    }

    #[test]
    fn test_encode_decode_no_loss() {
        let (ch_a, ch_b) = MemoryChannel::pair();

        let params = ReliableParams::new(4, 256)
            .without_arq()
            .without_heartbeat();

        let mut encoder = ReliableEncoder::new(params, ch_a);
        let mut decoder = ReliableDecoder::new(params, ch_b);

        // Send a frame
        let data = vec![0x42u8; 1000];
        let frame_id = encoder.send_frame(&data).unwrap();
        assert_eq!(frame_id, 0);

        // Receive all packets
        loop {
            match decoder.recv().unwrap() {
                RecvResult::Frame {
                    data: recv_data, ..
                } => {
                    // Data should match (may have padding)
                    assert!(recv_data.starts_with(&data));
                    break;
                }
                RecvResult::WouldBlock => break,
                _ => {}
            }
        }
    }

    #[test]
    fn test_frame_buffer_decode() {
        let mut fb = FrameBuffer::new(0, 4, 6, 32, 128); // 4 source, 2 parity, 128 bytes data

        // Add all source symbols
        for i in 0..4 {
            fb.add_symbol(i, false, &vec![(i + 1) as u8; 32], 128);
        }

        assert!(fb.can_decode());
        assert!(fb.all_sources_present());

        let decoded = fb.try_decode().unwrap();
        assert_eq!(decoded.len(), 4 * 32);
        assert_eq!(decoded[0], 1);
        assert_eq!(decoded[32], 2);
    }
}
