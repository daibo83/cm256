//! Network statistics tracking for adaptive FEC and congestion control.
//!
//! Inspired by nyxpsi's NetworkStats implementation.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Window size for RTT samples.
const RTT_WINDOW_SIZE: usize = 10;

/// EWMA smoothing factor for loss rate (0.1 = 10% new, 90% old).
const LOSS_EWMA_ALPHA: f64 = 0.1;

/// Network statistics tracker.
///
/// Tracks packet loss rate, RTT, and derives a network quality score
/// for adaptive FEC parameter tuning.
#[derive(Debug, Clone)]
pub struct NetworkStats {
    /// Exponentially weighted moving average of packet loss rate.
    loss_rate: f64,

    /// Recent RTT samples for averaging.
    rtt_samples: VecDeque<u32>,

    /// Minimum RTT observed (for BBR).
    min_rtt: Option<Duration>,

    /// When min_rtt was last updated.
    min_rtt_timestamp: Option<Instant>,

    /// Total packets sent (for loss calculation).
    packets_sent: u64,

    /// Total packets acknowledged.
    packets_acked: u64,

    /// Total bytes delivered (for bandwidth estimation).
    bytes_delivered: u64,

    /// Timestamp of last delivery sample.
    last_delivery_time: Option<Instant>,
}

impl Default for NetworkStats {
    fn default() -> Self {
        Self::new()
    }
}

impl NetworkStats {
    /// Create a new network stats tracker.
    pub fn new() -> Self {
        Self {
            loss_rate: 0.0,
            rtt_samples: VecDeque::with_capacity(RTT_WINDOW_SIZE),
            min_rtt: None,
            min_rtt_timestamp: None,
            packets_sent: 0,
            packets_acked: 0,
            bytes_delivered: 0,
            last_delivery_time: None,
        }
    }

    /// Record that a packet was sent.
    pub fn on_packet_sent(&mut self) {
        self.packets_sent += 1;
    }

    /// Record that packets were sent.
    pub fn on_packets_sent(&mut self, count: u64) {
        self.packets_sent += count;
    }

    /// Update stats from an ACK packet.
    ///
    /// # Arguments
    ///
    /// * `received_count` - Number of packets marked as received in the ACK bitmap
    /// * `total_count` - Total packets covered by the bitmap (typically 64)
    /// * `rtt_ms` - RTT sample from the ACK packet
    /// * `bytes_acked` - Number of bytes acknowledged
    pub fn on_ack(&mut self, received_count: u32, total_count: u32, rtt_ms: u32, bytes_acked: u64) {
        let now = Instant::now();

        // Update loss rate with EWMA
        if total_count > 0 {
            let sample_loss = 1.0 - (received_count as f64 / total_count as f64);
            self.loss_rate =
                LOSS_EWMA_ALPHA * sample_loss + (1.0 - LOSS_EWMA_ALPHA) * self.loss_rate;
        }

        // Update RTT samples
        if rtt_ms > 0 {
            if self.rtt_samples.len() >= RTT_WINDOW_SIZE {
                self.rtt_samples.pop_front();
            }
            self.rtt_samples.push_back(rtt_ms);

            // Update min RTT
            let rtt = Duration::from_millis(rtt_ms as u64);
            match self.min_rtt {
                None => {
                    self.min_rtt = Some(rtt);
                    self.min_rtt_timestamp = Some(now);
                }
                Some(min) if rtt < min => {
                    self.min_rtt = Some(rtt);
                    self.min_rtt_timestamp = Some(now);
                }
                _ => {}
            }
        }

        // Update delivery tracking
        self.packets_acked += received_count as u64;
        self.bytes_delivered += bytes_acked;
        self.last_delivery_time = Some(now);
    }

    /// Update with a simple loss event (packet lost vs received).
    pub fn update_loss(&mut self, packet_received: bool) {
        let sample = if packet_received { 0.0 } else { 1.0 };
        self.loss_rate = LOSS_EWMA_ALPHA * sample + (1.0 - LOSS_EWMA_ALPHA) * self.loss_rate;
    }

    /// Update with an RTT sample.
    pub fn update_rtt(&mut self, rtt_ms: u32) {
        if self.rtt_samples.len() >= RTT_WINDOW_SIZE {
            self.rtt_samples.pop_front();
        }
        self.rtt_samples.push_back(rtt_ms);

        let rtt = Duration::from_millis(rtt_ms as u64);
        let now = Instant::now();

        match self.min_rtt {
            None => {
                self.min_rtt = Some(rtt);
                self.min_rtt_timestamp = Some(now);
            }
            Some(min) if rtt < min => {
                self.min_rtt = Some(rtt);
                self.min_rtt_timestamp = Some(now);
            }
            _ => {}
        }
    }

    /// Get the smoothed loss rate (0.0 - 1.0).
    pub fn loss_rate(&self) -> f64 {
        self.loss_rate
    }

    /// Get the average RTT in milliseconds.
    pub fn avg_rtt_ms(&self) -> u32 {
        if self.rtt_samples.is_empty() {
            return 0;
        }
        let sum: u32 = self.rtt_samples.iter().sum();
        sum / self.rtt_samples.len() as u32
    }

    /// Get the average RTT as Duration.
    pub fn avg_rtt(&self) -> Duration {
        Duration::from_millis(self.avg_rtt_ms() as u64)
    }

    /// Get the minimum RTT observed.
    pub fn min_rtt(&self) -> Option<Duration> {
        self.min_rtt
    }

    /// Get the minimum RTT in milliseconds.
    pub fn min_rtt_ms(&self) -> u32 {
        self.min_rtt.map(|d| d.as_millis() as u32).unwrap_or(0)
    }

    /// Check if min_rtt needs to be re-probed (for BBR).
    ///
    /// Returns true if min_rtt hasn't been updated in `probe_interval`.
    pub fn should_probe_rtt(&self, probe_interval: Duration) -> bool {
        match self.min_rtt_timestamp {
            None => true,
            Some(ts) => ts.elapsed() > probe_interval,
        }
    }

    /// Reset min_rtt to force re-probing.
    pub fn reset_min_rtt(&mut self) {
        self.min_rtt = None;
        self.min_rtt_timestamp = None;
    }

    /// Get the network quality score (0.0 - 1.0).
    ///
    /// Higher is better. Combines loss rate and latency.
    /// Based on nyxpsi's network quality calculation.
    pub fn network_quality(&self) -> f64 {
        let packet_success_rate = 1.0 - self.loss_rate;

        let normalized_latency = if self.rtt_samples.is_empty() {
            0.5 // Default to middle if no data
        } else {
            let avg_rtt = self.avg_rtt_ms() as f64;
            // Normalize: 0ms -> 1.0, 1000ms -> 0.5, higher -> approaches 0
            1.0 / (1.0 + avg_rtt / 1000.0)
        };

        (normalized_latency + packet_success_rate) / 2.0
    }

    /// Get total packets sent.
    pub fn packets_sent(&self) -> u64 {
        self.packets_sent
    }

    /// Get total packets acknowledged.
    pub fn packets_acked(&self) -> u64 {
        self.packets_acked
    }

    /// Get total bytes delivered.
    pub fn bytes_delivered(&self) -> u64 {
        self.bytes_delivered
    }

    /// Estimate current bandwidth in bytes per second.
    ///
    /// Returns None if not enough data.
    pub fn estimated_bandwidth(&self) -> Option<u64> {
        let min_rtt = self.min_rtt?;
        if min_rtt.is_zero() {
            return None;
        }

        // BDP-based estimate: bytes_in_flight / RTT
        // This is a simplified estimate; BBR does more sophisticated tracking
        Some((self.bytes_delivered as f64 / min_rtt.as_secs_f64()) as u64)
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_stats() {
        let stats = NetworkStats::new();
        assert_eq!(stats.loss_rate(), 0.0);
        assert_eq!(stats.avg_rtt_ms(), 0);
        assert!(stats.min_rtt().is_none());
    }

    #[test]
    fn test_loss_rate_ewma() {
        let mut stats = NetworkStats::new();

        // Simulate 100% loss for a while
        for _ in 0..20 {
            stats.update_loss(false);
        }

        // Loss rate should approach 1.0
        assert!(stats.loss_rate() > 0.8);

        // Now simulate 100% success
        for _ in 0..50 {
            stats.update_loss(true);
        }

        // Loss rate should drop significantly
        assert!(stats.loss_rate() < 0.1);
    }

    #[test]
    fn test_rtt_tracking() {
        let mut stats = NetworkStats::new();

        stats.update_rtt(50);
        stats.update_rtt(60);
        stats.update_rtt(40);

        assert_eq!(stats.avg_rtt_ms(), 50);
        assert_eq!(stats.min_rtt_ms(), 40);
    }

    #[test]
    fn test_on_ack() {
        let mut stats = NetworkStats::new();

        // Simulate ACK with 60/64 received
        stats.on_ack(60, 64, 50, 60 * 1200);

        // Loss should be ~6.25% but EWMA starts from 0, so first sample is 0.1 * 0.0625 ≈ 0.00625
        assert!(stats.loss_rate() > 0.0 && stats.loss_rate() < 0.10);
        assert_eq!(stats.avg_rtt_ms(), 50);
        assert_eq!(stats.packets_acked(), 60);
    }

    #[test]
    fn test_network_quality() {
        let mut stats = NetworkStats::new();

        // Perfect conditions: no loss, low RTT
        for _ in 0..10 {
            stats.update_loss(true);
            stats.update_rtt(20);
        }

        let quality = stats.network_quality();
        assert!(quality > 0.9, "Quality should be high: {}", quality);

        // Poor conditions: high loss, high RTT
        let mut poor_stats = NetworkStats::new();
        for _ in 0..50 {
            poor_stats.update_loss(false);
            poor_stats.update_rtt(500);
        }

        let poor_quality = poor_stats.network_quality();
        assert!(
            poor_quality < 0.5,
            "Quality should be low: {}",
            poor_quality
        );
    }

    #[test]
    fn test_rtt_window() {
        let mut stats = NetworkStats::new();

        // Fill window with high RTT
        for _ in 0..RTT_WINDOW_SIZE {
            stats.update_rtt(100);
        }
        assert_eq!(stats.avg_rtt_ms(), 100);

        // Add low RTT samples, should push out old ones
        for _ in 0..RTT_WINDOW_SIZE {
            stats.update_rtt(20);
        }
        assert_eq!(stats.avg_rtt_ms(), 20);
    }
}
