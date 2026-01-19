//! BBR (Bottleneck Bandwidth and Round-trip propagation time) congestion control.
//!
//! Implements Google's BBR algorithm for bandwidth estimation and pacing.
//! Reference: https://research.google/pubs/pub45646/

use std::time::{Duration, Instant};

use super::stats::NetworkStats;

/// BBR operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BbrMode {
    /// Exponential growth to find bottleneck bandwidth.
    Startup,
    /// Drain the queue after startup overshoot.
    Drain,
    /// Steady state with periodic bandwidth probing.
    ProbeBW,
    /// Periodically probe for lower RTT.
    ProbeRTT,
}

/// Pacing gain cycle for ProbeBW mode.
const PACING_GAIN_CYCLE: [f64; 8] = [1.25, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

/// Startup pacing gain (2/ln(2) ≈ 2.89).
const STARTUP_PACING_GAIN: f64 = 2.89;

/// Drain pacing gain (1/startup_gain).
const DRAIN_PACING_GAIN: f64 = 1.0 / STARTUP_PACING_GAIN;

/// Startup cwnd gain.
const STARTUP_CWND_GAIN: f64 = 2.89;

/// Default cwnd gain for steady state.
const DEFAULT_CWND_GAIN: f64 = 2.0;

/// ProbeRTT cwnd (minimum packets to keep in flight).
const PROBE_RTT_CWND: u32 = 4;

/// ProbeRTT duration.
const PROBE_RTT_DURATION: Duration = Duration::from_millis(200);

/// Minimum RTT probe interval.
const MIN_RTT_PROBE_INTERVAL: Duration = Duration::from_secs(10);

/// Rounds to wait for bandwidth increase in startup.
const STARTUP_ROUNDS_WITHOUT_GROWTH: u32 = 3;

/// Bandwidth growth threshold for startup (25% growth).
const STARTUP_GROWTH_THRESHOLD: f64 = 1.25;

/// Default packet size for calculations.
const DEFAULT_PACKET_SIZE: u32 = 1200;

/// BBR congestion control state.
#[derive(Debug, Clone)]
pub struct BbrState {
    /// Current operating mode.
    mode: BbrMode,

    /// Estimated bottleneck bandwidth (bytes/sec).
    btl_bw: u64,

    /// Minimum RTT observed (propagation delay).
    rt_prop: Option<Duration>,

    /// When rt_prop was last updated.
    rt_prop_stamp: Option<Instant>,

    /// Current pacing rate (bytes/sec).
    pacing_rate: u64,

    /// Current congestion window (packets).
    cwnd: u32,

    /// Packets currently in flight.
    inflight: u32,

    /// Cycle index for ProbeBW mode.
    cycle_index: usize,

    /// When the current cycle phase started.
    cycle_stamp: Instant,

    /// Round count for startup exit detection.
    round_count: u64,

    /// Last round's bandwidth for growth detection.
    last_round_bw: u64,

    /// Rounds without significant bandwidth growth.
    rounds_without_growth: u32,

    /// ProbeRTT entry time.
    probe_rtt_start: Option<Instant>,

    /// Whether we're done with ProbeRTT.
    probe_rtt_done: bool,

    /// Initial cwnd.
    initial_cwnd: u32,

    /// Packet size for BDP calculations.
    packet_size: u32,

    /// RTT probe interval.
    rtt_probe_interval: Duration,
}

impl BbrState {
    /// Create a new BBR state with default parameters.
    pub fn new() -> Self {
        Self::with_config(10, DEFAULT_PACKET_SIZE, MIN_RTT_PROBE_INTERVAL)
    }

    /// Create a new BBR state with custom configuration.
    pub fn with_config(initial_cwnd: u32, packet_size: u32, rtt_probe_interval: Duration) -> Self {
        Self {
            mode: BbrMode::Startup,
            btl_bw: 0,
            rt_prop: None,
            rt_prop_stamp: None,
            pacing_rate: initial_cwnd as u64 * packet_size as u64, // Initial guess
            cwnd: initial_cwnd,
            inflight: 0,
            cycle_index: 0,
            cycle_stamp: Instant::now(),
            round_count: 0,
            last_round_bw: 0,
            rounds_without_growth: 0,
            probe_rtt_start: None,
            probe_rtt_done: false,
            initial_cwnd,
            packet_size,
            rtt_probe_interval,
        }
    }

    /// Get current mode.
    pub fn mode(&self) -> BbrMode {
        self.mode
    }

    /// Get estimated bottleneck bandwidth (bytes/sec).
    pub fn btl_bw(&self) -> u64 {
        self.btl_bw
    }

    /// Get minimum RTT.
    pub fn rt_prop(&self) -> Option<Duration> {
        self.rt_prop
    }

    /// Get current pacing rate (bytes/sec).
    pub fn pacing_rate(&self) -> u64 {
        self.pacing_rate
    }

    /// Get current congestion window (packets).
    pub fn cwnd(&self) -> u32 {
        self.cwnd
    }

    /// Get packets in flight.
    pub fn inflight(&self) -> u32 {
        self.inflight
    }

    /// Check if we can send more packets.
    pub fn can_send(&self) -> bool {
        self.inflight < self.cwnd
    }

    /// Get the pacing interval for the next packet.
    ///
    /// Returns the duration to wait before sending the next packet.
    pub fn pacing_interval(&self) -> Duration {
        if self.pacing_rate == 0 {
            return Duration::from_millis(1);
        }

        let interval_ns = (self.packet_size as u64 * 1_000_000_000) / self.pacing_rate;
        Duration::from_nanos(interval_ns)
    }

    /// Record that a packet was sent.
    pub fn on_send(&mut self) {
        self.inflight = self.inflight.saturating_add(1);
    }

    /// Record that packets were sent.
    pub fn on_send_n(&mut self, n: u32) {
        self.inflight = self.inflight.saturating_add(n);
    }

    /// Process an ACK and update BBR state.
    ///
    /// # Arguments
    ///
    /// * `packets_acked` - Number of packets acknowledged
    /// * `bytes_acked` - Number of bytes acknowledged
    /// * `rtt` - RTT sample from this ACK
    /// * `now` - Current time
    pub fn on_ack(&mut self, packets_acked: u32, bytes_acked: u64, rtt: Duration, now: Instant) {
        // Update inflight
        self.inflight = self.inflight.saturating_sub(packets_acked);

        // Update RTT
        self.update_rt_prop(rtt, now);

        // Update bandwidth estimate
        self.update_bandwidth(bytes_acked, rtt);

        // Increment round count (simplified: each ACK is a "round")
        self.round_count += 1;

        // State machine transitions
        match self.mode {
            BbrMode::Startup => self.check_startup_exit(),
            BbrMode::Drain => self.check_drain_exit(),
            BbrMode::ProbeBW => self.update_probe_bw(now),
            BbrMode::ProbeRTT => self.update_probe_rtt(now),
        }

        // Check if we should enter ProbeRTT
        if self.mode != BbrMode::ProbeRTT && self.should_probe_rtt(now) {
            self.enter_probe_rtt(now);
        }

        // Update pacing rate and cwnd
        self.update_pacing_rate();
        self.update_cwnd();
    }

    /// Update from NetworkStats (convenience method).
    pub fn update_from_stats(
        &mut self,
        stats: &NetworkStats,
        packets_acked: u32,
        bytes_acked: u64,
    ) {
        let rtt = stats.avg_rtt();
        if rtt.is_zero() {
            return;
        }
        self.on_ack(packets_acked, bytes_acked, rtt, Instant::now());
    }

    /// Update minimum RTT.
    fn update_rt_prop(&mut self, rtt: Duration, now: Instant) {
        if rtt.is_zero() {
            return;
        }

        match self.rt_prop {
            None => {
                self.rt_prop = Some(rtt);
                self.rt_prop_stamp = Some(now);
            }
            Some(current) if rtt <= current => {
                self.rt_prop = Some(rtt);
                self.rt_prop_stamp = Some(now);
            }
            _ => {}
        }
    }

    /// Update bandwidth estimate.
    fn update_bandwidth(&mut self, bytes_acked: u64, rtt: Duration) {
        if rtt.is_zero() || bytes_acked == 0 {
            return;
        }

        // Bandwidth = bytes / time
        let bw = (bytes_acked as f64 / rtt.as_secs_f64()) as u64;

        // Take max of recent samples (simplified windowed max)
        if bw > self.btl_bw {
            self.btl_bw = bw;
        }
    }

    /// Check if we should exit Startup mode.
    fn check_startup_exit(&mut self) {
        // Check for bandwidth plateau
        if self.btl_bw > 0 {
            let growth = self.btl_bw as f64 / self.last_round_bw.max(1) as f64;

            if growth < STARTUP_GROWTH_THRESHOLD {
                self.rounds_without_growth += 1;
            } else {
                self.rounds_without_growth = 0;
            }

            self.last_round_bw = self.btl_bw;

            if self.rounds_without_growth >= STARTUP_ROUNDS_WITHOUT_GROWTH {
                self.mode = BbrMode::Drain;
                self.rounds_without_growth = 0;
            }
        }
    }

    /// Check if we should exit Drain mode.
    fn check_drain_exit(&mut self) {
        // Exit drain when inflight drops to BDP
        let bdp = self.bdp_packets();
        if self.inflight <= bdp {
            self.mode = BbrMode::ProbeBW;
            self.cycle_index = 0;
            self.cycle_stamp = Instant::now();
        }
    }

    /// Update ProbeBW state.
    fn update_probe_bw(&mut self, now: Instant) {
        // Advance cycle every RTT
        if let Some(rtt) = self.rt_prop {
            if now.duration_since(self.cycle_stamp) >= rtt {
                self.cycle_index = (self.cycle_index + 1) % PACING_GAIN_CYCLE.len();
                self.cycle_stamp = now;
            }
        }
    }

    /// Update ProbeRTT state.
    fn update_probe_rtt(&mut self, now: Instant) {
        if let Some(start) = self.probe_rtt_start {
            // Wait for inflight to drain
            if self.inflight <= PROBE_RTT_CWND && !self.probe_rtt_done {
                self.probe_rtt_done = true;
            }

            // Exit after duration
            if self.probe_rtt_done && now.duration_since(start) >= PROBE_RTT_DURATION {
                self.exit_probe_rtt();
            }
        }
    }

    /// Check if we should enter ProbeRTT.
    fn should_probe_rtt(&self, now: Instant) -> bool {
        match self.rt_prop_stamp {
            Some(stamp) => now.duration_since(stamp) >= self.rtt_probe_interval,
            None => false,
        }
    }

    /// Enter ProbeRTT mode.
    fn enter_probe_rtt(&mut self, now: Instant) {
        self.mode = BbrMode::ProbeRTT;
        self.probe_rtt_start = Some(now);
        self.probe_rtt_done = false;
    }

    /// Exit ProbeRTT mode.
    fn exit_probe_rtt(&mut self) {
        self.mode = BbrMode::ProbeBW;
        self.probe_rtt_start = None;
        self.probe_rtt_done = false;
        self.cycle_index = 0;
        self.cycle_stamp = Instant::now();
    }

    /// Get current pacing gain.
    fn pacing_gain(&self) -> f64 {
        match self.mode {
            BbrMode::Startup => STARTUP_PACING_GAIN,
            BbrMode::Drain => DRAIN_PACING_GAIN,
            BbrMode::ProbeBW => PACING_GAIN_CYCLE[self.cycle_index],
            BbrMode::ProbeRTT => 1.0,
        }
    }

    /// Get current cwnd gain.
    fn cwnd_gain(&self) -> f64 {
        match self.mode {
            BbrMode::Startup => STARTUP_CWND_GAIN,
            BbrMode::Drain => STARTUP_CWND_GAIN, // Keep high during drain
            BbrMode::ProbeBW => DEFAULT_CWND_GAIN,
            BbrMode::ProbeRTT => 1.0,
        }
    }

    /// Update pacing rate based on current state.
    fn update_pacing_rate(&mut self) {
        let gain = self.pacing_gain();
        self.pacing_rate = (self.btl_bw as f64 * gain) as u64;

        // Minimum pacing rate
        if self.pacing_rate == 0 {
            self.pacing_rate = self.initial_cwnd as u64 * self.packet_size as u64;
        }
    }

    /// Update congestion window based on current state.
    fn update_cwnd(&mut self) {
        if self.mode == BbrMode::ProbeRTT {
            self.cwnd = PROBE_RTT_CWND;
            return;
        }

        let gain = self.cwnd_gain();
        let bdp = self.bdp_packets();

        self.cwnd = ((bdp as f64 * gain) as u32).max(self.initial_cwnd);
    }

    /// Calculate BDP in packets.
    fn bdp_packets(&self) -> u32 {
        let rtt = self.rt_prop.unwrap_or(Duration::from_millis(100));
        let bdp_bytes = self.btl_bw as f64 * rtt.as_secs_f64();
        ((bdp_bytes / self.packet_size as f64) as u32).max(1)
    }

    /// Reset BBR state.
    pub fn reset(&mut self) {
        *self = Self::with_config(self.initial_cwnd, self.packet_size, self.rtt_probe_interval);
    }
}

impl Default for BbrState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let bbr = BbrState::new();
        assert_eq!(bbr.mode(), BbrMode::Startup);
        assert!(bbr.can_send());
    }

    #[test]
    fn test_send_tracking() {
        let mut bbr = BbrState::new();

        bbr.on_send();
        assert_eq!(bbr.inflight(), 1);

        bbr.on_send_n(5);
        assert_eq!(bbr.inflight(), 6);
    }

    #[test]
    fn test_ack_reduces_inflight() {
        let mut bbr = BbrState::new();
        let now = Instant::now();

        bbr.on_send_n(10);
        assert_eq!(bbr.inflight(), 10);

        bbr.on_ack(5, 5 * 1200, Duration::from_millis(50), now);
        assert_eq!(bbr.inflight(), 5);
    }

    #[test]
    fn test_rtt_tracking() {
        let mut bbr = BbrState::new();
        let now = Instant::now();

        // First RTT sets rt_prop
        bbr.on_ack(1, 1200, Duration::from_millis(50), now);
        assert_eq!(bbr.rt_prop(), Some(Duration::from_millis(50)));

        // Lower RTT updates rt_prop
        bbr.on_ack(1, 1200, Duration::from_millis(30), now);
        assert_eq!(bbr.rt_prop(), Some(Duration::from_millis(30)));

        // Higher RTT doesn't update rt_prop
        bbr.on_ack(1, 1200, Duration::from_millis(60), now);
        assert_eq!(bbr.rt_prop(), Some(Duration::from_millis(30)));
    }

    #[test]
    fn test_bandwidth_tracking() {
        let mut bbr = BbrState::new();
        let now = Instant::now();

        // Simulate high bandwidth
        bbr.on_ack(10, 10 * 1200, Duration::from_millis(10), now);

        // btl_bw should be set (10 * 1200 bytes / 0.01 sec = 1.2 MB/s)
        assert!(bbr.btl_bw() > 0);
    }

    #[test]
    fn test_can_send_respects_cwnd() {
        let mut bbr = BbrState::with_config(5, 1200, Duration::from_secs(10));

        // Should be able to send initially
        assert!(bbr.can_send());

        // Fill up cwnd
        for _ in 0..5 {
            bbr.on_send();
        }

        // Should not be able to send when cwnd is full
        assert!(!bbr.can_send());
    }

    #[test]
    fn test_pacing_interval() {
        let mut bbr = BbrState::new();
        let now = Instant::now();

        // Set up some bandwidth
        bbr.on_ack(10, 10 * 1200, Duration::from_millis(10), now);

        let interval = bbr.pacing_interval();
        assert!(interval > Duration::ZERO);
        assert!(interval < Duration::from_secs(1));
    }
}
