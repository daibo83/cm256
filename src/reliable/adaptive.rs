//! Adaptive FEC tuner for dynamic redundancy adjustment.
//!
//! Adjusts FEC parameters based on observed network conditions.

use super::stats::NetworkStats;
use crate::streaming::StreamingParams;

/// Adaptive FEC parameter tuner.
///
/// Adjusts FEC redundancy based on observed packet loss rate.
/// Higher loss rates trigger more aggressive FEC protection.
#[derive(Debug, Clone)]
pub struct AdaptiveFec {
    /// Base delay (window size).
    delay: u8,

    /// Base step size.
    step_size: u8,

    /// Symbol size in bytes.
    symbol_bytes: usize,

    /// Current number of parities.
    current_parities: u8,

    /// Minimum parities.
    min_parities: u8,

    /// Maximum parities.
    max_parities: u8,

    /// Loss threshold for increasing parities.
    increase_threshold: f64,

    /// Loss threshold for decreasing parities.
    decrease_threshold: f64,

    /// Hysteresis counter to prevent rapid oscillation.
    stable_count: u32,

    /// Required stable readings before changing.
    stable_threshold: u32,
}

impl AdaptiveFec {
    /// Create a new adaptive FEC tuner.
    ///
    /// # Arguments
    ///
    /// * `delay` - FEC window size
    /// * `initial_parities` - Starting number of parity symbols
    /// * `step_size` - Parity generation frequency
    /// * `symbol_bytes` - Size of each symbol
    pub fn new(delay: u8, initial_parities: u8, step_size: u8, symbol_bytes: usize) -> Self {
        Self {
            delay,
            step_size,
            symbol_bytes,
            current_parities: initial_parities,
            min_parities: 1,
            max_parities: 4,
            increase_threshold: 0.10, // Increase if loss > 10%
            decrease_threshold: 0.03, // Decrease if loss < 3%
            stable_count: 0,
            stable_threshold: 5,
        }
    }

    /// Create from existing streaming params.
    pub fn from_params(params: StreamingParams) -> Self {
        Self::new(
            params.delay(),
            params.num_parities(),
            params.step_size(),
            params.symbol_bytes(),
        )
    }

    /// Set the parity range.
    pub fn set_parity_range(&mut self, min: u8, max: u8) {
        self.min_parities = min;
        self.max_parities = max;
        self.current_parities = self.current_parities.clamp(min, max);
    }

    /// Set the loss thresholds for adaptation.
    pub fn set_thresholds(&mut self, increase: f64, decrease: f64) {
        self.increase_threshold = increase;
        self.decrease_threshold = decrease;
    }

    /// Set the stability threshold (readings before changing).
    pub fn set_stable_threshold(&mut self, threshold: u32) {
        self.stable_threshold = threshold;
    }

    /// Get current number of parities.
    pub fn current_parities(&self) -> u8 {
        self.current_parities
    }

    /// Get current FEC overhead ratio.
    pub fn overhead(&self) -> f64 {
        self.current_parities as f64 / self.step_size as f64
    }

    /// Update FEC parameters based on network stats.
    ///
    /// Returns true if parameters changed.
    pub fn update(&mut self, stats: &NetworkStats) -> bool {
        let loss = stats.loss_rate();
        let old_parities = self.current_parities;

        if loss > self.increase_threshold && self.current_parities < self.max_parities {
            self.stable_count += 1;
            if self.stable_count >= self.stable_threshold {
                self.current_parities += 1;
                self.stable_count = 0;
            }
        } else if loss < self.decrease_threshold && self.current_parities > self.min_parities {
            self.stable_count += 1;
            if self.stable_count >= self.stable_threshold {
                self.current_parities -= 1;
                self.stable_count = 0;
            }
        } else {
            // Reset stability counter if we're in the middle range
            self.stable_count = 0;
        }

        self.current_parities != old_parities
    }

    /// Update with explicit loss rate.
    pub fn update_with_loss(&mut self, loss_rate: f64) -> bool {
        let old_parities = self.current_parities;

        if loss_rate > self.increase_threshold && self.current_parities < self.max_parities {
            self.stable_count += 1;
            if self.stable_count >= self.stable_threshold {
                self.current_parities += 1;
                self.stable_count = 0;
            }
        } else if loss_rate < self.decrease_threshold && self.current_parities > self.min_parities {
            self.stable_count += 1;
            if self.stable_count >= self.stable_threshold {
                self.current_parities -= 1;
                self.stable_count = 0;
            }
        } else {
            self.stable_count = 0;
        }

        self.current_parities != old_parities
    }

    /// Force a specific parity level (e.g., for testing).
    pub fn set_parities(&mut self, parities: u8) {
        self.current_parities = parities.clamp(self.min_parities, self.max_parities);
        self.stable_count = 0;
    }

    /// Get recommended parities for a given loss rate (stateless).
    ///
    /// This can be used to get the "ideal" setting without affecting state.
    pub fn recommended_parities(&self, loss_rate: f64) -> u8 {
        // Simple tiered approach based on loss rate
        let recommended = if loss_rate < 0.05 {
            1
        } else if loss_rate < 0.15 {
            2
        } else if loss_rate < 0.30 {
            3
        } else {
            4
        };

        recommended.clamp(self.min_parities, self.max_parities)
    }

    /// Get current streaming parameters.
    ///
    /// Returns None if parameters are invalid.
    pub fn get_params(&self) -> Option<StreamingParams> {
        StreamingParams::with_step_size(
            self.delay,
            self.current_parities,
            self.step_size,
            self.symbol_bytes,
        )
        .ok()
    }

    /// Reset to initial state.
    pub fn reset(&mut self, initial_parities: u8) {
        self.current_parities = initial_parities.clamp(self.min_parities, self.max_parities);
        self.stable_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let fec = AdaptiveFec::new(8, 2, 4, 1200);
        assert_eq!(fec.current_parities(), 2);
        assert!((fec.overhead() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_increase_on_loss() {
        let mut fec = AdaptiveFec::new(8, 2, 4, 1200);
        fec.set_stable_threshold(1); // Immediate response for testing

        // Simulate high loss
        let changed = fec.update_with_loss(0.20);

        assert!(changed);
        assert_eq!(fec.current_parities(), 3);
    }

    #[test]
    fn test_decrease_on_good_conditions() {
        let mut fec = AdaptiveFec::new(8, 3, 4, 1200);
        fec.set_stable_threshold(1);

        // Simulate very low loss
        let changed = fec.update_with_loss(0.01);

        assert!(changed);
        assert_eq!(fec.current_parities(), 2);
    }

    #[test]
    fn test_respects_bounds() {
        let mut fec = AdaptiveFec::new(8, 4, 4, 1200);
        fec.set_stable_threshold(1);

        // Try to increase beyond max
        fec.update_with_loss(0.50);
        assert_eq!(fec.current_parities(), 4); // Should stay at max

        // Set to min and try to decrease
        fec.set_parities(1);
        fec.update_with_loss(0.01);
        assert_eq!(fec.current_parities(), 1); // Should stay at min
    }

    #[test]
    fn test_hysteresis() {
        let mut fec = AdaptiveFec::new(8, 2, 4, 1200);
        fec.set_stable_threshold(3);

        // First two high-loss readings shouldn't change parities
        assert!(!fec.update_with_loss(0.20));
        assert_eq!(fec.current_parities(), 2);

        assert!(!fec.update_with_loss(0.20));
        assert_eq!(fec.current_parities(), 2);

        // Third reading should trigger change
        assert!(fec.update_with_loss(0.20));
        assert_eq!(fec.current_parities(), 3);
    }

    #[test]
    fn test_recommended_parities() {
        let fec = AdaptiveFec::new(8, 2, 4, 1200);

        assert_eq!(fec.recommended_parities(0.02), 1);
        assert_eq!(fec.recommended_parities(0.10), 2);
        assert_eq!(fec.recommended_parities(0.25), 3);
        assert_eq!(fec.recommended_parities(0.50), 4);
    }

    #[test]
    fn test_get_params() {
        let fec = AdaptiveFec::new(8, 2, 4, 1200);
        let params = fec.get_params().unwrap();

        assert_eq!(params.delay(), 8);
        assert_eq!(params.num_parities(), 2);
        assert_eq!(params.step_size(), 4);
        assert_eq!(params.symbol_bytes(), 1200);
    }
}
