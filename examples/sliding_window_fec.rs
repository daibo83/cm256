//! Sliding Window FEC Implementation
//!
//! This example demonstrates a sliding window Forward Error Correction (FEC) system
//! using CM256 for ultra-low latency reliable transport.
//!
//! Configuration:
//! - window_size: 15 data packets per encoding
//! - step_size: 5 packets to slide before re-encoding (67% overlap)
//! - redundancy: 3 recovery packets per window
//! - block_bytes: 1200 bytes per packet

use cm256::{decode, encode, Block, BlockMut, Params};
use std::collections::HashMap;

/// Configuration for the sliding window FEC system
#[derive(Debug, Clone)]
struct SlidingWindowConfig {
    window_size: usize,
    step_size: usize,
    redundancy: usize,
    block_bytes: usize,
}

impl SlidingWindowConfig {
    fn new(window_size: usize, step_size: usize, redundancy: usize, block_bytes: usize) -> Self {
        assert!(window_size > 0, "Window size must be positive");
        assert!(
            step_size > 0 && step_size <= window_size,
            "Step size must be in range (1, window_size]"
        );
        assert!(redundancy > 0, "Redundancy must be positive");
        assert!(block_bytes > 0, "Block bytes must be positive");

        Self {
            window_size,
            step_size,
            redundancy,
            block_bytes,
        }
    }

    /// Calculate the overlap percentage between consecutive windows
    fn overlap_percentage(&self) -> f32 {
        (1.0 - (self.step_size as f32 / self.window_size as f32)) * 100.0
    }
}

/// Represents a single data packet with sequence number and payload
#[derive(Debug, Clone)]
struct DataPacket {
    sequence: u32,
    payload: Vec<u8>,
}

impl DataPacket {
    fn new(sequence: u32, payload: Vec<u8>) -> Self {
        Self { sequence, payload }
    }
}

/// Represents a FEC (recovery) packet
#[derive(Debug, Clone)]
struct FecPacket {
    sequence: u32,       // Unique sequence for this FEC packet
    window_start: u32,   // Starting sequence of the window this FEC covers
    window_end: u32,     // Ending sequence of the window this FEC covers
    index_in_window: u8, // Index of this FEC packet in the recovery set (0..redundancy-1)
    payload: Vec<u8>,    // FEC data
}

impl FecPacket {
    fn new(
        sequence: u32,
        window_start: u32,
        window_end: u32,
        index_in_window: u8,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            sequence,
            window_start,
            window_end,
            index_in_window,
            payload,
        }
    }
}

/// Sliding window FEC encoder
struct SlidingWindowEncoder {
    config: SlidingWindowConfig,
    params: cm256::Params,
    window: Vec<DataPacket>,
    next_sequence: u32,
    next_fec_sequence: u32,
    packets_since_last_encode: usize,
}

impl SlidingWindowEncoder {
    fn new(config: SlidingWindowConfig) -> Self {
        let params = Params::new(config.window_size, config.redundancy, config.block_bytes)
            .expect("Invalid parameters");

        let window_size = config.window_size;

        Self {
            config: config.clone(),
            params,
            window: Vec::with_capacity(window_size),
            next_sequence: 0,
            next_fec_sequence: 0,
            packets_since_last_encode: 0,
        }
    }

    /// Add a new data packet and potentially generate FEC packets
    fn add_packet(&mut self, payload: Vec<u8>) -> Option<Vec<FecPacket>> {
        // Pad payload to block_bytes if necessary
        let mut padded_payload = payload;
        if padded_payload.len() < self.config.block_bytes {
            padded_payload.resize(self.config.block_bytes, 0);
        }

        // Create and add the data packet
        let packet = DataPacket::new(self.next_sequence, padded_payload);
        self.window.push(packet);
        self.next_sequence += 1;
        self.packets_since_last_encode += 1;

        // Check if we should generate FEC
        if self.window.len() >= self.config.window_size
            && self.packets_since_last_encode >= self.config.step_size
        {
            Some(self.generate_fec())
        } else {
            None
        }
    }

    /// Generate FEC packets for the current window
    fn generate_fec(&mut self) -> Vec<FecPacket> {
        println!("\n=== Generating FEC for window ===");
        println!(
            "  Window packets: [{}..{}]",
            self.window[0].sequence,
            self.window.last().unwrap().sequence
        );

        // Prepare blocks for encoding
        let blocks: Vec<Block> = self
            .window
            .iter()
            .enumerate()
            .map(|(i, packet)| Block::new(i as u8, &packet.payload))
            .collect();

        // Generate recovery data
        let mut recovery_data = vec![0u8; self.config.redundancy * self.config.block_bytes];
        encode(&self.params, &blocks, &mut recovery_data).expect("Encoding failed");

        // Create FEC packets
        let window_start = self.window[0].sequence;
        let window_end = self.window.last().unwrap().sequence;
        let mut fec_packets = Vec::new();

        for i in 0..self.config.redundancy {
            let start = i * self.config.block_bytes;
            let end = start + self.config.block_bytes;
            let payload = recovery_data[start..end].to_vec();

            let fec_packet = FecPacket::new(
                self.next_fec_sequence,
                window_start,
                window_end,
                i as u8,
                payload,
            );

            println!(
                "  FEC packet #{}: seq={}, covers window [{}..{}], index={}",
                i, self.next_fec_sequence, window_start, window_end, i
            );

            fec_packets.push(fec_packet);
            self.next_fec_sequence += 1;
        }

        // Slide the window forward
        self.slide_window();

        fec_packets
    }

    /// Slide the window forward by step_size
    fn slide_window(&mut self) {
        let packets_to_remove = self.config.step_size.min(self.window.len());

        println!("  Sliding window forward by {} packets", packets_to_remove);
        println!(
            "  Removing packets: {:?}",
            self.window
                .iter()
                .take(packets_to_remove)
                .map(|p| p.sequence)
                .collect::<Vec<_>>()
        );

        for _ in 0..packets_to_remove {
            self.window.remove(0);
        }

        self.packets_since_last_encode = 0;

        if !self.window.is_empty() {
            println!(
                "  New window: [{}..{}]",
                self.window[0].sequence,
                self.window.last().unwrap().sequence
            );
        }
    }

    /// Force generation of FEC even if step size hasn't been reached
    fn flush(&mut self) -> Option<Vec<FecPacket>> {
        if self.window.len() >= self.params.original_count() {
            self.packets_since_last_encode = self.config.step_size; // Force encode
            Some(self.generate_fec())
        } else {
            None
        }
    }
}

/// Sliding window FEC decoder
struct SlidingWindowDecoder {
    config: SlidingWindowConfig,
    params: cm256::Params,
    data_packets: HashMap<u32, Vec<u8>>,
    fec_packets: Vec<FecPacket>,
    recovered_packets: HashMap<u32, Vec<u8>>,
}

impl SlidingWindowDecoder {
    fn new(config: SlidingWindowConfig) -> Self {
        let params = Params::new(config.window_size, config.redundancy, config.block_bytes)
            .expect("Invalid parameters");

        Self {
            config,
            params,
            data_packets: HashMap::new(),
            fec_packets: Vec::new(),
            recovered_packets: HashMap::new(),
        }
    }

    /// Process a received data packet
    fn process_data_packet(&mut self, packet: DataPacket) {
        println!("  Received data packet: sequence={}", packet.sequence);
        self.data_packets.insert(packet.sequence, packet.payload);
        self.cleanup_old_fec_packets();
    }

    /// Process a received FEC packet
    fn process_fec_packet(&mut self, fec_packet: FecPacket) {
        println!(
            "  Received FEC packet: seq={}, covers [{}..{}], index={}",
            fec_packet.sequence,
            fec_packet.window_start,
            fec_packet.window_end,
            fec_packet.index_in_window
        );
        self.fec_packets.push(fec_packet);
    }

    /// Attempt to recover missing data packets using available FEC
    fn recover_missing_packets(&mut self) {
        if self.data_packets.is_empty() {
            return;
        }

        // Find gaps in received data packets
        let sequences: Vec<u32> = {
            let mut seqs: Vec<_> = self.data_packets.keys().cloned().collect();
            seqs.sort();
            seqs
        };

        let min_seq = sequences[0];
        let max_seq = sequences[sequences.len() - 1];

        // Find all missing sequences in range
        let missing: Vec<u32> = (min_seq..=max_seq)
            .filter(|seq| {
                !self.data_packets.contains_key(seq) && !self.recovered_packets.contains_key(seq)
            })
            .collect();

        if missing.is_empty() {
            return;
        }

        println!(
            "\n  Attempting to recover {} missing packets: {:?}",
            missing.len(),
            missing
        );

        // Group FEC packets by the window they cover
        let mut windows: std::collections::HashMap<(u32, u32), Vec<FecPacket>> =
            std::collections::HashMap::new();

        for fec_packet in &self.fec_packets {
            let key = (fec_packet.window_start, fec_packet.window_end);
            windows.entry(key).or_default().push(fec_packet.clone());
        }

        // Try each window that covers missing packets
        for ((window_start, window_end), fec_packets) in windows {
            // Check if this window covers any missing packets
            let covers_missing = missing
                .iter()
                .any(|seq| *seq >= window_start && *seq <= window_end);

            if !covers_missing {
                continue;
            }

            self.recover_window(window_start, window_end, &fec_packets);
        }
    }

    /// Recover packets for a specific window using multiple FEC packets
    fn recover_window(&mut self, window_start: u32, window_end: u32, fec_packets: &[FecPacket]) {
        let window_size = self.config.window_size;

        println!(
            "  Recovering window [{}, {}] with {} FEC packets",
            window_start,
            window_end,
            fec_packets.len()
        );

        // Collect all available blocks: data packets + FEC packets
        let mut all_blocks: Vec<Vec<u8>> = Vec::new();
        let mut all_indices: Vec<u8> = Vec::new();

        // Add data packets
        for seq in window_start..=window_end {
            if let Some(data) = self.data_packets.get(&seq) {
                let index = (seq - window_start) as u8;
                all_blocks.push(data.clone());
                all_indices.push(index);
            } else if let Some(data) = self.recovered_packets.get(&seq) {
                let index = (seq - window_start) as u8;
                all_blocks.push(data.clone());
                all_indices.push(index);
            }
        }

        // Add FEC packets
        for (i, fec) in fec_packets.iter().enumerate() {
            let index = window_size as u8 + i as u8;
            all_blocks.push(fec.payload.clone());
            all_indices.push(index);
        }

        // We need exactly window_size blocks for decoding
        if all_blocks.len() < window_size {
            println!(
                "    ✗ Not enough blocks: have {} (data + FEC), need {}",
                all_blocks.len(),
                window_size
            );
            return;
        }

        println!(
            "    Decoding window [{}, {}] with {} blocks...",
            window_start,
            window_end,
            all_blocks.len()
        );

        // Create decode blocks - we use ALL available blocks (data + FEC)
        // The decode function will pick the right ones
        let mut decode_blocks: Vec<BlockMut> = Vec::new();
        for (i, data) in all_blocks.iter_mut().enumerate() {
            decode_blocks.push(BlockMut::new(all_indices[i], data));
        }

        // Attempt decode
        match decode(&self.params, &mut decode_blocks) {
            Ok(decoded_indices) => {
                println!(
                    "    ✓ Successfully decoded! Recovered {} missing indices: {:?}",
                    decoded_indices.len(),
                    decoded_indices
                );
                // CM256 decode returns to indices that were recovered
                // The recovered data is now in the FEC blocks (indices >= window_size)
                // decoded_indices[i] corresponds to data now in decode_blocks at index (window_size + i)

                let mut newly_recovered = 0;
                let fec_start_index = window_size as u8;

                for (i, recovered_idx) in decoded_indices.iter().enumerate() {
                    if *recovered_idx < fec_start_index {
                        let seq = window_start + *recovered_idx as u32;

                        // Only count as recovered if we didn't already have it
                        if !self.data_packets.contains_key(&seq)
                            && !self.recovered_packets.contains_key(&seq)
                        {
                            // Find FEC block that now contains this recovered data
                            let fec_block_index = fec_start_index + i as u8;
                            if let Some(fec_block) =
                                decode_blocks.iter().find(|b| b.index == fec_block_index)
                            {
                                self.recovered_packets.insert(seq, fec_block.data.to_vec());
                                newly_recovered += 1;
                            }
                        }
                    }
                }

                if newly_recovered > 0 {
                    println!(
                        "    ✓ Recovered {} packets: {:?}",
                        newly_recovered, decoded_indices
                    );
                }
            }
            Err(e) => {
                println!("    ✗ Decode failed: {}", e);
            }
        }
    }

    /// Clean up old FEC packets that are no longer useful
    fn cleanup_old_fec_packets(&mut self) {
        if let Some(&max_data_seq) = self.data_packets.keys().max() {
            // Remove FEC packets whose window is entirely before the oldest data packet
            let oldest_useful = max_data_seq.saturating_sub(self.config.window_size as u32 * 2);
            self.fec_packets
                .retain(|fec| fec.window_end >= oldest_useful);
        }
    }

    /// Get all available packets
    fn get_all_packets(&self) -> Vec<(u32, Vec<u8>)> {
        let mut all: Vec<_> = self
            .data_packets
            .iter()
            .chain(self.recovered_packets.iter())
            .map(|(seq, data)| (*seq, data.clone()))
            .collect();
        all.sort_by_key(|(seq, _)| *seq);
        all
    }
}

/// Simulate network transmission with packet loss
fn simulate_transmission(
    encoder: &mut SlidingWindowEncoder,
    decoder: &mut SlidingWindowDecoder,
    total_packets: usize,
    loss_rate: f32,
    config: &SlidingWindowConfig,
) {
    println!("\n=== Simulating Transmission ===");
    println!("  Total packets: {}", total_packets);
    println!("  Loss rate: {:.1}%", loss_rate * 100.0);

    use rand::Rng;
    let mut rng = rand::thread_rng();

    for i in 0..total_packets {
        // Create a data packet with some identifiable content
        let mut payload =
            format!("Data packet {} - Test payload for FEC demonstration", i).into_bytes();
        // Pad to block_bytes to match what encoder expects
        if payload.len() < encoder.config.block_bytes {
            payload.resize(encoder.config.block_bytes, 0);
        }

        // Add packet to encoder and generate FEC if needed
        if let Some(fec_packets) = encoder.add_packet(payload) {
            // Send FEC packets
            for fec in fec_packets {
                // Simulate loss of FEC packets (less likely than data)
                if rng.gen::<f32>() > loss_rate * 0.5 {
                    decoder.process_fec_packet(fec);
                } else {
                    println!("  ✗ FEC packet {} lost (simulated)", fec.sequence);
                }
            }
        }

        // Simulate loss of data packets
        if rng.gen::<f32>() > loss_rate {
            let mut payload =
                format!("Data packet {} - Test payload for FEC demonstration", i).into_bytes();
            // Pad to block_bytes
            if payload.len() < encoder.config.block_bytes {
                payload.resize(encoder.config.block_bytes, 0);
            }
            let packet = DataPacket::new(i as u32, payload);
            decoder.process_data_packet(packet);
        } else {
            println!("  ✗ Data packet {} lost (simulated)", i);
        }

        // Attempt recovery every few packets and when we have FEC packets
        if i % 3 == 0 && !decoder.fec_packets.is_empty() {
            decoder.recover_missing_packets();
        }
    }

    // Flush any remaining FEC from encoder
    if let Some(fec_packets) = encoder.flush() {
        for fec in fec_packets {
            decoder.process_fec_packet(fec);
        }
    }

    // Final recovery attempt
    decoder.recover_missing_packets();
}

fn main() {
    println!("=================================================");
    println!("  Sliding Window FEC Demonstration");
    println!("=================================================");

    // Create configuration
    let config = SlidingWindowConfig {
        window_size: 15,
        step_size: 5,
        redundancy: 3,
        block_bytes: 1200,
    };

    println!("\nConfiguration:");
    println!("  Window size: {} packets", config.window_size);
    println!("  Step size: {} packets", config.step_size);
    println!("  Redundancy: {} FEC packets per window", config.redundancy);
    println!("  Block bytes: {} bytes", config.block_bytes);
    println!("  Overlap: {:.1}%", config.overlap_percentage());

    println!("\nExplanation:");
    println!(
        "  - Each window contains {} data packets",
        config.window_size
    );
    println!(
        "  - {} recovery packets are generated per window",
        config.redundancy
    );
    println!(
        "  - Window slides forward by {} packets at a time",
        config.step_size
    );
    println!(
        "  - {:.0}% overlap between consecutive windows",
        config.overlap_percentage()
    );
    println!("  - This overlap provides multiple recovery opportunities for burst loss");

    // Create encoder and decoder
    let mut encoder = SlidingWindowEncoder::new(config.clone());
    let mut decoder = SlidingWindowDecoder::new(config.clone());

    // Simulate transmission with loss
    simulate_transmission(&mut encoder, &mut decoder, 50, 0.15, &config);

    // Results
    println!("\n=== Results ===");
    let all_packets = decoder.get_all_packets();
    let total_expected = 50;
    let recovered = decoder.recovered_packets.len();
    let received = decoder.data_packets.len();
    let success_rate = (all_packets.len() as f32 / total_expected as f32) * 100.0;

    println!("  Total packets sent: {}", total_expected);
    println!("  Original packets received: {}", received);
    println!("  Packets recovered via FEC: {}", recovered);
    println!("  Unique packets available: {}", all_packets.len());
    println!("  Total packets available: {}", all_packets.len());
    println!("  Success rate: {:.1}%", success_rate);

    // Verify data integrity of recovered packets
    println!("\n=== Verification ===");
    let mut all_correct = true;
    let mut verified_count = 0;
    for (seq, data) in &all_packets {
        let expected =
            format!("Data packet {} - Test payload for FEC demonstration", seq).into_bytes();
        let mut padded_expected = expected;
        if padded_expected.len() < config.block_bytes {
            padded_expected.resize(config.block_bytes, 0);
        }
        if data == &padded_expected {
            verified_count += 1;
        } else {
            all_correct = false;
            println!("  ✗ Packet {} data mismatch!", seq);
        }
    }
    if all_correct {
        println!("  ✓ All verified packets have correct data!");
    }
    println!(
        "  Verified {}/{} packets",
        verified_count,
        all_packets.len()
    );

    if all_correct {
        println!("  ✓ All recovered packets have correct data!");
    }

    // Show which sequences were recovered
    if !decoder.recovered_packets.is_empty() {
        let recovered_seqs: Vec<_> = decoder.recovered_packets.keys().cloned().collect();
        println!("  Recovered sequences: {:?}", recovered_seqs);
    }

    // Show FEC packet statistics
    println!("\n=== FEC Statistics ===");
    println!(
        "  Total FEC packets generated: {}",
        encoder.next_fec_sequence
    );
    println!("  FEC packets received: {}", decoder.fec_packets.len());
    println!(
        "  FEC overhead: {:.1}%",
        ((encoder.next_fec_sequence as usize * encoder.config.redundancy) as f32
            / total_expected as f32)
            * 100.0
    );

    // Show window progression
    println!("\n=== Window Progression ===");
    println!(
        "  With window_size={}, step_size={}",
        config.window_size, config.step_size
    );
    println!("  Window 0 covers packets: [0-14]");
    println!("  Window 1 covers packets: [5-19]");
    println!("  Window 2 covers packets: [10-24]");
    println!("  Window 3 covers packets: [15-29]");
    println!("  ...");
    println!(
        "  Each packet is covered by approximately {:.0} different FEC groups",
        (config.window_size as f32 / config.step_size as f32)
    );

    println!("\n=== Key Insights ===");
    println!("1. Sliding window provides continuous FEC coverage");
    println!("2. 67% overlap means packets are protected by multiple FEC blocks");
    println!("3. Recovery is possible even when multiple packets are lost in a burst");
    println!("4. The system balances overhead, latency, and recovery capability");
    println!(
        "5. Each packet stays in the window for {} FEC generations",
        config.window_size / config.step_size
    );
    println!("6. This design is ideal for streaming applications where latency is critical");

    println!("\n=================================================");
}
