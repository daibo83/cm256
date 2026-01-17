//! Streaming FEC Demonstration
//!
//! This example demonstrates the diagonal interleaving streaming FEC system
//! for low-latency real-time applications like video/audio streaming.
//!
//! Run with: cargo run --release --example streaming_demo

use cm256::streaming::{StreamingDecoder, StreamingEncoder, StreamingParams};
use rand::Rng;

fn main() {
    println!("=================================================");
    println!("  Diagonal Interleaving Streaming FEC Demo");
    println!("=================================================\n");

    // Configuration
    let delay = 8; // Each parity covers 8 source symbols
    let num_parities = 2; // 2 parities per window (can recover burst of 2)
    let symbol_bytes = 1200; // MTU-sized packets
    let total_symbols = 100;
    let loss_rate = 0.15; // 15% packet loss

    let params =
        StreamingParams::new(delay, num_parities, symbol_bytes).expect("Invalid parameters");

    println!("Configuration:");
    println!("  Delay (window size): {} symbols", params.delay());
    println!("  Parities per window: {}", params.num_parities());
    println!("  Symbol size: {} bytes", params.symbol_bytes());
    println!("  Max burst recovery: {} symbols", params.max_burst());
    println!("  Overhead: {:.1}%", params.overhead() * 100.0);
    println!("  Total symbols to send: {}", total_symbols);
    println!("  Simulated loss rate: {:.1}%", loss_rate * 100.0);

    println!("\n--- Encoding and Transmission ---\n");

    let mut encoder = StreamingEncoder::new(params);
    let mut decoder = StreamingDecoder::new(params);
    // Keep enough history for this demo (in real use, you'd consume data as it arrives)
    decoder.set_history_windows((total_symbols / delay as usize + 1) as u16);
    let mut rng = rand::thread_rng();

    // Statistics
    let mut sources_sent = 0;
    let mut sources_lost = 0;
    let mut parities_sent = 0;
    let mut parities_lost = 0;

    // Store original data for verification
    let originals: Vec<Vec<u8>> = (0..total_symbols)
        .map(|i| {
            // Create identifiable data
            let mut data = vec![0u8; symbol_bytes];
            data[0] = (i >> 8) as u8;
            data[1] = (i & 0xFF) as u8;
            for j in 2..symbol_bytes {
                data[j] = ((i + j) % 256) as u8;
            }
            data
        })
        .collect();

    // Simulate encoding and lossy transmission
    for (i, data) in originals.iter().enumerate() {
        let result = encoder.add_source(data);
        sources_sent += 1;

        // Simulate source packet loss
        if rng.gen::<f32>() > loss_rate {
            decoder.add_source(result.source_seq, data);
        } else {
            sources_lost += 1;
            if i < 20 || i >= total_symbols - 5 {
                println!("  ✗ Source {} lost", result.source_seq);
            }
        }

        // Send parities (with lower loss rate - often sent via different path)
        for parity in &result.parities {
            parities_sent += 1;
            if rng.gen::<f32>() > loss_rate * 0.5 {
                decoder.add_parity(parity.end_seq, parity.parity_index, &parity.data);
            } else {
                parities_lost += 1;
            }
        }

        // Periodically attempt recovery
        if i % 10 == 9 {
            let recovered = decoder.try_recover();
            if !recovered.is_empty() {
                for (seq, _) in &recovered {
                    if i < 25 || i >= total_symbols - 5 {
                        println!("  ✓ Recovered source {}", seq);
                    }
                }
            }
        }
    }

    // Final recovery attempt
    let final_recovered = decoder.try_recover();
    for (seq, _) in &final_recovered {
        println!("  ✓ Recovered source {} (final)", seq);
    }

    println!("\n--- Results ---\n");

    // Count available symbols
    let mut available = 0;
    let mut verified = 0;
    for i in 0..total_symbols {
        if let Some(data) = decoder.get_source(i as u16) {
            available += 1;
            if data == &originals[i][..] {
                verified += 1;
            } else {
                println!("  ✗ Data mismatch at symbol {}", i);
            }
        }
    }

    println!("Transmission Statistics:");
    println!("  Sources sent: {}", sources_sent);
    println!(
        "  Sources lost: {} ({:.1}%)",
        sources_lost,
        100.0 * sources_lost as f32 / sources_sent as f32
    );
    println!("  Parities sent: {}", parities_sent);
    println!(
        "  Parities lost: {} ({:.1}%)",
        parities_lost,
        if parities_sent > 0 {
            100.0 * parities_lost as f32 / parities_sent as f32
        } else {
            0.0
        }
    );

    println!("\nRecovery Statistics:");
    println!("  Total symbols expected: {}", total_symbols);
    println!("  Symbols available: {}", available);
    println!("  Symbols verified correct: {}", verified);
    println!(
        "  Recovery rate: {:.1}%",
        100.0 * available as f32 / total_symbols as f32
    );

    let recovered_count = available as i32 - (sources_sent - sources_lost) as i32;
    if sources_lost > 0 {
        println!(
            "  Lost symbols recovered: {}/{} ({:.1}%)",
            recovered_count.max(0),
            sources_lost,
            if sources_lost > 0 {
                100.0 * recovered_count.max(0) as f32 / sources_lost as f32
            } else {
                0.0
            }
        );
    }

    println!("\n--- How It Works ---\n");
    println!("Diagonal Interleaving creates overlapping protection windows:");
    println!();
    println!("  Time:     0   1   2   3   4   5   6   7   8   9  10  11  ...");
    println!("  Source:   S₀  S₁  S₂  S₃  S₄  S₅  S₆  S₇  S₈  S₉  S₁₀ S₁₁ ...");
    println!("            ╲   ╲   ╲   ╲   ╲   ╲   ╲   ╲");
    println!("  Parity:               P₇              P₈              P₉  ...");
    println!("                        │               │               │");
    println!("                        └─ covers ──────┴─ covers ──────┘");
    println!("                          S₀..S₇           S₁..S₈");
    println!();
    println!("Key advantages over block FEC:");
    println!("  1. Lower latency: Don't wait for full block before sending");
    println!("  2. Better burst recovery: Overlapping windows catch bursts");
    println!("  3. Continuous protection: Every symbol is covered by multiple parities");
    println!(
        "  4. Configurable: delay={} gives ~{:.0}ms latency at 30fps",
        delay,
        delay as f32 * 33.3
    );

    println!("\n=================================================");
}
