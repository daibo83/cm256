//! Comparison benchmark: CM256 vs RaptorQ vs Wirehair
//!
//! Native targets (with Wirehair):
//!   RUSTFLAGS="-C target-cpu=native" cargo run --release --example compare_raptorq --features compare
//!
//! WASM targets (CM256 + RaptorQ - Wirehair requires SSE4.1):
//!   cargo run --release --example compare_raptorq --target wasm32-wasip1

use std::time::Instant;

fn main() {
    run_benchmarks();
}

#[cfg(feature = "compare")]
fn run_benchmarks() {
    // Test configurations: (k, m) pairs
    let test_configs = [
        (10, 2),
        (20, 4),
        (50, 10),
        (100, 20),
        (100, 30), // Original configuration for backward compatibility
    ];
    let block_bytes = 1296;

    println!("==============================================");
    println!("Erasure Code Comparison: CM256 vs RaptorQ vs Wirehair");
    println!("Block size: {} bytes", block_bytes);
    println!("==============================================");
    println!();

    // Print table header
    println!(
        "{:<12} {:>12} {:>12} {:>12}",
        "Config (k,m)", "CM256", "RaptorQ", "Wirehair"
    );
    println!(
        "{:<12} {:>12} {:>12} {:>12}",
        "", "(MB/s)", "(MB/s)", "(MB/s)"
    );
    println!("{}", "-".repeat(52));

    // Run benchmarks for each configuration
    for (original_count, recovery_count) in test_configs {
        let data_size = original_count * block_bytes;

        // Benchmark CM256
        let cm256_speed = benchmark_cm256(original_count, recovery_count, block_bytes);

        // Benchmark RaptorQ
        let raptorq_speed = benchmark_raptorq(data_size, block_bytes as u16, recovery_count);

        // Benchmark Wirehair
        let wirehair_speed = benchmark_wirehair(data_size, block_bytes, recovery_count);

        // Print results in table row
        println!(
            "k={:<3}, m={:<3} {:>12.2} {:>12.2} {:>12.2}",
            original_count, recovery_count, cm256_speed, raptorq_speed, wirehair_speed
        );
    }

    println!();
    println!("=== Summary ===");
    println!("CM256: O(k²) decode, O(k×m) encode - systematic Reed-Solomon");
    println!("RaptorQ: O(n) encode/decode - fountain code (unlimited recovery)");
    println!("Wirehair: O(n) encode/decode - fountain code (fast)");
}

#[cfg(not(feature = "compare"))]
fn run_benchmarks() {
    // Test configurations: (k, m) pairs
    let test_configs = [
        (10, 2),
        (20, 4),
        (50, 10),
        (100, 20),
        (100, 30), // Original configuration for backward compatibility
    ];
    let block_bytes = 1296;

    println!("==============================================");
    println!("Erasure Code Comparison: CM256 vs RaptorQ");
    println!("Block size: {} bytes", block_bytes);
    println!("Note: Wirehair requires SSE4.1 and is not included");
    println!("       (use --features compare on native targets)");
    println!("==============================================");
    println!();

    // Print table header
    println!("{:<12} {:>12} {:>12}", "Config (k,m)", "CM256", "RaptorQ");
    println!("{:<12} {:>12} {:>12}", "", "(MB/s)", "(MB/s)");
    println!("{}", "-".repeat(40));

    // Run benchmarks for each configuration
    for (original_count, recovery_count) in test_configs {
        let data_size = original_count * block_bytes;

        // Benchmark CM256
        let cm256_speed = benchmark_cm256(original_count, recovery_count, block_bytes);

        // Benchmark RaptorQ
        let raptorq_speed = benchmark_raptorq(data_size, block_bytes as u16, recovery_count);

        // Print results in table row
        println!(
            "k={:<3}, m={:<3} {:>12.2} {:>12.2}",
            original_count, recovery_count, cm256_speed, raptorq_speed
        );
    }

    println!();
    println!("=== Summary ===");
    println!("CM256: O(k²) decode, O(k×m) encode - systematic Reed-Solomon");
    println!("RaptorQ: O(n) encode/decode - fountain code (unlimited recovery)");
}

fn benchmark_cm256(original_count: usize, recovery_count: usize, block_bytes: usize) -> f64 {
    use cm256::{encode, Block, Params};

    let params = Params::new(original_count, recovery_count, block_bytes).unwrap();
    let data_size = original_count * block_bytes;

    // Create original data
    let orig_data: Vec<Vec<u8>> = (0..original_count)
        .map(|i| {
            (0..block_bytes)
                .map(|j| ((i * block_bytes + j) % 256) as u8)
                .collect()
        })
        .collect();

    let mut recovery_data = vec![0u8; recovery_count * block_bytes];

    // Warm up
    let blocks: Vec<Block> = orig_data
        .iter()
        .enumerate()
        .map(|(i, data)| Block::new(i as u8, data))
        .collect();
    encode(&params, &blocks, &mut recovery_data).unwrap();

    // Benchmark encode
    let iterations = 1000;
    let start = Instant::now();
    for _ in 0..iterations {
        let blocks: Vec<Block> = orig_data
            .iter()
            .enumerate()
            .map(|(i, data)| Block::new(i as u8, data))
            .collect();
        encode(&params, &blocks, &mut recovery_data).unwrap();
    }
    let encode_time = start.elapsed();
    let encode_throughput =
        (data_size * iterations) as f64 / encode_time.as_secs_f64() / 1_000_000.0;

    encode_throughput
}

fn benchmark_raptorq(data_size: usize, symbol_size: u16, repair_count: usize) -> f64 {
    use raptorq::{Encoder, ObjectTransmissionInformation};

    // Create test data
    let data: Vec<u8> = (0..data_size).map(|i| (i % 256) as u8).collect();

    // RaptorQ config
    let config = ObjectTransmissionInformation::new(data_size as u64, symbol_size, 1, 1, 8);

    // Warm up
    let encoder = Encoder::new(&data, config);
    let _packets: Vec<_> = encoder.get_encoded_packets(repair_count as u32);

    // Benchmark encode
    let iterations = 1000;
    let start = Instant::now();
    for _ in 0..iterations {
        let encoder = Encoder::new(&data, config);
        let _packets: Vec<_> = encoder.get_encoded_packets(repair_count as u32);
    }
    let encode_time = start.elapsed();
    let encode_throughput =
        (data_size * iterations) as f64 / encode_time.as_secs_f64() / 1_000_000.0;

    encode_throughput
}

#[cfg(feature = "compare")]
fn benchmark_wirehair(data_size: usize, block_bytes: usize, recovery_count: usize) -> f64 {
    use wirehair_wrapper::wirehair::{wirehair_init, WirehairEncoder};

    // Initialize wirehair library
    wirehair_init().expect("Failed to init wirehair");

    // Create test data
    let data: Vec<u8> = (0..data_size).map(|i| (i % 256) as u8).collect();

    // Warm up
    let encoder = WirehairEncoder::new(&data, data_size as u64, block_bytes as u32);
    let mut block = vec![0u8; block_bytes];
    let mut out_bytes: u32 = 0;
    for i in 0..recovery_count as u64 {
        let _ = encoder.encode(i, &mut block, block_bytes as u32, &mut out_bytes);
    }

    // Benchmark encode
    let iterations = 1000;
    let start = Instant::now();
    for _ in 0..iterations {
        let encoder = WirehairEncoder::new(&data, data_size as u64, block_bytes as u32);
        for i in 0..recovery_count as u64 {
            let _ = encoder.encode(i, &mut block, block_bytes as u32, &mut out_bytes);
        }
    }
    let encode_time = start.elapsed();
    let encode_throughput =
        (data_size * iterations) as f64 / encode_time.as_secs_f64() / 1_000_000.0;

    encode_throughput
}
