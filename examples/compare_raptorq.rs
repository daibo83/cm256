//! Comparison benchmark: CM256 vs RaptorQ vs Wirehair
//!
//! Run with: RUSTFLAGS="-C target-cpu=native" cargo run --release --example compare_raptorq

use std::time::Instant;

fn main() {
    // Test parameters
    let original_count = 100;
    let recovery_count = 30;
    let block_bytes = 1296;
    let data_size = original_count * block_bytes;
    
    println!("==============================================");
    println!("Erasure Code Comparison: CM256 vs RaptorQ vs Wirehair");
    println!("Data size: {} bytes, Block size: {} bytes", data_size, block_bytes);
    println!("==============================================");
    println!();
    
    // Benchmark CM256
    benchmark_cm256(original_count, recovery_count, block_bytes);
    
    // Benchmark RaptorQ
    benchmark_raptorq(data_size, block_bytes as u16, recovery_count);
    
    // Benchmark Wirehair
    benchmark_wirehair(data_size, block_bytes, recovery_count);
}

fn benchmark_cm256(original_count: usize, recovery_count: usize, block_bytes: usize) {
    use cm256::{Block, Params, encode};
    
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
    let encode_throughput = (data_size * iterations) as f64 / encode_time.as_secs_f64() / 1_000_000.0;
    
    println!("CM256 (Cauchy Reed-Solomon)");
    println!("  k={}, m={}, {} bytes/block", original_count, recovery_count, block_bytes);
    println!("  Encode: {:.2} MB/s", encode_throughput);
    println!();
}

fn benchmark_raptorq(data_size: usize, symbol_size: u16, repair_count: usize) {
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
    let encode_throughput = (data_size * iterations) as f64 / encode_time.as_secs_f64() / 1_000_000.0;
    
    let num_source = (data_size + symbol_size as usize - 1) / symbol_size as usize;
    println!("RaptorQ (RFC 6330 Fountain Code)");
    println!("  {} source symbols, {} bytes/symbol", num_source, symbol_size);
    println!("  Encode: {:.2} MB/s", encode_throughput);
    println!();
}

fn benchmark_wirehair(data_size: usize, block_bytes: usize, recovery_count: usize) {
    use wirehair_wrapper::wirehair::{WirehairEncoder, wirehair_init};
    
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
    let encode_throughput = (data_size * iterations) as f64 / encode_time.as_secs_f64() / 1_000_000.0;
    
    let num_blocks = (data_size + block_bytes - 1) / block_bytes;
    println!("Wirehair (Fountain Code)");
    println!("  {} blocks, {} bytes/block", num_blocks, block_bytes);
    println!("  Encode: {:.2} MB/s", encode_throughput);
    println!();
}
