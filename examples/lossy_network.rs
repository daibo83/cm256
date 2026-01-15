//! Demonstrate CM256: Shuffle all blocks (original + recovery), send only what's needed, recover

use cm256::{decode, encode, Block, BlockMut, Params};
use rand::seq::SliceRandom;

fn test_shuffled_recovery(
    original_count: usize,
    recovery_count: usize,
    block_bytes: usize,
    message: &str,
) {
    println!(
        "\n--- Test: {} original + {} recovery blocks ({} bytes each) ---",
        original_count, recovery_count, block_bytes
    );

    let params = Params::new(original_count, recovery_count, block_bytes).unwrap();
    let total_blocks = original_count + recovery_count;

    // Create original data blocks from message
    let message_bytes = message.as_bytes();
    let orig_data: Vec<Vec<u8>> = (0..original_count)
        .map(|i| {
            let start = (i * block_bytes) % message_bytes.len();
            let end = std::cmp::min(start + block_bytes, message_bytes.len());
            let mut block = vec![0u8; block_bytes];
            let copy_len = end - start;
            block[..copy_len].copy_from_slice(&message_bytes[start..end]);
            block
        })
        .collect();

    // Generate recovery blocks
    let mut recovery_data = vec![0u8; recovery_count * block_bytes];
    let blocks: Vec<Block> = orig_data
        .iter()
        .enumerate()
        .map(|(i, data)| Block::new(i as u8, data))
        .collect();
    encode(&params, &blocks, &mut recovery_data).unwrap();

    // Combine all blocks (original + recovery) into one pool
    let mut all_blocks: Vec<(u8, Vec<u8>)> = Vec::new();

    // Add original blocks
    for i in 0..original_count {
        all_blocks.push((i as u8, orig_data[i].clone()));
    }

    // Add recovery blocks
    for i in 0..recovery_count {
        let start = i * block_bytes;
        let end = start + block_bytes;
        all_blocks.push((
            (original_count + i) as u8,
            recovery_data[start..end].to_vec(),
        ));
    }

    println!(
        "Total blocks available: {} (mixed original + recovery)",
        total_blocks
    );
    println!(
        "Original payload size: {} bytes ({} blocks × {} bytes)",
        original_count * block_bytes,
        original_count,
        block_bytes
    );

    // Test sending different numbers of blocks
    let test_counts = [
        original_count - 2,
        original_count - 1,
        original_count,
        original_count + 2,
        original_count + 5,
    ];

    for &count_to_send in &test_counts {
        if count_to_send > total_blocks {
            continue;
        }

        // Shuffle all blocks randomly
        let mut rng = rand::thread_rng();
        let mut shuffled = all_blocks.clone();
        shuffled.shuffle(&mut rng);

        // Take only as many as we want to send
        let mut selected: Vec<(u8, Vec<u8>)> = shuffled.into_iter().take(count_to_send).collect();

        // Separate by block type for reporting
        let mut orig_received = 0;
        let mut rec_received = 0;
        let mut block_indices = Vec::new();

        for (index, _) in &selected {
            block_indices.push(*index);
            if *index < original_count as u8 {
                orig_received += 1;
            } else {
                rec_received += 1;
            }
        }

        block_indices.sort();

        println!(
            "\nSend {} blocks: {} original + {} recovery",
            count_to_send, orig_received, rec_received
        );
        println!("  Block indices: {:?}", block_indices);

        // Decode requires EXACTLY original_count blocks
        // If we have more, truncate to exactly original_count
        let selected_len = selected.len();
        let blocks_for_decode = selected_len.min(original_count);

        // Create mutable blocks for decoding
        let mut decode_blocks = Vec::new();
        for (index, data) in selected.iter_mut().take(blocks_for_decode) {
            decode_blocks.push(BlockMut::new(*index, data));
        }

        if selected_len != original_count {
            if selected_len < original_count {
                println!(
                    "  Note: Using all {} blocks (need {} for recovery)",
                    selected_len, original_count
                );
            } else {
                println!(
                    "  Note: Using first {} of {} blocks (decode needs exactly {})",
                    blocks_for_decode, selected_len, original_count
                );
            }
        }

        // Attempt to decode
        match decode(&params, &mut decode_blocks) {
            Ok(decoded_indices) => {
                println!("  ✓ Successfully recovered!");
                println!(
                    "    Decoded {} missing original blocks",
                    decoded_indices.len()
                );

                // Verify recovered data matches original
                let mut all_match = true;
                for block in &decode_blocks {
                    if block.index < original_count as u8 {
                        let recovered = &*block.data;
                        let original = &orig_data[block.index as usize];
                        if recovered != original.as_slice() {
                            all_match = false;
                            println!("    ✗ Block {} mismatch!", block.index);
                            break;
                        }
                    }
                }

                if all_match {
                    println!("    ✓ All blocks match original data!");
                }
            }
            Err(e) => {
                println!("  ✗ Failed to recover: {}", e);
                if count_to_send < original_count {
                    println!(
                        "    Reason: Not enough blocks (need {}, have {})",
                        original_count, count_to_send
                    );
                }
            }
        }
    }
}

fn main() {
    println!("=== CM256: Shuffled Block Recovery Demo ===");
    println!("Demonstrates that CM256 can recover from any N blocks,");
    println!("regardless of whether they are original or recovery blocks.\n");

    let message = "Hello, this is a test message sent over a lossy network! CM256 erasure coding helps recover lost packets.";
    let block_bytes = 16;

    println!("Message: \"{}\"", message);
    println!("Block size: {} bytes\n", block_bytes);

    test_shuffled_recovery(10, 5, block_bytes, message);
    test_shuffled_recovery(20, 10, block_bytes, message);
    test_shuffled_recovery(5, 5, block_bytes, message);

    println!("\n=== Key Insight ===");
    println!("CM256 can recover original payload using ANY N blocks,");
    println!("where N = original_count.");
    println!("The decoder doesn't care which blocks it receives,");
    println!("as long as it gets exactly original_count blocks!");
}
