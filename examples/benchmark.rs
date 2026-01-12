//! Benchmarks for CM256 encoder/decoder

use cm256::{Block, BlockMut, Params, decode, encode};
use std::time::Instant;

fn main() {
    // Benchmark parameters - same as C++ unit test
    let block_bytes = 1296;
    let original_count = 100;
    let recovery_count = 30;
    let trials = 1000;

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

    // Benchmark encoding
    let encode_start = Instant::now();
    for _ in 0..trials {
        let blocks: Vec<Block> = orig_data
            .iter()
            .enumerate()
            .map(|(i, data)| Block::new(i as u8, data))
            .collect();
        encode(&params, &blocks, &mut recovery_data).unwrap();
    }
    let encode_elapsed = encode_start.elapsed();
    let encode_us = encode_elapsed.as_micros() as f64 / trials as f64;
    let encode_mbps = data_size as f64 / encode_us;

    // Save recovery data for decode benchmark
    let recovery_copy = recovery_data.clone();

    // Benchmark decoding
    let decode_start = Instant::now();
    for _ in 0..trials {
        // Create fresh copies of recovery blocks for this trial
        let mut rec0: Vec<u8> = recovery_copy[0 * block_bytes..1 * block_bytes].to_vec();
        let mut rec1: Vec<u8> = recovery_copy[1 * block_bytes..2 * block_bytes].to_vec();
        let mut rec2: Vec<u8> = recovery_copy[2 * block_bytes..3 * block_bytes].to_vec();
        let mut rec3: Vec<u8> = recovery_copy[3 * block_bytes..4 * block_bytes].to_vec();
        let mut rec4: Vec<u8> = recovery_copy[4 * block_bytes..5 * block_bytes].to_vec();
        let mut rec5: Vec<u8> = recovery_copy[5 * block_bytes..6 * block_bytes].to_vec();
        let mut rec6: Vec<u8> = recovery_copy[6 * block_bytes..7 * block_bytes].to_vec();
        let mut rec7: Vec<u8> = recovery_copy[7 * block_bytes..8 * block_bytes].to_vec();
        let mut rec8: Vec<u8> = recovery_copy[8 * block_bytes..9 * block_bytes].to_vec();
        let mut rec9: Vec<u8> = recovery_copy[9 * block_bytes..10 * block_bytes].to_vec();
        let mut rec10: Vec<u8> = recovery_copy[10 * block_bytes..11 * block_bytes].to_vec();
        let mut rec11: Vec<u8> = recovery_copy[11 * block_bytes..12 * block_bytes].to_vec();
        let mut rec12: Vec<u8> = recovery_copy[12 * block_bytes..13 * block_bytes].to_vec();
        let mut rec13: Vec<u8> = recovery_copy[13 * block_bytes..14 * block_bytes].to_vec();
        let mut rec14: Vec<u8> = recovery_copy[14 * block_bytes..15 * block_bytes].to_vec();
        let mut rec15: Vec<u8> = recovery_copy[15 * block_bytes..16 * block_bytes].to_vec();
        let mut rec16: Vec<u8> = recovery_copy[16 * block_bytes..17 * block_bytes].to_vec();
        let mut rec17: Vec<u8> = recovery_copy[17 * block_bytes..18 * block_bytes].to_vec();
        let mut rec18: Vec<u8> = recovery_copy[18 * block_bytes..19 * block_bytes].to_vec();
        let mut rec19: Vec<u8> = recovery_copy[19 * block_bytes..20 * block_bytes].to_vec();
        let mut rec20: Vec<u8> = recovery_copy[20 * block_bytes..21 * block_bytes].to_vec();
        let mut rec21: Vec<u8> = recovery_copy[21 * block_bytes..22 * block_bytes].to_vec();
        let mut rec22: Vec<u8> = recovery_copy[22 * block_bytes..23 * block_bytes].to_vec();
        let mut rec23: Vec<u8> = recovery_copy[23 * block_bytes..24 * block_bytes].to_vec();
        let mut rec24: Vec<u8> = recovery_copy[24 * block_bytes..25 * block_bytes].to_vec();
        let mut rec25: Vec<u8> = recovery_copy[25 * block_bytes..26 * block_bytes].to_vec();
        let mut rec26: Vec<u8> = recovery_copy[26 * block_bytes..27 * block_bytes].to_vec();
        let mut rec27: Vec<u8> = recovery_copy[27 * block_bytes..28 * block_bytes].to_vec();
        let mut rec28: Vec<u8> = recovery_copy[28 * block_bytes..29 * block_bytes].to_vec();
        let mut rec29: Vec<u8> = recovery_copy[29 * block_bytes..30 * block_bytes].to_vec();

        // Create copies of original blocks we'll keep
        let mut orig30 = orig_data[30].clone();
        let mut orig31 = orig_data[31].clone();
        let mut orig32 = orig_data[32].clone();
        let mut orig33 = orig_data[33].clone();
        let mut orig34 = orig_data[34].clone();
        let mut orig35 = orig_data[35].clone();
        let mut orig36 = orig_data[36].clone();
        let mut orig37 = orig_data[37].clone();
        let mut orig38 = orig_data[38].clone();
        let mut orig39 = orig_data[39].clone();
        let mut orig40 = orig_data[40].clone();
        let mut orig41 = orig_data[41].clone();
        let mut orig42 = orig_data[42].clone();
        let mut orig43 = orig_data[43].clone();
        let mut orig44 = orig_data[44].clone();
        let mut orig45 = orig_data[45].clone();
        let mut orig46 = orig_data[46].clone();
        let mut orig47 = orig_data[47].clone();
        let mut orig48 = orig_data[48].clone();
        let mut orig49 = orig_data[49].clone();
        let mut orig50 = orig_data[50].clone();
        let mut orig51 = orig_data[51].clone();
        let mut orig52 = orig_data[52].clone();
        let mut orig53 = orig_data[53].clone();
        let mut orig54 = orig_data[54].clone();
        let mut orig55 = orig_data[55].clone();
        let mut orig56 = orig_data[56].clone();
        let mut orig57 = orig_data[57].clone();
        let mut orig58 = orig_data[58].clone();
        let mut orig59 = orig_data[59].clone();
        let mut orig60 = orig_data[60].clone();
        let mut orig61 = orig_data[61].clone();
        let mut orig62 = orig_data[62].clone();
        let mut orig63 = orig_data[63].clone();
        let mut orig64 = orig_data[64].clone();
        let mut orig65 = orig_data[65].clone();
        let mut orig66 = orig_data[66].clone();
        let mut orig67 = orig_data[67].clone();
        let mut orig68 = orig_data[68].clone();
        let mut orig69 = orig_data[69].clone();
        let mut orig70 = orig_data[70].clone();
        let mut orig71 = orig_data[71].clone();
        let mut orig72 = orig_data[72].clone();
        let mut orig73 = orig_data[73].clone();
        let mut orig74 = orig_data[74].clone();
        let mut orig75 = orig_data[75].clone();
        let mut orig76 = orig_data[76].clone();
        let mut orig77 = orig_data[77].clone();
        let mut orig78 = orig_data[78].clone();
        let mut orig79 = orig_data[79].clone();
        let mut orig80 = orig_data[80].clone();
        let mut orig81 = orig_data[81].clone();
        let mut orig82 = orig_data[82].clone();
        let mut orig83 = orig_data[83].clone();
        let mut orig84 = orig_data[84].clone();
        let mut orig85 = orig_data[85].clone();
        let mut orig86 = orig_data[86].clone();
        let mut orig87 = orig_data[87].clone();
        let mut orig88 = orig_data[88].clone();
        let mut orig89 = orig_data[89].clone();
        let mut orig90 = orig_data[90].clone();
        let mut orig91 = orig_data[91].clone();
        let mut orig92 = orig_data[92].clone();
        let mut orig93 = orig_data[93].clone();
        let mut orig94 = orig_data[94].clone();
        let mut orig95 = orig_data[95].clone();
        let mut orig96 = orig_data[96].clone();
        let mut orig97 = orig_data[97].clone();
        let mut orig98 = orig_data[98].clone();
        let mut orig99 = orig_data[99].clone();

        let mut decode_blocks = vec![
            BlockMut::new(100, &mut rec0),
            BlockMut::new(101, &mut rec1),
            BlockMut::new(102, &mut rec2),
            BlockMut::new(103, &mut rec3),
            BlockMut::new(104, &mut rec4),
            BlockMut::new(105, &mut rec5),
            BlockMut::new(106, &mut rec6),
            BlockMut::new(107, &mut rec7),
            BlockMut::new(108, &mut rec8),
            BlockMut::new(109, &mut rec9),
            BlockMut::new(110, &mut rec10),
            BlockMut::new(111, &mut rec11),
            BlockMut::new(112, &mut rec12),
            BlockMut::new(113, &mut rec13),
            BlockMut::new(114, &mut rec14),
            BlockMut::new(115, &mut rec15),
            BlockMut::new(116, &mut rec16),
            BlockMut::new(117, &mut rec17),
            BlockMut::new(118, &mut rec18),
            BlockMut::new(119, &mut rec19),
            BlockMut::new(120, &mut rec20),
            BlockMut::new(121, &mut rec21),
            BlockMut::new(122, &mut rec22),
            BlockMut::new(123, &mut rec23),
            BlockMut::new(124, &mut rec24),
            BlockMut::new(125, &mut rec25),
            BlockMut::new(126, &mut rec26),
            BlockMut::new(127, &mut rec27),
            BlockMut::new(128, &mut rec28),
            BlockMut::new(129, &mut rec29),
            BlockMut::new(30, &mut orig30),
            BlockMut::new(31, &mut orig31),
            BlockMut::new(32, &mut orig32),
            BlockMut::new(33, &mut orig33),
            BlockMut::new(34, &mut orig34),
            BlockMut::new(35, &mut orig35),
            BlockMut::new(36, &mut orig36),
            BlockMut::new(37, &mut orig37),
            BlockMut::new(38, &mut orig38),
            BlockMut::new(39, &mut orig39),
            BlockMut::new(40, &mut orig40),
            BlockMut::new(41, &mut orig41),
            BlockMut::new(42, &mut orig42),
            BlockMut::new(43, &mut orig43),
            BlockMut::new(44, &mut orig44),
            BlockMut::new(45, &mut orig45),
            BlockMut::new(46, &mut orig46),
            BlockMut::new(47, &mut orig47),
            BlockMut::new(48, &mut orig48),
            BlockMut::new(49, &mut orig49),
            BlockMut::new(50, &mut orig50),
            BlockMut::new(51, &mut orig51),
            BlockMut::new(52, &mut orig52),
            BlockMut::new(53, &mut orig53),
            BlockMut::new(54, &mut orig54),
            BlockMut::new(55, &mut orig55),
            BlockMut::new(56, &mut orig56),
            BlockMut::new(57, &mut orig57),
            BlockMut::new(58, &mut orig58),
            BlockMut::new(59, &mut orig59),
            BlockMut::new(60, &mut orig60),
            BlockMut::new(61, &mut orig61),
            BlockMut::new(62, &mut orig62),
            BlockMut::new(63, &mut orig63),
            BlockMut::new(64, &mut orig64),
            BlockMut::new(65, &mut orig65),
            BlockMut::new(66, &mut orig66),
            BlockMut::new(67, &mut orig67),
            BlockMut::new(68, &mut orig68),
            BlockMut::new(69, &mut orig69),
            BlockMut::new(70, &mut orig70),
            BlockMut::new(71, &mut orig71),
            BlockMut::new(72, &mut orig72),
            BlockMut::new(73, &mut orig73),
            BlockMut::new(74, &mut orig74),
            BlockMut::new(75, &mut orig75),
            BlockMut::new(76, &mut orig76),
            BlockMut::new(77, &mut orig77),
            BlockMut::new(78, &mut orig78),
            BlockMut::new(79, &mut orig79),
            BlockMut::new(80, &mut orig80),
            BlockMut::new(81, &mut orig81),
            BlockMut::new(82, &mut orig82),
            BlockMut::new(83, &mut orig83),
            BlockMut::new(84, &mut orig84),
            BlockMut::new(85, &mut orig85),
            BlockMut::new(86, &mut orig86),
            BlockMut::new(87, &mut orig87),
            BlockMut::new(88, &mut orig88),
            BlockMut::new(89, &mut orig89),
            BlockMut::new(90, &mut orig90),
            BlockMut::new(91, &mut orig91),
            BlockMut::new(92, &mut orig92),
            BlockMut::new(93, &mut orig93),
            BlockMut::new(94, &mut orig94),
            BlockMut::new(95, &mut orig95),
            BlockMut::new(96, &mut orig96),
            BlockMut::new(97, &mut orig97),
            BlockMut::new(98, &mut orig98),
            BlockMut::new(99, &mut orig99),
        ];

        decode(&params, &mut decode_blocks).unwrap();
    }
    let decode_elapsed = decode_start.elapsed();
    let decode_us = decode_elapsed.as_micros() as f64 / trials as f64;
    let decode_mbps = data_size as f64 / decode_us;

    println!(
        "Rust CM256 Benchmark (k={}, m={}, {} bytes/block)",
        original_count, recovery_count, block_bytes
    );
    println!("  Encode: {:.2} us, {:.2} MB/s", encode_us, encode_mbps);
    println!("  Decode: {:.2} us, {:.2} MB/s", decode_us, decode_mbps);
}
