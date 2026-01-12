# cm256-rs

A pure Rust implementation of the CM256 Cauchy Reed-Solomon erasure coding library, ported from the [original C++ implementation](https://github.com/catid/cm256).

## Features

- **Pure Rust** - No C/C++ dependencies, no FFI
- **SIMD Optimized** - AVX2, SSE3/SSSE3, and NEON support with automatic detection
- **Performance Parity** - Matches or exceeds C++ performance with AVX2
- **Safe API** - Memory-safe Rust interface with minimal `unsafe` in SIMD hot paths

## Performance

### x86_64 Benchmarks

Benchmarks on x86_64 with `k=100` original blocks, `m=30` recovery blocks, `1296` bytes/block:

| Implementation | Encode | Decode |
|----------------|--------|--------|
| **Rust + AVX2** | **1403 MB/s** | **1415 MB/s** |
| C++ + AVX2 (original) | 1283 MB/s | 1380 MB/s |
| Rust + SSE3 | 742 MB/s | 756 MB/s |
| Rust + WASM SIMD | 547 MB/s | 594 MB/s |
| Rust (scalar) | 71 MB/s | 76 MB/s |

### ARM (Apple M4) Benchmarks

| Implementation | Encode | Decode |
|----------------|--------|--------|
| **Rust + NEON** | **1351 MB/s** | **1592 MB/s** |
| C++ + NEON (original) | 1307 MB/s | 1424 MB/s |
| Rust + WASM SIMD | 1239 MB/s | 1234 MB/s |
| Rust (scalar) | 131 MB/s | 142 MB/s |

🏆 **Rust beats C++ on both x86 AVX2 and ARM NEON!**

### Run All Benchmarks

```bash
./run_benchmarks.sh
```

### Build for Maximum Performance

```bash
# Enable native CPU features (AVX2 on most modern x86_64)
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## Usage

```rust
use cm256::{Params, Block, encode_block, decode};

// Configure: 100 original blocks, 30 recovery blocks, 1024 bytes each
let params = Params::new(100, 30, 1024).unwrap();

// Original data blocks
let data: Vec<Vec<u8>> = (0..100)
    .map(|i| vec![i as u8; 1024])
    .collect();

let blocks: Vec<Block> = data.iter()
    .enumerate()
    .map(|(i, d)| Block { data: d, index: i as u8 })
    .collect();

// Generate recovery blocks
let mut recovery = vec![vec![0u8; 1024]; 30];
for i in 0..30 {
    encode_block(&params, &blocks, i, &mut recovery[i]);
}

// Simulate loss: lose blocks 0-29, keep recovery blocks
let mut available: Vec<Block> = blocks[30..].to_vec();
let mut recovery_blocks: Vec<Block> = recovery.iter()
    .enumerate()
    .map(|(i, r)| Block { data: r, index: (100 + i) as u8 })
    .collect();

// Decode lost blocks
let decoded = decode(&params, &blocks[30..], &recovery_blocks, &mut recovery).unwrap();
assert_eq!(decoded[0], data[0]);
```

## Architecture Support

| Architecture | SIMD Instructions | Bytes/Iteration |
|--------------|-------------------|-----------------|
| x86_64 + AVX2 | `vpshufb` (256-bit) | 128 (4x unrolled) |
| x86_64 + SSE3 | `pshufb` (128-bit) | 64 (4x unrolled) |
| aarch64 (ARM) | `vtbl` (128-bit) | 64 (4x unrolled) |
| wasm32 + SIMD | `i8x16_swizzle` (128-bit) | 64 (4x unrolled) |
| Other | Scalar lookup | 1 |

The SIMD implementation uses the nibble-based multiplication technique:
- Split each byte into high/low nibbles
- Use 16-entry lookup tables for each nibble
- XOR the results: `result = TABLE_LO[x & 0x0f] ^ TABLE_HI[x >> 4]`

This allows 16/32-byte parallel GF(256) multiplication using shuffle instructions.

## Algorithm

CM256 implements a systematic Cauchy Reed-Solomon code:

1. **Cauchy Matrix** - Uses a Cauchy matrix over GF(256) for optimal erasure correction
2. **LDU Decomposition** - Efficient O(n²) decoding using LDU matrix factorization
3. **GF(256) Arithmetic** - Polynomial 0x14d with generator 0x02

### Properties

- Systematic code: original blocks are transmitted unmodified
- Maximum Distance Separable (MDS): can recover from any `m` erasures with exactly `m` recovery blocks
- Block count: up to 256 total blocks (original + recovery)

## Building

```bash
# Standard build
cargo build --release

# With maximum SIMD optimization
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo run --release --example benchmark
```

## WASM Support

CM256 supports WebAssembly with SIMD acceleration using `i8x16_swizzle`:

```bash
# Install dependencies
rustup target add wasm32-wasip1
curl https://wasmtime.dev/install.sh -sSf | bash

# Build with SIMD
RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-wasip1

# Run benchmark
wasmtime --wasm simd target/wasm32-wasip1/release/examples/benchmark.wasm
```

WASM SIMD achieves ~550 MB/s encode / ~595 MB/s decode - about 7.5x faster than scalar and 74% of native SSE3 performance.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Credits

- Original C++ implementation: [catid/cm256](https://github.com/catid/cm256)
- GF(256) crate: [geky/gf256](https://github.com/geky/gf256)
