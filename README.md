# cm256-rs

A pure Rust implementation of the CM256 Cauchy Reed-Solomon erasure coding library, ported from the [original C++ implementation](https://github.com/catid/cm256).

## Features

- **Pure Rust** - No C/C++ dependencies, no FFI
- **SIMD Optimized** - AVX-512, AVX2, SSSE3, and NEON with **runtime detection** (no `-C target-cpu=native` required!)
- **Performance Parity** - Matches or exceeds C++ performance
- **Safe API** - Memory-safe Rust interface with minimal `unsafe` in SIMD hot paths
- **Streaming FEC** - Low-latency diagonal interleaving for real-time applications
- **Transport Abstraction** - Pluggable transport layer supporting UDP, QUIC, and custom transports

## Performance

### x86_64 Benchmarks

Benchmarks on x86_64 with `k=100` original blocks, `m=30` recovery blocks, `1296` bytes/block:

| Implementation | Encode | Decode |
|----------------|--------|--------|
| **Rust + AVX-512 (auto-detected)** | **1742 MB/s** | **1619 MB/s** |
| Rust + AVX2 (auto-detected) | 1380 MB/s | 1315 MB/s |
| C++ + AVX2 (original) | 1283 MB/s | 1380 MB/s |
| Rust + SSSE3 (auto-detected) | 745 MB/s | 755 MB/s |
| Rust + WASM SIMD | 487 MB/s | 584 MB/s |
| Rust (scalar) | 71 MB/s | 76 MB/s |

### ARM (Apple M4) Benchmarks

| Implementation | Encode | Decode |
|----------------|--------|--------|
| **Rust + NEON** | **1351 MB/s** | **1592 MB/s** |
| C++ + NEON (original) | 1307 MB/s | 1424 MB/s |
| Rust + WASM SIMD | 1239 MB/s | 1234 MB/s |
| Rust (scalar) | 131 MB/s | 142 MB/s |

🏆 **Rust beats C++ on x86 (36% faster with AVX-512) and ARM!**

### Run All Benchmarks

```bash
./run_benchmarks.sh
```

### Runtime SIMD Detection

SIMD instructions are automatically detected at runtime - no compiler flags needed!
The library automatically selects the best available: AVX-512 → AVX2 → SSSE3 → scalar.

```bash
# Just build normally - SIMD is auto-detected at runtime
cargo build --release
```

For maximum performance with compile-time detection (slightly faster due to inlining):

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## Usage

### Block FEC (Traditional)

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

### Streaming FEC (Low-Latency)

For real-time streaming applications where latency is critical:

```rust
use cm256::streaming::{StreamingEncoder, StreamingDecoder, StreamingParams};

// delay=8 symbols, 2 parities per window, 1200 bytes per symbol
let params = StreamingParams::new(8, 2, 1200).unwrap();

let mut encoder = StreamingEncoder::new(params);
let mut decoder = StreamingDecoder::new(params);

// Encode source data - parities generated when window is full
let data = vec![0x42u8; 1200];
let result = encoder.add_source(&data);

// Send source packet, then any parity packets
// result.parities contains generated parity symbols
for parity in &result.parities {
    // Send parity.data with parity.end_seq and parity.parity_index
}
```

## Streaming FEC Module

The `streaming` module provides diagonal interleaving FEC for low-latency applications:

```text
Time:     0   1   2   3   4   5   6   7   8   ...
Source:   S₀  S₁  S₂  S₃  S₄  S₅  S₆  S₇  S₈  ...
          ╲   ╲   ╲   ╲   ╲
Parity:       P₀  P₁  P₂  P₃  P₄  ...
              │   │   │   │   │
              └───┴───┴───┴───┴── Each Pᵢ covers [Sᵢ₋ₐ₊₁, Sᵢ]
```

### Key Features

- **Lower latency** than block FEC - parities generated per-packet
- **Burst recovery** - overlapping windows catch consecutive losses
- **Continuous protection** - every packet covered by multiple parity windows

### Transport Abstraction

The `transport` module provides traits for abstracting over datagram transports:

```rust
use cm256::transport::{DatagramSend, DatagramRecv};
use cm256::streaming::{TransportEncoder, TransportDecoder, StreamingParams};
use std::net::UdpSocket;

// Create connected UDP socket
let socket = UdpSocket::bind("0.0.0.0:0").unwrap();
socket.connect("127.0.0.1:9000").unwrap();

// Wrap with FEC encoder
let params = StreamingParams::new(8, 2, 1200).unwrap();
let mut encoder = TransportEncoder::new(params, socket);

// Send data - FEC encoding and transmission handled automatically
encoder.send(&[0x42; 1200]).unwrap();
```

**Supported Transports:**
- `std::net::UdpSocket` - Synchronous UDP
- `tokio::net::UdpSocket` - Async UDP (requires `tokio` feature)
- `quinn::Connection` - QUIC datagrams (requires `quinn` feature)
- `UnixDatagram` - Unix datagram sockets
- `MemoryChannel` - In-memory channels for testing
- `LossyChannel<T>` - Wrapper for simulating packet loss

### Async Transport Example

```rust
use cm256::streaming::{AsyncTransportEncoder, StreamingParams};
use tokio::net::UdpSocket;

async fn send_with_fec() -> std::io::Result<()> {
    let socket = UdpSocket::bind("0.0.0.0:0").await?;
    socket.connect("127.0.0.1:9000").await?;

    let params = StreamingParams::new(8, 2, 1200).unwrap();
    let mut encoder = AsyncTransportEncoder::new(params, socket);

    encoder.send(&[0x42; 1200]).await?;
    Ok(())
}
```

## Examples

### FFmpeg Streaming Example

Stream H.264 video with FEC protection over UDP or QUIC:

```bash
# Build with tokio support
cargo build --release --example ffmpeg_streaming

# UDP Transport (default)
# Terminal 1 - Receiver (start first)
cargo run --release --example ffmpeg_streaming -- recv | ffplay -f h264 -

# Terminal 2 - Sender
ffmpeg -re -i input.mp4 -c:v libx264 -tune zerolatency -f h264 - | \
    cargo run --release --example ffmpeg_streaming -- send

# QUIC Transport
# Terminal 1 - Receiver (QUIC server)
cargo run --release --example ffmpeg_streaming -- recv --transport quic | ffplay -f h264 -

# Terminal 2 - Sender (QUIC client)
ffmpeg -re -i input.mp4 -c:v libx264 -tune zerolatency -f h264 - | \
    cargo run --release --example ffmpeg_streaming -- send --transport quic
```

**Options:**
- `--transport <udp|quic>` - Transport type (default: udp)
- `--delay <N>` - FEC window size in packets (default: 15)
- `--parities <N>` - Parity packets per window (default: 1)
- `--step-size <N>` - Generate parities every N packets (default: 5)
- `--packet-size <N>` - Packet size in bytes (default: 1200, sender only - receiver auto-detects)

### Benchmark Example

```bash
cargo run --release --example benchmark
```

## Architecture Support

| Architecture | SIMD Instructions | Bytes/Iteration |
|--------------|-------------------|-----------------|
| x86_64 + AVX-512 | `vpshufb` (512-bit) | 256 (4x unrolled) |
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

### Optional Features

| Feature | Description |
|---------|-------------|
| `simd` | SIMD acceleration (enabled by default) |
| `tokio` | Async transport support for `tokio::net::UdpSocket` |
| `quinn` | QUIC datagram transport via `quinn::Connection` |

```bash
# Build with async transport support
cargo build --release --features tokio

# Build with QUIC support
cargo build --release --features quinn

# Build examples with all features
cargo build --release --examples --features "tokio quinn"
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
