#!/bin/bash
# Run all CM256 benchmarks across different SIMD levels

set -e

ARCH=$(uname -m)

echo "=============================================="
echo "CM256 Benchmark Suite"
echo "Architecture: $ARCH"
echo "=============================================="
echo ""

# Build C++ benchmark if needed
if [ ! -f "cm256/build/benchmark" ]; then
    echo "Building C++ benchmark..."
    mkdir -p cm256/build
    cd cm256/build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    cd ../..
    echo ""
fi

# Run C++ benchmark
if [ "$ARCH" = "x86_64" ]; then
    echo "----------------------------------------------"
    echo "C++ + AVX2 (original, -march=native)"
    echo "----------------------------------------------"
elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    echo "----------------------------------------------"
    echo "C++ + NEON (original, -march=native)"
    echo "----------------------------------------------"
else
    echo "----------------------------------------------"
    echo "C++ (original, -march=native)"
    echo "----------------------------------------------"
fi
./cm256/build/benchmark
echo ""

echo "----------------------------------------------"
echo "Rust (scalar, no SIMD)"
echo "----------------------------------------------"
cargo run --release --no-default-features --example benchmark 2>/dev/null
echo ""

# x86_64 specific benchmarks
if [ "$ARCH" = "x86_64" ]; then
    echo "----------------------------------------------"
    echo "Rust + SSE3 (SSSE3)"
    echo "----------------------------------------------"
    RUSTFLAGS="-C target-feature=+ssse3" cargo run --release --example benchmark 2>/dev/null
    echo ""

    echo "----------------------------------------------"
    echo "Rust + AVX2"
    echo "----------------------------------------------"
    RUSTFLAGS="-C target-feature=+avx2,+ssse3" cargo run --release --example benchmark 2>/dev/null
    echo ""

    # Check for AVX-512 support
    if grep -q avx512bw /proc/cpuinfo 2>/dev/null; then
        echo "----------------------------------------------"
        echo "Rust + AVX-512 (native CPU)"
        echo "----------------------------------------------"
        RUSTFLAGS="-C target-cpu=native" cargo run --release --example benchmark 2>/dev/null
        echo ""
    fi
fi

# ARM/aarch64 specific benchmarks
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    echo "----------------------------------------------"
    echo "Rust + NEON (native CPU)"
    echo "----------------------------------------------"
    RUSTFLAGS="-C target-cpu=native" cargo run --release --example benchmark 2>/dev/null
    echo ""
else
    echo "----------------------------------------------"
    echo "Rust + NEON (aarch64)"
    echo "----------------------------------------------"
    echo "⚠️  NEON benchmark skipped: not running on ARM"
    echo "   To run on ARM: use aarch64/arm64 hardware"
    echo ""
fi

# WASM SIMD benchmark
echo "----------------------------------------------"
echo "Rust + WASM SIMD (wasm32 + simd128)"
echo "----------------------------------------------"
if command -v wasmtime &> /dev/null && rustup target list --installed 2>/dev/null | grep -q wasm32-wasip1; then
    # Build WASM binary with SIMD
    echo "Building WASM binary..."
    RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-wasip1 --example benchmark 2>/dev/null
    
    # Run with wasmtime
    wasmtime --wasm simd target/wasm32-wasip1/release/examples/benchmark.wasm
    echo ""
else
    echo "⚠️  WASM benchmark skipped: missing dependencies"
    echo "   To enable:"
    echo "   1. Install wasmtime: curl https://wasmtime.dev/install.sh -sSf | bash"
    echo "   2. Add WASM target: rustup target add wasm32-wasip1"
    echo ""
fi

echo "----------------------------------------------"
echo "Erasure Code Comparison (CM256 vs RaptorQ vs Wirehair)"
echo "----------------------------------------------"
echo "(Using AVX2 for CM256 for fair comparison)"
RUSTFLAGS="-C target-feature=+avx2,+ssse3" cargo run --release --example compare_raptorq 2>/dev/null
echo ""

echo "=============================================="
echo "Benchmark complete!"
echo "=============================================="
