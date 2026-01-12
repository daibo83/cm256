#!/bin/bash
# Run all CM256 benchmarks across different SIMD levels

set -e

echo "=============================================="
echo "CM256 Benchmark Suite"
echo "=============================================="
echo ""

# Build C++ if needed
if [ ! -f "cm256/build/benchmark" ]; then
    echo "Building C++ benchmark..."
    mkdir -p cm256/build
    cd cm256/build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    cd ../..
    echo ""
fi

echo "----------------------------------------------"
echo "C++ + AVX2 (original, -march=native)"
echo "----------------------------------------------"
./cm256/build/benchmark
echo ""

echo "----------------------------------------------"
echo "Rust (scalar, no SIMD)"
echo "----------------------------------------------"
cargo run --release --example benchmark 2>/dev/null
echo ""

echo "----------------------------------------------"
echo "Rust + SSE3 (SSSE3)"
echo "----------------------------------------------"
RUSTFLAGS="-C target-feature=+ssse3" cargo run --release --example benchmark 2>/dev/null
echo ""

echo "----------------------------------------------"
echo "Rust + AVX2 (native CPU)"
echo "----------------------------------------------"
RUSTFLAGS="-C target-cpu=native" cargo run --release --example benchmark 2>/dev/null
echo ""

echo "=============================================="
echo "Benchmark complete!"
echo "=============================================="
