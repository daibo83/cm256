// Benchmark C++ cm256 encoder/decoder
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include "cm256.h"

int main() {
    if (cm256_init()) {
        fprintf(stderr, "cm256_init failed\n");
        return 1;
    }

    // Benchmark parameters - same as original unit test
    cm256_encoder_params params;
    params.BlockBytes = 1296;
    params.OriginalCount = 100;
    params.RecoveryCount = 30;

    const int TRIALS = 1000;
    const int data_size = params.OriginalCount * params.BlockBytes;

    uint8_t* orig_data = new uint8_t[256 * params.BlockBytes];
    uint8_t* recovery_data = new uint8_t[256 * params.BlockBytes];
    cm256_block blocks[256];

    // Initialize data
    for (int i = 0; i < params.BlockBytes * params.OriginalCount; ++i) {
        orig_data[i] = (uint8_t)i;
    }
    for (int i = 0; i < params.OriginalCount; ++i) {
        blocks[i].Block = orig_data + i * params.BlockBytes;
        blocks[i].Index = i;
    }

    // Benchmark encoding
    auto encode_start = std::chrono::high_resolution_clock::now();
    for (int trial = 0; trial < TRIALS; ++trial) {
        if (cm256_encode(params, blocks, recovery_data)) {
            fprintf(stderr, "encode failed\n");
            return 1;
        }
    }
    auto encode_end = std::chrono::high_resolution_clock::now();
    double encode_us = std::chrono::duration<double, std::micro>(encode_end - encode_start).count() / TRIALS;
    double encode_mbps = data_size / encode_us;

    // Prepare for decoding benchmark
    // Simulate losing first RecoveryCount blocks
    for (int i = 0; i < params.RecoveryCount && i < params.OriginalCount; ++i) {
        blocks[i].Block = recovery_data + params.BlockBytes * i;
        blocks[i].Index = params.OriginalCount + i;
    }

    // Benchmark decoding
    auto decode_start = std::chrono::high_resolution_clock::now();
    for (int trial = 0; trial < TRIALS; ++trial) {
        // Restore blocks for each trial
        for (int i = 0; i < params.RecoveryCount && i < params.OriginalCount; ++i) {
            memcpy(blocks[i].Block, recovery_data + params.BlockBytes * i, params.BlockBytes);
            blocks[i].Index = params.OriginalCount + i;
        }
        if (cm256_decode(params, blocks)) {
            fprintf(stderr, "decode failed\n");
            return 1;
        }
    }
    auto decode_end = std::chrono::high_resolution_clock::now();
    double decode_us = std::chrono::duration<double, std::micro>(decode_end - decode_start).count() / TRIALS;
    double decode_mbps = data_size / decode_us;

    printf("C++ CM256 Benchmark (k=%d, m=%d, %d bytes/block)\n", 
           params.OriginalCount, params.RecoveryCount, params.BlockBytes);
    printf("  Encode: %.2f us, %.2f MB/s\n", encode_us, encode_mbps);
    printf("  Decode: %.2f us, %.2f MB/s\n", decode_us, decode_mbps);

    delete[] orig_data;
    delete[] recovery_data;
    return 0;
}
