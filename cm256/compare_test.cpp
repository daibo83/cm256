// Test program to output encoded recovery blocks for comparison with Rust
#include <cstdio>
#include <cstdint>
#include <cstring>
#include "cm256.h"

// Print bytes as hex for comparison
void print_hex(const char* label, const uint8_t* data, int len) {
    printf("%s: ", label);
    for (int i = 0; i < len && i < 64; i++) {
        printf("%02x", data[i]);
    }
    if (len > 64) printf("...");
    printf("\n");
}

int main() {
    if (cm256_init()) {
        fprintf(stderr, "cm256_init failed\n");
        return 1;
    }

    // Test case 1: Simple 3 original, 2 recovery, 16 bytes each
    {
        printf("=== Test 1: 3 orig, 2 recovery, 16 bytes ===\n");
        
        cm256_encoder_params params;
        params.OriginalCount = 3;
        params.RecoveryCount = 2;
        params.BlockBytes = 16;

        // Create predictable data: block i filled with value (i+1)
        uint8_t orig0[16], orig1[16], orig2[16];
        memset(orig0, 1, 16);
        memset(orig1, 2, 16);
        memset(orig2, 3, 16);

        cm256_block blocks[3];
        blocks[0].Block = orig0;
        blocks[0].Index = 0;
        blocks[1].Block = orig1;
        blocks[1].Index = 1;
        blocks[2].Block = orig2;
        blocks[2].Index = 2;

        uint8_t recovery[32];
        if (cm256_encode(params, blocks, recovery)) {
            fprintf(stderr, "encode failed\n");
            return 1;
        }

        print_hex("orig0", orig0, 16);
        print_hex("orig1", orig1, 16);
        print_hex("orig2", orig2, 16);
        print_hex("rec0 ", recovery, 16);
        print_hex("rec1 ", recovery + 16, 16);
    }

    // Test case 2: XOR parity verification
    {
        printf("\n=== Test 2: XOR parity 4 bytes ===\n");
        
        cm256_encoder_params params;
        params.OriginalCount = 3;
        params.RecoveryCount = 1;
        params.BlockBytes = 4;

        uint8_t orig0[] = {0x11, 0x22, 0x33, 0x44};
        uint8_t orig1[] = {0x55, 0x66, 0x77, 0x88};
        uint8_t orig2[] = {0x99, 0xAA, 0xBB, 0xCC};

        cm256_block blocks[3];
        blocks[0].Block = orig0;
        blocks[1].Block = orig1;
        blocks[2].Block = orig2;

        uint8_t recovery[4];
        if (cm256_encode(params, blocks, recovery)) {
            fprintf(stderr, "encode failed\n");
            return 1;
        }

        print_hex("orig0", orig0, 4);
        print_hex("orig1", orig1, 4);
        print_hex("orig2", orig2, 4);
        print_hex("rec0 ", recovery, 4);
        
        // Expected: XOR of all three
        uint8_t expected[4];
        for (int i = 0; i < 4; i++) {
            expected[i] = orig0[i] ^ orig1[i] ^ orig2[i];
        }
        print_hex("expected", expected, 4);
    }

    // Test case 3: Multiple recovery blocks with varying data
    {
        printf("\n=== Test 3: 5 orig, 3 recovery, 32 bytes ===\n");
        
        cm256_encoder_params params;
        params.OriginalCount = 5;
        params.RecoveryCount = 3;
        params.BlockBytes = 32;

        uint8_t orig[5][32];
        cm256_block blocks[5];
        
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 32; j++) {
                orig[i][j] = (uint8_t)((i * 32 + j) % 256);
            }
            blocks[i].Block = orig[i];
            blocks[i].Index = i;
        }

        uint8_t recovery[96];
        if (cm256_encode(params, blocks, recovery)) {
            fprintf(stderr, "encode failed\n");
            return 1;
        }

        for (int i = 0; i < 5; i++) {
            char label[16];
            snprintf(label, sizeof(label), "orig%d", i);
            print_hex(label, orig[i], 32);
        }
        for (int i = 0; i < 3; i++) {
            char label[16];
            snprintf(label, sizeof(label), "rec%d ", i);
            print_hex(label, recovery + i * 32, 32);
        }
    }

    printf("\n=== All tests completed ===\n");
    return 0;
}
