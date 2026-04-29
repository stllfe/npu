/*
 * minimal_add.c - Minimal standalone NPU add operation test
 *
 * Build: gcc -o minimal_add minimal_add.c -I./include -lm
 * Run:   sudo ./minimal_add
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "rknnops.h"

// ALU algorithm codes from rknnops.h
#define ALU_ALGO_ADD 2

// Helper: print FP16 matrix
static void print_fp16_matrix(const char *name, __fp16 *data, int rows, int cols) {
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int r = 0; r < rows && r < 8; r++) {
        printf("  ");
        for (int c = 0; c < cols && c < 8; c++) {
            printf("%8.4f ", (float)data[r * cols + c]);
        }
        if (cols > 8) printf("...");
        printf("\n");
    }
    if (rows > 8) printf("  ...\n");
    printf("\n");
}

int main(void) {
    // Configuration
    const int rows = 4;
    const int cols = 4;
    const int size = rows * cols;
    const size_t total_elements = (size_t)rows * cols;

    printf("=== Minimal NPU Add Test (%dx%d) ===\n\n", rows, cols);

    // Allocate input buffers (host memory)
    __fp16 *a = (__fp16 *)malloc(total_elements * sizeof(__fp16));
    __fp16 *b = (__fp16 *)malloc(total_elements * sizeof(__fp16));
    if (!a || !b) {
        printf("Failed to allocate input buffers\n");
        free(a); free(b);
        return 1;
    }

    // Fill with simple test data
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        a[i] = (__fp16)((float)(i + 1) * 0.5f);  // 0.5, 1.0, 1.5, ...
        b[i] = (__fp16)((float)(i + 1) * 0.25f); // 0.25, 0.5, 0.75, ...
    }

    print_fp16_matrix("Input A", a, rows, cols);
    print_fp16_matrix("Input B", b, rows, cols);

    // Run NPU add operation
    printf("Running NPU add (algo=%d)...\n", ALU_ALGO_ADD);
    __fp16 *result = float16_alu_op(a, b, ALU_ALGO_ADD, size);
    if (!result) {
        printf("NPU operation failed\n");
        free(a); free(b);
        return 1;
    }

    // Unpack result (NPU outputs are spaced every 0x10 bytes)
    __fp16 *unpacked = (__fp16 *)malloc(total_elements * sizeof(__fp16));
    if (!unpacked) {
        printf("Failed to allocate unpack buffer\n");
        free(result); free(a); free(b);
        return 1;
    }

    const size_t stride_fp16 = 0x10 / sizeof(__fp16);  // = 8
    for (int i = 0; i < size; i++) {
        unpacked[i] = result[(size_t)i * stride_fp16];
    }

    print_fp16_matrix("NPU Result", unpacked, rows, cols);

    // Verify against CPU reference
    float max_diff = 0.0f;
    printf("CPU Reference:\n  ");
    for (int i = 0; i < size && i < 16; i++) {
        float expected = (float)a[i] + (float)b[i];
        float actual = (float)unpacked[i];
        float diff = fabsf(actual - expected);
        if (diff > max_diff) max_diff = diff;
        printf("%8.4f ", expected);
    }
    printf("\n\n");

    const float kAtol = 1e-3f;
    int matches = max_diff <= kAtol;
    printf("Verification: %s (max diff=%.6f, tolerance=%.6f)\n",
           matches ? "PASS" : "FAIL", max_diff, kAtol);

    // Cleanup
    free(result);  // NPU output buffer
    free(unpacked);
    free(a);
    free(b);

    return matches ? 0 : 1;
}
