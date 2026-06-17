#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "include/gemmini_testutils.h"
#include "include/matmul_fp8_64x64.h"

void solution(void) {

    const int tiles_I = MATMUL_M / DIM;
    const int tiles_J = MATMUL_N / DIM;
    const int tiles_K = MATMUL_K / DIM;

    const uint32_t a_base    = 0;
    const uint32_t b_base    = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;

    const uint32_t SPAD_DEST = 0;

    gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY,
        1, 1,
        0, 0,
        false,
        0, 0, 3, 0);

    gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
    gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);

    gemmini_config_ld(MATMUL_K * sizeof(elem_t));
    {
        const elem_t *A_ptr = (const elem_t *)A_in;
        for (int i = 0; i < tiles_I; i++) {

            const void *dram_ptr = (const void *)(A_ptr + (i * DIM) * MATMUL_K);
            uint32_t    sp_addr  = a_base + i * tiles_K * DIM;

            gemmini_block_mvin(dram_ptr, sp_addr, tiles_K);
        }
    }

    {
        const elem_t *B_ptr = (const elem_t *)B_in;
        for (int k = 0; k < tiles_K; k++) {

            const void *dram_ptr = (const void *)(B_ptr + (k * DIM) * MATMUL_N);
            uint32_t    sp_addr  = b_base + k * tiles_J * DIM;

            gemmini_block_mvin(dram_ptr, sp_addr, tiles_J);
        }
    }

    gemmini_config_st(OUT_COLS * sizeof(out_t));
    gemmini_mxquant_config_mvout((uint64_t)scale_factors,
        tiles_I, tiles_J, tiles_K,
        0, 0,
        1);

    gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0,
        a_base, BANK_NUM * BANK_ROWS, 0, SPAD_DEST,
        false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);

    gemmini_mx_read_smem(&C_hw[0][0], SPAD_DEST * 16, MATMUL_M * MATMUL_N);
    gemmini_fence();
}
