#include <string.h>

void solution(void) {
  // Tile counts
  const int tiles_I = MATMUL_M / DIM;
  const int tiles_J = MATMUL_N / DIM;
  const int tiles_K = MATMUL_K / DIM;

  // Scratchpad base addresses
  const int a_base  = 0;
  const int b_base  = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
  const int SPAD_DEST = 0;

  // Step 1: Configure execution (MX FP8 -> BF16, weight-stationary)
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY,
      1, 1, 0, 0, false, 0, 0, 3, 0);

  // Step 2: Load scale factors (A=sel 0, B=sel 1)
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);

  // Step 3: Load B tiles using mvin2 (port 1) — done once, amortized over all I tiles
  gemmini_extended3_config_ld(MATMUL_K * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1);
  for (int j = 0; j < tiles_J; j++) {
    int k = 0;
    for (; k + 4 <= tiles_K; k += 4) {
      gemmini_extended_mvin2(
          (void*)(((elem_t*)B_in) + j * DIM * MATMUL_K + k * DIM),
          b_base + (j * tiles_K + k) * DIM,
          4 * DIM, DIM);
    }
    for (; k < tiles_K; k++) {
      gemmini_extended_mvin2(
          (void*)(((elem_t*)B_in) + j * DIM * MATMUL_K + k * DIM),
          b_base + (j * tiles_K + k) * DIM,
          DIM, DIM);
    }
  }

  // Step 4: Configure mvin port 0 for A tiles
  gemmini_extended3_config_ld(MATMUL_K * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0);

  // Step 5: im2col + mvin A tiles.
  // OPTIMIZATION: Hoist pi/pj computation out of (c,dy,dx) loops,
  // and replace scalar element writes with contiguous memcpy over dx dimension.
  // Each (patch_idx, c, dy) row in A_buf gets CONV_KSZ consecutive bytes
  // from X_in[c][pi*CONV_KSZ+dy][pj*CONV_KSZ .. pj*CONV_KSZ+CONV_KSZ-1],
  // which are contiguous in memory — enabling vectorized/word-wide copies.
  for (int i = 0; i < tiles_I; i++) {
    for (int li = 0; li < DIM; li++) {
      int patch_idx = i * DIM + li;
      // Hoist division/modulo: computed once per patch_idx, not per (c,dy,dx)
      int pi = patch_idx / PATCHES_X;
      int pj = patch_idx % PATCHES_X;
      // Destination row in A_buf for this patch
      elem_t *dst_row = A_buf[patch_idx];
      for (int c = 0; c < CONV_C; c++) {
        for (int dy = 0; dy < CONV_KSZ; dy++) {
          // Source: CONV_KSZ contiguous bytes in X_in[c][pi*CONV_KSZ+dy][pj*CONV_KSZ]
          const elem_t *src = &X_in[c][pi * CONV_KSZ + dy][pj * CONV_KSZ];
          elem_t *dst = dst_row + (c * CONV_KSZ + dy) * CONV_KSZ;
          // Single contiguous copy of CONV_KSZ bytes — auto-vectorizable
          memcpy(dst, src, CONV_KSZ * sizeof(elem_t));
        }
      }
    }

    // Unrolled mvin for this i-row's K-tiles: issue up to 4 tiles per command.
    int k = 0;
    for (; k + 4 <= tiles_K; k += 4) {
      gemmini_extended_mvin(
          (void*)(((elem_t*)A_buf) + i * DIM * MATMUL_K + k * DIM),
          a_base + (i * tiles_K + k) * DIM,
          4 * DIM, DIM);
    }
    for (; k < tiles_K; k++) {
      gemmini_extended_mvin(
          (void*)(((elem_t*)A_buf) + i * DIM * MATMUL_K + k * DIM),
          a_base + (i * tiles_K + k) * DIM,
          DIM, DIM);
    }
  }

  // Step 6: Configure output stride and requantizer.
  // i_bound, j_bound, k_bound must match the tile counts used in loop_ws_spad.
  gemmini_config_st(OUT_COLS * sizeof(out_t));
  gemmini_mxquant_config_mvout((uint64_t)scale_factors,
      tiles_I, tiles_J, tiles_K, 0, 0, tiles_K);

  // Step 7: Execute fused tiled weight-stationary matmul over scratchpad.
  // funct 24 (LOOP_WS_CONFIG_SPAD_AB) is re-issued inside gemmini_loop_ws_spad
  // macro before every LOOP_WS — the one-shot mx_loop_spad_marker is satisfied.
  gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0,
      a_base, BANK_NUM * BANK_ROWS, 0, SPAD_DEST,
      false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);

  // Step 8: Drain BF16 results from mx_smem to DRAM, then fence.
  gemmini_mx_read_smem(&C_hw[0][0], SPAD_DEST * 16, MATMUL_M * MATMUL_N);
  gemmini_fence();
}
