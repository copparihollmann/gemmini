// AUTO-GENERATED outer-tiled MX matmul baseline (1x1024x1024) - Optimized Phase 5
void solution(void) {
  // Pre-pack B's scale factors into a contiguous layout once at the start
  // to avoid strided slices in the nested execution loop.
  static uint8_t B_scales_contiguous[BLOCKS_N][MATMUL_K / 32][BLK];

  // Perform one-time pre-packing of B scale factors at the beginning
  for (int j0 = 0; j0 < BLOCKS_N; j0++) {
    for (int g = 0; g < MATMUL_K / 32; g++) {
      memcpy(&B_scales_contiguous[j0][g][0], &B_scales_col[g][j0 * BLK], BLK);
    }
  }

  for (int i0 = 0; i0 < BLOCKS_M; i0++) {
    for (int j0 = 0; j0 < BLOCKS_N; j0++) {
      // Partial last M/N block: compute only the real rows/cols, pad the rest.
      int mrows = REAL_M - i0 * BLK; if (mrows > BLK) mrows = BLK; if (mrows <= 0) continue;
      int ncols = REAL_N - j0 * BLK; if (ncols > BLK) ncols = BLK; if (ncols <= 0) continue;

      int tiles = BLK / DIM, tiles_K = MATMUL_K / DIM;
      uint32_t a_base = 0;
      uint32_t b_base = BANK_NUM * BANK_ROWS - tiles * tiles_K * DIM;
      int SPAD_DEST = 128;
      int pad_i = BLK - mrows, pad_j = BLK - ncols;

      gemmini_flush(0);
      gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);
      
      // Pass direct DRAM pointers to scale factor loading functions, completely eliminating memcpy here.
      gemmini_mx_load_scales((uint64_t)A_scales_row, A_S_BYTES, 0);
      gemmini_mx_load_scales((uint64_t)B_scales_contiguous[j0], B_S_BYTES, 1);

      gemmini_config_ld(MATMUL_K * sizeof(elem_t));      // A rows: stride K (full block-aligned tiles)
      for (int i = 0; i < tiles; i++)
        for (int k = 0; k < tiles_K; k++)
          gemmini_extended_mvin((void*)(((elem_t*)A_in) + (i0 * BLK + i * DIM) * MATMUL_K + k * DIM),
                                a_base + (i * tiles_K + k) * DIM, DIM, DIM);

      gemmini_config_ld(MATMUL_N * sizeof(elem_t));      // B rows: stride N
      for (int j = 0; j < tiles; j++)                    // B[k][j0*BLK+j] -> slot (k*tiles + j)
        for (int k = 0; k < tiles_K; k++)
          gemmini_extended_mvin((void*)(((elem_t*)B_in) + (k * DIM) * MATMUL_N + (j0 * BLK + j * DIM)),
                                b_base + (k * tiles + j) * DIM, DIM, DIM);

      gemmini_config_st(BLK_OUTC * sizeof(out_t));
      gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles, tiles, tiles_K, 0, 0, 1);
      gemmini_loop_ws_spad(tiles, tiles, tiles_K, pad_i, pad_j, 0, a_base, BANK_NUM * BANK_ROWS, 0,
                           SPAD_DEST, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);

      // Accelerator-centric readout: per-row hardware DMA from smem straight into the
      // bf16-packed row-major output (block-internal smem row stride is BLK).
      for (int r = 0; r < mrows; r++)
        gemmini_mx_read_smem(&C_hw[i0 * BLK + r][(j0 * BLK) / BF16_PER_WORD],
                             SPAD_DEST * 16 + r * BLK, ncols);
    }
  }

  // Defer synchronization to after the entire batch has been successfully queued.
  gemmini_fence();
}
