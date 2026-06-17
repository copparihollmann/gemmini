void solution(void) {

  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);

#ifdef SPIKE_SIM
  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);
  gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);
#else
  load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_A, (uint8_t *) &A_scales_row, MATMUL_M, MATMUL_K);
  load_scale_factors((volatile uint64_t *) GEMMINI_SF_MEM_B, (uint8_t *) &B_scales_col, MATMUL_N, MATMUL_K);
#endif

  // ---- MVIN A: tile (i,k) -> a_base + (i*tiles_K + k)*DIM ----
  // Use mvin port 0 for A, batch up to MAX_BLOCK_LEN=4 consecutive k tiles
  gemmini_extended3_config_ld(MATMUL_M * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0);

  for (int i = 0; i < tiles_I; i++) {
    int k = 0;
    while (k < tiles_K) {
      int batch = tiles_K - k;
      if (batch > MAX_BLOCK_LEN) batch = MAX_BLOCK_LEN;
      elem_t *dram_ptr = ((elem_t*)A_in) + i * DIM * MATMUL_M + k * DIM;
      uint32_t sp_addr = a_base + (i * tiles_K + k) * DIM;
      gemmini_extended_mvin((void *) dram_ptr, sp_addr, batch * DIM, DIM);
      k += batch;
    }
  }

  // ---- MVIN B: tile (j,k) -> b_base + (j*tiles_K + k)*DIM ----
  // Use mvin2 port for B to allow overlap with A loads/compute
  gemmini_extended3_config_ld(MATMUL_M * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1);

  for (int j = 0; j < tiles_J; j++) {
    int k = 0;
    while (k < tiles_K) {
      int batch = tiles_K - k;
      if (batch > MAX_BLOCK_LEN) batch = MAX_BLOCK_LEN;
      elem_t *dram_ptr = ((elem_t*)B_in) + j * DIM * MATMUL_M + k * DIM;
      uint32_t sp_addr = b_base + (j * tiles_K + k) * DIM;
      gemmini_extended_mvin2((void *) dram_ptr, sp_addr, batch * DIM, DIM);
      k += batch;
    }
  }

  int SPAD_DEST = 128;

  gemmini_config_st(OUT_COLS * sizeof(out_t));
  gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, tiles_J);

  // ---- Compute ----
  gemmini_loop_ws_spad(
      tiles_I, tiles_J, tiles_K,
      0, 0, 0,
      a_base,
      BANK_NUM * BANK_ROWS,
      0,
      SPAD_DEST,
      false, false,
      false, false, false,
      NO_ACTIVATION,
      0, 0,
      false,
      0x38);

  gemmini_fence();
#ifndef SPIKE_SIM
  gemmini_poll_until_ready();
#endif
  gemmini_fence();

#ifdef SPIKE_SIM
  gemmini_mx_read_smem(&C_hw[0][0], SPAD_DEST * 16, MATMUL_M * MATMUL_N);
#else
  uint64_t* smem_start_addr = ((uint64_t*)SMEM) + SPAD_DEST * 2;
  printf("Address: %p \n", smem_start_addr);
  for (int i = 0; i < MATMUL_M; i ++) {
    for (int j = 0; j < OUT_COLS; j++) {
        C_hw[i][j] = *(smem_start_addr + (i*OUT_COLS + j));
    }
  }
#endif

  gemmini_fence();
}
