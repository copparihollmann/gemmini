void solution(void) {
  // ---- 1) S1 = Q @ K^T on Gemmini (bf16 out) ----
  {
    const int tiles_I = ATT_S / DIM;
    const int tiles_J = ATT_S / DIM;
    const int tiles_K = ATT_D / DIM;
    const uint32_t a_base = 0;
    const uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;

    gemmini_flush(0);

    // Step 1: Configure execution (FP8 E4M3 in, BF16 out)
    gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY,
                                1, 1, 0, 0, false,
                                0, 0, 3, 0);

    // Step 2: Load scale factors for A (Q) and B (KT)
    gemmini_mx_load_scales((uint64_t)&Q_scales_row,  sizeof(Q_scales_row),  0);
    gemmini_mx_load_scales((uint64_t)&KT_scales_col, sizeof(KT_scales_col), 1);

    // Step 4: MVIN A tiles (Q) via mvin port, B tiles (KT) via mvin2 port
    // Configure stride for A: Q is [ATT_S][ATT_D], row stride = ATT_D bytes
    gemmini_extended3_config_ld(ATT_D * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0);
    // Configure stride for B: KT is [ATT_D][ATT_S], row stride = ATT_S bytes
    gemmini_extended3_config_ld(ATT_S * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1);

    // Interleave A and B tile loads using separate mvin ports
    for (int i = 0; i < tiles_I; i++) {
      for (int k = 0; k < tiles_K; k++) {
        gemmini_extended_mvin(
          (void*)(((elem_t*)Q_in) + (size_t)(i * DIM) * ATT_D + k * DIM),
          a_base + (i * tiles_K + k) * DIM,
          DIM, DIM);
      }
    }

    for (int j = 0; j < tiles_J; j++) {
      for (int k = 0; k < tiles_K; k++) {
        gemmini_extended_mvin2(
          (void*)(((elem_t*)KT_in) + (size_t)(k * DIM) * ATT_S + j * DIM),
          b_base + (k * tiles_J + j) * DIM,
          DIM, DIM);
      }
    }

    // Step 5: Configure output stride and requantizer
    gemmini_config_st(S_OUT_COLS * sizeof(out_t));
    gemmini_mxquant_config_mvout((uint64_t)scale_factors,
                                 tiles_I, tiles_J, tiles_K,
                                 0, 0, tiles_J);

    // Step 6: Execute fused tiled matmul (one-shot marker re-issued inside macro)
    gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K,
                         0, 0, 0,
                         a_base, BANK_NUM * BANK_ROWS,
                         0, SPAD_DEST,
                         false, false,
                         false, false, false,
                         NO_ACTIVATION, 0, 0, false, 0x38);

    // Step 7: Drain BF16 results from mx_smem to DRAM
    gemmini_mx_read_smem(&S1_hw[0][0], SPAD_DEST * 16, ATT_S * ATT_S);
    gemmini_fence();
  }

  // ---- 2) P = softmax(S1) on CPU; quantize to fp8 + per-32-group scales ----
  for (int i = 0; i < ATT_S; i++) {
    float row[ATT_S];

    // Pass 1: Unpack BF16 -> float, find maxv
    float maxv = -1e30f;
    for (int j = 0; j < ATT_S; j++) {
      uint16_t b = (uint16_t)((S1_hw[i][j / 4] >> ((j % 4) * 16)) & 0xFFFF);
      row[j] = bf16_to_f(b);
      if (row[j] > maxv) maxv = row[j];
    }

    // Pass 2: Compute exp(row[j] - maxv), accumulate sum
    float sum = 0.0f;
    for (int j = 0; j < ATT_S; j++) {
      row[j] = exp_f(row[j] - maxv);
      sum += row[j];
    }

    float inv_sum = 1.0f / sum;

    // Pass 3 (fused): normalize + per-group max scan + per-group quantize
    for (int g = 0; g < GROUPS_S; g++) {
      float gmax = 0.0f;
      for (int j = g * 32; j < (g + 1) * 32; j++) {
        row[j] *= inv_sum;
        float a = f_abs(row[j]);
        if (a > gmax) gmax = a;
      }
      int e = 0;
      if (gmax > 0.0f) e = f_exp(gmax / 16.0f) + 1;
      float inv_scale = pow2i(-e);
      P_scales[g][i] = (uint8_t)(e + 127);
      for (int j = g * 32; j < (g + 1) * 32; j++)
        P_q[i][j] = fp8_e4m3_rtz(row[j] * inv_scale);
    }
  }

  // ---- 3) O = P @ V on Gemmini (bf16 out) ----
  {
    const int tiles_I = ATT_S / DIM;
    const int tiles_J = ATT_D / DIM;
    const int tiles_K = ATT_S / DIM;
    const uint32_t a_base = 0;
    const uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;

    gemmini_flush(0);

    // Step 1: Configure execution (FP8 E4M3 in, BF16 out)
    gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY,
                                1, 1, 0, 0, false,
                                0, 0, 3, 0);

    // Step 2: Load scale factors for A (P) and B (V)
    gemmini_mx_load_scales((uint64_t)&P_scales, sizeof(P_scales), 0);
    gemmini_mx_load_scales((uint64_t)&V_scales_col, sizeof(V_scales_col), 1);

    // Step 4: MVIN A tiles (P_q) via mvin port, B tiles (V) via mvin2 port
    // Configure stride for A: P_q is [ATT_S][ATT_S], row stride = ATT_S bytes
    gemmini_extended3_config_ld(ATT_S * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0);
    // Configure stride for B: V is [ATT_S][ATT_D], row stride = ATT_D bytes
    gemmini_extended3_config_ld(ATT_D * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 1);

    // Load A tiles (P_q)
    for (int i = 0; i < tiles_I; i++) {
      for (int k = 0; k < tiles_K; k++) {
        gemmini_extended_mvin(
          (void*)(((elem_t*)P_q) + (size_t)(i * DIM) * ATT_S + k * DIM),
          a_base + (i * tiles_K + k) * DIM,
          DIM, DIM);
      }
    }

    // Load B tiles (V) via mvin2 port
    for (int j = 0; j < tiles_J; j++) {
      for (int k = 0; k < tiles_K; k++) {
        gemmini_extended_mvin2(
          (void*)(((elem_t*)V_in) + (size_t)(k * DIM) * ATT_D + j * DIM),
          b_base + (k * tiles_J + j) * DIM,
          DIM, DIM);
      }
    }

    // Step 5: Configure output stride and requantizer
    gemmini_config_st(D_OUT_COLS * sizeof(out_t));
    gemmini_mxquant_config_mvout((uint64_t)scale_factors,
                                 tiles_I, tiles_J, tiles_K,
                                 0, 0, tiles_J);

    // Step 6: Execute fused tiled matmul
    gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K,
                         0, 0, 0,
                         a_base, BANK_NUM * BANK_ROWS,
                         0, SPAD_DEST,
                         false, false,
                         false, false, false,
                         NO_ACTIVATION, 0, 0, false, 0x38);

    // Step 7: Drain BF16 results from mx_smem to DRAM
    gemmini_mx_read_smem(&O_hw[0][0], SPAD_DEST * 16, ATT_S * ATT_D);
    gemmini_fence();
  }
}
