void solution(void) {
  /* ── Phase 1 : Q × KT → S1 (fp6 e3m2, S=128, D=64) ─────────────────── */
  {
    const int ti = ATT_S / DIM / 2;   /* 4 */
    const int tj = ATT_S / DIM / 2;   /* 4 */
    const int tk = ATT_D / DIM;        /* 4 */
    const uint32_t ab = 0;
    const uint32_t bb = BANK_NUM * BANK_ROWS - tk * tj * DIM;
    const int SD = 128; /* SPAD_DEST word offset */

    gemmini_flush(0);

    /* 1. Configure execution (fp6 e3m2: act=1, wgt=1, out=3/BF16, uselut=1) */
    gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY,
        1, 1, 0, 0, false, 1, 1, 3, 1);

    /* 2. Load scale factors: sel=0 → A (Q), sel=1 → B (KT) */
    gemmini_mx_load_scales((uint64_t)&Q_scales_row,  sizeof(Q_scales_row),  0);
    gemmini_mx_load_scales((uint64_t)&KT_scales_col, sizeof(KT_scales_col), 1);

    /* 3. Load LUTs: sel=0 → B (KT weight LUT), sel=1 → A (Q activation LUT) */
    gemmini_mx_load_lut((uint64_t)KT_lp, 1, 0);
    gemmini_mx_load_lut((uint64_t)Q_lp,  1, 1);

    /* 4. MVIN A tiles (Q): stride = ATT_D * sizeof(elem_t) = 64 bytes */
    gemmini_extended3_config_ld(ATT_D * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0);
    for (int i = 0; i < ti; i++) {
      for (int k = 0; k < tk; k += MAX_BLOCK_LEN) {
        int blk = (k + MAX_BLOCK_LEN <= tk) ? MAX_BLOCK_LEN : (tk - k);
        gemmini_extended_mvin(
            (void*)(((elem_t*)Q_hw) + i * DIM * ATT_D + k * DIM),
            ab + (i * tk + k) * DIM,
            blk * DIM, DIM);
      }
    }

    /* 4. MVIN B tiles (KT): KT stored transposed, row stride = ATT_S/2 * sizeof(elem_t)
         KT_in is fp6-packed: ATT_S/2 bytes per row of DIM elements */
    gemmini_extended3_config_ld(ATT_S * sizeof(elem_t) / 2, MVIN_SCALE_IDENTITY, false, 1);
    for (int k = 0; k < tk; k++) {
      for (int j = 0; j < tj; j += MAX_BLOCK_LEN) {
        int blk = (j + MAX_BLOCK_LEN <= tj) ? MAX_BLOCK_LEN : (tj - j);
        gemmini_extended_mvin2(
            (void*)(((elem_t*)KT_in) + k * DIM * (ATT_S / 2) + j * DIM),
            bb + (k * tj + j) * DIM,
            blk * DIM, DIM);
      }
    }

    /* 5. Configure output stride and requantizer */
    gemmini_config_st((ATT_S / BF16_PER_WORD) * sizeof(out_t));
    gemmini_mxquant_config_mvout((uint64_t)scale_factors,
        ti, tj, tk, 0, 0, 8);

    /* 6. Compute */
    gemmini_loop_ws_spad(ti, tj, tk, 0, 0, 0,
        ab, BANK_NUM * BANK_ROWS, 0, SD,
        false, false, false, false, false,
        NO_ACTIVATION, 0, 0, false, 0x38);

    /* 7. Drain results */
    gemmini_mx_read_smem(&S1_hw[0][0], SD * 16, ATT_S * ATT_S);
    gemmini_fence();
  }

  /* ── Softmax (CPU) ────────────────────────────────────────────────────── */
  for (int i = 0; i < ATT_S; i++) {
    float row[ATT_S];
    float mx = -1e30f;
    for (int j = 0; j < ATT_S; j++) {
      uint16_t b = (uint16_t)((S1_hw[i][j / 4] >> ((j % 4) * 16)) & 0xFFFF);
      row[j] = bf16f(b);
      if (row[j] > mx) mx = row[j];
    }
    float sm = 0.f;
    for (int j = 0; j < ATT_S; j++) {
      row[j] = expf2(row[j] - mx);
      sm += row[j];
    }
    for (int j = 0; j < ATT_S; j++) row[j] /= sm;

    for (int g = 0; g < GKP; g++) {
      float gm = 0.f;
      for (int j = g * 32; j < g * 32 + 32; j++) {
        float a = fabsf2(row[j]);
        if (a > gm) gm = a;
      }
      int e = 0;
      if (gm > 0.f) e = fexp2(gm / 0.5f) + 1;
      float isc = pow2i(-e);
      P_scales[g][i] = (uint8_t)(e + 127);
      for (int j = g * 32; j < g * 32 + 32; j++) {
        uint8_t idx = enc_P(row[j] * isc);
        if ((i & 1) == 0)
          P_hw[i >> 1][j] = (P_hw[i >> 1][j] & 0xF0) | idx;
        else
          P_hw[i >> 1][j] = (P_hw[i >> 1][j] & 0x0F) | (idx << 4);
      }
    }
  }

  /* ── Phase 2 : P × V → O (fp4/fp6, S=128, D=64) ─────────────────────── */
  {
    const int ti = ATT_S / DIM / 2;   /* 4 */
    const int tj = ATT_D / DIM / 2;   /* 2 */
    const int tk = ATT_S / DIM;        /* 8 */
    const uint32_t ab = 0;
    const uint32_t bb = BANK_NUM * BANK_ROWS - tk * tj * DIM;
    const int SD = 128; /* SPAD_DEST word offset */

    gemmini_flush(0);

    /* 1. Configure execution (fp6 e3m2: act=1, wgt=1, out=3/BF16, uselut=1) */
    gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY,
        1, 1, 0, 0, false, 1, 1, 3, 1);

    /* 2. Load scale factors: sel=0 → A (P), sel=1 → B (V) */
    gemmini_mx_load_scales((uint64_t)&P_scales,    sizeof(P_scales),    0);
    gemmini_mx_load_scales((uint64_t)&V_scales_col, sizeof(V_scales_col), 1);

    /* 3. Load LUTs: sel=0 → B (V weight LUT), sel=1 → A (P activation LUT) */
    gemmini_mx_load_lut((uint64_t)V_lp, 1, 0);
    gemmini_mx_load_lut((uint64_t)P_lp, 1, 1);

    /* 4. MVIN A tiles (P): stride = ATT_S * sizeof(elem_t) = 128 bytes
         P_hw is fp4-packed: ATT_S bytes wide (2 fp4 per byte → ATT_S/2 bytes,
         but original uses ATT_S*sizeof(elem_t) for stride) */
    gemmini_extended3_config_ld(ATT_S * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, 0);
    for (int i = 0; i < ti; i++) {
      for (int k = 0; k < tk; k += MAX_BLOCK_LEN) {
        int blk = (k + MAX_BLOCK_LEN <= tk) ? MAX_BLOCK_LEN : (tk - k);
        gemmini_extended_mvin(
            (void*)(((elem_t*)P_hw) + i * DIM * ATT_S + k * DIM),
            ab + (i * tk + k) * DIM,
            blk * DIM, DIM);
      }
    }

    /* 4. MVIN B tiles (V): stride = ATT_D/2 * sizeof(elem_t) */
    gemmini_extended3_config_ld(ATT_D * sizeof(elem_t) / 2, MVIN_SCALE_IDENTITY, false, 1);
    for (int k = 0; k < tk; k++) {
      for (int j = 0; j < tj; j += MAX_BLOCK_LEN) {
        int blk = (j + MAX_BLOCK_LEN <= tj) ? MAX_BLOCK_LEN : (tj - j);
        gemmini_extended_mvin2(
            (void*)(((elem_t*)V_in) + k * DIM * (ATT_D / 2) + j * DIM),
            bb + (k * tj + j) * DIM,
            blk * DIM, DIM);
      }
    }

    /* 5. Configure output stride and requantizer */
    gemmini_config_st((ATT_D / BF16_PER_WORD) * sizeof(out_t));
    gemmini_mxquant_config_mvout((uint64_t)scale_factors,
        ti, tj, tk, 0, 0, 8);

    /* 6. Compute */
    gemmini_loop_ws_spad(ti, tj, tk, 0, 0, 0,
        ab, BANK_NUM * BANK_ROWS, 0, SD,
        false, false, false, false, false,
        NO_ACTIVATION, 0, 0, false, 0x38);

    /* 7. Drain results */
    gemmini_mx_read_smem(&O_hw[0][0], SD * 16, ATT_S * ATT_D);
    gemmini_fence();
  }
}
