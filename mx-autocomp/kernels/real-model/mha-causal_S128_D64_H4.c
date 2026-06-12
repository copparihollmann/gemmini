void solution(void) {

  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);

  gemmini_mx_load_scales((uint64_t)Qss[0], QSS_BYTES, 0);
  gemmini_mx_load_scales((uint64_t)KTss[0], QSS_BYTES, 1);

  for (int h = 0; h < ATT_H; h++) {
    int kv = h / (ATT_H / ATT_HKV);
    const elem_t *Q_in = Qs[h];  const uint8_t *Q_sc = Qss[h];
    const elem_t *KT_in = KTs[kv]; const uint8_t *KT_sc = KTss[kv];
    const elem_t *V_in = Vs[kv];  const uint8_t *V_sc = Vss[kv];

    {
      int tiles_I = ATT_S / DIM, tiles_J = ATT_S / DIM, tiles_K = ATT_D / DIM;
      uint32_t a_base = 0;
      uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
      gemmini_flush(0);
      
      gemmini_config_ld(ATT_D * sizeof(elem_t));
      for (int i = 0; i < tiles_I; i++)
        for (int k = 0; k < tiles_K; k += 2) { int bb = (tiles_K - k < 2) ? (tiles_K - k) : 2;
          gemmini_extended_mvin((void*)(((elem_t*)Q_in) + i * DIM * ATT_D + k * DIM),
                                a_base + (i * tiles_K + k) * DIM, bb * DIM, DIM); }
      gemmini_config_ld(ATT_S * sizeof(elem_t));
      for (int k = 0; k < tiles_K; k++)
        for (int j = 0; j < tiles_J; j += 2) { int bb = (tiles_J - j < 2) ? (tiles_J - j) : 2;
          gemmini_extended_mvin((void*)(((elem_t*)KT_in) + (k * DIM) * ATT_S + j * DIM),
                                b_base + (k * tiles_J + j) * DIM, bb * DIM, DIM); }
      gemmini_config_st(S_OUT_COLS * sizeof(out_t));
      gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);
      gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0,
                           SPAD_DEST, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);
      gemmini_mx_read_smem(&S1_hw[0][0], SPAD_DEST * 16, ATT_S * ATT_S);
      gemmini_fence();
    }

    for (int i = 0; i < ATT_S; i++) {
      int64_t q[ATT_S];
      int64_t mq = -((int64_t)1 << 62);
      
      for (int j = 0; j < ATT_S; j += 8) {
        uint64_t w0 = S1_hw[i][j / 4];
        uint64_t w1 = S1_hw[i][j / 4 + 1];

#if ATT_CAUSAL

        if (j > i) { q[j] = -((int64_t)1 << 62); } 
        else { int64_t qq = mx_ibf16_to_q((uint16_t)(w0 & 0xFFFF)); q[j] = qq; if (qq > mq) mq = qq; }
        
        if (j + 1 > i) { q[j + 1] = -((int64_t)1 << 62); } 
        else { int64_t qq = mx_ibf16_to_q((uint16_t)((w0 >> 16) & 0xFFFF)); q[j + 1] = qq; if (qq > mq) mq = qq; }
        
        if (j + 2 > i) { q[j + 2] = -((int64_t)1 << 62); } 
        else { int64_t qq = mx_ibf16_to_q((uint16_t)((w0 >> 32) & 0xFFFF)); q[j + 2] = qq; if (qq > mq) mq = qq; }
        
        if (j + 3 > i) { q[j + 3] = -((int64_t)1 << 62); } 
        else { int64_t qq = mx_ibf16_to_q((uint16_t)((w0 >> 48) & 0xFFFF)); q[j + 3] = qq; if (qq > mq) mq = qq; }

        if (j + 4 > i) { q[j + 4] = -((int64_t)1 << 62); } 
        else { int64_t qq = mx_ibf16_to_q((uint16_t)(w1 & 0xFFFF)); q[j + 4] = qq; if (qq > mq) mq = qq; }
        
        if (j + 5 > i) { q[j + 5] = -((int64_t)1 << 62); } 
        else { int64_t qq = mx_ibf16_to_q((uint16_t)((w1 >> 16) & 0xFFFF)); q[j + 5] = qq; if (qq > mq) mq = qq; }
        
        if (j + 6 > i) { q[j + 6] = -((int64_t)1 << 62); } 
        else { int64_t qq = mx_ibf16_to_q((uint16_t)((w1 >> 32) & 0xFFFF)); q[j + 6] = qq; if (qq > mq) mq = qq; }
        
        if (j + 7 > i) { q[j + 7] = -((int64_t)1 << 62); } 
        else { int64_t qq = mx_ibf16_to_q((uint16_t)((w1 >> 48) & 0xFFFF)); q[j + 7] = qq; if (qq > mq) mq = qq; }
#else

        int64_t qq0 = mx_ibf16_to_q((uint16_t)(w0 & 0xFFFF)); q[j] = qq0; if (qq0 > mq) mq = qq0;
        int64_t qq1 = mx_ibf16_to_q((uint16_t)((w0 >> 16) & 0xFFFF)); q[j + 1] = qq1; if (qq1 > mq) mq = qq1;
        int64_t qq2 = mx_ibf16_to_q((uint16_t)((w0 >> 32) & 0xFFFF)); q[j + 2] = qq2; if (qq2 > mq) mq = qq2;
        int64_t qq3 = mx_ibf16_to_q((uint16_t)((w0 >> 48) & 0xFFFF)); q[j + 3] = qq3; if (qq3 > mq) mq = qq3;

        int64_t qq4 = mx_ibf16_to_q((uint16_t)(w1 & 0xFFFF)); q[j + 4] = qq4; if (qq4 > mq) mq = qq4;
        int64_t qq5 = mx_ibf16_to_q((uint16_t)((w1 >> 16) & 0xFFFF)); q[j + 5] = qq5; if (qq5 > mq) mq = qq5;
        int64_t qq6 = mx_ibf16_to_q((uint16_t)((w1 >> 32) & 0xFFFF)); q[j + 6] = qq6; if (qq6 > mq) mq = qq6;
        int64_t qq7 = mx_ibf16_to_q((uint16_t)((w1 >> 48) & 0xFFFF)); q[j + 7] = qq7; if (qq7 > mq) mq = qq7;
#endif
      }

      int64_t p[ATT_S];
      int64_t sm = 0;
      int64_t gmax_arr[GROUPS_S];
      for (int g = 0; g < GROUPS_S; g++) {
        int64_t gmax = 0;
        for (int j = g * 32; j < (g + 1) * 32; j++) { int64_t pv = mx_iexp(q[j] - mq); p[j] = pv; sm += pv; if (pv > gmax) gmax = pv; }
        gmax_arr[g] = gmax;
      }
      int ls = mx_ilog2((uint64_t)sm);
      uint64_t inv_sm = (((uint64_t)1 << (ls + MX_FRAC)) + ((uint64_t)sm >> 1)) / (uint64_t)sm;
      for (int g = 0; g < GROUPS_S; g++) {
        int64_t gmax = gmax_arr[g];
        if (gmax == 0) { P_scales[g][i] = 0; for (int j = g * 32; j < (g + 1) * 32; j++) P_q[i][j] = 0; continue; }

        int e = mx_ilog2((uint64_t)gmax) - ls - MX_P_TARGET_LOG2;
        P_scales[g][i] = (uint8_t)(e + 127);
        for (int j = g * 32; j < (g + 1) * 32; j++) P_q[i][j] = mx_ienc_fp8(p[j], inv_sm, ls, e);
      }
    }

    {
      int tiles_I = ATT_S / DIM, tiles_J = ATT_D / DIM, tiles_K = ATT_S / DIM;
      uint32_t a_base = 0;
      uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
      gemmini_flush(0);
      
      gemmini_mx_load_scales((uint64_t)&P_scales, sizeof(P_scales), 0);
      gemmini_mx_load_scales((uint64_t)V_sc, VSS_BYTES, 1);
      gemmini_config_ld(ATT_S * sizeof(elem_t));
      for (int i = 0; i < tiles_I; i++)
        for (int k = 0; k < tiles_K; k += 2) { int bb = (tiles_K - k < 2) ? (tiles_K - k) : 2;
          gemmini_extended_mvin((void*)(((elem_t*)P_q) + i * DIM * ATT_S + k * DIM),
                                a_base + (i * tiles_K + k) * DIM, bb * DIM, DIM); }
      gemmini_config_ld(ATT_D * sizeof(elem_t));
      for (int k = 0; k < tiles_K; k++)
        for (int j = 0; j < tiles_J; j += 2) { int bb = (tiles_J - j < 2) ? (tiles_J - j) : 2;
          gemmini_extended_mvin((void*)(((elem_t*)V_in) + (k * DIM) * ATT_D + j * DIM),
                                b_base + (k * tiles_J + j) * DIM, bb * DIM, DIM); }
      gemmini_config_st(D_OUT_COLS * sizeof(out_t));
      gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);
      gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0,
                           SPAD_DEST, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);
      gemmini_mx_read_smem(&O_hw[h][0][0], SPAD_DEST * 16, ATT_S * ATT_D);

      if (h + 1 < ATT_H) {
        int next_kv = (h + 1) / (ATT_H / ATT_HKV);
        gemmini_mx_load_scales((uint64_t)Qss[h+1], QSS_BYTES, 0);
        gemmini_mx_load_scales((uint64_t)KTss[next_kv], QSS_BYTES, 1);
      }

      gemmini_fence();
    }
  }
}
