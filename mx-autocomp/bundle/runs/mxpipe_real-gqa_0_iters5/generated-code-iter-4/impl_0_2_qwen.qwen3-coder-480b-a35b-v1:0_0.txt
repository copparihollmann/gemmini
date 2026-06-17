#include <stdint.h>
#include "include/gemmini_testutils.h"

// AUTO-GENERATED MHA kernel with optimized RISC-V Bitmanip BF16 unpacking.
void solution(void) {
  for (int h = 0; h < ATT_H; h++) {
    int kv = h / (ATT_H / ATT_HKV);
    const elem_t *Q_in = Qs[h];  const uint8_t *Q_sc = Qss[h];
    const elem_t *KT_in = KTs[kv]; const uint8_t *KT_sc = KTss[kv];
    const elem_t *V_in = Vs[kv];  const uint8_t *V_sc = Vss[kv];

    // ---- 1) S1 = Q @ K^T on Gemmini (bf16 out) ----
    {
      int tiles_I = ATT_S / DIM, tiles_J = ATT_S / DIM, tiles_K = ATT_D / DIM;
      uint32_t a_base = 0;
      uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
      gemmini_flush(0);
      gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);
      gemmini_mx_load_scales((uint64_t)Q_sc, QSS_BYTES, 0);
      gemmini_mx_load_scales((uint64_t)KT_sc, QSS_BYTES, 1);
      gemmini_config_ld(ATT_D * sizeof(elem_t));
      for (int i = 0; i < tiles_I; i++) {
        for (int k = 0; k < tiles_K; k += 2) {
          int bb = (tiles_K - k < 2) ? (tiles_K - k) : 2;
          gemmini_extended_mvin((void*)(((elem_t*)Q_in) + i * DIM * ATT_D + k * DIM),
                                a_base + (i * tiles_K + k) * DIM, bb * DIM, DIM);
        }
      }
      gemmini_config_ld(ATT_S * sizeof(elem_t));   // B=KT[D][S] stride N=S
      for (int k = 0; k < tiles_K; k++) {
        for (int j = 0; j < tiles_J; j += 2) {
          int bb = (tiles_J - j < 2) ? (tiles_J - j) : 2;
          gemmini_extended_mvin((void*)(((elem_t*)KT_in) + (k * DIM) * ATT_S + j * DIM),
                                b_base + (k * tiles_J + j) * DIM, bb * DIM, DIM);
        }
      }
      gemmini_config_st(S_OUT_COLS * sizeof(out_t));
      gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);
      gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0,
                           SPAD_DEST, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);
      gemmini_mx_read_smem(&S1_hw[0][0], SPAD_DEST * 16, ATT_S * ATT_S);
      gemmini_fence();
    }

    // ---- 2) FLOAT-FREE integer softmax + fp8 requant (I-BERT iexp) ----
    for (int i = 0; i < ATT_S; i++) {
      int64_t q[ATT_S];
      int64_t mq = -((int64_t)1 << 62);
      for (int j = 0; j < ATT_S; j += 4) {
        uint64_t w = S1_hw[i][j / 4];
        uint64_t r0, r1, r2, r3;

        // Perform parallel extraction of 4x 16-bit halfwords from the 64-bit value using Bitmanip 'zext.h'
        asm (
            "zext.h %0, %4\n\t"
            "srli %1, %4, 16\n\t"
            "zext.h %1, %1\n\t"
            "srli %2, %4, 32\n\t"
            "zext.h %2, %2\n\t"
            "srli %3, %4, 48"
            : "=&r"(r0), "=&r"(r1), "=&r"(r2), "=&r"(r3)
            : "r"(w)
        );

        // Process unpacked values (l = 0)
        {
          int col = j + 0;
#if ATT_CAUSAL
          if (col > i) {
            q[col] = -((int64_t)1 << 62);
          } else {
#endif
            int64_t qq = mx_ibf16_to_q((uint16_t)r0);
            q[col] = qq;
            if (qq > mq) mq = qq;
#if ATT_CAUSAL
          }
#endif
        }

        // Process unpacked values (l = 1)
        {
          int col = j + 1;
#if ATT_CAUSAL
          if (col > i) {
            q[col] = -((int64_t)1 << 62);
          } else {
#endif
            int64_t qq = mx_ibf16_to_q((uint16_t)r1);
            q[col] = qq;
            if (qq > mq) mq = qq;
#if ATT_CAUSAL
          }
#endif
        }

        // Process unpacked values (l = 2)
        {
          int col = j + 2;
#if ATT_CAUSAL
          if (col > i) {
            q[col] = -((int64_t)1 << 62);
          } else {
#endif
            int64_t qq = mx_ibf16_to_q((uint16_t)r2);
            q[col] = qq;
            if (qq > mq) mq = qq;
#if ATT_CAUSAL
          }
#endif
        }

        // Process unpacked values (l = 3)
        {
          int col = j + 3;
#if ATT_CAUSAL
          if (col > i) {
            q[col] = -((int64_t)1 << 62);
          } else {
#endif
            int64_t qq = mx_ibf16_to_q((uint16_t)r3);
            q[col] = qq;
            if (qq > mq) mq = qq;
#if ATT_CAUSAL
          }
#endif
        }
      }

      int64_t p[ATT_S];
      int64_t sm = 0;
      int64_t gmax_arr[GROUPS_S];
      for (int g = 0; g < GROUPS_S; g++) {
        int64_t gmax = 0;
        for (int j = g * 32; j < (g + 1) * 32; j++) {
          int64_t pv = mx_iexp(q[j] - mq);
          p[j] = pv;
          sm += pv;
          if (pv > gmax) gmax = pv;
        }
        gmax_arr[g] = gmax;
      }
      int ls = mx_ilog2((uint64_t)sm);
      uint64_t inv_sm = (((uint64_t)1 << (ls + MX_FRAC)) + ((uint64_t)sm >> 1)) / (uint64_t)sm;
      for (int g = 0; g < GROUPS_S; g++) {
        int64_t gmax = gmax_arr[g];
        if (gmax == 0) {
          P_scales[g][i] = 0;
          for (int j = g * 32; j < (g + 1) * 32; j++) P_q[i][j] = 0;
          continue;
        }
        int e = mx_ilog2((uint64_t)gmax) - ls - MX_P_TARGET_LOG2;
        P_scales[g][i] = (uint8_t)(e + 127);
        for (int j = g * 32; j < (g + 1) * 32; j++) {
          P_q[i][j] = mx_ienc_fp8(p[j], inv_sm, ls, e);
        }
      }
    }

    // ---- 3) O = P @ V on Gemmini (bf16 out) ----
    {
      int tiles_I = ATT_S / DIM, tiles_J = ATT_D / DIM, tiles_K = ATT_S / DIM;
      uint32_t a_base = 0;
      uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;
      gemmini_flush(0);
      gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);
      gemmini_mx_load_scales((uint64_t)&P_scales, sizeof(P_scales), 0);
      gemmini_mx_load_scales((uint64_t)V_sc, VSS_BYTES, 1);
      gemmini_config_ld(ATT_S * sizeof(elem_t));
      for (int i = 0; i < tiles_I; i++) {
        for (int k = 0; k < tiles_K; k += 2) {
          int bb = (tiles_K - k < 2) ? (tiles_K - k) : 2;
          gemmini_extended_mvin((void*)(((elem_t*)P_q) + i * DIM * ATT_S + k * DIM),
                                a_base + (i * tiles_K + k) * DIM, bb * DIM, DIM);
        }
      }
      gemmini_config_ld(ATT_D * sizeof(elem_t));   // B=V[S][D] stride N=D
      for (int k = 0; k < tiles_K; k++) {
        for (int j = 0; j < tiles_J; j += 2) {
          int bb = (tiles_J - j < 2) ? (tiles_J - j) : 2;
          gemmini_extended_mvin((void*)(((elem_t*)V_in) + (k * DIM) * ATT_D + j * DIM),
                                b_base + (k * tiles_J + j) * DIM, bb * DIM, DIM);
        }
      }
      gemmini_config_st(D_OUT_COLS * sizeof(out_t));
      gemmini_mxquant_config_mvout((uint64_t)scale_factors, tiles_I, tiles_J, tiles_K, 0, 0, 1);
      gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0,
                           SPAD_DEST, false, false, false, false, false, NO_ACTIVATION, 0, 0, false, 0x38);
      gemmini_mx_read_smem(&O_hw[h][0][0], SPAD_DEST * 16, ATT_S * ATT_D);
      gemmini_fence();
    }
  }
}
