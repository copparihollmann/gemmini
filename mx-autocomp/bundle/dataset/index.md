# MX-Gemmini autocomp run index

60 MX runs | 1133 total attempts | 42,045,874 input + 8,726,474 output tokens | $149.17 run-attributed cost.

Cycle counts are under each run's metric (spike / faithful-cycles). Spike speedups are inflated ~1.5-2x vs RTL; see results/MX_AUTOCOMP_RESULTS.md for the silicon-true RTL numbers.

| family | prob_type | prob_id | shape_MxKxN | metric | baseline_cyc | best_cyc | speedup | num_iters | num_attempts | input_tokens | output_tokens | cost_usd | run_dir |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| autocomp-generated | gemmini-mx-attention | 0 | S128xD64 | latency | 762147 | 672193 | 1.1338 | 2 | 6 | 309216 | 32448 | 1.4144 | mxpipe_gemmini-mx-attention_0_iters2 |
| autocomp-generated | gemmini-mx-attention | 0 | S128xD64 | latency | 824009 | 791111 | 1.0416 | 5 | 27 | 383570 | 35301 | 1.6802 | mxpipe_gemmini-mx-attention_0_iters5 |
| autocomp-generated | gemmini-mx-attention | 0 | S128xD64 | latency | 762147 | 672193 | 1.1338 | 5 | 33 | 698261 | 87030 | 3.4002 | mxpipe_gemmini-mx-attention_0_iters6 |
| autocomp-generated | gemmini-mx-attn-fp4 | 0 | S128xD64 | latency | 1229275 | 1229275 | 1.0 | 2 | 6 | 154557 | 16049 | 0.7044 | mxpipe_gemmini-mx-attn-fp4_0_iters2 |
| autocomp-generated | gemmini-mx-attn-fp6 | 0 | S128xD64 | latency | 1230247 | 1230247 | 1.0 | 2 | 4 | 136954 | 13889 | 0.6192 | mxpipe_gemmini-mx-attn-fp6_0_iters2 |
| autocomp-generated | gemmini-mx-attn-fp6 | 0 | S128xD64 | latency | 6546779 | 6514161 | 1.005 | 5 | 24 | 355212 | 33299 | 1.5651 | mxpipe_gemmini-mx-attn-fp6_0_iters5 |
| autocomp-generated | gemmini-mx-conv | 0 | 2x32x4x32 (NCHW patch) | latency | 8210 | 7963 | 1.031 | 2 | 6 | 260561 | 21384 | 1.1024 | mxpipe_gemmini-mx-conv_0_iters2 |
| autocomp-generated | gemmini-mx-conv | 0 | 2x32x4x32 (NCHW patch) | latency | 8210 | 6226 | 1.3187 | 4 | 21 | 839543 | 92783 | 3.9104 | mxpipe_gemmini-mx-conv_0_iters4 |
| autocomp-generated | gemmini-mx-conv | 1 | 2x32x4x32 (NCHW patch) | latency | 15611 | 15611 | 1.0 | 2 | 4 | 180904 | 15974 | 0.7823 | mxpipe_gemmini-mx-conv_1_iters2 |
| autocomp-generated | gemmini-mx-conv | 1 | 2x32x4x32 (NCHW patch) | latency | 15611 | 15576 | 1.0022 | 4 | 21 | 390271 | 29590 | 1.6147 | mxpipe_gemmini-mx-conv_1_iters4 |
| autocomp-generated | gemmini-mx-flash-attn | 0 | S128xD64 | latency | 1581085 | 1581085 | 1.0 | 2 | 4 | 224106 | 23700 | 1.0278 | mxpipe_gemmini-mx-flash-attn_0_iters2 |
| autocomp-generated | gemmini-mx-flash-attn | 0 | S128xD64 | latency | 1316934 | 1316576 | 1.0003 | 5 | 21 | 331376 | 38380 | 1.5698 | mxpipe_gemmini-mx-flash-attn_0_iters5 |
| autocomp-generated | gemmini-mx-flash-fp4 | 0 | S128xD64 | latency | 1855175 | 1855175 | 1.0 | 2 | 4 | 123443 | 13472 | 0.5724 | mxpipe_gemmini-mx-flash-fp4_0_iters2 |
| autocomp-generated | gemmini-mx-flash-fp6 | 0 | S128xD64 | latency | 1834414 |  |  | 0 | 0 |  |  |  | mxpipe_gemmini-mx-flash-fp6_0_iters2 |
| autocomp-generated | gemmini-mx-flash-fp6 | 0 | S128xD64 | latency | 7098437 | 7098437 | 1.0 | 5 | 15 | 303768 | 34296 | 1.4257 | mxpipe_gemmini-mx-flash-fp6_0_iters5 |
| autocomp-generated | gemmini-mx-matmul | 0 |  | latency | 294 | 256 | 1.1484 | 2 | 4 | 185927 | 10602 | 0.7168 | mxpipe_gemmini-mx-matmul_0_iters2 |
| autocomp-generated | gemmini-mx-matmul | 0 |  | latency | 294 | 92 | 3.1957 | 4 | 21 | 842281 | 54614 | 3.3461 | mxpipe_gemmini-mx-matmul_0_iters4 |
| autocomp-generated | gemmini-mx-matmul | 0 |  | latency | 279 | 92 | 3.0326 | 5 | 27 | 1099979 | 94055 | 4.7108 | mxpipe_gemmini-mx-matmul_0_iters5 |
| autocomp-generated | gemmini-mx-matmul | 1 |  | latency | 96 | 95 | 1.0105 | 2 | 6 | 261044 | 16289 | 1.0275 | mxpipe_gemmini-mx-matmul_1_iters2 |
| autocomp-generated | gemmini-mx-matmul | 1 |  | latency | 96 | 70 | 1.3714 | 4 | 21 | 847578 | 59613 | 3.4369 | mxpipe_gemmini-mx-matmul_1_iters4 |
| autocomp-generated | gemmini-mx-matmul | 1 |  | latency | 97 | 73 | 1.3288 | 5 | 27 | 372067 | 29701 | 1.5617 | mxpipe_gemmini-mx-matmul_1_iters5 |
| autocomp-generated | gemmini-mx-matmul | 2 |  | latency | 673 | 616 | 1.0925 | 2 | 6 | 275094 | 21466 | 1.1473 | mxpipe_gemmini-mx-matmul_2_iters2 |
| autocomp-generated | gemmini-mx-matmul | 2 |  | latency | 673 | 124 | 5.4274 | 4 | 21 | 845691 | 61585 | 3.4608 | mxpipe_gemmini-mx-matmul_2_iters4 |
| autocomp-generated | gemmini-mx-matmul | 2 |  | latency | 611 | 166 | 3.6807 | 5 | 24 | 366020 | 31531 | 1.571 | mxpipe_gemmini-mx-matmul_2_iters5 |
| autocomp-generated | gemmini-mx-matmul | 3 |  | latency | 1277 | 1277 | 1.0 | 2 | 4 | 187359 | 12053 | 0.7429 | mxpipe_gemmini-mx-matmul_3_iters2 |
| autocomp-generated | gemmini-mx-matmul | 3 |  | latency | 1277 | 1097 | 1.1641 | 4 | 21 | 847767 | 51621 | 3.3176 | mxpipe_gemmini-mx-matmul_3_iters4 |
| autocomp-generated | gemmini-mx-matmul | 3 |  | latency | 1149 | 382 | 3.0079 | 5 | 27 | 401656 | 32043 | 1.6856 | mxpipe_gemmini-mx-matmul_3_iters5 |
| autocomp-generated | gemmini-mx-matmul-fp4 | 0 | 128x128x256 | latency | 675 | 163 | 4.1411 | 5 | 21 | 335400 | 31307 | 1.4758 | mxpipe_gemmini-mx-matmul-fp4_0_iters5 |
| autocomp-generated | gemmini-mx-matmul-fp4 | 0 | 128x128x256 | faithful-cycles | 18909 | 18698 | 1.0113 | 7 | 26 | 1323285 | 531980 | 3.5779 | mxpipe_gemmini-mx-matmul-fp4_0_iters7 |
| autocomp-generated | gemmini-mx-matmul-fp6 | 0 | 128x128x256 | latency | 680 | 168 | 4.0476 | 5 | 24 | 349459 | 24518 | 1.4161 | mxpipe_gemmini-mx-matmul-fp6_0_iters5 |
| autocomp-generated | gemmini-mx-matmul-fp6 | 0 | 128x128x256 | faithful-cycles | 18914 | 18703 | 1.0113 | 7 | 26 | 1236412 | 579048 | 3.6219 | mxpipe_gemmini-mx-matmul-fp6_0_iters7 |
| autocomp-generated | gemmini-mx-matmul-tiled | 0 | 128x128x128 | latency | 2016019 | 2016019 | 1.0 | 4 | 12 | 186020 | 17292 | 0.8174 | mxpipe_gemmini-mx-matmul-tiled_0_iters4 |
| autocomp-generated | gemmini-mx-matmul-tiled | 0 | 128x128x128 | latency | 1619689 | 1619689 | 1.0 | 4 | 12 | 249091 | 20902 | 1.0608 | mxpipe_gemmini-mx-matmul-tiled_0_iters5 |
| autocomp-generated | gemmini-mx-matmul-tiled | 1 | 128x128x128 | latency | 16745669 | 16700492 | 1.0027 | 4 | 18 | 149034 | 12616 | 0.6363 | mxpipe_gemmini-mx-matmul-tiled_1_iters4 |
| golden-fork | fork-mm-fp8-128 | 0 | 128x128x128 | latency | 853 | 217 | 3.9309 | 3 | 15 | 205084 | 15449 | 0.847 | mxpipe_fork-mm-fp8-128_0_iters6 |
| golden-fork | fork-mm-fp8-256 | 0 | 256x256x256 | latency | 1607 | 442 | 3.6357 | 6 | 33 | 1392710 | 111887 | 5.8564 | mxpipe_fork-mm-fp8-256_0_iters6 |
| real-model | real-attn-causal | 0 | S128xD64 causal | latency | 651887 | 498946 | 1.3065 | 5 | 24 | 1274748 | 391285 | 3.112 | mxpipe_real-attn-causal_0_iters5 |
| real-model | real-bitvla | 0 | 32x256x256 | latency | 1596 |  |  | 0 | 0 |  |  |  | mxpipe_real-bitvla_0_iters1 |
| real-model | real-bitvla | 0 | 32x256x256 | latency | 1596 | 1004 | 1.5896 | 5 | 24 | 1128116 | 362055 | 2.6932 | mxpipe_real-bitvla_0_iters5 |
| real-model | real-gemv-1024 | 0 | 1x1024x1024 | latency | 61441 | 52014 | 1.1812 | 5 | 27 | 1244458 | 355218 | 2.7987 | mxpipe_real-gemv-1024_0_iters5 |
| real-model | real-gemv-3072 | 0 | 1x1024x3072 | latency | 184241 | 184241 | 1.0 | 5 | 15 | 733235 | 305283 | 2.0729 | mxpipe_real-gemv-3072_0_iters5 |
| real-model | real-gqa | 0 | S64xD64xH4kv2 | latency | 786929 | 759108 | 1.0366 | 5 | 24 | 1282094 | 454433 | 3.4056 | mxpipe_real-gqa_0_iters5 |
| real-model | real-groot-patch | 0 | 41x512x512 | latency | 17428 | 17428 | 1.0 | 3 | 6 | 291863 | 25252 | 1.2544 | mxpipe_real-groot-patch_0_iters3 |
| real-model | real-groot-patch | 0 | 41x512x512 | latency | 17428 | 10678 | 1.6321 | 5 | 36 | 1506262 | 754064 | 9.2144 | mxpipe_real-groot-patch_0_iters5 |
| real-model | real-groot-patch | 0 | 41x512x512 | latency | 17428 | 15341 | 1.136 | 6 | 45 | 2473252 | 248726 | 10.0356 | mxpipe_real-groot-patch_0_iters6 |
| real-model | real-llm-attn | 0 | 8x512x512 | latency | 16048 | 14000 | 1.1463 | 3 | 8 | 459440 | 40443 | 1.985 | mxpipe_real-llm-attn_0_iters3 |
| real-model | real-llm-attn | 0 | 8x512x512 | latency | 16048 | 8760 | 1.832 | 5 | 30 | 1362921 | 332663 | 2.8144 | mxpipe_real-llm-attn_0_iters5 |
| real-model | real-llm-attn | 0 | 8x512x512 | latency | 16048 | 14000 | 1.1463 | 2 | 12 | 200617 | 22280 | 0.8424 | mxpipe_real-llm-attn_0_iters6 |
| real-model | real-mha | 0 | S64xD64xH4 | latency | 787770 | 664624 | 1.1853 | 5 | 27 | 1422858 | 535921 | 3.9684 | mxpipe_real-mha_0_iters5 |
| real-model | real-mha-causal | 0 | S64xD64xH4 causal | latency | 738594 | 638750 | 1.1563 | 5 | 24 | 1236074 | 448464 | 3.2902 | mxpipe_real-mha-causal_0_iters5 |
| real-model | real-pi05-ffn | 0 | 256x256x512 | latency | 43239 | 31829 | 1.3585 | 2 | 6 | 285814 | 20924 | 1.1713 | mxpipe_real-pi05-ffn_0_iters3 |
| real-model | real-pi05-ffn | 0 | 256x256x512 | latency | 43239 | 20871 | 2.0717 | 5 | 30 | 1378612 | 412218 | 3.1903 | mxpipe_real-pi05-ffn_0_iters5 |
| real-model | real-pi05-ffn | 0 | 256x256x512 | faithful-cycles | 427143 | 341347 | 1.2513 | 7 | 28 | 1413528 | 425677 | 3.2355 | mxpipe_real-pi05-ffn_0_iters7 |
| real-model | real-smolvla-ffn | 0 | 50x256x512 | latency | 10305 | 10298 | 1.0007 | 3 | 10 | 567884 | 53802 | 2.5107 | mxpipe_real-smolvla-ffn_0_iters3 |
| real-model | real-smolvla-ffn | 0 | 50x256x512 | latency | 10305 | 4974 | 2.0718 | 5 | 30 | 1397261 | 388965 | 3.058 | mxpipe_real-smolvla-ffn_0_iters5 |
| real-model | real-smolvla-ffn | 0 | 50x256x512 | latency | 10305 | 10298 | 1.0007 | 6 | 36 | 992664 | 96361 | 3.9811 | mxpipe_real-smolvla-ffn_0_iters6 |
| real-model | real-smolvla-vis | 0 | 113x256x256 | latency | 10608 | 10608 | 1.0 | 3 | 6 | 361905 | 34428 | 1.6021 | mxpipe_real-smolvla-vis_0_iters3 |
| real-model | real-smolvla-vis | 0 | 113x256x256 | latency | 10608 | 5659 | 1.8745 | 5 | 30 | 1356075 | 412431 | 3.0882 | mxpipe_real-smolvla-vis_0_iters5 |
| real-model | real-smolvla-vis | 0 | 113x256x256 | latency | 10608 | 7743 | 1.37 | 6 | 48 | 2476055 | 251709 | 10.0834 | mxpipe_real-smolvla-vis_0_iters6 |
| real-model | smolvla-act | 0 | 128x320x320 | latency | 15922 | 15922 | 1.0 | 5 | 15 | 549438 | 46105 | 2.3399 | mxpipe_smolvla-act_0_iters5 |
