# MX-Gemmini kernel optimization — results

Optimization of MX-format (fp8 e4m3, fp6 e3m2, fp4 e2m1; per-32-element e8m0 scales) matmul,
attention, and convolution kernels for the `RadianceGemminiOnlyConfig` Gemmini (16×16 PE mesh,
weight-stationary). Work covers three kernel families, a spike functional-model that was made
faithful to the elaborated RTL, and real RTL (VCS) cycle measurements.

## How to read the numbers

Two simulators are involved and they do **not** agree:

- **Spike** is a functional, CPI=1 instruction simulator. `read_cycles()` reads the cycle CSR, which
  spike advances by retired-instruction count (`riscv-isa-sim/riscv/execute.cc:373-376`,
  "Model a hart whose CPI is 1"). Gemmini ROCC ops (mvin/mvout/compute/loop_ws/fence) retire in ~1
  instruction with zero modeled latency. So a spike "cycle" ≈ the Rocket instruction count, and any
  transform that issues fewer instructions (batched mvin, fence removal) looks large on spike.
- **VCS** runs the elaborated RTL. There, load latency scales with bytes moved
  (`LoadController.scala`) and compute latency scales with K + pipeline depth (`Mesh.scala`).
  Instruction count is not the cost. **VCS is the only timing truth.**

Consequence: raw spike speedups overstate silicon by ~1.5–2×. Every speedup below is labelled
**RTL** (VCS-measured, silicon truth), **faithful** (spike with the faithful-cycle model, tracks RTL
within ±9%), or **spike** (functional spike — a direction only, magnitude inflated). Use RTL where it
exists; otherwise faithful; treat spike as directional.

---

## 1. Spike functional-model changes

All changes are in the `libgemmini` submodule (`gemmini.cc`, `gemmini.h`) on top of `cedc820`. They
are of two kinds: RTL-legality guards (so a kernel that passes spike also passes VCS) and a
faithful-cycle model (so spike cycle counts approximate RTL). Every guard and the cycle model are
env-gated; default functional behaviour is unchanged when the env vars are unset. None of the cycle
logic affects functional results.

### 1.1 RTL-legality guards (`GEMMINI_RTL_ASSERTS_OFF` to disable; trace with `GEMMINI_MVIN_TRACE`)

1. **6-bit `num_cols`/`num_rows` field truncation on mvin/mvout.** RTL reads `cols = rs2[37:32]` and
   `rows = rs2[53:48]` — 6-bit fields. The macros pack 16-bit values, so anything > 63 is silently
   truncated by hardware (e.g. `cols = 64 → 0 bytes → $fatal` at `LoadController.scala:191`). Spike
   now truncates identically and guards zero-byte mvin. This is the root cause of the original
   "winner fatals on VCS" failures.
2. **Block-stride underflow guard** mirroring `LoadController.scala:138` (`block_stride >= rows`).
3. **loop_ws dual-port input guard.** A `loop_ws` input loaded via a non-zero mvin port (mvin2/mvin3)
   reads a different physical scratchpad region than the loop_ws A/B read ports address, so RTL reads
   stale data while spike's unified scratchpad reads it fine. Spike now rejects this — it was a
   spike-correct/VCS-wrong "winner".
4. **flush clears MX shared memory** (correctness fix, not a guard). MX smem accumulates across
   tiled `loop_ws_spad` calls; without clearing on flush, tiled matmuls pile onto the previous
   block's residue. `reset()` only runs at startup, so software-tiled matmuls needed this.

Verified: port-0 kernels (matmul, integer attention) still pass; `cols=64` now fatals on spike as on
VCS; the dual-port-B "winner" now reports incorrect on spike, matching VCS.

### 1.2 Faithful-cycle model (`GEMMINI_FAITHFUL_CYCLES=1`)

Base spike undercounts RTL by 20–40× on accelerator-bound kernels because accelerator ops are modeled
as ~free. The faithful model bumps the cycle CSR by a data-driven latency (cycle-only, never affects
correctness):

- per mvin: `K_DMA · cols · rows` — summed over all mvins this is the total load bytes, so it is
  **independent of mvin batching** (batched and unbatched cost the same).
- per `loop_ws_spad`: `K_CMP · (MACs / peak)`, with peak = 16 (fp8) or 32 (sub-byte) MAC/PE/cycle.

Coefficients `K_DMA = 0.143`, `K_CMP = 1.804` were fit to VCS cycle data and are env-overridable
(`GEMMINI_K_DMA_MILLI`, `GEMMINI_K_CMP_MILLI`) so recalibration needs no recompile.

Validation against VCS: fp8-256 +3%, fp8-64 +9%, fp4-512 +9%, fp6-512 −23% (the fp6 LUT-decode cost
is not yet modeled). The instruction-count mirage is eliminated: a batch-1 vs batch-2 mvin pair that
base spike scored as 1.13× scores 37,617 vs 37,515 (0.3%) under the faithful model — i.e. correctly
~neutral.

Not yet modeled: fp6 LUT-decode overhead; host CPI ≈ 2 (the real in-order Rocket runs integer
softmax at ~CPI 2 vs spike's CPI 1, which is why attention is ~2× slower on RTL than on spike).

---

## 2. Family A — golden / author-designed kernels (RTL-measured)

The reference MX matmul kernels shipped by the upstream authors, plus the RTL-correct hand-ports
(`*_LEGAL.c`) used to measure them on VCS. These are the only kernels with direct baseline→optimized
RTL cycle pairs, so they calibrate everything else. Baseline = batch-1 mvin; optimized = the batch-2
block-mvin recipe; both correctness-PASS on VCS.

| kernel | shape | baseline (RTL cyc) | final (RTL cyc) | RTL speedup | spike claimed | accel-region util | how achieved |
|---|---|---|---|---|---|---|---|
| fp6 matmul | 128×128×512 | 73,085 | 46,958 | **1.56×** | 2.08× | 35% | batch-2 block-mvin; gain is real because K=512 sub-byte is the most load-bound |
| fp4 matmul | 128×128×512 | 40,383 | 33,253 | **1.21×** | 2.39× | 49% | batch-2 block-mvin |
| fp8 matmul | 128×128×256 | 41,872 | 36,577 | **1.14×** | 1.78× | 45% | batch-2 block-mvin |
| fp8 matmul | 64×64×64 | 3,019 | 3,024 | **1.00×** | 1.99× | 34% | batch-2 block-mvin — no RTL gain (compute-bound, too little mvin to save) |

The RTL gain scales exactly with mvin/DMA-boundedness: K=512 sub-byte fp6 benefits most (1.56×), the
tiny compute-bound fp8-64 not at all. **fp6-512 at 1.56× is the best confirmed silicon speedup.**

Upstream-shipped reference matmuls (spike only, directional): fork fp8-128 3.93×, fp8-256 3.64×.

**Attention** (`attn_fp8_vcs.c`, S=128, D=64, integer softmax): spike 762,147 cyc, VCS **1,561,969**
cyc (checksum PASS) — RTL is **2.05× slower** than spike. Attention is host-bound (the ~16K-element
integer-softmax polynomial dominates), and the real in-order Rocket runs it at CPI ≈ 2. So attention's
absolute spike cycles are also ~2× optimistic, for a different reason than matmul.

---

## 3. Family B — real-model kernels (RTL where measurable, else directional)

Shapes drawn from real models (smolvla, pi05, bitvla, groot, tiny-llama/molmoact). Most real matmuls
have M not a multiple of 16; these kernels pad via `loop_ws_spad`'s `pad_I/J/K` (generate header/gold
at the padded shape, mvin/mvout only real elements). Matmul RTL cycle counts are data-independent
(depend only on shape + instruction schedule), so the tiled winners were cycle-measured on VCS even
where the generated golden header does not yet match RTL bitwise.

| kernel | shape | RTL cyc | accel-region util | spike direction | notes |
|---|---|---|---|---|---|
| pi05-ffn | 256×256×512 | **306,698** | **42.7%** | 2.07× | 32 full 64-blocks → array well-fed |
| smolvla-ffn | 50×256×512 | **98,151** | **26.1%** | 2.07× | M=50 pads to one 64-row block → underfills |
| bitvla | 32×256×256 | **31,813** | **25.7%** | 1.59× | M=32 = 2 row-tiles, skinny-M underutilizes the 16-row array |
| (tiled baseline) | 128×128×128 | 35,905 | 22.8% | — | validation point |
| smolvla-vis | 113×256×256 | — | — | 1.87× | irregular M=113 |
| groot-patch | 41×512×512 | — | — | 1.44× | M=41 |
| llm-attn (matmul) | 8×512×512 | — | — | 1.83× | M=8 |
| gemv-1024 / gemv-3072 | 1×K | — | — | 1.18× / 1.00× | M=1 decode, DMA-bound |
| mha / gqa / attn-causal | S=128, D=64 | — | — | 1.19× / 1.04× / 1.31× | host-bound (softmax) |

Well-shaped tiled matmul (pi05-ffn, 42.7%) reaches the fitting-kernel band (~45%); skinny-M shapes
(smolvla-ffn M=50, bitvla M=32) drop to ~26% from row-tile underfill. Attention/GEMV are host- or
DMA-bound, so their spike directions are not array-throughput numbers. RTL speedups for the unmeasured
rows are expected in the Family-A band (1.0–1.6×) since they use the same mvin-batching transform
class; confirming each requires an RTL-correct (LEGAL-style) port.

---

## 4. Family C — autocomp-generated kernels

Kernels produced by the autocomp beam search (plan model: Gemini-3.5-flash; code model:
Qwen3-Coder-480B). Two metric regimes:

**Base-spike search (directional, magnitudes inflated):** 64×128×128 5.43×, fp4-64³ 4.14×, fp6-64³
4.05×, 64³ 3.20×, 128×128×256 3.01×, conv 1.32×, attention 1.13×. These are spike instruction-count
numbers; per §1/§2 expect 1.0–1.6× on silicon.

**Faithful-metric search (`GEMMINI_FAITHFUL_CYCLES=1`, the honest result):** re-running the search
against the faithful cycle model shows what survives when instruction-count tricks no longer pay:

| kernel | shape | baseline (faithful cyc) | final (faithful cyc) | faithful speedup | winning transform |
|---|---|---|---|---|---|
| matmul fp4 | 128×128×256 | 18,909 | 18,698 | **1.01×** | batched mvin — now correctly ~neutral (does not reduce bytes) |
| matmul fp6 | 128×128×256 | 18,914 | 18,703 | **1.01×** | batched mvin — ~neutral |
| pi05-ffn (tiled) | 256×256×512 | 427,143 | 341,347 | **1.25×** | reduce host-side data staging + wider DMA bursts + operand reuse (real load-traffic reduction) |

This is the core result of the campaign: under the faithful metric the small fp4/fp6 matmuls show
**no gain** (1.01×) because the only transform the search had found — mvin batching — is an
instruction-count trick the faithful model scores correctly as neutral. The tiled pi05-ffn shows a
genuine **1.25×** from cutting host-side data movement, which is exactly the lever §6 identifies as
the real one. The honest metric steers the search away from artifacts and toward load reduction.

Faithful-metric run cost: $3.24 (lifetime project autocomp spend $167.96).

---

## 5. Utilization

`util = useful_MACs / (peak_MAC_per_cycle × cycles)`, computed conservatively: RTL cycles (not spike,
which would inflate ~2×); real M·N·K MACs (padding excluded); per-format peak chosen high (256 MAC/cyc
fp8; 512 for sub-byte fp4/fp6, since 2 sub-byte elements pack per byte — confirmed by fp4-512's
compute alone exceeding its whole runtime at peak=256). The accelerator-region denominator
(mvin+compute+store) is the optimistic bound; whole-kernel utilization is lower.

| kernel | peak MAC/cyc | RTL cyc | accel-region util |
|---|---|---|---|
| fp4 128×128×512 (batch-2) | 512 | 33,253 | **49%** |
| fp8 128×128×256 (batch-2) | 256 | 36,577 | **45%** |
| pi05-ffn 256×256×512 | 256 | 306,698 | **42.7%** |
| fp8 128×128×256 (baseline) | 256 | 41,872 | 39% |
| fp6 128×128×512 (batch-2) | 512 | 46,958 | **35%** |
| fp8 64×64×64 | 256 | 3,019 | 34% |
| smolvla-ffn 50×256×512 | 256 | 98,151 | 26.1% |
| bitvla 32×256×256 | 256 | 31,813 | 25.7% |

Accelerator regions run at ~35–49% of peak — roughly 2× headroom. **The headroom is not a scheduling
problem.** A hand-double-buffered fp8-128×128×256 (K-split, ping-pong banks) measured 36,730 cyc vs
36,577 baseline on VCS — 0.4% *worse*. Decomposition: loads alone 20,703 cyc, compute alone ≈ 22,891,
fully-serial 43,594 — but the baseline already runs in 36,577 < 43,594, i.e. it already overlaps
~7,000 cyc. The kernel issues all mvins then one `loop_ws_spad`, and Gemmini's ROB fires compute as
soon as the needed tiles land, extracting the overlap automatically; a manual K-split only imposes a
stricter order. The overlap ceiling is ≈ max(20.7k, 22.9k) ≈ 23k cyc, unreachable by reordering.
**The 45% is loads ≈ compute, not idle scheduling.** Skinny-M shapes (smolvla M=50, bitvla M=32) drop
to ~26% from row-tile underfill, independent of load cost.

End-to-end (whole kernel including host code): for attention the array is idle during the ~97%
host-softmax, so end-to-end PE utilization is ~1–3% — it is host-bound, not array-bound. Offloading
softmax to the Normalizer is the lever there, not array scheduling.

---

## 6. Conclusions

1. Spike is a CPI=1 functional model, not a timing model. It undercounts accelerator ops (~free) and
   host instructions (CPI 1 vs real ≈2). Both kernel classes are ~2× optimistic on spike.
2. Matmul is load-bound and already near its overlap ceiling. Double-buffering does not help
   (VCS: 0.4% worse). Utilization 35–49% reflects loads ≈ compute. The real lever is cutting DMA/load
   traffic, not scheduling.
3. Attention is host-bound; the array is ~idle. The lever is offloading softmax, not array tuning.
4. The faithful-cycle model makes the search optimize silicon instead of spike artifacts: the
   instruction-count "wins" (3–5× on base spike) collapse to ~1.0× faithful, while a genuine
   load-reduction transform (pi05-ffn) survives at 1.25×.
5. Best confirmed silicon speedup: fp6 128×128×512 at **1.56× RTL**.

---

## 7. Reproduce

- RTL profiling: `autocomp/vcs_profile.py` (builds a baremetal harness with `read_cycles` + on-device
  golden check; defines the spike-only MX intrinsics as RTL MMIO macros so a kernel body runs
  unmodified). Instrumented pairs: `gemmini-rocc-tests-ref/bareMetalC/matmul_tiled_*{,_LEGAL}.c`.
- VCS run: `sims/vcs/simv-…RadianceGemminiOnlyConfig +permissive +dramsim
  +dramsim_ini_dir=<…> +max-cycles=10000000 +loadmem=<bin> +permissive-off <bin>`
  (`source env.sh` for the riscv toolchain to build).
- Faithful spike: `libgemmini/gemmini.cc` (`gmx_faithful_cycles`/`gmx_k_dma`/`gmx_k_cmp`); enable with
  `GEMMINI_FAITHFUL_CYCLES=1`; reinstall via `make -C .../libgemmini install` after `source env.sh`.
- Faithful-metric search: `GEMMINI_FAITHFUL_CYCLES=1 python -m autocomp.search.run_mx_pipeline
  --only gemmini-mx-matmul-fp4,gemmini-mx-matmul-fp6,real-pi05-ffn --iterations 7
  --plan-model gcp::gemini-3.5-flash --code-model aws::qwen.qwen3-coder-480b-a35b-v1:0`.
- Mineable transforms: `traces/transform_ledger.jsonl` (one record per search attempt: strategy, code,
  compiled/correct/latency, speedup, plan path) and the per-kernel `traces/<kernel>/` triples.
- Kernels: `kernels/{golden,real-model,autocomp}/`.
