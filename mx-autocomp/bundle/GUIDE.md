# MX-Gemmini autocomp corpus — guide

This folder is self-contained. It holds the full record of the autocomp optimization campaign over
MX-format (fp8/fp6/fp4) Gemmini kernels: every kernel's starting point, its full beam-search
optimization journey (every transformation tried), where it landed, and the token/$ cost of getting
there. It is meant to be moved as one unit and handed to another agent to mine for compiler
heuristics. Nothing here references files outside this folder.

## What's in here

```
GUIDE.md            this file
results/            MX_AUTOCOMP_RESULTS.md   the analysed results (3 kernel families, baseline->final
                                             cycles, measured RTL utilization, how each speedup was got)
                    SPIKE_MODEL_CHANGES.md   the spike functional-model changelog (RTL guards + faithful model)
                    MODULES.md               cross-repo map: which repo/branch holds the live sources
kernels/golden/     author-designed + RTL-correct (LEGAL) kernels (.c)
kernels/real-model/ best kernel per real-workload shape (.c)
kernels/autocomp/   best autocomp-generated kernels incl. faithful-metric winners (.c)
dataset/
  index.csv,index.md   one row per run: family, shape, baseline cyc, best cyc, speedup, metric,
                       #iters, #attempts, tokens, cost, run dir
  journeys/<run>.json  per-run optimization journey (baseline -> kept beam per iter -> best + cost)
  journeys.jsonl       flat, one row per ATTEMPT (every plan->code->outcome), joinable to runs/
  transform_ledger.jsonl   the per-attempt ledger (strategy -> outcome -> speedup)
runs/<run>/         FULL raw artifacts for every run (nothing dropped) — see "Raw run layout" below
cost/               project-spend.jsonl + project-spend-total.json (whole-project token/$ ledger)
tools/bundle_runs.py   the script that regenerated dataset/ from runs/ (for reproducing/extending)
```

## How to read the dataset

`dataset/index.csv` is the entry point — one row per run. **Cycle counts are under each run's
`metric`**: most are base-spike `latency` (inflated ~1.5–2× vs real silicon); the three
`faithful-cycles` runs (fp4/fp6 matmul, pi05-ffn iters7) track RTL within ±9%. For silicon-true RTL
speedups see `results/MX_AUTOCOMP_RESULTS.md` — do NOT read raw spike speedups as silicon.

`dataset/journeys/<run>.json` fields:
- `baseline.score` — starting cycles (sol0, the first generated kernel). `baseline.code_excerpt`.
- `best.score` / `best.iteration_reached` — landing point and when it was found.
- `speedup_vs_baseline` — baseline/best.
- `kept_beam_per_iter` — `{iter: [{cand_idx, score, parent_score, is_seed, strategy_snippet}]}`.
  This is the surviving beam at each iteration (the trunk of the search tree). `is_seed=true` marks an
  iter-0 candidate that was a prior-run winner injected as a seed (not the fresh sol0 baseline).
- `attempts` — every plan→code→eval attempt (the full breadth, including failures): `iteration`,
  `parent_idx` (which kept candidate it was derived from), `cand_idx`, `strategy`, `compiled`,
  `correct`, `latency`, `parent_latency`, `speedup`, `outcome`
  (improved|correct_no_gain|regressed|incorrect|compile_error), `plan_path` (into `runs/`).
- `cost` — `total_input_tokens`, `total_output_tokens`, `estimated_cost_usd`, `by_model_usd`,
  `run_total_s`, `per_iteration`.

`dataset/journeys.jsonl` — same `attempts` rows flattened across all runs, each annotated with
`run`, `family`, `shape_MxKxN`, `baseline_score`, `speedup_vs_baseline`, `metric`. This is the table
to mine for "which transformation helps which shape, and what it cost."

## Reconstructing a kernel's start → journey → end

1. Start: `journeys/<run>.json` → `baseline.score` (and `runs/<run>/candidates-iter-0/candidate_0.txt`
   for the full baseline code; `parent=None, plan=None`).
2. Journey: walk `kept_beam_per_iter` iteration by iteration — each node's `parent_score` links it to
   the prior beam; the full breadth of what was tried (and rejected) at each step is in `attempts`
   (filter by `iteration`). The actual plan text and generated code are in `runs/<run>/` (below).
3. End: `best.score` at `best.iteration_reached`.

## Raw run layout (`runs/<run>/`)

Per iteration N the search writes:
- `candidates-iter-N/candidate_*.txt` — the kept beam (Python `CodeCandidate` repr: `parent=` nested,
  `code=`, `score=` cycles, `plan=`). iter-0 candidate_0 = baseline.
- `generated-plans-iter-N/plan_parent{P}_{model}_{C}.txt` — the optimization plan/transformation;
  `_full.txt` = full LLM response; `prompt_*` = the prompt sent.
- `generated-code-iter-N/impl_{P}_{C}_{model}_{S}.txt` — the generated kernel code.
- `eval-results-iter-N/code_{C}_result.txt` — `{compiled, correct, latency, ...}` (spike/faithful eval).
- `reimplemented-code-iter-N/` + `eval-reimplemented-results-iter-N/` — retries of failed candidates.
- `run_metadata.json` (beam_size, models, metric, simulator, seeds), `run_metrics.json`
  (per-iteration tokens/cost/durations, best_score), `metrics-iter-N.json`, `cost_ledger.jsonl`
  (per-LLM-call tokens/$), `cost_live.json`.

Filename join: `plan_parent{P}_{model}_{C}.txt` ↔ `impl_{P}_{C}_..` ↔ `eval-results/code_{C}_result.txt`;
`{P}` indexes the parent in `candidates-iter-{N-1}/`.

## Headline numbers + the spike↔RTL truth rule

- Best confirmed **silicon (RTL)** speedup: **fp6 128×128×512 = 1.56×** (golden/LEGAL). Also fp4-512
  1.21×, fp8-256 1.14×. Array utilization 35–49% (load-bound; double-buffering does not help).
- Best **faithful-metric** software win: **pi05-ffn = 1.25×** via cutting host data-staging.
- Anti-pattern proven: fp4/fp6 mvin-batching = base-spike 4.1×/4.0× but **1.01× faithful** — an
  instruction-count trick that does nothing on silicon. Treat large base-spike speedups skeptically.
- Cost: whole project **$167.96** (3289 LLM calls, 2026-06-03→11); run-attributed across these 59 MX
  runs ≈ **$149** / 42.0M input + 8.7M output tokens. Per-run cost is in each `journeys/*.json`.

## Cross-repo map (live sources, if you have network access)

See `results/MODULES.md`. Summary: branch `feat/gemmini-mx-autocomp` on
`github.com/copparihollmann/{gemmini,libgemmini,gemmini-rocc-tests,autocomp,chipyard}`. Live kernels in
gemmini-rocc-tests; spike model in libgemmini; generators + this bundle's source data
(`output/mxpipe_*`) in autocomp. This bundle was generated from the autocomp `output/` dirs.

## Regenerating the dataset

From an autocomp checkout with the runs under `output/`:
`python -m autocomp.backend.gemmini.mx_pipeline.bundle_runs --out <dir>`
(`tools/bundle_runs.py` is the copy used here). It composes `transform_log.py` (per-attempt ledger),
the `candidates-iter-*` beam state, and `run_metrics.json` cost.
