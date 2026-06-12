# Modules to check out for the MX-Gemmini autocomp work

This work spans several repositories. Every modified repo carries a branch named
`feat/gemmini-mx-autocomp`. This `gemmini` repo is the hub: `mx-autocomp/` holds the results report,
the spike-model changelog, copies of the best kernels for all three families, and the curated
autocomp traces, so a reviewer needs only this directory to understand the outcome. The other repos
hold the live, buildable sources.

## Repositories and branches

| repo | remote | branch | what it carries |
|---|---|---|---|
| gemmini (this repo) | `github.com/ucb-bar/gemmini` | `feat/gemmini-mx-autocomp` | `mx-autocomp/` deliverables; submodule pointers below |
| libgemmini | `github.com/ucb-bar/libgemmini` | `feat/gemmini-mx-autocomp` | spike functional-model changes (`gemmini.cc`, `gemmini.h`) — RTL-legality guards + faithful-cycle model |
| gemmini-rocc-tests | `github.com/<fork>/gemmini-rocc-tests` (fork of `Rakanic/gemmini-rocc-tests`) | `feat/gemmini-mx-autocomp` | MX baremetal kernels, LEGAL RTL-correct ports, VCS test harnesses, mxgen golden headers |
| chipyard | `github.com/ucb-bar/chipyard` | `feat/gemmini-mx-autocomp` | top-level submodule pointer bump to this gemmini branch |
| autocomp | `github.com/ucb-bar/autocomp` | `feat/gemmini-mx-autocomp` | MX-pipeline generators (matmul/attention/conv), `vcs_profile.py`, `transform_log.py`, MX problems in `run_mx_pipeline.py` |

`<fork>` is the personal fork of the Rakanic repo; substitute your GitHub account.

## Checkout recipe

```bash
# top level
git clone -b feat/gemmini-mx-autocomp github.com/ucb-bar/chipyard.git chipyard-mx
cd chipyard-mx
git submodule update --init generators/gemmini
cd generators/gemmini
git checkout feat/gemmini-mx-autocomp

# spike model
git submodule update --init software/libgemmini
git -C software/libgemmini checkout feat/gemmini-mx-autocomp

# kernels (fork)
git submodule update --init software/gemmini-rocc-tests
git -C software/gemmini-rocc-tests remote set-url origin github.com/<fork>/gemmini-rocc-tests.git
git -C software/gemmini-rocc-tests fetch origin feat/gemmini-mx-autocomp
git -C software/gemmini-rocc-tests checkout feat/gemmini-mx-autocomp

# search tool (separate working tree)
git clone -b feat/gemmini-mx-autocomp github.com/ucb-bar/autocomp.git
```

## Build / run notes

- Spike model: after checking out libgemmini, `source env.sh` then
  `make -C generators/gemmini/software/libgemmini install`. Enable the faithful-cycle model with
  `GEMMINI_FAITHFUL_CYCLES=1`.
- VCS kernels build via the bareMetalC sub-make (`source env.sh` for the riscv toolchain); run on
  `sims/vcs/simv-…RadianceGemminiOnlyConfig` (flags in `MX_AUTOCOMP_RESULTS.md` §7).
- The autocomp branch contains MX-pipeline code only; muon and unrelated work are not on this branch.
