package chipyard

import org.chipsalliance.cde.config.Config

// ------------------------------------------------------------------
// DSE config for the mlc cycle-model plan (Phase E, config coverage): a
// Gemmini pinned to the OUTPUT-STATIONARY dataflow (vs the default
// `Dataflow.BOTH`, which supports WS and OS at runtime), otherwise
// IDENTICAL to `GemminiRocketConfig` (default 16x16 mesh). Derived from
// GemminiConfigs.defaultConfig with only `dataflow` set to Dataflow.OS.
// This exercises the OTHER systolic schedule so the M2 model's dataflow
// term is validated against RTL in both modes (WS via GemminiRocketConfig,
// OS here) rather than only weight-stationary -- a real anti-overfit axis
// (different accumulation/drain schedule, same DIM). Added as a separate
// file (not editing GemminiConfigs.scala) so it is trivially removable.
// See modeling/mlc/bench/configs.py (gemmini_dim16_os).
// ------------------------------------------------------------------
class GemminiOSRocketConfig extends Config(
  new gemmini.DefaultGemminiConfig(
    gemmini.GemminiConfigs.defaultConfig.copy(dataflow = gemmini.Dataflow.OS)) ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig)
