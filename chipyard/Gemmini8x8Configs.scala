package chipyard

import org.chipsalliance.cde.config.Config

// ------------------------------------------------------------------
// DSE config for the mlc cycle-model plan (Phase E, config coverage): a
// Gemmini with an 8x8 systolic mesh (vs the default 16x16), otherwise
// IDENTICAL to `GemminiRocketConfig`. Derived from
// GemminiConfigs.defaultConfig (the exact known-good config that produced
// the working 16x16 .fir) with only meshRows/meshColumns dropped to 8 --
// the mirror of Gemmini32x32RocketConfig, in the other direction. This
// gives the M2 cycle model a THIRD mesh point (8/16/32) for the anti-
// overfit config sweep: the same predict_dataflow_cycles must order and
// bound RTL cycles across a 4x range of DIM with no refit. Added as a
// separate file (not editing GemminiConfigs.scala) so it is trivially
// removable. See modeling/mlc/bench/configs.py (gemmini_dim8_ws).
// ------------------------------------------------------------------
class Gemmini8x8RocketConfig extends Config(
  new gemmini.DefaultGemminiConfig(
    gemmini.GemminiConfigs.defaultConfig.copy(meshRows = 8, meshColumns = 8)) ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig)
