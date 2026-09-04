package chipyard

import org.chipsalliance.cde.config.Config

// ------------------------------------------------------------------
// DSE config for the mlc cycle-model plan (Phase 2): a Gemmini with a
// 32x32 systolic mesh (vs the default 16x16), otherwise IDENTICAL to
// `GemminiRocketConfig`. Derived from GemminiConfigs.defaultConfig (the
// exact known-good config that produced the working 16x16 .fir) with
// only meshRows/meshColumns bumped to 32. NOTE: we do NOT use
// gemmini.DSEConfigs.largerDimConfig because the DSE baseConfig sets
// acc_scale_args latency=0, which trips `assert(acc_scale_latency > 0)`
// in GemminiConfigs.scala:165 under the current gemmini source (the DSE
// fragments are stale). defaultConfig already uses spatialArrayOutputType
// = SInt(20.W) -- the same width DSEConfigs.largerDimConfig chose for a
// 32-row mesh. Added as a separate file (not editing GemminiConfigs.scala)
// so it is trivially removable. See modeling/runs/circt-arc/gemmini/dse/RUN.md.
// ------------------------------------------------------------------
class Gemmini32x32RocketConfig extends Config(
  new gemmini.DefaultGemminiConfig(
    gemmini.GemminiConfigs.defaultConfig.copy(meshRows = 32, meshColumns = 32)) ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig)
