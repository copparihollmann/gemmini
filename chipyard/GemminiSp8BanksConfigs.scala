package chipyard

import org.chipsalliance.cde.config.Config

// ------------------------------------------------------------------
// Single-fact architectural sweep config: a Gemmini with 8 scratchpad
// banks (sp_banks = 8) vs the default 4, otherwise IDENTICAL to
// `GemminiRocketConfig` (default 16x16 mesh, WS/OS BOTH dataflow).
// Derived from GemminiConfigs.defaultConfig with ONLY sp_banks flipped.
// sp_capacity stays 256KB, so sp_bank_entries halves (4096 -> 2048),
// exercising the scratchpad-bank axis with no other change. Added as a
// separate file (not editing GemminiConfigs.scala) so it is trivially
// removable.
// ------------------------------------------------------------------
class GemminiSp8BanksRocketConfig extends Config(
  new gemmini.DefaultGemminiConfig(
    gemmini.GemminiConfigs.defaultConfig.copy(sp_banks = 8)) ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig)
