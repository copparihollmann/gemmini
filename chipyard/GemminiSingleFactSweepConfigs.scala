package chipyard

import org.chipsalliance.cde.config.Config

// ------------------------------------------------------------------
// Single-fact architectural sweep configs. Each variant flips exactly
// ONE knob vs GemminiConfigs.defaultConfig (which backs
// GemminiRocketConfig), otherwise IDENTICAL. Derived the same way as
// GemminiSp8BanksRocketConfig. Kept in a separate file so each is
// trivially removable and never touches GemminiConfigs.scala.
// ------------------------------------------------------------------

// DMA width axis: dma_maxbytes 64 -> 128 (doubles max DMA beat bytes).
class GemminiDma256RocketConfig extends Config(
  new gemmini.DefaultGemminiConfig(
    gemmini.GemminiConfigs.defaultConfig.copy(dma_maxbytes = 128)) ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig)

// Accumulator banks: acc_banks 2 -> 4.
class GemminiAcc4BanksRocketConfig extends Config(
  new gemmini.DefaultGemminiConfig(
    gemmini.GemminiConfigs.defaultConfig.copy(acc_banks = 4)) ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig)

// Execute reservation-station entries: 16 -> 32.
class GemminiRs32ExRocketConfig extends Config(
  new gemmini.DefaultGemminiConfig(
    gemmini.GemminiConfigs.defaultConfig.copy(reservation_station_entries_ex = 32)) ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig)

// TLB size: 4 -> 16 entries.
class GemminiTlb16RocketConfig extends Config(
  new gemmini.DefaultGemminiConfig(
    gemmini.GemminiConfigs.defaultConfig.copy(tlb_size = 16)) ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig)

// Max in-flight memory requests: 16 -> 32.
class GemminiInflight32RocketConfig extends Config(
  new gemmini.DefaultGemminiConfig(
    gemmini.GemminiConfigs.defaultConfig.copy(max_in_flight_mem_reqs = 32)) ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig)
