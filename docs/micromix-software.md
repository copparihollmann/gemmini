# MicroMX software integration

The kernels are in `npu-exploration/baremetal/mxgemmini`. The package contains
projection and attention APIs, transformer-layer examples, offline data tools,
measured configuration profiles and functional tests. Start with
[MicroMX projection kernels](../npu-exploration/baremetal/mxgemmini/docs/micromix.md)
or the [decode replay](../npu-exploration/baremetal/mxgemmini/docs/MICROMIX_DECODE.md).

## Checkout

The integration branch is `kernels/micro-mx`. The NPU changes are in the
official `ucb-bar/npu-exploration` repository; Gemmini, libgemmini and the RoCC
interface use the supporting forks below.

```sh
git clone --branch kernels/micro-mx https://github.com/copparihollmann/gemmini.git
cd gemmini
git submodule update --init npu-exploration software/libgemmini software/gemmini-rocc-tests
git -C software/gemmini-rocc-tests submodule update --init --recursive
```

| Component | Repository |
|---|---|
| NPU kernels/compiler | [ucb-bar/npu-exploration](https://github.com/ucb-bar/npu-exploration/tree/kernels/micro-mx) |
| Functional model | [copparihollmann/libgemmini](https://github.com/copparihollmann/libgemmini/tree/kernels/micro-mx) |
| Device interface | [copparihollmann/gemmini-rocc-tests](https://github.com/copparihollmann/gemmini-rocc-tests/tree/kernels/micro-mx) |

Use the parent repository's pinned submodule commits. This integration extends
Gemmini's `gemmini-mx-cleanup-mixed-prec` base `b6c04720` and the NPU's
`compiler-dev` base `881a89fa`. The RoCC addition provides the shared-accumulator
loop interface; libgemmini adds direct E3M2 accumulation and full-BF16
small-product support. Unrelated parameter defaults and examples are retained.

## Image and toolchain

Applications explicitly include `mx_micromix_dim16.h` for the matching MX/VPU
image: DIM16, 4,096 rows per scratchpad bank, 512 accumulator rows and 128 LUT
codebooks, with BF16 multiplication and accumulation. This per-application
view does not change the default Gemmini parameters or existing FP8 builds.
A hardware run needs a compatible image; building an ELF does not select or
program one.

Provide a RISC-V bare-metal toolchain through `RISCV` and put `dtc` on `PATH`
for Spike. NumPy is required for generator tests.

```sh
make -C software/libgemmini check-mx-products
make -C software/libgemmini RISCV=/path/to/riscv-tools libgemmini_mxbf16_dim16.so
make -C npu-exploration/baremetal/mxgemmini micromix-test micromix-gen-test
make -C npu-exploration/baremetal/mxgemmini micromix-attention-test
make -C npu-exploration/baremetal/mxgemmini micromix-spike-test \
  RISCV=/path/to/riscv-tools
```

The full-BF16 model retains small products; reduced-precision model profiles
keep their existing floor. See [libgemmini](../software/libgemmini/README.md)
for supported controls and the difference between the software E4M3FN codec
and the physical image's FP8 encoding domain.

## Measurement scope

The profiles identify measured image/workload configurations. Host and Spike
tests check packing, buffer guards, lifetimes, projection outputs and attention
arithmetic. Native TinyLlama prefill and the Qwen mixed16 row layout also have
functional FPGA checks. These checks establish correctness, not a new
repeated-trial speed result for every native binary. Mixed16 remains opt-in;
all performance profiles retain canonical row heights.

Examples take user-supplied captures, weights and independent reference files.
Checkpoints, generated blobs and hardware run logs are kept outside the source
repositories. The workload is one real transformer layer, not complete
inference or a perplexity evaluation. Same-mix kernel agreement and
quantization error relative to an unquantized model are separate results.
