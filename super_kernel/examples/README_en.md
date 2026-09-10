# SuperKernel Sample Guide

## Overview

This directory contains two groups of Python samples:

- `jit/`: builds and runs fused kernels through the SuperKernel JIT interface.
- `aot/`: enables SuperKernel optimization during TorchAir `npugraph_ex` static compilation.

## Directory Structure

```text
examples/
├── run_example.sh                                      # Unified sample runner
├── jit/
│   ├── example01_super_kernel_base/                 # Basic SuperKernel usage
│   ├── example02_super_kernel_profiling/            # Profiling comparison
│   └── example03_super_kernel_runtime_ascendc_only/ # Minimal AscendC and Runtime sample
└── aot/
    ├── _lib/                              # Shared Bash functions for AOT samples
    ├── example01_dual_stream/             # Two streams with NPU events
    ├── example02_sk_options/              # SuperKernel options
    └── example03_kernel_pybind/           # Pybind custom operator fusion
```

## Prerequisites

Follow the [source build guide](../../docs/en/build.md), then install the Python dependencies in
[requirements.txt](requirements.txt).

## Running the Samples

`--npu-arch` specifies the NPU architecture for running the samples. Supported values are `dav-2201` and `dav-3510`. The samples use the currently visible NPU.

```bash
bash super_kernel/examples/run_example.sh --npu-arch=dav-2201
bash super_kernel/examples/run_example.sh --npu-arch=dav-3510
```

With `dav-2201`, the command runs every JIT and AOT Python sample in sequence. The JIT samples do not support `dav-3510`, so that target skips JIT and runs only the AOT samples. An explicit `--npu-arch` is required to select the applicable samples. Each sample handles its own compilation parameters.

## References

- For SuperKernel options, see the [TorchAir SuperKernel guide](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/advanced/superkernel.md).
- [Ascend Extension for PyTorch user guide](https://www.hiascend.com/document/redirect/pytorchuserguide)
