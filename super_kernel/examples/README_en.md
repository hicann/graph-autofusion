# SuperKernel Sample Guide

## Overview

This directory contains two groups of Python samples:

- `jit/`: builds and runs fused kernels through the SuperKernel JIT interface.
- `aot/`: enables SuperKernel optimization during TorchAir `npugraph_ex` static compilation.

## Directory Structure

```text
examples/
├── _lib/                                  # Shared Bash functions for AOT samples
├── jit/
│   ├── super_kernel_base/                 # Basic SuperKernel usage
│   ├── super_kernel_profiling/            # Profiling comparison
│   └── super_kernel_runtime_ascendc_only/ # Minimal AscendC and Runtime sample
└── aot/
    ├── dual_stream/                       # Two streams with NPU events
    ├── net01_sk_options/                  # SuperKernel options
    └── net03_pybind/                      # Pybind custom operator fusion
```

## Prerequisites

Follow the [source build guide](../../docs/en/build.md), then install the Python dependencies in
[requirements.txt](requirements.txt).

## Running the Samples

`--npu-arch` specifies the AOT compilation target. Supported values are `dav-2201` and `dav-3510`. It does not select a device; use `NPU_DEVICE_ID` or `ASCEND_DEVICE_ID` for that purpose.

```bash
bash build.sh --run_example --module=superkernel --no-autofuse --npu-arch=dav-2201 -j 8
bash build.sh --run_example --module=superkernel --no-autofuse --npu-arch=dav-3510 -j 8
```

The command runs every JIT and AOT Python sample in sequence. An explicit `--npu-arch` is required to avoid guessing the AOT compilation target.

## References

- [TorchAir SuperKernel guide](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/advanced/superkernel.md)
- [Ascend Extension for PyTorch user guide](https://www.hiascend.com/document/redirect/pytorchuserguide)
