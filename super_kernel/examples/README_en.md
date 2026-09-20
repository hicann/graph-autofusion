# SuperKernel Sample Guide

## Overview

This directory contains two groups of Python samples:

- `jit/`: builds and runs fused kernels through the SuperKernel JIT interface.
- `aot/`: enables SuperKernel optimization during TorchAir `npugraph_ex` static compilation.

## Directory Structure

```text
examples/
├── run_example.sh                                   # Unified sample runner
├── jit/
│   ├── example01_super_kernel_base/                 # Basic SuperKernel usage
│   ├── example02_super_kernel_profiling/            # Profiling comparison
│   └── example03_super_kernel_runtime_ascendc_only/ # Minimal AscendC and Runtime sample
└── aot/
    ├── scripts/                                     # Shared Bash functions for AOT samples
    ├── example01_dual_stream/                       # Two streams with NPU events
    ├── example02_sk_options/                        # SuperKernel options
    └── example03_kernel_pybind/                     # Pybind custom operator fusion
```

## Prerequisites

The samples support the following product models:

- Ascend 950PR/Ascend 950DT
- Atlas A3 training series products/Atlas A3 inference series products
- Atlas A2 training series products/Atlas A2 inference series products

Follow the [source build guide](../../docs/en/build.md) and install the officially released matching
PyTorch and TorchNPU versions. See the
[Ascend Extension for PyTorch user guide](https://www.hiascend.com/document/redirect/pytorchuserguide).
Finally, install the Python dependencies of these samples:

```bash
pip install -r super_kernel/examples/requirements.txt
```

## Running the Samples

`--npu-arch` specifies the NPU architecture and must match the product model in use:

| `--npu-arch` | Corresponding Products |
| --- | --- |
| `dav-3510` | Ascend 950 series products, such as Ascend 950PR and Ascend 950DT |
| `dav-2201` | Atlas A3 training/inference series products and Atlas A2 training/inference series products |

Run the following command from the repository root for Ascend 950 series products:

```bash
bash super_kernel/examples/run_example.sh --npu-arch=dav-3510
```

For Atlas A3 or Atlas A2 products:

```bash
bash super_kernel/examples/run_example.sh --npu-arch=dav-2201
```

The entry script automatically runs the samples supported by the product model specified by `--npu-arch`. The support scope and running instructions vary between samples; see the `README_en.md` in each sample directory for details.

## References

- For SuperKernel options, see the [TorchAir SuperKernel guide](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/advanced/superkernel.md).
