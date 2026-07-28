# PyTorch Inductor Examples

## Description

These examples demonstrate how to use `torch.compile` to perform operator fusion in PyTorch models.

The following three examples are currently provided:

* `add + ge`: Fuses the addition and comparison operators into a single operator.
* `mul + reducesum`: Fuses the multiplication and sum-reduction operators into a single operator.
* `gather + add`: Constructs a graph pattern containing index gathering and element-wise addition.

> **Note:** Gather fusion is not currently supported. After [issue175](https://gitcode.com/cann/graph-autofusion/issues/175) is resolved, `gather` will be able to fuse with `add`.

NPU Profiling is enabled in all three examples. You can inspect the generated profiling data to view the fused kernels.

## Directory Structure

```text
pytorch
├── README.md
├── README_en.md
├── af_pointwise
│   ├── README.md
│   ├── README_en.md
│   └── af_add_ge.py              # Fuses add + ge
├── af_reduce
│   ├── README.md
│   ├── README_en.md
│   └── af_mul_reducesum.py       # Fuses mul + reducesum
└── af_gather
    ├── README.md
    ├── README_en.md
    └── af_gather_add.py          # Constructs the gather + add graph pattern
```

## Prerequisites

Before running these examples, carefully read the [PyTorch Environment Installation Guide](../../../docs/env_install/pytorch/env_pytorch.md) and complete the following steps:

1. Ensure that the CANN package version is `9.0.0` or later. Install the toolkit and ops packages correctly by using [CANN Quick Installation](https://www.hiascend.com/cann/download?versionId=745&ids=d802%2Ch0501%2Ch0602%2Ch0701). For more information, see the [Installation Guide](../../../docs/zh/quick_install.md).

2. Ensure that the `torch_npu` version is `2.9.0` or later. You can use the [environment quick installation script](../../../scripts/env_install/pytorch/setup_torch_npu_daily.sh) to quickly install the Python environment and `torch_npu`.

## Setting Environment Variables

Run the following commands whenever you open a new terminal:

```bash
# Activate the Python environment.
source /mnt/workspace/env/venv/torch210_daily/bin/activate

# Set the CANN installation path based on the actual installation location.
export CANN_INSTALL_PATH=/home/developer/Ascend

# Load CANN environment variables.
source $CANN_INSTALL_PATH/cann/set_env.sh

# Assume that the examples run on device 0.
export ASCEND_DEVICE_ID=0
```

## Running the Examples

### add + ge Fusion

```bash
cd af_pointwise
python af_add_ge.py
```

### mul + reducesum Fusion

```bash
cd af_reduce
python af_mul_reducesum.py
```

### gather + add Graph Pattern

```bash
cd af_gather
python af_gather_add.py
```

## Expected Results

After an example finishes running, a `profiling` directory is generated in the current directory.

You can view operator execution details in the following directory:

```text
profiling/PROF_<timestamp>/mindstudio_profiler_output
```

Open the following file:

```text
op_summary_<timestamp>.csv
```

If the operator list contains a kernel whose name starts with `autofused_`, the related operators have been successfully fused into a single fused operator.

## References

* [Autofuse Overview and Quick Start](../../README.md)
* [Profiling Performance Analysis Tool Guide](https://hiascend.com/document/redirect/CannCommunityToolProfiling)
* [Accuracy Debugging Tool Guide](https://hiascend.com/document/redirect/CannCommunityToolAccucacy)
