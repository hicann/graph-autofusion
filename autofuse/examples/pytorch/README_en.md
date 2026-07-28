# PyTorch Inductor + AscendC Example Demonstration

## Description

Use the AscendC backend of `torch.compile` to perform operator fusion for PyTorch networks.

The following two examples are currently included:

- `add + ge`: fuses the addition and comparison operators into a single operator;
- `mul + reducesum`: fuses the multiplication and sum-reduction operators into a single operator.

NPU Profiling is enabled for both examples. The generated profiling files can be used to view the fused Kernel.

## Directory Structure

```text
pytorch
├── README.md
├── README_en.md
├── af_pointwise
│   ├── README.md
│   ├── README_en.md
│   └── af_add_ge.py              # Fuse add + ge
└── af_reduce
    ├── README.md
    ├── README_en.md
    └── af_mul_reducesum.py       # Fuse mul + reducesum
```

## Prerequisites

Before running the examples, carefully read the [PyTorch Environment Installation Guide](../../../docs/env_install/pytorch/env_pytorch.md) and complete the following steps:

1. CANN version `9.0.0` or later is required. Install the Toolkit and OPS packages correctly through [CANN Quick Installation](https://www.hiascend.com/cann/download?versionId=745&ids=d802%2Ch0501%2Ch0602%2Ch0701). For details, see the [Installation Guide](../../../docs/zh/quick_install.md).
2. `torch_npu` version `2.9.0` or later is required. You can use the [Quick Environment Installation Script](../../../scripts/env_install/pytorch/setup_torch_npu_daily.sh) to quickly install the Python environment and `torch_npu`.

## Setting Environment Variables

Run the following commands each time you open a new terminal:

```bash
# Activate the environment.
source /mnt/workspace/env/venv/torch210_daily/bin/activate

# Set the CANN installation path according to the actual installation location.
export CANN_INSTALL_PATH=/home/developer/Ascend

# Load CANN environment variables.
source $CANN_INSTALL_PATH/cann/set_env.sh

# Assume the example runs on device 0.
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

## Expected Results

After the program finishes running, a `profiling` directory is generated in the current directory.

Operator execution details can be viewed in the following directory:

```text
profiling/PROF_timestamp/mindstudio_profiler_output
```

Open the following file:

```text
op_summary_timestamp.csv
```

If the operator list contains a Kernel whose name starts with `autofused_`, the related operators have been successfully fused into a single fused operator.

## References

- [Autofuse Introduction and Quick Start](../../README.md)
- [Profiling Performance Analysis Tool Guide](https://hiascend.com/document/redirect/CannCommunityToolProfiling)
