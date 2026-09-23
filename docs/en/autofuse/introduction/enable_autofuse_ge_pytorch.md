# Enable AutoFuse on the GE Path (PyTorch)

This document describes how to enable AutoFuse automatic operator fusion based on the GE path (PyTorch framework). A PyTorch model enters the GE path after being compiled by TorchAir, and AutoFuse is enabled in the same way as in the TensorFlow scenario (see [Enable AutoFuse on the GE Path (TensorFlow)](./enable_autofuse_ge_tensorflow.md)). This document uses `Abs + ReLU + Exp` operator fusion as an example to demonstrate how to configure and run a fusion example and verify the fusion result.

> **Constraint note**: The GE path supports dynamic shapes and is suitable for scenarios with varying input shapes. The Inductor path currently supports static shapes only. See [Enable AutoFuse on the Inductor Path](./enable_autofuse_inductor.md).

## Environment Preparation

### Runtime Requirements

| Dependency | Requirement |
| :--- | :--- |
| Hardware and basic software | Prepare hardware equipped with an Ascend AI processor and install the matching driver, firmware, and CANN packages. For installation instructions, see [CANN Software Installation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta1/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum). |
| PyTorch and TorchNPU plugin | Select compatible versions according to the official releases. For version information, see the [official documentation](https://www.hiascend.com/document/detail/zh/Pytorch/latest/installguide/swinstall/docs/zh/installation_guide/installation_description.md). |
| TorchAir | Use a version compatible with PyTorch/TorchNPU. For installation and source build requirements, see the [TorchAir official repository](https://gitcode.com/Ascend/torchair). |
| GCC | 9.5.0 or later; 9.5.0 is recommended. |
| CMake | 3.20.0 or later; 3.20.0 is recommended. |

### Set Environment Variables

Before running the program, set the CANN environment variables:

```bash
source /usr/local/Ascend/cann/set_env.sh
```

`/usr/local/Ascend/` is the default installation path when CANN is installed by the root user. Replace it with the actual installation path as needed.

## Enable AutoFuse

On the GE path, enable AutoFuse through an environment variable:

```bash
export AUTOFUSE_FLAGS="--enable_autofuse=true"
```

After `--enable_autofuse=true` is configured, basic AutoFuse fusion is enabled (minimal configuration), which supports automatic fusion between Elemwise and Broadcast operators. The `Abs + ReLU + Exp` chain in this example is an Elemwise operator chain.

A PyTorch model must be compiled with the TorchAir backend to enter the GE path. AutoFuse completes operator fusion during GE graph compilation:

```python
import torchair

config = torchair.CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)
model = torch.compile(
    model,
    backend=npu_backend,
    dynamic=True,
)
```

`dynamic=True` enables dynamic-shape compilation. After the first compilation, the same compiled result supports inputs of different shapes without recompilation.

For more options, see [Environment Variable Reference](#environment-variable-reference).

## Example Code

The following example demonstrates `Abs + ReLU + Exp` operator fusion. It uses four groups of `float16` inputs with different shapes, performs 100 inference runs on the NPU, and includes NPU Profiling. A `profiling` directory is generated after execution for inspecting fusion results and performance data.

```python
import torch
import torch_npu
import torchair
import torch.nn as nn

DEVICE = "npu:0"
torch.npu.set_device(DEVICE)


class MyModel(nn.Module):
    def forward(self, x):
        return torch.exp(torch.relu(torch.abs(x)))


model = MyModel().to(DEVICE)

config = torchair.CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)
model = torch.compile(
    model,
    backend=npu_backend,
    dynamic=True,
)

model.eval()

# Dynamic-shape inputs: one compiled result supports multiple shapes without recompilation
inputs = [
    torch.randn(shape, dtype=torch.float16, device=DEVICE)
    for shape in [(128, 192), (64, 256), (256, 64), (100, 100)]
]

experimental_config = torch_npu.profiler._ExperimentalConfig(
    export_type=[torch_npu.profiler.ExportType.Text],
    profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
    msprof_tx=False,
    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    l2_cache=False,
    op_attr=False,
    data_simplification=False,
    record_op_args=False,
    gc_detect_threshold=None,
)

with torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ],
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./profiling"),
    record_shapes=True,
    profile_memory=False,
    with_stack=False,
    with_modules=False,
    with_flops=False,
    experimental_config=experimental_config,
) as prof:
    for x in inputs:
        for _ in range(25):
            model(x)
```

## Verify Fusion Results

After execution, a `profiling` directory is generated in the current directory. The Profiling-exported `op_summary_*.csv` file is usually located at:

```text
profiling/
└── xxx_timestamp_ascend_pt/
    └── PROF_timestamp_xxx/
        └── mindstudio_profiler_output/
            └── op_summary_timestamp.csv
```

Open the `op_summary_*.csv` file for the current run and inspect the operator list. If a fused Kernel whose name starts with `autofuse_` appears, the corresponding operators have been fused. In this example, the fused Kernel is named `autofuse_pointwise_0_Abs_Relu_Exp`. In dynamic-shape scenarios, the same fused Kernel appears multiple times in the execution records with different input shapes. Kernel names may vary across versions; determine the result together with the operator types and execution records.

## Performance Comparison Before and After Fusion

To evaluate the performance benefits of AutoFuse, collect Profiling data in the following two scenarios:

1. **AutoFuse enabled**: Set the environment variable `AUTOFUSE_FLAGS="--enable_autofuse=true"`.
2. **AutoFuse disabled**: Set the environment variable `AUTOFUSE_FLAGS="--enable_autofuse=false"` as the baseline.

The two scenarios should use the same inputs, execution count, and Profiling configuration, and compare execution time over the same computation range while distinguishing the initial graph-compilation overhead from the steady-state execution time after warm-up. For operators with significant input/output data movement, also examine `aiv_mte2_time` and `aiv_mte3_time` in the Profiling data.

For details about the Profiling performance-analysis tool, see the [Profiling Performance Analysis Tool Guide](https://hiascend.com/document/redirect/CannCommunityToolProfiling).

## Environment Variable Reference

For all environment variables and control options involved in AutoFuse operation and debugging, see the [AutoFuse Environment Variable Reference](./environment_variables.md).
