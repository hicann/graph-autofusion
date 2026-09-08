# AutoFuse Usage Guide with PyTorch

This document describes how to enable AutoFuse automatic operator fusion based on the PyTorch framework (Inductor path), and uses `add + ge` operator fusion as an example to demonstrate how to configure and run a fusion example and verify the fusion result.

## Environment Preparation

### Runtime Requirements

| Dependency | Requirement |
| :--- | :--- |
| Hardware and basic software | Prepare hardware equipped with an Ascend AI processor and install the matching driver, firmware, and CANN packages. For installation instructions, see [CANN Software Installation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta1/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum). |
| PyTorch and TorchNPU plugin | Select compatible versions according to the official releases. For version information, see the [official documentation](https://www.hiascend.com/document/detail/zh/Pytorch/latest/installguide/swinstall/docs/zh/installation_guide/installation_description.md). |
| GCC | 9.5.0 or later; 9.5.0 is recommended. |
| CMake | 3.20.0 or later; 3.20.0 is recommended. |

### Set Environment Variables

Before running the program, set the CANN environment variables:

```bash
source /usr/local/Ascend/cann/set_env.sh
```

`/usr/local/Ascend/` is the default installation path when CANN is installed by the root user. Replace it with the actual installation path as needed.

## Enable AutoFuse

Specify the AscendC backend in `torch.compile` to enable AutoFuse:

```python
model = torch.compile(
    model,
    options={"npu_backend": "ascendc"},
)
```

`options={"npu_backend": "ascendc"}` selects the AscendC backend and enables AutoFuse.

AutoFuse can also be enabled with the decorator form:

```python
@torch.compile(options={"npu_backend": "ascendc"})
def test_add_ge(x, y, z):
    return torch.ge(torch.add(x, y), z)
```

## Example Code

The following example demonstrates `add + ge` operator fusion. It uses `float32` inputs with shape `[128, 50]`, performs 100 inference runs on the NPU, and includes NPU Profiling. A `profiling` directory is generated after execution for inspecting fusion results and performance data.

```python
import torch
import torch_npu
import torch.nn as nn

DEVICE = "npu:0"
torch.npu.set_device(DEVICE)


class MyModel(nn.Module):
    def forward(self, x, y, z):
        return torch.ge(torch.add(x, y), z)


model = MyModel().to(DEVICE)
model = torch.compile(
    model,
    options={"npu_backend": "ascendc"},
)

x = torch.randn(128, 50, device=DEVICE)
y = torch.randn(128, 50, device=DEVICE)
z = torch.randn(128, 50, device=DEVICE)

model.eval()

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
    for _ in range(100):
        model(x, y, z)
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

Open the `op_summary_*.csv` file for the current run and inspect the operator list. If a fused Kernel whose name starts with `autofused_` appears, the corresponding operators have been fused. Kernel names may vary across versions; determine the result together with the operator types and execution records.

## Performance Comparison Before and After Fusion

To evaluate the performance benefits of AutoFuse, collect Profiling data in the following two scenarios:

1. **AutoFuse enabled**: Keep the `torch.compile(..., options={"npu_backend": "ascendc"})` configuration.
2. **AutoFuse disabled**: Comment out or remove the `torch.compile` configuration so that the model runs in PyTorch Eager mode as the unfused baseline.

The two scenarios should use the same inputs, execution count, and Profiling configuration, and compare execution time over the same computation range while distinguishing the initial compilation overhead from the steady-state execution time after warm-up. For operators with significant input/output data movement, also examine `aiv_mte2_time` and `aiv_mte3_time` in the Profiling data.

For details about the Profiling performance-analysis tool, see the [Profiling Performance Analysis Tool Guide](https://hiascend.com/document/redirect/CannCommunityToolProfiling).

## Environment Variable Reference

For all environment variables and control options involved in AutoFuse operation and debugging, see the [AutoFuse Environment Variable Reference](./environment_variables.md).
