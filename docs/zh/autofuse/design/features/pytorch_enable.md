# PyTorch 框架下的 AutoFuse 使用指南

本文介绍基于 PyTorch 框架（Inductor 路径）启用 AutoFuse 自动算子融合功能的方法，并以 `add + ge` 算子融合为例，演示如何配置和运行融合用例，以及如何验证融合结果。

## 环境准备

### 运行环境要求

| 依赖项                   | 要求                                                                                                                                                                                                                                               |
| :----------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 硬件与基础软件           | 准备搭载昇腾 AI 处理器的硬件环境，并安装匹配的驱动固件和 CANN 软件包。安装步骤请参见 [CANN 软件安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta1/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum)。 |
| PyTorch 和 TorchNPU 插件 | 按照官方发布的配套版本选择，具体版本信息请参见 [官方文档](https://www.hiascend.com/document/detail/zh/Pytorch/latest/installguide/swinstall/docs/zh/installation_guide/installation_description.md)。                                                |
| GCC                      | 9.5.0 及以上，建议 9.5.0。                                                                                                                                                                                                                         |
| CMake                    | 3.20.0 及以上，建议 3.20.0。                                                                                                                                                                                                                       |

### 设置环境变量

运行程序前，请先设置 CANN 环境变量：

```bash
source /usr/local/Ascend/cann/set_env.sh
```

其中，`/usr/local/Ascend/` 是以 root 用户安装 CANN 时的默认路径，请根据实际安装路径替换。

## 启用 AutoFuse

在 `torch.compile` 中指定 AscendC 后端即可启用 AutoFuse：

```python
model = torch.compile(
    model,
    options={"npu_backend": "ascendc"},
)
```

其中，`options={"npu_backend": "ascendc"}` 用于选择 AscendC 后端并启用 AutoFuse。

也可以使用装饰器形式启用 AutoFuse：

```python
@torch.compile(options={"npu_backend": "ascendc"})
def test_add_ge(x, y, z):
    return torch.ge(torch.add(x, y), z)
```

## 示例代码

以下以 `add + ge` 算子融合为例，展示完整的示例代码。示例使用形状为 `[128, 50]`、数据类型为 `float32` 的输入，在 NPU 上执行 100 次推理，并内置 NPU Profiling。执行后会生成 `profiling` 目录，便于查看融合结果和性能数据。

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

## 验证融合结果

执行后会在当前目录生成 `profiling` 目录，由 Profiling 导出的 `op_summary_*.csv` 文件通常位于以下路径：

```text
profiling/
└── xxx_时间戳_ascend_pt/
    └── PROF_时间戳_xxx/
        └── mindstudio_profiler_output/
            └── op_summary_时间戳.csv
```

打开本次运行对应的 `op_summary_*.csv`，查看其中的算子列表。如果出现名称以 `autofused_` 开头的融合 Kernel，则表示相关算子已完成融合。具体 Kernel 名称可能随版本变化，应结合算子类型和执行记录进行判断。

## 融合前后性能对比

如需评估 AutoFuse 的性能收益，可以采集以下两种场景的 Profiling 数据：

1. **启用 AutoFuse**：保留 `torch.compile(..., options={"npu_backend": "ascendc"})` 配置。
2. **未启用 AutoFuse**：注释或移除 `torch.compile` 配置，使模型回退到 PyTorch Eager 模式执行，作为未融合场景的对照。

两种场景应使用相同的输入、执行次数和 Profiling 配置，并比较相同计算范围内的执行时间，同时区分首次编译开销和预热后的稳定执行时间。对于输入、输出搬运占比较高的算子，还可以进一步关注 Profiling 中的 `aiv_mte2_time` 和 `aiv_mte3_time`。

详细的 Profiling 性能分析工具使用方法，请参见 [Profiling 性能分析工具指南](https://hiascend.com/document/redirect/CannCommunityToolProfiling)。

## 环境变量参考

AutoFuse 运行过程中涉及的所有环境变量和控制项，请参见 [AutoFuse 相关环境变量参考](./environment_variables.md)。
