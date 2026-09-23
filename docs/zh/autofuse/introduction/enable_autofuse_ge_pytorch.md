# GE 路径下使能 AutoFuse（PyTorch）

本文介绍基于 GE 路径（PyTorch 框架）启用 AutoFuse 自动算子融合功能的方法。PyTorch 模型经 TorchAir 编译后进入 GE 路径执行，AutoFuse 的开启方式与 TensorFlow 场景一致（参见 [GE 路径下使能 AutoFuse（TensorFlow）](./enable_autofuse_ge_tensorflow.md)）。本文以 `Abs + ReLU + Exp` 算子融合为例，演示如何配置和运行融合用例，以及如何验证融合结果。

> **约束说明**：GE 路径支持动态 Shape，适合输入形状变化的场景；Inductor 路径当前仅支持静态 Shape，参见 [Inductor 路径下使能 AutoFuse](./enable_autofuse_inductor.md)。

## 环境准备

### 运行环境要求

| 依赖项                   | 要求                                                                                                                                                                                                                                                |
| :--------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 硬件与基础软件           | 准备搭载昇腾 AI 处理器的硬件环境，并安装匹配的驱动固件和 CANN 软件包。安装步骤请参见[CANN 软件安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta1/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum)。 |
| PyTorch 和 TorchNPU 插件 | 按照官方发布的配套版本选择，具体版本信息请参见[官方文档](https://www.hiascend.com/document/detail/zh/Pytorch/latest/installguide/swinstall/docs/zh/installation_guide/installation_description.md)。                                                |
| TorchAir                 | 使用与 PyTorch/TorchNPU 配套的版本，安装及源码编译要求请参见[TorchAir 官方仓库](https://gitcode.com/Ascend/torchair)。                                                                                                                              |
| GCC                      | 9.5.0 及以上，建议 9.5.0。                                                                                                                                                                                                                          |
| CMake                    | 3.20.0 及以上，建议 3.20.0。                                                                                                                                                                                                                        |

### 设置环境变量

运行程序前，请先设置 CANN 环境变量：

```bash
source /usr/local/Ascend/cann/set_env.sh
```

其中，`/usr/local/Ascend/` 是以 root 用户安装 CANN 时的默认路径，请根据实际安装路径替换。

## 启用 AutoFuse

GE 路径下，通过环境变量启用 AutoFuse：

```bash
export AUTOFUSE_FLAGS="--enable_autofuse=true"
```

配置 `--enable_autofuse=true` 后，即可开启基础 AutoFuse 融合功能（最简配置），支持 Elemwise 算子与 Broadcast 算子之间的自动融合。本示例中的 `Abs + ReLU + Exp` 属于 Elemwise 算子链。

PyTorch 模型需要通过 TorchAir 后端编译，才能进入 GE 路径。AutoFuse 在 GE 图编译阶段完成算子融合：

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

其中，`dynamic=True` 用于开启动态 Shape 编译，首次编译后同一份编译结果可支持不同形状的输入，无需按 Shape 重新编译。

更多配置请参见 [环境变量参考](#环境变量参考)。

## 示例代码

以下以 `Abs + ReLU + Exp` 算子融合为例，展示完整的示例代码。示例使用数据类型为 `float16`、形状各不相同的 4 组输入，在 NPU 上执行 100 次推理，并内置 NPU Profiling。执行后会生成 `profiling` 目录，便于查看融合结果和性能数据。

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

# 动态 Shape 输入：同一份编译结果支持多种形状，无需重新编译
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

## 验证融合结果

执行后会在当前目录生成 `profiling` 目录，由 Profiling 导出的 `op_summary_*.csv` 文件通常位于以下路径：

```text
profiling/
└── xxx_时间戳_ascend_pt/
    └── PROF_时间戳_xxx/
        └── mindstudio_profiler_output/
            └── op_summary_时间戳.csv
```

打开本次运行对应的 `op_summary_*.csv`，查看其中的算子列表。如果出现名称以 `autofuse_` 开头的融合 Kernel，则表示相关算子已完成融合，本示例中融合 Kernel 名称为 `autofuse_pointwise_0_Abs_Relu_Exp`。动态 Shape 场景下，同一个融合 Kernel 会以不同的输入形状多次出现在执行记录中。具体 Kernel 名称可能随版本变化，应结合算子类型和执行记录进行判断。

## 融合前后性能对比

如需评估 AutoFuse 的性能收益，可以采集以下两种场景的 Profiling 数据：

1. **启用 AutoFuse**：配置环境变量 `AUTOFUSE_FLAGS="--enable_autofuse=true"`。
2. **未启用 AutoFuse**：配置环境变量 `AUTOFUSE_FLAGS="--enable_autofuse=false"`，作为对照场景。

两种场景应使用相同的输入、执行次数和 Profiling 配置，并比较相同计算范围内的执行时间，同时区分首次图编译开销和预热后的稳定执行时间。对于输入、输出搬运占比较高的算子，还可以进一步关注 Profiling 中的 `aiv_mte2_time` 和 `aiv_mte3_time`。

详细的 Profiling 性能分析工具使用方法，请参见 [Profiling 性能分析工具指南](https://hiascend.com/document/redirect/CannCommunityToolProfiling)。

## 环境变量参考

AutoFuse 运行及调测过程中涉及的环境变量和控制项，请参见 [AutoFuse 相关环境变量参考](./environment_variables.md)。
