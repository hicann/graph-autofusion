# TensorFlow 框架下的 AutoFuse 使用指南

本文介绍基于 TensorFlow 1.x 框架（GE 路径）启用 AutoFuse 自动算子融合功能的方法，并以 `Abs + ReLU + Exp` 算子融合为例，演示如何配置和运行融合用例，以及如何验证融合结果。

## 环境准备

### 运行环境要求

| 依赖项                        | 要求                                                                                                                                                                                                                                               |
| :---------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 硬件与基础软件                | 准备搭载昇腾 AI 处理器的硬件环境，并安装匹配的驱动固件和 CANN 软件包。安装步骤请参见 [CANN 软件安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta1/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum)。 |
| TensorFlow 和 TF Adapter 插件 | 按照官方发布的配套版本选择，具体版本信息请参见 [官方文档](https://www.hiascend.com/document/detail/zh/TensorFlowCommunity/latest/releasenote/releasenote_01.html)。                                                                                  |
| GCC                           | 9.5.0 及以上，建议 9.5.0。                                                                                                                                                                                                                         |
| CMake                         | 3.20.0 及以上，建议 3.20.0。                                                                                                                                                                                                                       |

### 设置环境变量

运行程序前，请先设置 CANN 环境变量：

```bash
source /usr/local/Ascend/cann/set_env.sh
```

其中，`/usr/local/Ascend/` 是以 root 用户安装 CANN 时的默认路径，请根据实际安装路径替换。

## 启用 AutoFuse

TensorFlow 图模式通过环境变量启用 AutoFuse：

```bash
export AUTOFUSE_FLAGS="--enable_autofuse=true"
```

配置 `--enable_autofuse=true` 后，即可开启基础 AutoFuse 融合功能（最简配置），支持 Elemwise 算子与 Broadcast 算子之间的自动融合。本示例中的 `Abs + ReLU + Exp` 属于 Elemwise 算子链。

更多配置请参见 [环境变量参考](#环境变量参考)。

## 示例代码

以下以 `Abs + ReLU + Exp` 算子融合为例，展示完整的示例代码。示例使用形状为 `[128, 192]`、数据类型为 `float16` 的输入，在 NPU 上执行 100 次推理，并内置 NPU Profiling。执行后会生成 `profiling` 目录，便于查看融合结果和性能数据。

```python
import glob
import os
import subprocess
import numpy as np
import tensorflow as tf
import npu_bridge

PROFILING_DIR = os.path.abspath("./profiling")
PROFILING_OPTIONS = (
    '{"output":"%s","training_trace":"on","task_time":"on",'
    '"hccl":"on","aicpu":"on","aic_metrics":"PipeUtilization","msproftx":"off"}'
) % PROFILING_DIR


def configure_npu(session_config):
    custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["graph_run_mode"].i = 0
    custom_op.parameter_map["profiling_mode"].b = True
    custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(PROFILING_OPTIONS)
    return session_config


def get_profile_dirs():
    return set(glob.glob(os.path.join(PROFILING_DIR, "PROF_*")))


def export_new_profiling(profile_dirs_before):
    new_profile_dirs = sorted(get_profile_dirs() - profile_dirs_before)
    for profile_dir in new_profile_dirs:
        subprocess.run(
            ["msprof", "--export=on", "--output={}".format(profile_dir)],
            check=True,
        )
    return new_profile_dirs


input_tensor = tf.placeholder(tf.float16, shape=[128, 192], name="input")
abs_result = tf.abs(input_tensor, name="abs")
relu_result = tf.nn.relu(abs_result, name="relu")
output_tensor = tf.exp(relu_result, name="exp")

np.random.seed(0)
input_data = np.random.uniform(-1.0, 1.0, size=(128, 192)).astype(np.float16)

profile_dirs_before = get_profile_dirs()
session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
configure_npu(session_config)

with tf.Session(config=session_config) as session:
    for _ in range(100):
        session.run(output_tensor, feed_dict={input_tensor: input_data})

export_new_profiling(profile_dirs_before)
```

## 验证融合结果

执行后会在当前目录生成 `profiling` 目录，由 Profiling 导出的 `op_summary_*.csv` 文件通常位于以下路径：

```text
profiling/
└── PROF_时间戳_xxx/
    └── mindstudio_profiler_output/
        └── op_summary_时间戳.csv
```

打开本次运行对应的 `op_summary_*.csv`，查看其中的算子列表。如果出现名称以 `autofused_` 开头的融合 Kernel，则表示相关算子已完成融合。具体 Kernel 名称可能随版本变化，应结合算子类型和执行记录进行判断。

## 融合前后性能对比

如需评估 AutoFuse 的性能收益，可以采集以下两种场景的 Profiling 数据：

1. **启用 AutoFuse**：配置环境变量 `AUTOFUSE_FLAGS="--enable_autofuse=true"`。
2. **未启用 AutoFuse**：配置环境变量 `AUTOFUSE_FLAGS="--enable_autofuse=false"`，作为未融合场景的对照组。

两种场景应使用相同的输入数据、执行次数和 Profiling 配置，并比较相同计算范围内的执行时间，同时区分首次图编译开销和预热后的稳定执行时间。对于输入、输出搬运占比较高的算子，还可以进一步关注 Profiling 中的 `aiv_mte2_time` 和 `aiv_mte3_time`。

详细的 Profiling 性能分析工具使用方法，请参见 [Profiling 性能分析工具指南](https://hiascend.com/document/redirect/CannCommunityToolProfiling)。

## 环境变量参考

AutoFuse 运行及调测过程中涉及的环境变量和控制项，请参见 [AutoFuse 相关环境变量参考](./environment_variables.md)。
