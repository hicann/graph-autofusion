# AutoFuse Usage Guide with TensorFlow

This document describes how to enable AutoFuse automatic operator fusion based on the TensorFlow 1.x framework (GE path), and uses `Abs + ReLU + Exp` operator fusion as an example to demonstrate how to configure and run a fusion example and verify the fusion result.

## Environment Preparation

### Runtime Requirements

| Dependency | Requirement |
| :--- | :--- |
| Hardware and basic software | Prepare hardware equipped with an Ascend AI processor and install the matching driver, firmware, and CANN packages. For installation instructions, see [CANN Software Installation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta1/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum). |
| TensorFlow and TF Adapter plugin | Select compatible versions according to the official releases. For version information, see the [official documentation](https://www.hiascend.com/document/detail/zh/TensorFlowCommunity/latest/releasenote/releasenote_01.html). |
| GCC | 9.5.0 or later; 9.5.0 is recommended. |
| CMake | 3.20.0 or later; 3.20.0 is recommended. |

### Set Environment Variables

Before running the program, set the CANN environment variables:

```bash
source /usr/local/Ascend/cann/set_env.sh
```

`/usr/local/Ascend/` is the default installation path when CANN is installed by the root user. Replace it with the actual installation path as needed.

## Enable AutoFuse

Enable AutoFuse in TensorFlow graph mode through an environment variable:

```bash
export AUTOFUSE_FLAGS="--enable_autofuse=true"
```

After `--enable_autofuse=true` is configured, the basic AutoFuse fusion function (minimal configuration) is enabled, supporting automatic fusion for Elemwise and Broadcast operators. In this example, `Abs + ReLU + Exp` is an Elemwise operator chain.

For more configuration options, see the [Environment Variable Reference](#environment-variable-reference).

## Example Code

The following example demonstrates `Abs + ReLU + Exp` operator fusion. It uses `float16` inputs with shape `[128, 192]`, performs 100 inference runs on the NPU, and includes NPU Profiling. A `profiling` directory is generated after execution for inspecting fusion results and performance data.

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

## Verify Fusion Results

After execution, a `profiling` directory is generated in the current directory. The Profiling-exported `op_summary_*.csv` file is usually located at:

```text
profiling/
└── PROF_timestamp_xxx/
    └── mindstudio_profiler_output/
        └── op_summary_timestamp.csv
```

Open the `op_summary_*.csv` file for the current run and inspect the operator list. If a fused Kernel whose name starts with `autofused_` appears, the corresponding operators have been fused. Kernel names may vary across versions; determine the result together with the operator types and execution records.

## Performance Comparison Before and After Fusion

To evaluate the performance benefits of AutoFuse, collect Profiling data in the following two scenarios:

1. **AutoFuse enabled**: Set the environment variable `AUTOFUSE_FLAGS="--enable_autofuse=true"` and run the example in TensorFlow graph mode.
2. **AutoFuse disabled**: Set the environment variable `AUTOFUSE_FLAGS="--enable_autofuse=false"` and run the model in TensorFlow graph mode without fusion as the baseline.

The two scenarios should use the same input data, execution count, and Profiling configuration, and compare execution time over the same computation range while distinguishing the initial graph compilation overhead from the steady-state execution time after warm-up. For operators with significant input/output data movement, also examine `aiv_mte2_time` and `aiv_mte3_time` in the Profiling data.

For details about the Profiling performance-analysis tool, see the [Profiling Performance Analysis Tool Guide](https://hiascend.com/document/redirect/CannCommunityToolProfiling).

## Environment Variable Reference

For all environment variables and control options involved in AutoFuse operation and debugging, see the [AutoFuse Environment Variable Reference](./environment_variables.md).
