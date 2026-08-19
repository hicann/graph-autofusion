# Elementwise + Elementwise Fusion Example (abs + relu + exp)

## Function Description

AutoFuse fuses the three elementwise operators `abs + relu + exp`. The script uses the `--mode` parameter to select the TensorFlow version:

| Mode | TensorFlow Version | NPU Integration Method | Graph API |
|------|--------------------|------------------------|-----------|
| `tf1` | TensorFlow 1.15.0 | `npu_bridge` (registered through import side effects) | `tf.placeholder` + `Session` + `NpuOptimizer` |
| `tf2-compat` | TensorFlow 2.6.5 | `npu_device.compat.enable_v1()` | `tf.compat.v1.placeholder` + `tf.compat.v1.Session` |

## Execution Commands

Run all the following commands from the graph-autofusion repository root.

```bash
# TensorFlow 1.15 environment
source scripts/env_install/tensorflow/env/activate_tf1.sh
export AUTOFUSE_FLAGS="--enable_autofuse=true"
python3 autofuse/examples/tensorflow/af_tf_eleandele/test_abs_relu_exp.py --mode tf1

# TensorFlow 2.6.5 environment (compatibility mode)
source scripts/env_install/tensorflow/env/activate_tf2.sh
export AUTOFUSE_FLAGS="--enable_autofuse=true"
python3 autofuse/examples/tensorflow/af_tf_eleandele/test_abs_relu_exp.py --mode tf2-compat
```

## Expected Result

The script constructs an `abs → relu → exp` computation graph and performs 100 inference steps on the NPU. If the script finishes without errors, the example has executed successfully. Use graph dump files or Profiling data to confirm whether fusion takes effect.

Profiling is already configured in the script. After running the commands above, check `./profiling/PROF_*/mindstudio_profiler_output/op_summary_*.csv`. When fusion takes effect, an AutoFuse fused Kernel containing `Abs`, `Relu`, and `Exp` can be observed, typically with a name similar to `autofuse_pointwise_0_Abs_Relu_Exp`, while the corresponding standalone `Abs`, `Relu`, and `Exp` Kernels no longer appear.
