# Elementwise + Broadcast Fusion Example (abs + add + relu)

## Function Description

AutoFuse fuses the `abs + add + relu` operators. `data1` has shape `[128, 192]`, while `data2` has shape `[192]`. During the `add` operation, `data2` is broadcast to `[128, 192]`. This example verifies fusion between Elementwise and Broadcast operators.

The script uses the `--mode` parameter to select the TensorFlow version:

| Mode | TensorFlow Version | NPU Integration Method | Graph API |
|------|--------------------|------------------------|-----------|
| `tf1` | TensorFlow 1.15.0 | `npu_bridge` (registered through import side effects) | `tf.placeholder` + `Session` + `NpuOptimizer` |
| `tf2-compat` | TensorFlow 2.6.5 | `npu_device.compat.enable_v1()` | `tf.compat.v1.placeholder` + `tf.compat.v1.Session` |

## Execution Commands

Activate the corresponding TensorFlow environment from the graph-autofusion repository root and make sure AutoFuse is enabled:

```bash
# TensorFlow 1.15 environment
source scripts/env_install/tensorflow/env/activate_tf1.sh
export AUTOFUSE_FLAGS="--enable_autofuse=true"
python3 autofuse/examples/tensorflow/af_tf_eleandbroadcast/test_abs_add_relu.py --mode tf1
```

Or:

```bash
# TensorFlow 2.6.5 environment (compatibility mode)
source scripts/env_install/tensorflow/env/activate_tf2.sh
export AUTOFUSE_FLAGS="--enable_autofuse=true"
python3 autofuse/examples/tensorflow/af_tf_eleandbroadcast/test_abs_add_relu.py --mode tf2-compat
```

## Expected Result

The script constructs an `abs → add(Broadcast) → relu` computation graph and performs 100 inference steps on the NPU. If the script finishes without errors, the example has executed successfully. Use graph dump files or Profiling data to confirm whether fusion takes effect.

Profiling is already configured in the script. After running the commands above, check `./profiling/PROF_*/mindstudio_profiler_output/op_summary_*.csv`. When fusion takes effect, an AutoFuse fused Kernel containing `Abs`, `Add`, and `Relu` can be observed, typically with a name similar to `autofuse_pointwise_0_Abs_Add_Relu`, while the corresponding standalone `Abs`, `Add`, and `Relu` Kernels no longer appear.
