# Elementwise + Reduce Fusion Example (abs + reduce_sum)

## Function Description

AutoFuse fuses the `abs + reduce_sum` operators. The input shape is `[128, 192]`. The graph first performs the Elementwise `abs` operation and then performs the Reduce `reduce_sum` operation along `axis=1`. This example verifies fusion between Elementwise and Reduce operators.

The script uses the `--mode` parameter to select the TensorFlow version:

| Mode | TensorFlow Version | NPU Integration Method | Graph API |
|------|--------------------|------------------------|-----------|
| `tf1` | TensorFlow 1.15.0 | `npu_bridge` (registered through import side effects) | `tf.placeholder` + `Session` + `NpuOptimizer` |
| `tf2-compat` | TensorFlow 2.6.5 | `npu_device.compat.enable_v1()` | `tf.compat.v1.placeholder` + `tf.compat.v1.Session` |

## Execution Commands

Reduce fusion is disabled by default. Activate the corresponding TensorFlow environment from the graph-autofusion repository root, and then explicitly enable Reduce fusion:

```bash
# TensorFlow 1.15 environment
source scripts/env_install/tensorflow/env/activate_tf1.sh
export AUTOFUSE_FLAGS="--enable_autofuse=true;--autofuse_enable_pass=reduce"
python3 autofuse/examples/tensorflow/af_tf_eleandreduce/test_abs_reducesum.py --mode tf1
```

Or:

```bash
# TensorFlow 2.6.5 environment (compatibility mode)
source scripts/env_install/tensorflow/env/activate_tf2.sh
export AUTOFUSE_FLAGS="--enable_autofuse=true;--autofuse_enable_pass=reduce"
python3 autofuse/examples/tensorflow/af_tf_eleandreduce/test_abs_reducesum.py --mode tf2-compat
```

## Expected Result

The script constructs an `abs → reduce_sum` computation graph and performs 100 inference steps on the NPU. If the script finishes without errors, the example has executed successfully. Use graph dump files or Profiling data to confirm whether fusion takes effect.

Profiling is already configured in the script. After running the commands above, check `./profiling/PROF_*/mindstudio_profiler_output/op_summary_*.csv`. When fusion takes effect, a fused Kernel with the `autofuse_reduce_` prefix can be observed. The fused Kernel contains the `Abs` and ReduceSum computations, while the corresponding standalone `Abs` and `ReduceSum` Kernels no longer appear. The exact fused Kernel name may vary slightly between CANN versions.
