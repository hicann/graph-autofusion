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
python3 autofuse/examples/tensorflow/af_tf_eleandele/test_abs_relu_exp.py --mode tf1

# TensorFlow 2.6.5 environment (compatibility mode)
source scripts/env_install/tensorflow/env/activate_tf2.sh
python3 autofuse/examples/tensorflow/af_tf_eleandele/test_abs_relu_exp.py --mode tf2-compat
```

## Expected Result

The script constructs an `abs → relu → exp` computation graph and performs 100 inference steps on the NPU. If no error is reported, the fusion is successful. The three operators are fused into an `AscBackend`-type fused operator named `autofuse_pointwise_0_Abs_Relu_Exp`, which is executed as a single Kernel on the NPU.

To view the fusion result, enable profiling (the script already includes the profiling configuration). After execution, check `PROF_*/mindstudio_profiler_output/op_summary_*.csv` under the `./profiling` directory in the repository root. If `autofuse_pointwise_0_Abs_Relu_Exp` appears in the profiling data and the standalone `Abs`, `Relu`, and `Exp` kernels no longer appear, it indicates that the three operators have been fused into a single fused operator.
