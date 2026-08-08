# Elementwise + Elementwise Fusion Example (abs + relu + exp)

## Function Description

AutoFuse fuses the three elementwise operators `abs + relu + exp`. The script uses the `--mode` parameter to select the TensorFlow version:

| Mode | TensorFlow Version | NPU Integration Method | Graph API |
|------|--------------------|------------------------|-----------|
| `tf1` | TensorFlow 1.15.0 | `npu_bridge` (registered through import side effects) | `tf.placeholder` + `Session` + `NpuOptimizer` |
| `tf2-compat` | TensorFlow 2.6.5 | `npu_device.compat.enable_v1()` | `tf.compat.v1.placeholder` + `tf.compat.v1.Session` |

## Execution Commands

```bash
# TensorFlow 1.15 environment
source scripts/env_install/env/activate_tf1.sh
python3 test_abs_relu_exp.py --mode tf1

# TensorFlow 2.6.5 environment (compatibility mode)
source scripts/env_install/env/activate_tf2.sh
python3 test_abs_relu_exp.py --mode tf2-compat
```

## Expected Result

The script constructs an `abs → relu → exp` computation graph and performs 100 inference steps on the NPU. If no error is reported, the fusion is successful. The three operators are fused into an `AscBackend`-type fused operator named `autofuse_pointwise_0_Abs_Relu_Exp`, which is executed as a single Kernel on the NPU.

To view the fusion result, enable Profiling, which is already configured in the script. After execution is complete, check `PROF_*/mindstudio_profiler_output/op_summary_*.csv` in the `./profiling` directory. If the only Kernel is named `autofuse_pointwise_0_Abs_Relu_Exp`, the three operators have been fused into a single fused operator.
