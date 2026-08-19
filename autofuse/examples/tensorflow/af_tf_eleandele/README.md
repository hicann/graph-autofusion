# Elementwise + Elementwise 融合样例（abs + relu + exp）

## 用例功能

AutoFuse 融合 `abs + relu + exp` 三个 Elementwise 算子。脚本通过 `--mode` 参数选择 TensorFlow 版本：

| 模式 | TF 版本 | NPU 接入方式 | 图 API |
|------|---------|-------------|--------|
| `tf1` | TF 1.15.0 | `npu_bridge`（import 副作用注册） | `tf.placeholder` + `Session` + `NpuOptimizer` |
| `tf2-compat` | TF 2.6.5 | `npu_device.compat.enable_v1()` | `tf.compat.v1.placeholder` + `tf.compat.v1.Session` |

## 执行命令

以下命令均在 graph-autofusion 仓库根目录执行。

```bash
# TF 1.15 环境
source scripts/env_install/tensorflow/env/activate_tf1.sh
export AUTOFUSE_FLAGS="--enable_autofuse=true"
python3 autofuse/examples/tensorflow/af_tf_eleandele/test_abs_relu_exp.py --mode tf1

# TF 2.6.5 环境（兼容模式）
source scripts/env_install/tensorflow/env/activate_tf2.sh
export AUTOFUSE_FLAGS="--enable_autofuse=true"
python3 autofuse/examples/tensorflow/af_tf_eleandele/test_abs_relu_exp.py --mode tf2-compat
```

## 预期执行结果

脚本构造 `abs → relu → exp` 计算图，并在 NPU 上执行 100 步推理。脚本执行无报错表示用例执行成功；是否发生融合，需要通过 Dump 图或 Profiling 进一步确认。

脚本已内置 Profiling 配置。按照上述命令执行后，在 `./profiling/PROF_*/mindstudio_profiler_output/op_summary_*.csv` 中查看算子执行情况。融合生效时，可以观察到包含 `Abs`、`Relu`、`Exp` 的 AutoFuse 融合 Kernel（通常类似 `autofuse_pointwise_0_Abs_Relu_Exp`），且不再出现对应的独立 `Abs`、`Relu`、`Exp` Kernel。
