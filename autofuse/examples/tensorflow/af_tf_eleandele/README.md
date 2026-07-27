# elementwise + elementwise 融合样例（abs + relu + exp）

## 用例功能

autofuse 融合 `abs + relu + exp` 三个 elementwise 算子。脚本通过 `--mode` 参数选择 TensorFlow 版本：

| 模式 | TF 版本 | NPU 接入方式 | 图 API |
|------|---------|-------------|--------|
| `tf1` | TF 1.15.0 | `npu_bridge`（import 副作用注册） | `tf.placeholder` + `Session` + `NpuOptimizer` |
| `tf2-compat` | TF 2.6.5 | `npu_device.compat.enable_v1()` | `tf.compat.v1.placeholder` + `tf.compat.v1.Session` |

。

## 执行命令

```bash
# TF 1.15 环境
source scripts/env_install/env/activate_tf1.sh
python3 test_abs_relu_exp.py --mode tf1

# TF 2.6.5 环境（兼容模式）
source scripts/env_install/env/activate_tf2.sh
python3 test_abs_relu_exp.py --mode tf2-compat
```

## 预期执行结果

脚本构造 `abs → relu → exp` 计算图，在 NPU 上执行 100 步推理，无报错即表示融合成功。三个算子被融合为一个 `AscBackend` 类型的融合算子 `autofuse_pointwise_0_Abs_Relu_Exp`，在 NPU 上以单个 kernel 执行。

如需查看融合效果，可开启 profiling（脚本已内置 profiling 配置），执行完成后在 `./profiling` 目录下查看 `PROF_*/mindstudio_profiler_output/op_summary_*.csv`，此时仅有算子名为 `autofuse_pointwise_0_Abs_Relu_Exp` 的 kernel，表示三个算子已融合为一个融合算子。
