# Elementwise + Reduce 融合样例（abs + reduce_sum）

## 用例功能

AutoFuse 融合 `abs + reduce_sum` 两个算子。输入 shape 为 `[128, 192]`，先执行 Elementwise 类型的 `abs`，再沿 `axis=1` 执行 Reduce 类型的 `reduce_sum`，用于验证 Elementwise 与 Reduce 类算子的融合能力。

脚本通过 `--mode` 参数选择 TensorFlow 版本：

| 模式 | TF 版本 | NPU 接入方式 | 图 API |
|------|---------|-------------|--------|
| `tf1` | TF 1.15.0 | `npu_bridge`（import 副作用注册） | `tf.placeholder` + `Session` + `NpuOptimizer` |
| `tf2-compat` | TF 2.6.5 | `npu_device.compat.enable_v1()` | `tf.compat.v1.placeholder` + `tf.compat.v1.Session` |

## 执行命令

Reduce 融合默认不使能。先在 graph-autofusion 仓库根目录激活对应 TensorFlow 环境，再显式开启 Reduce 融合：

```bash
# TF 1.15 环境
source scripts/env_install/tensorflow/env/activate_tf1.sh
export AUTOFUSE_FLAGS="--enable_autofuse=true;--autofuse_enable_pass=reduce"
python3 autofuse/examples/tensorflow/af_tf_eleandreduce/test_abs_reducesum.py --mode tf1
```

或：

```bash
# TF 2.6.5 环境（兼容模式）
source scripts/env_install/tensorflow/env/activate_tf2.sh
export AUTOFUSE_FLAGS="--enable_autofuse=true;--autofuse_enable_pass=reduce"
python3 autofuse/examples/tensorflow/af_tf_eleandreduce/test_abs_reducesum.py --mode tf2-compat
```

## 预期执行结果

脚本构造 `abs → reduce_sum` 计算图，并在 NPU 上执行 100 步推理。脚本执行无报错表示用例执行成功；是否发生融合，需要通过 Dump 图或 Profiling 进一步确认。

脚本已内置 Profiling 配置。按照上述命令执行后，在 `./profiling/PROF_*/mindstudio_profiler_output/op_summary_*.csv` 中查看算子执行情况。融合生效时，可以观察到 `autofuse_reduce_` 前缀的融合 Kernel，其中包含 `Abs` 和 ReduceSum 计算，且不再出现对应的独立 `Abs`、`ReduceSum` Kernel。不同 CANN 版本生成的融合 Kernel 名称可能略有差异。
