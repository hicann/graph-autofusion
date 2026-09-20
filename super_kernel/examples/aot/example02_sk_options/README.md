# example02_sk_options

## 用例功能

该样例展示 SuperKernel optimize/debug options，并使用 eager 输出校验静态编译结果。

## 环境依赖

支持如下产品型号：

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

请先参考[源码构建指南](../../../../docs/zh/build.md)完成环境准备，并按照官方发布的配套版本安装
PyTorch 2.7.1 与 TorchNPU 2.7.1.post10（CANN 9.1.0 及以上、Python 3.9.x 及以上），
版本配套与安装参见《[Ascend Extension for PyTorch 用户指南](https://www.hiascend.com/document/redirect/pytorchuserguide)》，
再安装本样例的 [Python 依赖](../../requirements.txt)：

```bash
pip install -r super_kernel/examples/requirements.txt
```

## 执行命令

`--npu-arch` 用于指定目标 NPU 架构，样例会根据该参数选择对应的 attention 网络。应根据实际使用的产品型号选择参数值：

| `--npu-arch` 取值 | 对应产品 |
| --- | --- |
| `dav-2201` | Atlas A2 训练/推理系列产品、Atlas A3 训练/推理系列产品 |
| `dav-3510` | Ascend 950 系列产品（如 Ascend 950PR、Ascend 950DT） |

在样例目录下，Atlas A2 或 Atlas A3 系列产品执行：

```bash
bash run.sh --npu-arch=dav-2201
```

Ascend 950 系列产品执行：

```bash
bash run.sh --npu-arch=dav-3510
```

## 预期执行结果

样例会分别执行 eager 模型和启用 SuperKernel 静态编译的模型，并使用 `rtol=1e-3`、`atol=1e-2` 校验两个输出的一致性。校验通过后会打印 eager 和 compiled 输出的 shape、dtype 和均值，关键日志如下：

```text
eager add_out: shape=<shape>, dtype=<dtype>, mean=<value>
eager <attention_output>: shape=<shape>, dtype=<dtype>, mean=<value>
compiled add_out: shape=<shape>, dtype=<dtype>, mean=<value>
compiled <attention_output>: shape=<shape>, dtype=<dtype>, mean=<value>
真值校验通过
测试完成!
```

`dav-2201` 和 `dav-3510` 使用的 attention 网络不同，因此输出名称、shape 和均值以实际日志为准。`run.sh` 还会检查 `static_kernel_compile_outputs` 目录下是否生成静态 Kernel `.run` 包；真值校验和产物检查均通过后，命令才会成功返回。
