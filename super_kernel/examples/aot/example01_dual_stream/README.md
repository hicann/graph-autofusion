# example01_dual_stream

## 用例功能

该样例使用两个 NPU stream 和 event 建立控制依赖，并通过 `auto_op_parallel` 启用 SuperKernel 自动算子并行。样例分别运行启用和关闭 SuperKernel 的网络，并检查输出一致性。

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

`--npu-arch` 指定样例的目标 NPU 架构，应根据实际运行样例的产品型号选择：

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

样例会先后运行启用 SuperKernel 的 SK 版本和未启用 SuperKernel 的 Non-SK 版本，然后使用 `atol=1e-3`、`rtol=1e-3` 比较两者的 `dq_res_2` 输出。执行成功时，各输出的 `allclose` 结果均为 `PASS`，并输出：

```text
SK 版本运行完成
Non-SK 版本运行完成
...
output[<index>] allclose(atol=0.001, rtol=0.001): PASS
...
测试通过: SK 与 Non-SK 输出一致!
```

最大绝对误差和平均绝对误差的具体数值可能随软硬件环境变化。`run.sh` 还会检查 `static_kernel_compile_outputs` 目录下是否生成静态 Kernel `.run` 包；精度比较和产物检查均通过后，命令才会成功返回。
