# example02_sk_options

## 用例功能

该样例展示 SuperKernel optimize/debug options，并使用 eager 输出校验静态编译结果。

## 环境依赖

支持如下产品型号：

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

请先参考[源码构建指南](../../../../docs/zh/build.md)完成环境准备，并按照官方发布的Pytorch与TorchNPU版本进行配套安装，
参见《[Ascend Extension for PyTorch 用户指南](https://www.hiascend.com/document/redirect/pytorchuserguide)》，
再安装对应的 Python 依赖：

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

样例执行成功时会输出如下关键日志：

```text
execute sample success
```
