# example03_kernel_pybind

## 用例功能

该样例使用 `bisheng` 编译带 `SK_BIND` 的 AscendC custom add kernel，通过 pybind 注册为 PyTorch 算子，再由 `npugraph_ex` 静态编译并启用 SuperKernel。

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

- 说明：该样例的 `run.sh` 将架构参数显式传给扩展安装脚本；安装脚本仅为本次 Python 构建设置 `SK_NPU_ARCH`，供 `setup.py` 配置 `bisheng --npu-arch`。扩展通过当前 Python（支持 venv）安装到样例的 `tmp/python_packages`，通过 `PYTHONPATH` 导入，退出时仅清理该目录，不卸载当前环境中的同名包。样例使用当前可见 NPU。

## 预期执行结果

`run.sh` 会先使用 `bisheng` 编译并将自定义 add 算子扩展临时安装到样例目录，然后执行 SuperKernel 静态编译模型。样例使用 `rtol=1e-3`、`atol=1e-3` 比较 NPU 上的 SuperKernel 输出和 CPU `torch.add` golden 结果。执行成功时，关键日志如下：

```text
Installed ACLGraph add op_extension package from <path>.
create cpu inputs begin
create cpu inputs end
----------------------- run with sk -----------------------------
with sk: torch.compile begin
with sk: move inputs to npu begin
with sk: move inputs to npu end
with sk: compiled call begin
with sk: copy output to cpu begin
with sk end
测试通过：with sk 输出与 golden 一致
```

`run.sh` 会检查静态 Kernel 编译错误，并在退出时删除本次样例临时安装的 Python 扩展、卸载本次安装的静态 Kernel。`static_kernel_compile_outputs` 下的本地编译产物会保留到下次运行前的清理阶段。清理时可能输出告警；脚本会保留原始执行返回码。
