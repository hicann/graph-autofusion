# SuperKernel 样例使用指导

## 功能描述

本目录提供两类 SuperKernel Python 样例：

- `jit/`：通过 SuperKernel JIT 接口完成融合、编译和运行。
- `aot/`：通过 TorchAir `npugraph_ex` 静态编译并启用 SuperKernel 优化。

## 目录结构

```text
examples/
├── run_example.sh                                      # 样例统一运行入口
├── jit/
│   ├── example01_super_kernel_base/                 # SuperKernel 基础功能
│   ├── example02_super_kernel_profiling/            # SuperKernel profiling 对比
│   └── example03_super_kernel_runtime_ascendc_only/ # AscendC + Runtime 极简样例
└── aot/
    ├── scripts/                           # AOT 样例公共 Bash 函数
    ├── example01_dual_stream/             # 双流与 NPU Event 控制边
    ├── example02_sk_options/              # SuperKernel options
    └── example03_kernel_pybind/           # Pybind 自定义算子融合
```

## 环境依赖

支持如下产品型号：

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

请先参考[源码构建指南](../../docs/zh/build.md)完成环境准备，并按照官方发布的Pytorch与TorchNPU版本进行配套安装，
参见《[Ascend Extension for PyTorch 用户指南](https://www.hiascend.com/document/redirect/pytorchuserguide)》，
再安装对应的 Python 依赖：

```bash
pip install -r super_kernel/examples/requirements.txt
```

## 运行说明

`--npu-arch` 指定运行样例的 NPU 架构，应根据实际使用的产品型号选择：

| `--npu-arch` 取值 | 对应产品 |
| --- | --- |
| `dav-2201` | Atlas A2 训练/推理系列产品、Atlas A3 训练/推理系列产品 |
| `dav-3510` | Ascend 950 系列产品（如 Ascend 950PR、Ascend 950DT） |

Atlas A2 或 Atlas A3 系列产品，在仓库根目录执行：

```bash
bash super_kernel/examples/run_example.sh --npu-arch=dav-2201
```

Ascend 950 系列产品，在仓库根目录执行：

```bash
bash super_kernel/examples/run_example.sh --npu-arch=dav-3510
```

统一入口会根据 `--npu-arch` 自动运行当前产品型号支持的样例。各样例的支持范围和运行方法详见样例目录下的 `README.md`。

## 参考

- SuperKernel option 参考 [TorchAir SuperKernel 使用说明](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/advanced/superkernel.md)。
