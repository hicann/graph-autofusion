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
    ├── scripts/                              # AOT 样例公共 Bash 函数
    ├── example01_dual_stream/             # 双流与 NPU Event 控制边
    ├── example02_sk_options/              # SuperKernel options
    └── example03_kernel_pybind/           # Pybind 自定义算子融合
```

## 前置说明

请先参考[源码构建指南](../../docs/zh/build.md)完成环境准备，并安装
[requirements.txt](requirements.txt) 中的 Python 依赖。

## 运行样例

`--npu-arch` 指定运行样例对应的 NPU 架构，当前支持 `dav-2201` 和 `dav-3510`。样例使用当前可见 NPU。

```bash
bash super_kernel/examples/run_example.sh --npu-arch=dav-2201
bash super_kernel/examples/run_example.sh --npu-arch=dav-3510
```

`dav-2201` 会依次运行全部 JIT 和 AOT Python 样例；`dav-3510` 不支持这些 JIT 样例，因此会跳过 JIT，仅运行 AOT 样例。运行 SuperKernel 样例时必须显式传入 `--npu-arch`，用于选择适用的样例；样例内部的编译参数由各样例处理。

AOT 样例退出时由 TorchAir 清理静态 kernel；旧式安装的兜底卸载仅接受本次编译产物对应、位于当前 CANN 安装目录且通过路径校验的 `uninstall.sh`。日志中的其他路径不会执行，清理失败会输出告警。

## 参考

- SuperKernel option 参考 [TorchAir SuperKernel 使用说明](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/advanced/superkernel.md)。
- [Ascend Extension for PyTorch 用户指南](https://www.hiascend.com/document/redirect/pytorchuserguide)
