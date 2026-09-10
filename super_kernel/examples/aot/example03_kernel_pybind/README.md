# example03_kernel_pybind

该样例使用 `bisheng` 编译带 `SK_BIND` 的 AscendC custom add kernel，通过 pybind 注册为 PyTorch 算子，再由 `npugraph_ex` 静态编译并启用 SuperKernel。

在仓库根目录执行：

```bash
bash super_kernel/examples/aot/example03_kernel_pybind/run.sh --npu-arch=dav-2201
```

该样例支持 `dav-2201` 和 `dav-3510`。样例的 `run.sh` 将架构参数显式传给扩展安装脚本；安装脚本仅为本次 Python 构建设置 `SK_NPU_ARCH`，供 `setup.py` 配置 `bisheng --npu-arch`。扩展安装在样例隔离的 Python user base 中，运行结束后自动卸载。样例使用当前可见 NPU。

option 参考 [TorchAir SuperKernel 使用说明](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/advanced/superkernel.md)。
