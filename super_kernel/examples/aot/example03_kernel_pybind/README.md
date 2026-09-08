# example03_kernel_pybind

该样例使用 `bisheng` 编译带 `SK_BIND` 的 AscendC custom add kernel，通过 pybind 注册为 PyTorch 算子，再由 `npugraph_ex` 静态编译并启用 SuperKernel。

在仓库根目录执行：

```bash
bash build.sh --run_example --module=superkernel --no-autofuse --npu-arch=dav-2201 -j 8
```

`build.sh` 会把架构传递给 `bisheng --npu-arch`。扩展安装在样例隔离的 Python user base 中，运行结束后自动卸载。

option 参考 [TorchAir SuperKernel 使用说明](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/advanced/superkernel.md)。
