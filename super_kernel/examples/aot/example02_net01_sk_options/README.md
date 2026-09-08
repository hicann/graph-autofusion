# example02_net01_sk_options

该样例展示 SuperKernel optimize/debug options，并使用 eager 输出校验静态编译结果。`dav-2201` 和 `dav-3510` 使用各自支持的 attention 网络。

在仓库根目录执行：

```bash
bash build.sh --run_example --module=superkernel --no-autofuse --npu-arch=dav-2201 -j 8
```

option 参考 [TorchAir SuperKernel 使用说明](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/advanced/superkernel.md)。
