# example02_sk_options

该样例展示 SuperKernel optimize/debug options，并使用 eager 输出校验静态编译结果。

在仓库根目录执行：

```bash
bash super_kernel/examples/aot/example02_sk_options/run.sh --npu-arch=dav-2201
```

该样例支持 `dav-2201` 和 `dav-3510`，根据目标架构选择对应的 attention 网络，并使用当前可见 NPU。

option 参考 [TorchAir SuperKernel 使用说明](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/advanced/superkernel.md)。
