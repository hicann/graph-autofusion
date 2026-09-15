# example01_dual_stream

该样例使用两个 NPU stream 和 event 建立控制依赖，并通过 `auto_op_parallel` 启用 SuperKernel 自动算子并行。样例分别运行启用和关闭 SuperKernel 的网络，并检查输出一致性。

在仓库根目录执行：

```bash
bash super_kernel/examples/aot/example01_dual_stream/run.sh --npu-arch=dav-2201
```

该样例支持 `dav-2201` 和 `dav-3510`，并使用当前可见 NPU。

option 参考 [TorchAir SuperKernel 使用说明](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/advanced/superkernel.md)。
