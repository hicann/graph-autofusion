# example01_dual_stream

该样例使用两个 NPU stream 和 event 建立控制依赖，并通过 `auto_op_parallel` 启用 SuperKernel 自动算子并行。样例分别运行启用和关闭 SuperKernel 的网络，并检查输出一致性。

在仓库根目录执行：

```bash
bash build.sh --run_example --module=superkernel --no-autofuse --npu-arch=dav-2201 -j 8
```

架构也可以指定为 `dav-3510`。设备由 `NPU_DEVICE_ID` 或 `ASCEND_DEVICE_ID` 选择。
