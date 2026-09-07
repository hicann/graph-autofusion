# dual_stream

This sample creates control dependencies with two NPU streams and events, and enables automatic SuperKernel operator parallelism through `auto_op_parallel`. It compares the outputs produced with and without SuperKernel optimization.

Run from the repository root:

```bash
bash build.sh --run_example --module=superkernel --no-autofuse --npu-arch=dav-2201 -j 8
```

Use `dav-3510` for that compilation target. Select the device with `NPU_DEVICE_ID` or `ASCEND_DEVICE_ID`.
