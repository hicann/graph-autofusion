# example01_dual_stream

This sample creates control dependencies with two NPU streams and events, and enables automatic SuperKernel operator parallelism through `auto_op_parallel`. It compares the outputs produced with and without SuperKernel optimization.

Run from the repository root:

```bash
bash super_kernel/examples/run_example.sh --npu-arch=dav-2201
```

This sample supports `dav-2201` and `dav-3510`, and uses the currently visible NPU.

For options, see the [TorchAir SuperKernel guide](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/advanced/superkernel.md).
