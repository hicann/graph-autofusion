# example02_sk_options

This sample demonstrates SuperKernel optimize and debug options. It checks the statically compiled outputs against eager execution.

Run from the repository root:

```bash
bash super_kernel/examples/run_example.sh --npu-arch=dav-2201
```

This sample supports `dav-2201` and `dav-3510`, and selects the corresponding attention network for the target architecture. Select the device with `NPU_DEVICE_ID` or `ASCEND_DEVICE_ID`.

For options, see the [TorchAir SuperKernel guide](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/advanced/superkernel.md).
