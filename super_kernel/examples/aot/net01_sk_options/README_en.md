# net01_sk_options

This sample demonstrates SuperKernel optimize and debug options. It checks the statically compiled outputs against eager execution. The `dav-2201` and `dav-3510` targets use their respective supported attention networks.

Run from the repository root:

```bash
bash build.sh --run_example --module=superkernel --no-autofuse --npu-arch=dav-2201 -j 8
```

For options, see the [TorchAir SuperKernel guide](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/advanced/superkernel.md).
