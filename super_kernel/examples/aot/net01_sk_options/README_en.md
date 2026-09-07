# net01_sk_options

This sample demonstrates the public SuperKernel optimize and debug options documented by TorchAir. It checks the statically compiled outputs against eager execution. The `dav-2201` and `dav-3510` targets use their respective supported attention networks.

Run from the repository root:

```bash
bash build.sh --run_example --module=superkernel --no-autofuse --npu-arch=dav-2201 -j 8
```

The options come from the [TorchAir SuperKernel guide](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/advanced/superkernel.md). Internal and undocumented options are intentionally omitted.
