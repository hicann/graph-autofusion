# example03_kernel_pybind

This sample compiles an AscendC custom add kernel with `SK_BIND` by using `bisheng`, registers it as a PyTorch operator through pybind, and enables SuperKernel during `npugraph_ex` static compilation.

Run from the repository root:

```bash
bash super_kernel/examples/aot/example03_kernel_pybind/run.sh --npu-arch=dav-2201
```

This sample supports `dav-2201` and `dav-3510`. The sample's `run.sh` explicitly passes the architecture to the extension installation script, which sets `SK_NPU_ARCH` only for that Python build so `setup.py` can configure `bisheng --npu-arch`. The extension is installed into an isolated Python user base and uninstalled after the sample exits. The sample uses the currently visible NPU.

For options, see the [TorchAir SuperKernel guide](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/advanced/superkernel.md).
