# net03_pybind

This sample compiles an AscendC custom add kernel with `SK_BIND` by using `bisheng`, registers it as a PyTorch operator through pybind, and enables SuperKernel during `npugraph_ex` static compilation.

Run from the repository root:

```bash
bash build.sh --run_example --module=superkernel --no-autofuse --npu-arch=dav-2201 -j 8
```

`build.sh` forwards the target to `bisheng --npu-arch`. The extension is installed into an isolated Python user base and uninstalled after the sample exits.
