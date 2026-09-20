# example03_kernel_pybind

This sample compiles an AscendC custom add kernel with `SK_BIND` by using `bisheng`, registers it as a PyTorch operator through pybind, and enables SuperKernel during `npugraph_ex` static compilation.

## Environment Requirements

This sample supports the following product models:

- Ascend 950PR/Ascend 950DT
- Atlas A3 training series products/Atlas A3 inference series products
- Atlas A2 training series products/Atlas A2 inference series products

Follow the [source build guide](../../../../docs/en/build.md), then install PyTorch 2.7.1 and
TorchNPU 2.7.1.post10 (CANN 9.1.0 or later, Python 3.9.x or later) as officially released matching
versions. For version compatibility and installation, see the
[Ascend Extension for PyTorch user guide](https://www.hiascend.com/document/redirect/pytorchuserguide).
Finally, install the sample's [Python dependencies](../../requirements.txt):

```bash
pip install -r super_kernel/examples/requirements.txt
```

## Execution Command

`--npu-arch` specifies the target NPU architecture and must match the product model in use:

| `--npu-arch` | Corresponding Products |
| --- | --- |
| `dav-2201` | Atlas A2 training/inference series products and Atlas A3 training/inference series products |
| `dav-3510` | Ascend 950 series products, such as Ascend 950PR and Ascend 950DT |

In the sample directory, run the following command for Atlas A2 or Atlas A3 products:

```bash
bash run.sh --npu-arch=dav-2201
```

For Ascend 950 series products:

```bash
bash run.sh --npu-arch=dav-3510
```

Note: The sample's `run.sh` explicitly passes the architecture to the extension installation script, which sets `SK_NPU_ARCH` only for that Python build so `setup.py` can configure `bisheng --npu-arch`. The extension is installed with the current Python (including venv) into the sample's `tmp/python_packages` and imported through `PYTHONPATH`. Cleanup only removes that directory and does not uninstall an existing package from the current environment. The sample uses the currently visible NPU.

For options, see the [TorchAir SuperKernel guide](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/advanced/superkernel.md).

## Expected Result

The `run.sh` script first uses `bisheng` to build the custom add operator extension and temporarily installs it under the sample directory. It then runs the statically compiled model with SuperKernel enabled. The sample compares the NPU SuperKernel output against the CPU `torch.add` golden result using `rtol=1e-3` and `atol=1e-3`. On success, the key log messages are similar to the following:

```text
Installed ACLGraph add op_extension package from <path>.
create cpu inputs begin
create cpu inputs end
----------------------- run with sk -----------------------------
with sk: torch.compile begin
with sk: move inputs to npu begin
with sk: move inputs to npu end
with sk: compiled call begin
with sk: copy output to cpu begin
with sk end
测试通过：with sk 输出与 golden 一致
```

The `run.sh` script checks for static Kernel compilation errors. On exit, it removes the Python extension temporarily installed by the sample and uninstalls static Kernels installed by the current run. Local build artifacts under `static_kernel_compile_outputs` remain until the cleanup phase of the next run. Cleanup may emit warnings, while the script preserves the original execution return code.
