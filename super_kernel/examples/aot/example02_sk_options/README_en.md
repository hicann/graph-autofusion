# example02_sk_options

This sample demonstrates SuperKernel optimize and debug options. It checks the statically compiled outputs against eager execution.

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

This sample selects the corresponding attention network for the target architecture and uses the currently visible NPU.

For options, see the [TorchAir SuperKernel guide](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/advanced/superkernel.md).

## Expected Result

The sample runs both the eager model and the statically compiled model with SuperKernel enabled, then validates their two outputs using `rtol=1e-3` and `atol=1e-2`. After validation passes, it prints the shape, data type, and mean value of the eager and compiled outputs. The key log messages are similar to the following:

```text
eager add_out: shape=<shape>, dtype=<dtype>, mean=<value>
eager <attention_output>: shape=<shape>, dtype=<dtype>, mean=<value>
compiled add_out: shape=<shape>, dtype=<dtype>, mean=<value>
compiled <attention_output>: shape=<shape>, dtype=<dtype>, mean=<value>
真值校验通过
测试完成!
```

The attention networks used for `dav-2201` and `dav-3510` are different, so the output names, shapes, and mean values depend on the selected architecture. The `run.sh` script also checks that a static Kernel `.run` package is generated under `static_kernel_compile_outputs`. The command succeeds only when both output validation and artifact checks pass.
