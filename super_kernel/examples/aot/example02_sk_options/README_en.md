# example02_sk_options

This sample demonstrates SuperKernel optimize and debug options. It checks the statically compiled outputs against eager execution.

## Environment Requirements

This sample supports the following product models:

- Ascend 950PR/Ascend 950DT
- Atlas A3 training series products/Atlas A3 inference series products
- Atlas A2 training series products/Atlas A2 inference series products

Follow the [source build guide](../../../../docs/en/build.md) and install the officially released matching
PyTorch and TorchNPU versions. See the
[Ascend Extension for PyTorch user guide](https://www.hiascend.com/document/redirect/pytorchuserguide).
Finally, install the sample's Python dependencies:

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

On success, the sample prints the following key log:

```text
execute sample success
```
