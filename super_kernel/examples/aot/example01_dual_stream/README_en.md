# example01_dual_stream

This sample creates control dependencies with two NPU streams and events, and enables automatic SuperKernel operator parallelism through `auto_op_parallel`. It compares the outputs produced with and without SuperKernel optimization.

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

This sample uses the currently visible NPU.

For options, see the [TorchAir SuperKernel guide](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/advanced/superkernel.md).

## Expected Result

The sample runs the SK version with SuperKernel enabled and the Non-SK version without SuperKernel, then compares their `dq_res_2` outputs using `atol=1e-3` and `rtol=1e-3`. On success, every output reports `PASS` for the `allclose` check, and the log includes:

```text
SK 版本运行完成
Non-SK 版本运行完成
...
output[<index>] allclose(atol=0.001, rtol=0.001): PASS
...
测试通过: SK 与 Non-SK 输出一致!
```

The maximum and mean absolute differences may vary with the hardware and software environment. The `run.sh` script also checks that a static Kernel `.run` package is generated under `static_kernel_compile_outputs`. The command succeeds only when both the accuracy comparison and artifact check pass.
