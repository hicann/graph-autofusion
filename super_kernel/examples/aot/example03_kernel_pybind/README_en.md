# example03_kernel_pybind

## Use Case

This sample demonstrates SuperKernel support for custom operators: an AscendC operator implemented by a developer can
be integrated with PyTorch and participate in TorchAir `npugraph_ex` static compilation and SuperKernel optimization.
The sample uses its own `add_custom` implementation for the core computation instead of a corresponding `torch_npu`
operator, covering the complete path from operator compilation and registration to SuperKernel execution.

Key features:

- Uses `SK_BIND` to bind a SuperKernel entry point to the custom AscendC operator.
- Registers the operator as `torch.ops.ascendc_ops.add_custom` through pybind and `torch.library`.
- Invokes the custom operator in an explicit SuperKernel scope and validates its output against CPU `torch.add`.

## Directory Structure

```text
example03_kernel_pybind/
├── README.md                              # Chinese documentation
├── README_en.md                           # English documentation
├── main.py                                # Register, compile, run, and validate the custom operator
├── run.sh                                 # Build the extension, run the sample, and check artifacts
├── ops/aclgraph_add_ops/
│   ├── csrc/
│   │   ├── add_custom.asc                 # AscendC operator, SuperKernel entry point, and SK_BIND
│   │   └── pybind11.asc                   # pybind module definition
│   ├── op_extension/
│   │   ├── __init__.py                    # Load the pybind extension
│   │   └── _torch_library.py              # Register the custom PyTorch operator
│   ├── setup.py                           # Build the extension with bisheng
│   ├── install.sh                         # Install the extension into the sample's temporary directory
│   └── uninstall.sh                       # Remove the temporary extension
├── log/                                   # Compilation logs (generated at runtime)
├── tmp/                                   # Contains run.log (generated at runtime)
└── static_kernel_compile_outputs/         # Static kernel artifacts (generated at runtime)
```

## Prerequisites

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

Source the CANN environment before running the sample and make sure the `bisheng` command is available.

## Use Case Details

```mermaid
flowchart TB
    source["add_custom.asc<br/>AscendC custom add + SK_BIND"] --> extension["bisheng compilation<br/>pybind extension"]
    extension --> registered["torch.library registration<br/>torch.ops.ascendc_ops.add_custom"]
    registered --> model["CustomAddModel<br/>explicit SuperKernel scope"]
    inputs["CPU inputs<br/>x / y"] --> golden["torch.add<br/>golden"]
    inputs --> npu["Copy to NPU"]
    npu --> model
    model --> compiled["npugraph_ex static compilation<br/>SuperKernel optimization and execution"]
    compiled --> output[output]
    golden --> check["torch.allclose<br/>rtol=1e-3, atol=1e-3"]
    output --> check
```

`add_custom.asc` defines both a regular kernel entry point and an `__sk__` SuperKernel entry point, then binds them
with `SK_BIND`. The compiled pybind extension launches the kernel, while `torch.library` provides the Meta and NPU
implementations required for `torch.compile` capture. `CustomAddModel` places the operator in the `custom_add_sk`
scope, and `npugraph_ex` performs static compilation and SuperKernel execution.

## Key Configuration

| Configuration | Sample value | Function and effect |
| --- | --- | --- |
| `static_kernel_compile` | `True` | Enables static kernel compilation. |
| `super_kernel_optimize` | `True` | Enables SuperKernel optimization. |
| `debug_per_op_max_core_num` | `1` | Builds a separate scope for each fusible operator and creates a debug configuration using the maximum available cores. |

For the complete option reference, see the
[TorchAir SuperKernel guide](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/advanced/superkernel.md).

## Execution Command

`--npu-arch` specifies the target NPU architecture and must match the product model in use:

| `--npu-arch` | Corresponding Products |
| --- | --- |
| `dav-3510` | Ascend 950 series products, such as Ascend 950PR and Ascend 950DT |
| `dav-2201` | Atlas A3 training/inference series products and Atlas A2 training/inference series products |

In the sample directory, run the following command for Ascend 950 series products:

```bash
bash run.sh --npu-arch=dav-3510
```

For Atlas A3 or Atlas A2 products:

```bash
bash run.sh --npu-arch=dav-2201
```

The script passes this value to `setup.py` to configure `bisheng --npu-arch` and uses the currently visible NPU.
The custom operator extension is installed under `tmp/python_packages/`, imported through `PYTHONPATH`, and removed
when the sample exits without affecting a package with the same name in the current Python environment.

## Expected Result

When the custom operator output matches the CPU `torch.add` result, the output includes:

```text
Test passed: the with sk output matches the golden result
execute sample success
```

To inspect the results:

- The run log is written to both the terminal and `tmp/run.log`.
- Static kernel compilation artifacts are under `static_kernel_compile_outputs/`. If no `.run` package is generated
  and a `*_compile_error.log` file exists, `run.sh` fails and reports the error log path.
