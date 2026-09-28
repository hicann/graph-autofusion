# example02_sk_options

## Use Case

This sample demonstrates SuperKernel's flexible tuning and diagnostic capabilities for complex fusion scenarios and
how to configure the related options for TorchAir `npugraph_ex` ahead-of-time (AOT) compilation. It compares a compiled
attention network with an eager baseline to verify result consistency under the configured option set.

Key features:

- Optimization options cover operator parallelism, DCCI cache coherency, early start, and aggressive fusion strategies.
- Debug options cover full-core synchronization, operator execution tracing, cross-core synchronization checks, and
  per-operator maximum-core execution.
- Uses one `torch.compile` entry point to combine optimization and debug switches, with eager execution as the
  consistency baseline.

## Directory Structure

```text
example02_sk_options/
├── README.md                             # Chinese documentation
├── README_en.md                          # English documentation
├── main-dav-2201.py                      # dav-2201 attention network and option configuration
├── main-dav-3510.py                      # dav-3510 attention network and option configuration
├── run.sh                                # Selects the target architecture, runs the sample, and checks artifacts
├── log/                                  # Log directory (generated at runtime)
├── tmp/                                  # Contains run.log (generated at runtime)
└── static_kernel_compile_outputs/        # Static kernel artifacts, including a .run package (generated at runtime)
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

## Use Case Details

```mermaid
flowchart TB
    inputs["Inputs"] --> attention1["Attention"]
    attention1 --> moe["MoE"]
    moe --> attention2["Attention"]

    attention2 --> attention_output["Attention output"]
    attention2 --> attention3["Attention"]
    attention3 --> grouped_matmul["Grouped Matmul"]
    attention3 --> add["Add"]
    grouped_matmul --> add
    add --> add_output["Add output"]
```

This sample demonstrates SuperKernel fusion optimization, execution tuning, and diagnostics for complex Attention
networks, with result consistency verified against the eager baseline.

## Option Reference

The sample uses `torch.compile` `options` to demonstrate SuperKernel capabilities in static compilation, fusion
optimization, execution tuning, and diagnostics.

| Group | Configuration entry | Demonstrated capability |
| --- | --- | --- |
| Basic options | Top-level `options` | Static compilation and SuperKernel fusion. |
| Optimization options | `super_kernel_optimize_options` | Operator scheduling, cache coherency, early start, and fusion strategies. |
| Debug options | `super_kernel_debug_options` | Synchronization, execution tracing, cross-core checks, and per-operator diagnostics. |

### Basic Options

| Option | Sample value |
| --- | --- |
| `static_kernel_compile` | `True` |
| `super_kernel_optimize` | `True` |

### Optimization Options

The following configuration demonstrates SuperKernel execution optimization for complex fusion scenarios:

| Option | Sample value |
| --- | --- |
| `auto_op_parallel` | `0` |
| `dcci_before_kernel_start` | `[".*"]` |
| `dcci_after_kernel_end` | `[".*"]` |
| `dcci_disable_on_kernel` | `[".*"]` |
| `early_start` | `1` |
| `aggressive_opt_strategies.value_breaker_bypass` | `0b10` |
| `aggressive_opt_strategies.task_breaker_bypass` | `0b00` |

### Debug Options

The following configuration demonstrates SuperKernel diagnostic capabilities. All debug options are set to `0` in
this sample:

| Option | Sample value |
| --- | --- |
| `debug_sync_all` | `0` |
| `debug_op_exec_trace` | `0` |
| `debug_cross_core_sync_check` | `0` |
| `debug_per_op_max_core_num` | `0` |

## Execution Command

`--npu-arch` specifies the target NPU architecture. The sample selects the corresponding attention network based on
this argument, so choose the value that matches the product model in use:

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

The script selects `main-dav-2201.py` or `main-dav-3510.py` based on `--npu-arch` and uses the currently visible NPU.

## Expected Result

When the SuperKernel static compilation result matches the eager baseline, the output includes the following key log:

```text
execute sample success
```
