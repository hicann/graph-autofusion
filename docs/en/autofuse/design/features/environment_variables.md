# AutoFuse Environment Variable Reference

This document summarizes commonly used environment variables and control options for AutoFuse during operation and debugging with PyTorch and TensorFlow.

## Contents

- [Shared AutoFuse Environment Variables](#shared-autofuse-environment-variables)
	- [`AUTOFUSE_FLAGS` Options](#autofuse_flags-options)
	- [`AUTOFUSE_DFX_FLAGS` Options](#autofuse_dfx_flags-options)
- [Framework-specific Environment Variables](#framework-specific-environment-variables)
	- [PyTorch](#pytorch)
	- [TensorFlow](#tensorflow)
- [Notes](#notes)

## Shared AutoFuse Environment Variables

The following variables control core AutoFuse functions and apply to both TensorFlow and PyTorch:

| Environment Variable | Applicable Frameworks | Description |
| :--- | :--- | :--- |
| `AUTOFUSE_FLAGS` | TensorFlow, PyTorch | Controls AutoFuse functions. Multiple options are separated by semicolons. TensorFlow requires `--enable_autofuse=true`; PyTorch enables AutoFuse through the `torch.compile` backend by default, while this variable is used for extended functions. |
| `AUTOFUSE_DFX_FLAGS` | TensorFlow, PyTorch | Controls fusion-graph dumps, code-generation debugging, and Auto Tiling debugging. Multiple options are separated by semicolons. |

## `AUTOFUSE_FLAGS` Options

When `--enable_autofuse=true` is not enabled, other AutoFuse options have no effect.

**Enable only the basic AutoFuse fusion function (minimal configuration):**

```bash
export AUTOFUSE_FLAGS="--enable_autofuse=true"
```

| Option | Applicable Frameworks | Description |
| :--- | :--- | :--- |
| `--enable_autofuse` | TensorFlow | Accepts `true` or `false`; disabled by default. Controls automatic fusion globally. |
| `--autofuse_enable_pass` | TensorFlow | Enables specified extended fusion capabilities. Currently, only `reduce` and `concat` are available, and multiple values are separated by commas. Other fusion types will be enabled in subsequent versions. Extended fusion is disabled by default. |
| `--autofuse_disable_pass` | TensorFlow | Accepts `reduce`, `concat`, or a comma-separated combination. It cannot configure the same value as `--autofuse_enable_pass`. |
| `--autofuse_enhance_precision_blacklist` | TensorFlow | Accepts AscIR operator type strings separated by commas; default: empty. `all` is also supported. Disabling precision enhancement may affect accuracy. |
| `--recomputation_threshold` | TensorFlow | Range: `0` to `255`; default: `1`. Sets the automatic-fusion recomputation threshold. |
| `--max_fusion_size` | TensorFlow | Range: `0` to the maximum `uint64_t` value. The default is implementation-defined; `0` disables fusion. |
| `--autofuse_enable_pgo` | TensorFlow, PyTorch | Accepts `true` or `false`; disabled by default. Enables PGO tuning, which supports static graphs only and requires `mspti` according to the applicable version requirements. |
| `--experimental_enable_jit_executor_v2` | TensorFlow | Accepts `true` or `false`; disabled by default. Enables split-graph compilation; dynamic bucketing, resource operators, and some control-flow scenarios are unsupported. |

Example:

```bash
export AUTOFUSE_FLAGS="--enable_autofuse=true;--autofuse_enable_pass=reduce,concat"
```

## `AUTOFUSE_DFX_FLAGS` Options

`AUTOFUSE_DFX_FLAGS` is used to debug AutoFuse compilation, Auto Tiling, and fusion results.

| Option | Applicable Frameworks | Description |
| :--- | :--- | :--- |
| `--codegen_compile_debug` | TensorFlow, PyTorch | Accepts `true` or `false`; disabled by default. When `true`, preserves Kernel, Tiling, CMake projects, and compilation results, and generates fusion-graph dumps. |
| `--debug_dir` | TensorFlow, PyTorch | A valid directory path; defaults to the current execution directory. Requires `--codegen_compile_debug=true`. |
| `--autofuse_att_algorithm` | TensorFlow | Accepts `AxesReorder` (default) or experimental `HighPerf`. |
| `--att_accuracy_level` | TensorFlow, PyTorch | Accepts `0` or `1`; default: `1`. `1` indicates high-accuracy solving and `0` indicates low-accuracy solving. |
| `--att_enable_multicore_ub_tradeoff` | TensorFlow, PyTorch | Accepts `true` or `false`; disabled by default. Enables the trade-off strategy between multicore utilization and UB utilization. |
| `--att_ub_threshold` | TensorFlow, PyTorch | Range: `0` to `100`; default: `20`. Requires `--att_enable_multicore_ub_tradeoff=true`. |
| `--att_corenum_threshold` | TensorFlow | Range: `0` to `100`; default: `40`. Requires `--att_enable_multicore_ub_tradeoff=true`. |
| `--att_profiling` | TensorFlow, PyTorch | Accepts `true` or `false`; disabled by default. Used mainly to locate Auto Tiling performance issues. |
| `--disable_lifting` | TensorFlow | Accepts `true` or `false`; enabled by default. Recommended only for locating AscBackend rollback issues. |
| `--autofuse_pgo_algo` | TensorFlow | Accepts `core_select` (default) or `pruning`. |
| `--autofuse_pgo_step_max` | TensorFlow | A power of 2 from `2` to `1024`; default: `16`. Effective only when `--autofuse_pgo_algo=pruning`. |
| `--autofuse_pgo_topn` | TensorFlow | Accepts `0` or any positive integer; default: `5`. `0` selects all candidates. |
| `--skip_node_names_cfg` | TensorFlow | A valid `.ini` configuration path used to specify operator names or types to skip during fusion. |

Example:

```bash
export AUTOFUSE_DFX_FLAGS="--codegen_compile_debug=true;--debug_dir=/path/to/dump"
```

## Framework-specific Environment Variables

### PyTorch

These variables are used for PyTorch compilation or runtime debugging and do not apply to TensorFlow graph mode:

| Environment Variable | Description | Usage |
| :--- | :--- | :--- |
| `TORCH_COMPILE_DEBUG` | Enables PyTorch compilation debugging and saves intermediate artifacts under `torch_compile_debug`. Directories prefixed with `autofused_` usually indicate fused operators generated by the AscendC backend. | `export TORCH_COMPILE_DEBUG=1` |
| `TORCHINDUCTOR_FORCE_DISABLE_CACHES` | Disables Inductor caches and forces recompilation on every execution. This increases compilation and graph-launch overhead and is intended for debugging. | `export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` |
| `ASCEND_LAUNCH_BLOCKING` | Makes Ascend Kernels execute synchronously to locate the first failing Kernel. This reduces performance and is recommended only for troubleshooting. | `export ASCEND_LAUNCH_BLOCKING=1` |

### TensorFlow

TensorFlow currently has no framework-exclusive environment variables; all TensorFlow-related options are included in the shared AutoFuse variables.

## Notes

- `AUTOFUSE_FLAGS` and `AUTOFUSE_DFX_FLAGS` apply to GE/AutoFuse graph-compilation scenarios. TensorFlow examples must set these variables before importing TensorFlow and `npu_bridge`.
- PyTorch examples select the AscendC backend through `torch.compile(..., options={"npu_backend": "ascendc"})` and normally do not need `AUTOFUSE_FLAGS` to enable AutoFuse.
- Debugging variables add compilation or runtime overhead. Unset them after troubleshooting.
- For details, see the [AUTOFUSE_FLAGS reference](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/programug/graphdevg/autofuse_1_0061.html) and [AUTOFUSE_DFX_FLAGS reference](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/programug/graphdevg/autofuse_1_0062.html).
