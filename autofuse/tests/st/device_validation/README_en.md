# Autofuse Device Validation

This guide is for developers who add or migrate device-validation cases. It covers the case contract, SoC profile, codegen/JIT, device execution, precision, performance, profiler dependencies, and failure diagnosis.

## 1. Core Model

Device validation has three contracts:

- `cases/<case_id>/case.json`: declares supported `backend`, SoC, dtypes, shapes, and compile/functional/precision/performance capabilities.
- `profiles/<soc>.json`: declares `soc_version`, ASCIR platform, resources, ABI, dtypes, and tools for a device.
- Runner/JIT: consumes the case and profile, runs codegen, Device compilation, ACL launch, D2H, and report generation.

When adding a case or SoC, do not add chip branches to framework code. The SoC must match exactly in the case `support_matrix` and the profile; mismatches must stop execution before a device task is launched.

## 2. Environment Setup

```bash
source <CANN installation directory>/set_env.sh
export PYTHONPATH=autofuse/tests/st:$PYTHONPATH
export DEVICE_VALIDATION_RUNNER=$PWD/build/autofuse/tests/st/device_validation/device_validation_runner
export AUTOFUSE_DEVICE_JIT=$PWD/autofuse/tests/st/device_validation/tools/jit_adapter.py
```

`<CANN installation directory>` is a placeholder; use your actual installation location (for example `/usr/local/Ascend/ascend-toolkit/set_env.sh`). This repository does not bind to any fixed path. `set_env.sh` exports `ASCEND_HOME_PATH`; the CANN package path and runtime probing below rely on it. If it is not exported, confirm with `echo $ASCEND_HOME_PATH` or set it manually to the actual installation directory.

Check the device and its exact runtime variant:

```bash
npu-smi info
python3 - <<'PY'
import ctypes
import os
lib = ctypes.CDLL(os.path.join(os.environ["ASCEND_HOME_PATH"], "lib64", "libacl_rt.so"))
lib.aclrtGetSocName.restype = ctypes.c_char_p
print(lib.aclrtGetSocName().decode())
PY
```

Do not generalize an `Ascend910` display into `Ascend910B`, `Ascend910B1/B2/B3/B4`, or `Ascend910_93`. Use the exact `aclrtGetSocName()` result.

Limit build parallelism:

```bash
cmake --build build --target pyautofuse device_validation_ut device_validation_runner -j 8
```

### Use the latest autofuse sources from this repository (default; development self-validation)

The codegen/ATT pipeline loads the autofusion implementation through `autofuse.pyautofuse`. **The goal of development-time self-validation is to exercise the latest sources in this repository**, so by default you should use the repository-built `pyautofuse` and place it first in `PYTHONPATH`:

```bash
# Building pyautofuse requires a local .dev_env (third-party dependency config; not committed)
# Without .dev_env, pyautofuse cannot be built - see "Fallback to CANN package" below
cmake --build build --target pyautofuse -j 8

# Put the repository build artifacts first in PYTHONPATH
export PYTHONPATH=$PWD/build/autofuse/compiler/py_module:$PWD/autofuse/tests/st:$PYTHONPATH
```

Verify the module source (confirm you are running the latest repository code, not the CANN bundled version):

```bash
python3 -c "import autofuse.pyautofuse; print(autofuse.pyautofuse.__file__)"
# Expected: starts with $PWD/build/autofuse/compiler/py_module (repository build artifact)
# If it prints a path under $ASCEND_HOME_PATH/python/site-packages,
# the module was not built or PYTHONPATH ordering is wrong
```

Only after you modify sources under the repository `autofuse/` tree (optimize/codegen, etc.) does the build artifact contain the latest changes; rebuild pyautofuse before rerunning the case after code changes.

### Fallback to the CANN-bundled autofuse (fallback only; NOT the latest sources)

Without `.dev_env` or an unbuilt pyautofuse, codegen reports `No module named autofuse`. In that case add the complete CANN Python package, not only the directory containing a bare `pyautofuse.so`:

```bash
export PYTHONPATH=$PWD/autofuse/compiler/python:$PWD/autofuse/tests/st:$ASCEND_HOME_PATH/python/site-packages:$PYTHONPATH
```

Note: with this path the case validates the **CANN-bundled autofuse**, not the latest repository sources - it cannot serve as self-validation evidence for changes in this repository's autofuse (use the self-check command above to tell them apart). In CI or environments where pyautofuse cannot be built, falling back to the CANN package is an acceptable baseline validation.

`$ASCEND_HOME_PATH/python/site-packages` relies on the environment variable exported by `set_env.sh`; if it is not exported, run `echo $ASCEND_HOME_PATH` to confirm or replace it with the actual CANN Python package directory.

## 3. New Case Layout

```text
cases/<case_id>/
├── case.json
├── input_ascir.py
├── gen_input.py
├── reference.py
└── steps/
    ├── op_a.py
    └── op_b.py
```

- `case.json`: inputs, outputs, support matrix, and variants.
- `input_ascir.py`: fused ASCIR/codegen entry.
- `gen_input.py`: deterministic input generation with a fixed seed and relevant boundary values.
- `reference.py`: independent golden implementation; do not call the tested AscendC API.
- `steps/*.py`: unfused decomposition, with an input/output and ABI contract for each step.

## 4. Building a Case from Scratch (Tutorial)

This section walks through the full five-step development flow for a minimal case. The real sample to read side by side is `cases/isinf_maskedfill_fusion/` (a fused ASCIR entry plus three unfused steps: isinf, logical_or, masked_fill). Chapters 5-8 are the reference descriptions of each stage; use them together while writing.

### 4.1 Step 1: Create the Directory and case.json

Directory layout (same as Chapter 3):

```text
cases/my_case/
├── case.json
├── input_ascir.py
├── gen_input.py
├── reference.py
└── steps/
    ├── step_codegen.py     # optional: shared codegen skeleton for multiple steps
    ├── op_a.py
    └── op_b.py
```

Minimal runnable `case.json` template:

```json
{
  "schema_version": 1,
  "case_id": "my_case",
  "graph_name": "my_case",
  "inputs": [{"shape": [1], "dtype": "float16", "dynamic": true}],
  "outputs": [{"shape": [1], "dtype": "float16", "dynamic": true}],
  "verification": {"functional": true, "precision": true, "atol": 0.001, "rtol": 0.001},
  "performance": {"required": false, "profiler": true, "metric": "runner_wall_clock"},
  "support_matrix": [{
    "case_id": "my_case",
    "backend": "ascendc_real_device",
    "soc": "ascend910_9362",
    "compile": "required",
    "functional": "required",
    "precision": "required",
    "performance": "optional",
    "dtypes": ["float16"],
    "input_dtypes": ["float16"],
    "output_dtypes": ["float16"],
    "shapes": [[128, 128]]
  }],
  "variants": {"fused": {"codegen_entry": "input_ascir.py", "graph": "my_case"}}
}
```

Field-by-field notes:

- `schema_version`: fixed to `1`.
- `case_id`: must match the directory name.
- `graph_name`: fused graph name used by codegen/JIT.
- `inputs`/`outputs`: placeholder shape and dtype declarations; `dynamic: true` means the actual shape comes from the runtime `--shape` argument.
- `verification`: `functional`/`precision` switches and tolerances (`atol`/`rtol`).
- `performance`: whether performance is `required`, whether `profiler` runs, and the default `metric`.
- `support_matrix[].soc`: must match the `profile` field of `profiles/<soc>.json` exactly (string equality).
- `support_matrix[].input_dtypes`/`output_dtypes`: must cover the real inputs, outputs, and unfused intermediate outputs.
- `support_matrix[].shapes`: list aligned, non-aligned, and tail shapes explicitly; undeclared shapes are not run.
- `variants.fused`: `codegen_entry` points to the entry script name and `graph` corresponds to `graph_name`; the unfused variant is covered in 4.6.

Verification:

```bash
python3 -c "import json; json.load(open('autofuse/tests/st/device_validation/cases/my_case/case.json'))"
pytest autofuse/tests/st/device_validation -q
```

When host tests pass, the case is loadable and support-matrix/contract validation succeeds.

### 4.2 Step 2: Write the Fused Entry input_ascir.py

Skeleton (the complete, readable graph-building code is in `cases/isinf_maskedfill_fusion/input_ascir.py`):

```python
import argparse
import json
from pathlib import Path

from autofuse.pyautofuse import ascir, Autofuser, AutofuserOptions


def build_graph(shape):
    # Describe the fused graph with ascir.ops/SizeExpr/Axis; see the
    # _create_* / _compute_and_output helpers in the sample input_ascir.py
    ...


def generate_codegen(shape, output_dir, profile):
    platform = json.loads(Path(profile).read_text(encoding="utf-8"))["ascir"]
    ascir.utils.set_platform(
        platform["platform"], platform["core_type"], platform["ub_size"]
    )
    fuser = Autofuser(AutofuserOptions())
    fused = fuser.schedule(build_graph(shape))
    tiling, host, device = fuser.codegen(fused)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "tiling.h").write_text(tiling)
    (output / "host_impl.cpp").write_text(host)
    (output / "device_impl.cpp").write_text(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output-dir", required=True)
    options = parser.parse_args()
    generate_codegen((options.rows, options.cols), options.output_dir, options.profile)
```

Key points:

- The four argument names `--rows/--cols/--profile/--output-dir` are the runner call contract; do not rename them.
- The artifacts are `tiling.h`, `host_impl.cpp`, and `device_impl.cpp`; `abi_metadata.json` may be written explicitly by codegen (the unfused `steps/step_codegen.py` does this) or omitted, in which case the runner derives and validates it from the launch signature in `device_impl.cpp`.
- Do not hardcode shape/SoC: shape comes from `--rows/--cols`, platform information comes from the `ascir` field of `--profile`.

Verification: run codegen once manually

```bash
python3 autofuse/tests/st/device_validation/cases/my_case/input_ascir.py \
  --rows 128 --cols 128 \
  --profile autofuse/tests/st/device_validation/profiles/ascend910_9362.json \
  --output-dir /tmp/my_case_codegen
ls /tmp/my_case_codegen
```

and confirm the three `.h/.cpp` artifacts exist and `host_impl.cpp` contains an `AutofuseLaunch` signature.

### 4.3 Step 3: Write gen_input.py and reference.py

`gen_input.py` skeleton (fixed seed, injected boundary values):

```python
import numpy as np


def generate_inputs(shape, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-4, 4, size=shape).astype(np.float16)
    # Inject NaN/Inf/0/boundary values as needed; see the sample gen_input.py
    x.reshape(-1)[0] = np.inf
    return x
```

`reference.py` skeleton (independent implementation; the golden must not depend on the tested kernel):

```python
import numpy as np


def compute_reference(x):
    return np.where(np.isinf(x), 1.0, x).astype(np.float16)
```

Key points:

- `gen_input.py` must use a fixed seed so generation is reproducible.
- `reference.py` may only use independent math/framework implementations; it is forbidden to call the tested AscendC API or the fused operator under test.

Verification: two runs with the same seed produce identical input bytes; the reference output dtype/shape matches the `outputs` declaration in `case.json`.

### 4.4 Step 4: Declare the Profile Link

- `support_matrix[].soc` and the `profile` field of `profiles/<soc>.json` must match by exact string equality: no fuzzy or prefix matching; case or separator differences fail to match.
- Before running on a device, confirm the exact chip variant with `aclrtGetSocName()` (see Chapter 2); the check snippet:

```bash
python3 - <<'PY'
import ctypes
import os
lib = ctypes.CDLL(os.path.join(os.environ["ASCEND_HOME_PATH"], "lib64", "libacl_rt.so"))
lib.aclrtGetSocName.restype = ctypes.c_char_p
print(lib.aclrtGetSocName().decode())
PY
```

- Use the returned name as the evidence for the profile `soc_version`; parameters such as `ascir.platform/core_type/ub_size` must come from real compile/Device facts for that chip, never copied from another SoC's profile (see Chapter 6).

Verification: `--mode prepare` generates a manifest without a SoC mismatch (command in 4.5).

### 4.5 Step 5: Run through the Validation Order

The order and commands are in Chapter 8; acceptance criteria per stage:

```text
1. Host tests: pytest autofuse/tests/st/device_validation -q
   Criterion: no failures; case loading and support-matrix/contract validation pass
2. --mode prepare --shape 128 128 --warmup 0
   Criterion: generating inputs/golden and manifest.json is success; the normal
   prepare outcome is prepare_only (stage_status=not_applicable, reason=prepare_only);
   stage_status=passed never occurs here
3. --variant fused --mode functional --warmup 0
   Criterion: stage_status=passed, precision.passed=true, mismatch_count=0, soc_profile correct
4. --variant unfused --mode functional --warmup 0
   Criterion: every step passes its own ABI validation; final golden matches fused
   (declare unfused steps in case.json first, see 4.6; otherwise the run reports
   `unfused variant must declare steps`)
5. --variant all --mode functional --warmup 0
   Criterion: fused and unfused both pass
6. --mode performance --metric runner_wall_clock --warmup 2 --repeat 5
   Criterion: the report has repeat samples with real unit/timing_source
7. --mode performance --profiler --metric device_kernel_duration --warmup 1 --repeat 3
   Criterion: profiler export succeeds; on failure report profiler_export_failed, never host time as device duration
```

`prepare_only`, `skipped`, or `not_applicable` in a functional/performance stage is not a functional pass; `stage_status=passed` applies only to functional/performance stages.

### 4.6 Writing Unfused Steps

- Each `steps/*.py` file is an independent CLI that follows the `run_step_codegen` contract in `steps/step_codegen.py`: it accepts `--rows/--cols/--profile/--output-dir`, while the op name and input/output dtypes are fixed inside the script. For example:

```python
from step_codegen import run_step_codegen

if __name__ == "__main__":
    run_step_codegen("isinf_graph", "IsInf", ("float16",), "uint8")
```

- In `case.json`, `variants.unfused.steps[i]` declares each step: `script` points to the script above; in `inputs`, `"$previous"` means the output file of the previous step, other entries look like `{"file": "input_N.bin", "dtype": ..., "shape": ...}`; `outputs` declares the artifacts of this step.
- Each step has its own inputs, outputs, and ABI: `_write_artifacts` in `steps/step_codegen.py` writes a dedicated `abi_metadata.json` per step, and the runner validates each step's ABI individually; do not reuse the fused ABI.
- The `input_dtypes`/`output_dtypes` of `case.json` must cover unfused intermediate outputs (for example the uint8 mask), otherwise matrix validation fails.

### 4.6.1 Optional aclnn Mode (Direct CANN aclnn Op)

Besides ASCIR steps (`steps/*.py` + codegen + JIT), a `steps` entry may declare `"aclnn": "<OpName>"` to use a CANN native aclnn op directly (for example `IsInf`/`LogicalOr`/`MaskedFillScalar`/`MaskedFillTensor`); in this case `script` is not needed:

```json
"variants": {
  "unfused": {
    "steps": [
      {"name": "isinf", "aclnn": "IsInf",
       "inputs": [{"file": "input_0.bin", "dtype": "float16", "shape": [1]}],
       "outputs": [{"file": "step_0_output_0.bin", "dtype": "uint8", "shape": [1]}]}
    ]
  }
}
```

Flow differences:

- Codegen/JIT is skipped: no `tiling.h`/`host_impl.cpp`/`device_impl.cpp` artifacts, `jit_adapter.py` is not invoked, and `abi_metadata.json` is not needed;
- The flat request carries `aclnn_op`; the runner dispatches to the aclnn step executor (`backend/aclnn_executor.cpp`) and calls the CANN aclnn API directly;
- Unknown ops report `unknown_aclnn_op`.

JIT environment requirement: when all steps are aclnn, `AUTOFUSE_DEVICE_JIT` is not required; fused runs or steps with ASCIR entries still require it.

### 4.6.2 How aclnn steps are invoked (runner execution sequence)

aclnn steps execute on device following the uniform CANN "query-allocate-create-bind-execute-sync-free" call sequence (implemented as an op dispatch table in `backend/aclnn_executor.cpp`; `IsInf`/`LogicalOr`/`MaskedFillScalar`/`MaskedFillTensor` are supported, new ops extend the same pattern):

```cpp
// 1) Query the workspace requirement (op-specific)
aclnnXxxGetWorkspaceSize(desc, &workspace_size);
// 2) Allocate workspace (device memory)
aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_NORMAL_ONLY);
// 3) Describe input/output tensors as device memory views (shape/dtype/format=ACL_FORMAT_ND)
aclCreateTensor(shape, dims, dtype, 0, nullptr, 0, ACL_FORMAT_ND, addr, &tensor);
// 4) Create the op executor and bind inputs/outputs
aclnnCreateXxx(&executor);
aclnnSetXxxInputTensor(executor, input_tensor);   // per input
aclnnSetXxxOutputTensor(executor, output_tensor); // output
// 5) Execute asynchronously (task enqueued on the stream)
aclnnXxx(executor, stream, workspace, workspace_size, ...);
// 6) Synchronize until completion (required before safe release)
aclrtSynchronizeStream(stream);
// 7) Release (after sync) - workspace and tensor descriptors
aclDestroyTensor(tensor);  aclrtFree(workspace);
```

Key semantics:

- **`Execute` is asynchronous enqueue**: after `aclnnXxx(...)` returns the kernel may not have run yet; you must `aclrtSynchronizeStream` before releasing workspace/tensors (consistent with the async-copy memory lifecycle constraint for AscendC);
- Input data is first copied H2D by the runner according to the flat request `tensor_specs` from `.bin` files; outputs are D2H after sync, then flow through the same decode/precision/sampling pipeline as ASCIR steps;
- `warmup/repeat` sampling and msprof device-duration collection are identical for aclnn and ASCIR steps (kernel-name filtering uses the aclnn op name, e.g. `IsInf` matches the `IsInf_xxxx` task record);

The `isinf_maskedfill_fusion` example case provides `--variant unfused_aclnn` to reproduce the aclnn single-op chain directly (no `AUTOFUSE_DEVICE_JIT` needed).

| Aspect | ASCIR step | aclnn step |
|--------|-----------|------------|
| Execution stack | Autofuse stack (steps/*.py + codegen + JIT) | CANN native aclnn op |
| Supported ops | Any Autofuse op that can be decomposed | Ops registered by the CANN on the target SoC |
| Artifacts | `tiling.h`/`host_impl`/`device_impl` + `abi_metadata.json` | None |
| Unknown op | N/A (fails at codegen/JIT stage) | `unknown_aclnn_op` |

`$previous` chaining and the input/output file contract are unchanged; the precision and report flow is identical to ASCIR steps; msprof kernel-name filtering uses the aclnn op name. aclnn steps and ASCIR steps can be mixed in the same `steps` list.

FAQ: How do I choose between the two unfused step modes? — Use ASCIR steps when you need to validate codegen/JIT artifacts of an Autofuse op itself; use aclnn steps when the op is not in Autofuse's supported scope or you only want to validate the data-flow and precision framework against a CANN native op.

### 4.7 Common Pitfalls

- **SoC name does not match exactly**: writing `soc` as `Ascend910`, `ascend910`, or `ascend910_9362` (inconsistent case/separators) does not match the profile `ascend910_9362`, and the run fails with a SoC mismatch before any device task.
- **Dynamic shape missing**: `inputs`/`outputs` without `dynamic: true`, or `shapes` not covering non-aligned/tail shapes, fails shape validation or silently misses tail paths.
- **Golden uses the tested kernel**: if `reference.py` calls the tested AscendC/fused operator, `mismatch_count=0` is guaranteed and proves nothing.
- **Warmup has no effect on the sample count**: wall-clock sampling always returns exactly `repeat` samples regardless of warmup (`--warmup >= --repeat` never causes a sample mismatch); warmup only excludes the first N samples. The report fails with `performance samples do not match repeat` only when profiler records are missing (dropped by msprof), so `samples` is fewer than `repeat`, and performance is declared `required` for that stage.
- **Host time presented as device time**: `runner_wall_clock` is a coarse ms-level reference; fusion-gain comparisons must use `device_kernel_duration` (us); never fake device kernel duration when profiler export fails.
- **Copying profile parameters from another SoC**: `platform/core_type/ub_size` must come from facts of the actual chip; copying from another SoC produces untrustworthy device results.

## 5. Writing case.json

Minimal structure:

```json
{
  "schema_version": 1,
  "case_id": "my_case",
  "graph_name": "my_case",
  "inputs": [{"shape": [1], "dtype": "float16", "dynamic": true}],
  "outputs": [{"shape": [1], "dtype": "float16", "dynamic": true}],
  "verification": {"functional": true, "precision": true, "atol": 0.001, "rtol": 0.001},
  "performance": {"required": false, "profiler": true, "metric": "runner_wall_clock"},
  "support_matrix": [{
    "case_id": "my_case",
    "backend": "ascendc_real_device",
    "soc": "ascend910_9362",
    "compile": "required",
    "functional": "required",
    "precision": "required",
    "performance": "optional",
    "dtypes": ["float16"],
    "input_dtypes": ["float16"],
    "output_dtypes": ["float16"],
    "shapes": [[128, 128]]
  }],
  "variants": {"fused": {"codegen_entry": "input_ascir.py", "graph": "my_case"}}
}
```

Rules:

- `soc` must exactly match the profile `profile` field.
- `input_dtypes`/`output_dtypes` must cover actual inputs, outputs, and unfused intermediate outputs.
- Aligned, non-aligned, tail, and boundary shapes must be explicitly listed.
- A required capability failure blocks that stage; an unavailable optional capability is skipped or not applicable.
- Unfused steps pass outputs with `"$previous"`; fused and unfused use the same final golden output.

## 6. Writing a Profile

Platform fields must come from CANN/runtime/Device compilation facts. Do not copy parameters from another SoC:

```json
{
  "profile": "ascend910_9362",
  "soc_version": "Ascend910_9362",
  "allowed_abi": "AutofuseLaunchV2,AutofuseLaunch",
  "dtypes": ["float16", "uint8"],
  "simulator_backend": "ascendc_simulator",
  "real_device_backend": "ascendc_real_device",
  "profiler": "optional",
  "tools": {"toolkit": "ASCEND_HOME_PATH", "profiler": "msprof"},
  "ascir": {"platform": "2201", "core_type": 40, "ub_size": 196608},
  "resources": {"max_block_dimension": 40}
}
```

If no reliable chip-level fact exists for `max_tiling_bytes` or `max_workspace_bytes`, omit them. Do not use a default value or an ABI type range as a hardware limit.

## 7. Codegen, Inputs, and Golden Outputs

`input_ascir.py` should accept `--rows`, `--cols`, `--profile`, and `--output-dir`, read platform information from the profile, and generate:

```text
tiling.h
host_impl.cpp
device_impl.cpp
abi_metadata.json
```

Do not hardcode shape or SoC in codegen. Use a fixed seed in `gen_input.py`. Keep `reference.py` independent from the tested kernel and define dtype, shape, tolerance, NaN, and Inf behavior explicitly.

## 8. Recommended Validation Order

```text
1. host JSON/schema/support-matrix tests
2. --mode prepare
3. real-codegen preflight
4. codegen/JIT/Device compile
5. fused functional/precision
6. unfused functional/precision
7. --variant all
8. runner_wall_clock performance
9. device_kernel_duration profiler
10. full pytest/gtest/CTest
```

### Prepare

```bash
PYTHONPATH=autofuse/tests/st:$PYTHONPATH \
python3 -m device_validation.tools.run_device_validation \
  --case autofuse/tests/st/device_validation/cases/my_case \
  --soc-profile ascend910_9362 \
  --profile autofuse/tests/st/device_validation/profiles/ascend910_9362.json \
  --backend ascendc_real_device --device 0 \
  --mode prepare --shape 128 128 --warmup 0
```

`prepare` generates inputs, goldens, and a manifest without launching a device kernel.

### Functional/precision

```bash
PYTHONPATH=autofuse/tests/st:$PYTHONPATH \
python3 -m device_validation.tools.run_device_validation \
  --case autofuse/tests/st/device_validation/cases/isinf_maskedfill_fusion \
  --soc-profile ascend910_9362 \
  --profile autofuse/tests/st/device_validation/profiles/ascend910_9362.json \
  --backend ascendc_real_device --device 0 \
  --variant fused --mode functional --shape 128 128 --warmup 0
```

Run the same command with `[128,130]` and `[127,129]`; use `--variant unfused` and `--variant all` for comparison.

Acceptance requires:

```text
stage_status=passed
precision.passed=true
precision.mismatch_count=0
soc_profile=<expected profile>
run_parameters.selected_shape=<requested shape>
```

`prepare_only`, `skipped`, `not_applicable`, and `mismatch_count=0` with `precision.passed=false` are not functional passes.

## 9. Performance and SQLite Profiler Dependency

Metrics:

- `runner_wall_clock`: host launch+sync wall clock in `ms`; coarse-grained reference only.
- `device_kernel_duration`: profiler-exported device AI Core duration in `us`; use this for formal fusion-gain comparisons.

Use multiple samples:

```bash
--mode performance --metric runner_wall_clock --warmup 2 --repeat 5
--mode performance --profiler --metric device_kernel_duration --warmup 1 --repeat 3
```

CANN `msprof_analysis.so` may depend on the exact name `libsqlite3.so`, while a distribution commonly provides only `libsqlite3.so.0` or `libsqlite3.so.0.x.y`. Missing the unversioned name affects only:

- profiler offline export;
- `task_time_*.csv` generation;
- `device_kernel_duration`;
- fused/unfused device-duration comparison.

It does not affect case/profile validation, codegen/JIT, AscendC Device compilation, ACL functional/precision, or `runner_wall_clock`.

Use a real, architecture-compatible library path (`/path/to/...` is illustrative; replace it with the actual compatible library directory):

```bash
export DEVICE_VALIDATION_SQLITE_LIB_PATH=/path/to/sqlite/libsqlite3.so
```

Verify it (paths are illustrative; use the actual library files):

```bash
file /path/to/libsqlite3.so
readelf -d /path/to/libsqlite3.so | grep SONAME
LD_LIBRARY_PATH=$(dirname /path/to/libsqlite3.so):$LD_LIBRARY_PATH \
python3 -c 'import ctypes; ctypes.CDLL("libsqlite3.so"); print("sqlite load passed")'
```

Do not commit SQLite binaries, private compatibility directories, or system symlinks. On export failure, the report must mark `profiler_export_failed`/`profiler_export_unavailable`; host time must never be presented as device duration.

## 10. CMake/CTest Tiers

- Host pytest runs directly: `pytest autofuse/tests/st/device_validation -q`.
- Host CTest is registered when the device-validation subdirectory is included by configuration.
- Runner contract, real-codegen, and hardware are independently opt-in through CMake options.
- Runner contract must not carry the `real_codegen` label.
- CMake must reference only existing Python test files.

```bash
cmake --build build --target device_validation_ut device_validation_runner -j 8
./build/autofuse/tests/ut/device_validation/device_validation_ut
ctest --test-dir build/autofuse/tests/st --output-on-failure -L '^device_validation$' -j 8
ctest --test-dir build/autofuse/tests/st --output-on-failure -L '^device_validation_runner$' -j 8
```

## 11. Troubleshooting

| Stage | Typical error | First checks |
|---|---|---|
| Profile | SoC mismatch | `aclrtGetSocName()`, case `soc`, profile `profile` |
| Codegen | `No module named autofuse` | Complete CANN Python package, `PYTHONPATH`, `LD_LIBRARY_PATH` |
| JIT | Generated artifact missing | Codegen stdout/stderr, ABI metadata |
| Matrix | `invalid_artifact_path` | Parent/nested artifacts, traversal, symlink |
| ABI | `abi_mismatch` | Fused top-level ABI, unfused step ABI |
| Runtime | ACL failure | Device, buffers, stream, shared libraries |
| Precision | Mismatch | Golden, dtype, shape, tolerances |
| Profiler | `libsqlite3.so`/export failure | SQLite ELF architecture, SONAME, analysis root |
| CTest | No tests found | CMake option, label, working directory |

Fix the first valid error in the log before investigating cascading errors.

## 12. Submission Checklist

```text
[ ] case.json contract is complete and SoC exactly matches the profile
[ ] Profile platform/resource fields have CANN or Device evidence
[ ] No new framework-core SoC branch was added
[ ] Aligned, non-aligned, and tail shapes pass
[ ] Fused/unfused/all precision passes
[ ] Profiler results have a real metric, unit, and timing_source
[ ] Host pytest, device_validation_ut, and CTest pass
[ ] README commands run from the repository root
[ ] Build, artifact, and SQLite compatibility files are not committed
```

## 13. Ascend910_9362 Reference Run

For this case, on CANN 9.2, `Ascend910_9362`, device 0, the pre-existing declared shapes (128x128, 128x130, 127x129) passed fused/unfused functional and precision validation with `mismatch_count=0`. `512x512` was measured on Ascend950PR (fused/unfused, including aclnn mode) and passed, while the `512x512` entry for `Ascend910_9362` is only a declaration extension awaiting on-device validation. Profiler export also succeeded: fused p50 was about `4.72 us` and unfused p50 about `15.52 us`. These are measurements for this environment and case, not general commitments for other SoCs, CANN versions, or cases.
