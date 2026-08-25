# Autofuse Device Validation

本文档面向新增或迁移 device validation 用例的开发者，覆盖 case contract、SoC profile、codegen/JIT、真机执行、精度、性能和故障定位。

## 1. 核心模型

device validation 由三类契约组成：

- `cases/<case_id>/case.json`：声明用例支持的 `backend`、SoC、dtype、shape 和 compile/functional/precision/performance 能力。
- `profiles/<soc>.json`：声明设备的 `soc_version`、ASCIR platform、core/resource、ABI、dtype 和工具。
- runner/JIT：消费 case/profile，执行 codegen、Device 编译、ACL launch、D2H 和 report 生成。

新增用例或 SoC 时，不要在框架核心添加芯片判断。SoC 必须同时出现在 case `support_matrix` 和对应 profile 中；两者不匹配时不得执行设备任务。

## 2. 环境准备

```bash
source <CANN 安装目录>/set_env.sh
export PYTHONPATH=autofuse/tests/st:$PYTHONPATH
export DEVICE_VALIDATION_RUNNER=$PWD/build/autofuse/tests/st/device_validation/device_validation_runner
export AUTOFUSE_DEVICE_JIT=$PWD/autofuse/tests/st/device_validation/tools/jit_adapter.py
```

`<CANN 安装目录>` 为占位，以实际安装位置为准（例如 `/usr/local/Ascend/ascend-toolkit/set_env.sh`）。本仓库不绑定任何固定路径。`set_env.sh` 会导出 `ASCEND_HOME_PATH`，后续的 CANN 包路径、运行时探测都基于该环境变量；若未导出，可用 `echo $ASCEND_HOME_PATH` 确认，或手动指定实际安装目录。

先确认：

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

运行前必须用 `aclrtGetSocName()` 确认实际变体，不能把 `Ascend910` 的显示名称泛化为 `Ascend910B`、`Ascend910B1/B2/B3/B4` 或 `Ascend910_93`。

构建测试目标时限制并行度：

```bash
cmake --build build --target pyautofuse device_validation_ut device_validation_runner -j 8
```

### 使用仓库最新 autofuse 源码（默认，开发自验证）

用例的 codegen/ATT 过程通过 `autofuse.pyautofuse` 加载 autofusion 实现。**开发阶段自验证的目标是测试本仓库最新源码**，因此默认应使用仓库构建的 `pyautofuse`，并在 `PYTHONPATH` 中置于首位：

```bash
# 构建 pyautofuse 依赖开发者本地 .dev_env（第三方依赖配置，不入库）
# 缺少 .dev_env 时 pyautofuse 无法构建，见下方"回落到 CANN 包"
cmake --build build --target pyautofuse -j 8

# PYTHONPATH 首位放仓库构建产物
export PYTHONPATH=$PWD/build/autofuse/compiler/py_module:$PWD/autofuse/tests/st:$PYTHONPATH
```

来源自检（确认跑的是仓库最新代码，而非 CANN 包内置版本）：

```bash
python3 -c "import autofuse.pyautofuse; print(autofuse.pyautofuse.__file__)"
# 期望输出以 $PWD/build/autofuse/compiler/py_module 开头（仓库构建产物）
# 若输出位于 $ASCEND_HOME_PATH/python/site-packages，说明未构建或 PYTHONPATH 顺序不对
```

只有修改了仓库 `autofuse/` 源码（optimize/codegen 等）后，这里的构建产物才包含最新改动；改完代码需重新构建 pyautofuse 再跑用例。

### 回落到 CANN 包内置 autofuse（兜底，非最新源码）

没有 `.dev_env` 或未构建 pyautofuse 时，codegen 会报 `No module named autofuse`。此时把完整 CANN Python 包加入路径（不能只加入裸的 `pyautofuse.so` 所在目录）：

```bash
export PYTHONPATH=$PWD/autofuse/compiler/python:$PWD/autofuse/tests/st:$ASCEND_HOME_PATH/python/site-packages:$PYTHONPATH
```

注意：此路径下用例验证的是 **CANN 包内置的 autofuse**，不是仓库最新源码，不能作为本仓库 autofuse 改动的自验证依据（可用上面的自检命令区分）。CI 或无法构建 pyautofuse 的环境中，回落到 CANN 包是可接受的基线验证。

`$ASCEND_HOME_PATH/python/site-packages` 基于 `set_env.sh` 导出的环境变量；如果未导出，先 `echo $ASCEND_HOME_PATH` 确认，或手动替换为实际 CANN Python 包目录。

## 3. 新用例目录

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

- `case.json`：输入、输出、支持矩阵和 variants。
- `input_ascir.py`：fused ASCIR/codegen 入口。
- `gen_input.py`：固定 seed 生成输入，覆盖边界值、NaN/Inf（如果适用）。
- `reference.py`：独立 golden 计算，不调用被测 AscendC API。
- `steps/*.py`：unfused 分解步骤，每一步有自己的输入输出和 ABI。

## 4. 从零开发一个用例（教程）

本节按五步走通一个最小用例的完整开发流程。可直接对照的真实样例：`cases/isinf_maskedfill_fusion/`（fused ascir 入口 + isinf/logical_or/masked_fill 三个 unfused steps）。第 5-8 章是对各环节的参考说明，编写时两者配合使用。

### 4.1 第一步：建目录与 case.json

目录结构（与第 3 章一致）：

```text
cases/my_case/
├── case.json
├── input_ascir.py
├── gen_input.py
├── reference.py
└── steps/
    ├── step_codegen.py     # 可选：多 step 共用的 codegen 骨架
    ├── op_a.py
    └── op_b.py
```

最小可跑 `case.json` 模板：

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

字段逐条说明：

- `schema_version`：固定为 `1`。
- `case_id`：必须与目录名一致。
- `graph_name`：fused graph 名，供 codegen/JIT 标识。
- `inputs`/`outputs`：tensor 声明的占位 shape 与 dtype；`dynamic: true` 表示实际 shape 由运行参数 `--shape` 决定。
- `verification`：`functional`/`precision` 开关与容差（`atol`/`rtol`）。
- `performance`：性能是否 `required`、是否 `profiler`、默认 `metric`。
- `support_matrix[].soc`：必须与 `profiles/<soc>.json` 的 `profile` 字段**精确一致**（字符串全等）。
- `support_matrix[].input_dtypes`/`output_dtypes`：必须覆盖真实输入、输出和 unfused 中间输出。
- `support_matrix[].shapes`：显式列出对齐、非对齐、tail 等 shape；不声明则不会运行。
- `variants.fused`：`codegen_entry` 指向入口脚本文件名，`graph` 与 `graph_name` 对应；unfused 变体见 4.6。

验证点：

```bash
python3 -c "import json; json.load(open('autofuse/tests/st/device_validation/cases/my_case/case.json'))"
pytest autofuse/tests/st/device_validation -q
```

host 测试无失败，说明 case 能被读取且 support-matrix/contract 校验通过。

### 4.2 第二步：写 fused 入口 input_ascir.py

骨架（完整可阅读的图构建代码见 `cases/isinf_maskedfill_fusion/input_ascir.py`）：

```python
import argparse
import json
from pathlib import Path

from autofuse.pyautofuse import ascir, Autofuser, AutofuserOptions


def build_graph(shape):
    # 用 ascir.ops/SizeExpr/Axis 描述 fused 图，见样例 input_ascir.py 的
    # _create_* / _compute_and_output 系列函数
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

要点：

- 四个参数 `--rows/--cols/--profile/--output-dir` 的名字是 runner 调用契约，不能改。
- 产物为 `tiling.h`、`host_impl.cpp`、`device_impl.cpp`；`abi_metadata.json` 可以在 codegen 中显式写出（unfused steps 的 `steps/step_codegen.py` 就是显式写出的），也可以不写——runner 会从 `device_impl.cpp` 的 launch signature 自动补齐并校验。
- 不要硬编码 shape/SoC：shape 来自 `--rows/--cols`，平台信息来自 `--profile` 的 `ascir` 字段。

验证点：手工执行一次 codegen

```bash
python3 autofuse/tests/st/device_validation/cases/my_case/input_ascir.py \
  --rows 128 --cols 128 \
  --profile autofuse/tests/st/device_validation/profiles/ascend910_9362.json \
  --output-dir /tmp/my_case_codegen
ls /tmp/my_case_codegen
```

确认三个 `.h/.cpp` 产物存在，`host_impl.cpp` 含 `AutofuseLaunch` 签名。

### 4.3 第三步：写 gen_input.py 与 reference.py

`gen_input.py` 骨架（固定 seed，注入边界值）：

```python
import numpy as np


def generate_inputs(shape, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-4, 4, size=shape).astype(np.float16)
    # 按需注入 NaN/Inf/0/边界值，见样例 gen_input.py
    x.reshape(-1)[0] = np.inf
    return x
```

`reference.py` 骨架（独立实现，golden 不依赖被测 kernel）：

```python
import numpy as np


def compute_reference(x):
    return np.where(np.isinf(x), 1.0, x).astype(np.float16)
```

要点：

- `gen_input.py` 必须固定 seed，保证生成可复现。
- `reference.py` 只能使用独立数学/框架实现，**禁止调用被测 AscendC API 或被测融合算子**。

验证点：同一 seed 运行两次输入字节一致；reference 输出的 dtype/shape 与 `case.json` 的 `outputs` 声明一致。

### 4.4 第四步：声明 profile 关联

- case `support_matrix[].soc` 与 `profiles/<soc>.json` 的 `profile` 字段是字符串全等匹配，没有 fuzzy/前缀匹配；大小写、分隔符不同都会不匹配。
- 跑设备前用 `aclrtGetSocName()` 确认芯片变体（见第 2 章），检查片段：

```bash
python3 - <<'PY'
import ctypes
import os
lib = ctypes.CDLL(os.path.join(os.environ["ASCEND_HOME_PATH"], "lib64", "libacl_rt.so"))
lib.aclrtGetSocName.restype = ctypes.c_char_p
print(lib.aclrtGetSocName().decode())
PY
```

- 拿到的名字作为 profile 的 `soc_version` 事实依据；`ascir.platform/core_type/ub_size` 等参数必须来自该芯片的实际编译/Device 事实，禁止复制其他 SoC 的 profile（见第 6 章）。

验证点：`--mode prepare` 能生成 manifest，不报 SoC mismatch（命令见 4.5）。

### 4.5 第五步：按验证顺序跑通

验证顺序与命令见第 8 章，每个阶段的通过判据：

```text
1. host 测试：pytest autofuse/tests/st/device_validation -q
   判据：无失败，case 读取与 support-matrix/contract 校验通过
2. --mode prepare --shape 128 128 --warmup 0
   判据：生成 input/golden 与 manifest.json 即成功；prepare 阶段的正常输出是
   prepare_only（stage_status=not_applicable、reason=prepare_only），不会出现 passed
3. --variant fused --mode functional --warmup 0
   判据：stage_status=passed、precision.passed=true、mismatch_count=0、soc_profile 正确
4. --variant unfused --mode functional --warmup 0
   判据：每个 step 的 ABI 校验通过，最终 golden 与 fused 一致
   （需先按 4.6 在 case.json 声明 unfused steps，否则报 `unfused variant must declare steps`）
5. --variant all --mode functional --warmup 0
   判据：fused 与 unfused 都通过
6. --mode performance --metric runner_wall_clock --warmup 2 --repeat 5
   判据：report 的 samples 数量等于 repeat，unit/timing_source 真实
7. --mode performance --profiler --metric device_kernel_duration --warmup 1 --repeat 3
   判据：profiler 导出成功；失败应报 profiler_export_failed，不得用 host 时间冒充
```

functional/performance 阶段出现 `prepare_only`/`skipped`/`not_applicable` 均不能算功能通过；`passed` 判据仅用于 functional/performance 阶段。

### 4.6 unfused steps 编写说明

- `steps/*.py` 每个文件是一个独立 CLI，契约与 `steps/step_codegen.py` 的 `run_step_codegen` 一致：接收 `--rows/--cols/--profile/--output-dir`，op 名称与输入输出 dtype 在脚本内固定。例如：

```python
from step_codegen import run_step_codegen

if __name__ == "__main__":
    run_step_codegen("isinf_graph", "IsInf", ("float16",), "uint8")
```

- `case.json` 的 `variants.unfused.steps[i]` 声明每个 step：`script` 指向上面的脚本；`inputs` 中的 `"$previous"` 表示取上一步的输出文件，其余形如 `{"file": "input_N.bin", "dtype": ..., "shape": ...}`；`outputs` 声明本步产物。
- 每个 step 有自己的输入输出与 ABI：`steps/step_codegen.py` 的 `_write_artifacts` 会为每个 step 显式写出自己的 `abi_metadata.json`，runner 对每一步单独校验 ABI，不能复用 fused 的 ABI。
- `case.json` 的 `input_dtypes`/`output_dtypes` 必须覆盖 unfused 的中间输出（如 uint8 mask），否则 matrix 校验失败。

### 4.6.1 可选 aclnn 模式（直接调用 CANN aclnn 算子）

`steps` 条目除了 ASCIR 步骤（`steps/*.py` + codegen + JIT），还可以声明 `"aclnn": "<OpName>"` 直接使用 CANN 原生 aclnn 算子（如 `IsInf`/`LogicalOr`/`MaskedFillScalar`/`MaskedFillTensor`），此时不需要 `script`：

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

流程差异：

- 跳过 codegen/JIT：不产 `tiling.h`/`host_impl.cpp`/`device_impl.cpp`，不调用 `jit_adapter.py`，也不需要 `abi_metadata.json`；
- flat request 携带 `aclnn_op`；runner 走 aclnn step executor（`backend/aclnn_executor.cpp`）直接调用 CANN aclnn API；
- 未知算子报 `unknown_aclnn_op`。

JIT 环境要求：全部步骤为 aclnn 时无需设置 `AUTOFUSE_DEVICE_JIT`；fused 或含 ASCIR 步骤时仍需要。

### 4.6.2 aclnn 调用方式（runner 内部执行序列）

aclnn 步骤在设备上的执行遵循 CANN 统一的"查询-分配-创建-绑定-执行-同步-释放"调用序列（`backend/aclnn_executor.cpp` 按算子分发表实现，`IsInf`/`LogicalOr`/`MaskedFillScalar`/`MaskedFillTensor` 已支持，新算子按同模式扩展）：

```cpp
// 1) 查询 workspace 需求（算子特有）
aclnnXxxGetWorkspaceSize(desc, &workspace_size);
// 2) 分配 workspace（设备内存）
aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_NORMAL_ONLY);
// 3) 用设备内存视图描述输入/输出 tensor（shape/dtype/format=ACL_FORMAT_ND）
aclCreateTensor(shape, dims, dtype, 0, nullptr, 0, ACL_FORMAT_ND, addr, &tensor);
// 4) 创建算子执行器并绑定输入输出
aclnnCreateXxx(&executor);
aclnnSetXxxInputTensor(executor, input_tensor);   // 逐个输入
aclnnSetXxxOutputTensor(executor, output_tensor); // 输出
// 5) 异步执行（任务入队 stream）
aclnnXxx(executor, stream, workspace, workspace_size, ...);
// 6) 同步等待执行完成（必须：此后才能安全释放）
aclrtSynchronizeStream(stream);
// 7) 释放（同步之后）——workspace 与 tensor 描述符
aclDestroyTensor(tensor);  aclrtFree(workspace);
```

关键语义：

- **`Execute` 是异步入队**：`aclnnXxx(...)` 返回后 kernel 可能尚未执行，必须先 `aclrtSynchronizeStream` 再释放 workspace/tensor（与 AscendC 异步拷贝的内存生命周期约束一致）；
- 输入数据先由 runner 按 flat request 的 `tensor_specs` 从 `.bin` H2D，输出在同步后 D2H，再走与 ASCIR 步骤相同的 decode/精度/采样流程；
- `warmup/repeat` 采样与 msprof 设备时长采集对 aclnn 步骤与 ASCIR 步骤完全一致（kernel 名过滤使用 aclnn 算子名，如 `IsInf` 会命中 `IsInf_xxxx` 任务记录）；

示例用例 `isinf_maskedfill_fusion` 提供 `--variant unfused_aclnn`，可直接复现 aclnn 单算子链路（无需 `AUTOFUSE_DEVICE_JIT`）。

| 维度 | ASCIR 步骤 | aclnn 步骤 |
|------|-----------|-----------|
| 执行栈 | Autofuse 同栈（steps/*.py + codegen + JIT） | CANN 原生 aclnn 算子 |
| 适用算子 | 任意可 decompose 的 Autofuse 算子 | 需要目标 SoC 的 CANN 已注册该算子 |
| 产物 | `tiling.h`/`host_impl`/`device_impl` + `abi_metadata.json` | 无 |
| 未知算子 | 不适用（codegen/JIT 阶段失败） | `unknown_aclnn_op` |

`$previous` 串接与输入输出文件契约不变；精度与报告流程和 ASCIR 步骤一致；msprof 的 kernel 名过滤使用 aclnn 算子名。aclnn 步骤与 ASCIR 步骤可在同一个 `steps` 列表中混用。

FAQ：如何选用两种 unfused 步骤模式？——需要验证 Autofuse 自身算子的 codegen/JIT 产物时用 ASCIR 步骤；算子不在 Autofuse 支持范围、或只想验证数据流与精度框架时，用 aclnn 步骤直接执行 CANN 原生算子。

### 4.7 常见误区

- **SoC 名不精确匹配**：`soc` 写成 `Ascend910`、`ascend910` 或 `ascend910_9362`（大小写/下划线不一致）都匹配不上 profile 的 `ascend910_9362`，运行时报 SoC mismatch，不会执行设备任务。
- **dynamic shape 缺失**：`inputs`/`outputs` 不声明 `dynamic: true`，或 `shapes` 不覆盖非对齐/tail shape，会导致 shape 校验失败或漏测 tail 路径。
- **golden 用了被测 kernel**：`reference.py` 调用被测 AscendC/融合算子，mismatch_count=0 是必然结果，无法证明正确性。
- **warmup 与样本数无关**：wall-clock 采样数恒等于 `repeat`，与 warmup 大小无关（`--warmup >= --repeat` 不会造成样本数不匹配），warmup 只决定剔除前 N 次样本。仅当 profiler 记录不足（msprof 丢记录）导致 `samples` 少于 `repeat`，且该阶段 performance 声明 `required` 时，report 才报 `performance samples do not match repeat`。
- **把 host 时间当设备时间**：`runner_wall_clock` 是 ms 级粗粒度参考，融合收益对比必须用 `device_kernel_duration`（us）；profiler 导出失败时不得伪装成设备内核时长。
- **复制其他 SoC 的 profile 参数**：platform/core_type/ub_size 必须来自实际芯片事实，从 `Ascend910B` 复制到别的 SoC 会产出不可信的设备结果。

## 5. 编写 case.json

最小结构：

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

注意：

- `soc` 必须与 profile 的 `profile` 字段精确一致。
- `input_dtypes`/`output_dtypes` 必须覆盖实际输入、输出和 unfused 中间输出。
- 非对齐、tail、边界 shape 应显式列入 `shapes`。
- `required` 失败必须阻断对应阶段；`optional` 能力不可用只能标记 skipped/not-applicable。
- `unfused` 使用 `"$previous"` 传递上一步输出；fused/unfused 必须使用相同最终 golden。

## 6. 编写 profile

profile 的平台字段必须来自 CANN/runtime/Device 编译事实，禁止复制其他 SoC 参数：

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

`max_tiling_bytes` 和 `max_workspace_bytes` 没有可靠芯片级事实时应省略，不得使用默认值或 ABI 类型范围猜测。

## 7. Codegen、输入和 golden

`input_ascir.py` 应接收 `--rows`、`--cols`、`--profile`、`--output-dir`，从 profile 获取平台信息，并生成：

```text
tiling.h
host_impl.cpp
device_impl.cpp
abi_metadata.json
```

codegen 不应硬编码 shape/SoC。`gen_input.py` 使用固定 seed，`reference.py` 使用独立数学/框架实现；不得调用被测 kernel 生成 golden。

## 8. 推荐验证顺序

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

`prepare` 只生成输入、golden 和 manifest，不发设备任务。

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

分别替换为 `[128,130]`、`[127,129]`；对照验证时使用 `--variant unfused` 和 `--variant all`。

通过条件：

```text
stage_status=passed
precision.passed=true
precision.mismatch_count=0
soc_profile=<expected profile>
run_parameters.selected_shape=<requested shape>
```

`prepare_only`、`skipped`、`not_applicable`，以及 `mismatch_count=0` 但 `precision.passed=false`，都不能算功能通过。

## 9. 性能与 SQLite profiler 依赖

指标含义：

- `runner_wall_clock`：host launch+sync 墙钟时间，单位 `ms`，只能作为粗粒度参考。
- `device_kernel_duration`：Profiler 导出的设备 AI Core 时间，单位 `us`，正式融合收益应使用此指标。

使用多次采样：

```bash
--mode performance --metric runner_wall_clock --warmup 2 --repeat 5
--mode performance --profiler --metric device_kernel_duration --warmup 1 --repeat 3
```

CANN 的 `msprof_analysis.so` 可能依赖精确名称 `libsqlite3.so`，而发行版通常只提供 `libsqlite3.so.0` 或 `libsqlite3.so.0.x.y`。缺少无版本名库时只影响：

- profiler 离线 export
- `task_time_*.csv` 生成
- `device_kernel_duration`
- fused/unfused 设备时长对比

不影响：

- case/profile 校验
- codegen/JIT
- AscendC Device 编译
- ACL functional/precision
- `runner_wall_clock`

可使用真实、匹配架构的兼容库目录（`/path/to/...` 仅为示意，请替换为实际兼容库所在目录）：

```bash
export DEVICE_VALIDATION_SQLITE_LIB_PATH=/path/to/sqlite/libsqlite3.so
```

验证库文件（路径仅为示意，使用实际库文件）：

```bash
file /path/to/libsqlite3.so
readelf -d /path/to/libsqlite3.so | grep SONAME
LD_LIBRARY_PATH=$(dirname /path/to/libsqlite3.so):$LD_LIBRARY_PATH \
python3 -c 'import ctypes; ctypes.CDLL("libsqlite3.so"); print("sqlite load passed")'
```

禁止提交 SQLite 二进制、私有兼容目录或系统 symlink。Profiler 失败时 report 必须为 `profiler_export_failed`/`profiler_export_unavailable`，不得把 host 时间冒充设备内核时间。

## 10. CMake/CTest 分层

- host pytest 可直接运行：`pytest autofuse/tests/st/device_validation -q`。
- host CTest 在 device validation 子目录纳入配置后注册。
- runner contract、real-codegen、hardware 分别由对应 CMake option opt-in。
- runner contract 不得带 `real_codegen` label。
- CMake 不得引用不存在的 Python 文件。

常用命令：

```bash
cmake --build build --target device_validation_ut device_validation_runner -j 8
./build/autofuse/tests/ut/device_validation/device_validation_ut
ctest --test-dir build/autofuse/tests/st --output-on-failure -L '^device_validation$' -j 8
ctest --test-dir build/autofuse/tests/st --output-on-failure -L '^device_validation_runner$' -j 8
```

## 11. 故障诊断

| 阶段 | 典型错误 | 优先检查 |
|---|---|---|
| Profile | SoC mismatch | `aclrtGetSocName()`、case `soc`、profile `profile` |
| Codegen | `No module named autofuse` | CANN 完整 Python 包、`PYTHONPATH`、`LD_LIBRARY_PATH` |
| JIT | generated artifact missing | codegen stdout/stderr、ABI metadata |
| Matrix | `invalid_artifact_path` | parent/nested artifact、traversal、symlink |
| ABI | `abi_mismatch` | fused 顶层 ABI、unfused step ABI |
| Runtime | ACL failure | device、buffer、stream、动态库 |
| Precision | mismatch | golden、dtype、shape、容差 |
| Profiler | `libsqlite3.so`/export failure | SQLite ELF 架构、SONAME、analysis root |
| CTest | no tests found | CMake option、label、工作目录 |

先处理日志中的首个有效错误，不要把级联错误当作多个根因。

## 12. 提交前 Checklist

```text
[ ] case.json contract 完整，SoC 与 profile 精确匹配
[ ] profile 平台/资源参数有 CANN 或 Device 事实依据
[ ] 没有新增框架核心 SoC 硬编码
[ ] 对齐、非对齐、tail shape 均通过
[ ] fused/unfused/all precision 通过
[ ] profiler 结果有真实 metric、unit、timing_source
[ ] host pytest、device_validation_ut、CTest 通过
[ ] README 命令可从仓库根目录执行
[ ] build、artifact、SQLite 私有兼容库未提交
```

## 13. Ascend910_9362 实测参考

当前 case 在 CANN 9.2、`Ascend910_9362`、device 0 上的既有 shape（128x128、128x130、127x129）已完成 fused/unfused functional/precision，均为 `mismatch_count=0`；`512x512` 已在 Ascend950PR 完成 fused/unfused（含 aclnn 模式）实测通过，`Ascend910_9362` 的 `512x512` 仅为声明扩展（待该设备实测）。Profiler 设备时长也已成功导出：fused p50 约 `4.72 us`，unfused p50 约 `15.52 us`。这些是当前环境和当前 case 的实测结果，不是其他 SoC、CANN 版本或用例的通用承诺。
