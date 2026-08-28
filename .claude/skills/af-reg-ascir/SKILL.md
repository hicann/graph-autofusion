---
name: af-reg-ascir
description: |
  识别 graph-autofusion 新增或更新 ASCIR 的完整修改面，并给出注册、regbase、Codegen、Python、UT/ST/E2E、上板故障的最小修改方案。
  **必须触发的场景**：用户提到 ASCIR 注册/更新、ascir op、AscendC regbase、dtype、tmp buffer、pyascir、ASCIR UT/ST/E2E、上板编译或运行错误。
  **必须触发ATT性能建模的场景**：用户提到 ASCIR reg API 外抛 for 循环、ATT 性能建模、perf 公式/性能模型、repeat_time/call_count、af-perf-modeler、性能模型缺失或需复核、新增/更新涉及外抛的 ASCIR（如 Load/Store/NDDMA/Transpose/Cast/Compare/Where/IsNan/IsFinite/IsInf 或融合进 VectorFunc 的 elewise）。
---

# graph-autofusion ASCIR 注册辅助

辅助用户在 graph-autofusion 中完成 ASCIR 算子的新增、修改、数据类型扩展、tmp buffer 注册，以及配套 UT/ST 生成与验证。

---

## 触发场景

用户请求包含以下任一意图时，必须使用本 skill：

- 新增 ASCIR 算子、注册 ASCIR 算子、添加 `REG_ASC_IR`。
- 修改已有 ASCIR 算子的注册、实现类、API 名称、输入输出、ComputeType 或 dtype。
- 为 ASCIR 算子增加 dtype 支持、类型转换逻辑或 cast-map。
- 为 ASCIR 算子添加或修改 tmp buffer 注册、`CalcTmpBufSize`、`CalcXxxTmpSizeV2`。
- 为 ASCIR 注册变更补充 UT、ST、E2E 测试或测试图函数。
- 排查 ASCIR 注册、codegen、ATT、pyascir、性能模型、测试编译失败。
- 涉及外抛 for 循环的 ASCIR reg API（必须触发 ATT 性能建模），或外抛未命中时需做 vector bound 兜底校验。

---

## 执行原则

- 先明确用户目标是“新增算子”还是“修改已有算子”，再决定最小修改范围。
- 不做无关重构，不批量格式化无关文件，不新增与需求无关的抽象。
- 修改前必须检索现有同类算子，优先复用项目已有模式。
- 修改后必须验证，至少完成静态检索；涉及编译或测试时按 af-build-runner 和 af-test-developer 执行。
- 所有编译命令必须限制并行度，优先使用 `-j 8`。
- 不自动提交代码，除非用户明确要求 commit。

---

## 信息收集

### 新增 ASCIR 算子

若用户未提供完整信息，一次性收集必需参数：

```text
OpName(PascalCase) InputNum(0/1/2/3) OutputNum(1) ComputeType(elewise/reduction/...) DataType(float32,float16,...) TmpBuf(true/false)
```

可选参数：

```text
version(v1/v2，默认 v2) test(true/false，默认 false) cast-map(如 float16:float32,int8:int32)
```

### 修改已有 ASCIR 算子

先确认以下信息：

- 目标算子名与版本：`OpName`、`v1` 或 `v2`。
- 修改类型：注册项、dtype、tmp buffer、API 名、ATT、Codegen、Python API、性能模型、UT/ST。
- 是否需要保持外部 API/OpType/节点名兼容。
- 是否需要补充或更新 UT/ST。

### 批量 dtype 注册

优先让用户提供 JSON 配置；如果用户没有 JSON，则先整理为等价结构再执行。

```json
[
  {
    "ascir_name": "Abs",
    "ascir_dtype": "DT_FLOAT16",
    "version": "v2",
    "enabled": true,
    "dtype_config": {
      "requires_cast": false,
      "cast_to_dtype": null
    },
    "st_config": {
      "generate": true,
      "dims": 3,
      "version": "v2"
    }
  }
]
```

---

## 关键路径

根据仓库实际文件为准，当前工程常见修改点包括：

- V2 ASCIR 注册表：`autofuse/v35/ascir/generator/ascir_builtin_ops_v2.cpp`。
- V1 ASCIR 注册表：`autofuse/ascir/generator/ascir_builtin_ops_v1.cpp`。
- 注册宏与注册表实现：`autofuse/inc/graph_metadef/graph/ascendc_ir/ascir_register.h`、`autofuse/graph_metadef/graph/ascendc_ir/generator/ascir_register.cc`、`autofuse/inc/graph_metadef/graph/ascendc_ir/ascir_registry.h`。
- V2 ATT 实现：`autofuse/v35/ascir/generator/v2_ascir_att_impl.h`。
- V2 Codegen 实现：`autofuse/v35/ascir/generator/v2_ascir_codegen_impl.h`。
- V2 tmp buffer 声明与实现：`default_reg_func_v2.h`、`autofuse/v35/ascir/reg_func/`。
- V2 性能模型：`autofuse/v35/att/api_perf_register/ascir_api_perf_v2.cpp`，Reduce 类性能模型在 `ascir_reduce_api_perf_v2.cpp`。
- Python C 扩展：`autofuse/compiler/py_module/pyascir.h` 维护 `REGISTERED_OPS`；`pyascir.cpp` 创建 `ascir.ops.OpName` 并暴露 IR 属性。
- Python 易用 API：`autofuse/compiler/python/ascir_api.py`。
- ASCIR/Codegen/Perf UT：`autofuse/tests/ut/ascir/`、`autofuse/tests/ut/codegen/`、`autofuse/tests/v35/ut/att/gen_model_info/api_perf_register/`、`autofuse/tests/v35/ut/codegen/`。
- ASCIR/Perf/Python ST：`autofuse/tests/v35/st/att/gen_model_info/`、`autofuse/tests/st/codegen/ascir_tool/`、`autofuse/tests/st/python/`。
- 测试脚本：`scripts/test/run_autofuse_test.sh`。

修改前必须通过检索确认准确路径，不要只凭路径名称假设。

---

## 当前工程注册机制

- `REG_ASC_IR(OpName)` 会创建 `AscirRegister` 静态对象，静态对象拷贝构造时调用 `AscirRegistry::RegisterAscIr` 完成注册。
- `AscirRegister` 链式接口常用 `.Input()`、`.Output()`、`.DynamicInput()`、`.DynamicOutput()`、`.Attr<T>()`、`.ComputeType()`、`.DataTypes()`、`.SameTmpBufSizeFromFirstInput()`、`.Impl()`。
- V2 注册通常使用 `.Impl(v2_soc_versions, {...})`，当前 v35 典型 SoC 列表在注册表中为 `3510`、`5102`。
- `.Impl()` 中通常同时绑定 ATT creator、Codegen creator 和 dtype symbol 映射。
- 注册查找按当前平台字符串和 ASCIR type 获取 ATT/Codegen 实现，修改 OpType 或 API 名时必须区分“ASCIR type 名”和“AscendC API 调用名”。
- 新增 V35/V2 算子优先修改 `autofuse/v35/ascir/generator/` 目录，只有明确需要兼容 V1 时才同步修改 V1 路径。

---

## 新增 ASCIR 算子流程

### Step 1: 检查是否已存在

检索 `OpName`、`REG_ASC_IR(OpName)`、`kOpName`、`OpNameAscIrCodegenImpl`、`OpNameAscIrAttImpl`。

- 已存在：停止新增流程，改为“修改已有算子”流程。
- 不存在：继续注册。

### Step 2: 注册 ATT 实现类

在 ATT ASCIR 实现注册文件中新增一行注册宏，放在现有同类注册附近。

```cpp
REG_ASC_IR_ATT_CLASS_DEFINE(OpName);
REG_ASC_IR_ATT_V2_CLASS_DEFINE(OpName);
```

选择 v1/v2 对应宏，不要两个都无条件添加。

### Step 3: 创建 Codegen 实现类

类名规则：

- v1：`OpNameAscIrCodegenImpl`
- v2：`OpNameAscIrCodegenImplV2`

必须实现或确认以下能力：

- `GetApiCallName()`：根据输入数量、版本和 tmp buffer 选择调用类型。
- `GetApiName()`：默认返回 `OpName`，若 AscendC API 名不同则返回实际 API 名。
- `LoadApiHeaderFiles()`：返回项目内 regbase 源码头，常见如 `{op_name}_reg_base.h`；有 helper 时按“helper 在前、主 API 在后”的依赖顺序列出。
- `IncludeApiHeaderFiles()`：当主 API 或任一 helper 调用 CANN 官方 AscendC 接口时，返回其真实官方头文件；不能用 `LoadApiHeaderFiles()` 替代。
- `GetConversionDtype()`：仅在需要 dtype 转换时添加。
- `CalcTmpBufSize()`：仅在需要 tmp buffer 时添加。
- `IsNodeValid()`：存在 scalar、shape、dtype 或特殊属性约束时必须补充校验。

常见 `GetApiCallName()` 选择：

| 版本 | 输入数量 | TmpBuf=false | TmpBuf=true |
|------|----------|--------------|-------------|
| v1 | 1 | `UnaryApiCall` | `UnaryTmpApiCall` |
| v1 | 2 | `BinaryApiCall` | `BinaryTmpApiCall` |
| v2 | 1 | `UnaryApiCall` | `UnaryApiTmpCall` |
| v2 | 2 | `BinaryApiCallV2` | `BinaryApiCallV2` / `BinaryApiTmpCall` |

当前工程还存在 `UnaryBitWidthChangeApiCallV2`、Reduce 专用 ApiCall、微 API 相关接口等特殊路径。不要只按输入数量机械选择，必须先在 `v2_ascir_codegen_impl.h` 中找同类算子参考。

#### `IncludeApiHeaderFiles()` 强制判定

创建或更新 Codegen 类时，必须读取主 regbase 文件及其全部 helper，搜索外部 API 调用，不能因为本地 UT 或生成文本通过就省略官方头。

- 如果实现只使用生成 kernel 公共环境已提供的基础类型/指令，且同类算子无需额外官方头，可以不重写，并说明证据。
- 如果出现 `AscendC::Adds`、`AscendC::Div`、`AscendC::Power`、`Lgamma`、`ERFC::` 等调用，必须对照 CANN 头文件和已有同类实现添加最小头集合。
- 项目内 `*_reg_base.h` 只能放在 `LoadApiHeaderFiles()`；`basic_api/...`、`adv_api/...` 等 CANN 官方头只能放在 `IncludeApiHeaderFiles()`。
- 依赖可能位于 helper 中，不能只扫描主 `{op_name}.h`。

普通只加载项目 regbase 的形式：

```cpp
[[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
  return {"example_op_reg_base.h"};
}
```

同时依赖项目 helper 和 CANN 官方 API 的形式：

```cpp
[[nodiscard]] std::vector<std::string> LoadApiHeaderFiles([[maybe_unused]] bool is_dynamic) const override {
  return {
      "example_op_helper_reg_base.h",
      "example_op_reg_base.h",
  };
}

[[nodiscard]] std::vector<std::string> IncludeApiHeaderFiles() const override {
  return {
      "basic_api/kernel_operator_vec_binary_intf.h",
      "basic_api/kernel_operator_vec_binary_scalar_intf.h",
      "adv_api/math/erfc.h",
      "adv_api/math/lgamma.h",
      "adv_api/math/power.h",
  };
}
```

上面 5 个官方头是 Igamma/Igammac 的实际示例，不是所有算子的固定模板。每个算子必须根据实际调用裁剪，禁止无差别复制。

验证时检查生成 kernel：项目 helper 的源码应被内联，官方头应出现在 kernel 顶部的 `#include` 中；两者缺一都可能只在真实 Device 编译时暴露。

### Step 4: 注册 tmp buffer

仅当 `TmpBuf=true` 时执行。

需要完成：

- 在 `default_reg_func.h` 或 `default_reg_func_v2.h` 添加声明。
- 在 reg_func 目录新增实现文件。
- 在 Codegen 实现类中新增 `CalcTmpBufSize()`。
- 如果可以复用注册层公共策略，可评估 `.SameTmpBufSizeFromFirstInput()`；否则优先在 Codegen 类中重写 `CalcTmpBufSize()`。

函数命名：

```cpp
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcOpNameTmpSize(const ge::AscNode &node);
std::vector<std::unique_ptr<ge::TmpBufDesc>> CalcOpNameTmpSizeV2(const ge::AscNode &node);
```

实现策略：

- 优先查找 asc-devkit 或本仓同类算子的 tmp buffer 计算逻辑。
- 固定系数场景使用 `CALC_PROC * 256` 模式。
- shape 相关场景参考 `exp2_v2.cpp`、`pow_v2.cpp` 等动态实现。
- 当前工程会将 `node->attr.tmp_buffers` 传递到 tiling data 和 kernel codegen，tmp buffer id/size 需要与 codegen 复用逻辑一致。
- 找不到可靠参考时，使用项目已有兜底逻辑并在交付说明中标明需人工确认。

### Step 5: 添加 ASCIR 注册

在 ASCIR 注册表中添加或修改：

```cpp
REG_ASC_IR(OpName)
    .Input("x", "T")
    .Output("y", "T")
    .DataType("T", TensorType{DT_FLOAT})
    .ComputeType(ge::ComputeType::kComputeElewise)
    .Impl(v2_soc_versions, {
        ge::ascir::AscIrImplCreator<ge::ascir::OpNameAscIrAttImplV2>(),
        ge::ascir::AscIrImplCreator<ge::ascir::OpNameAscIrCodegenImplV2>(),
        {{"T", TensorType{DT_FLOAT}}}
    });
```

必须按用户指定 dtype 精确添加，不要额外添加未要求的数据类型。

### Step 6: 补齐周边注册

按现有模式补齐：

- `att_const_values.h` 或现有常量定义位置：`inline const std::string kOpName = "OpName";`
- V2 性能模型：在 `ascir_api_perf_v2.cpp` 中补 `ApiPerfRegisterV2(kOpName, ...)`，注册名会对齐到 `OpNameV2`。
- `pyascir.h` 的 `REGISTERED_OPS` 宏列表：严格按 C++ 注册顺序补 `OP(OpName)`，由 `pyascir.cpp` 创建并暴露 `ascir.ops.OpName`。
- Python API 函数：按输入输出数量选择 `_common_in_N_out_1_normal_op` 等既有封装。

ATT 宏默认返回 `OpNameV2` 作为 perf 名称，性能模型注册名必须与此一致；如果手写 ATT 类改了名称，perf 注册也要同步。

### Step 7: 生成或更新测试

若用户要求 `test=true`、新增 dtype 需要覆盖，或修改影响 codegen/运行行为，必须补充 UT/ST。

推荐用例名：

```text
{lower_op_name}_{lower_dtype_without_dt_prefix}
```

ST 生成通常需要：

- share graph 函数声明和实现。
- `{case_name}_test` 目录。
- `CMakeLists.txt`。
- `{case_name}_backend_generator.cpp`。
- `test_e2e_{case_name}_expect_kernel.cpp`。
- 父级 CMakeLists 与 `scripts/test/run_autofuse_test.sh` 目标列表。

---

## 修改已有 ASCIR 流程

### Step 1: 建立引用清单

先检索以下关键词并列出影响范围：

- `OpName`
- `REG_ASC_IR(OpName)`
- `kOpName`
- `OpNameAscIrAttImpl`
- `OpNameAscIrCodegenImpl`
- `OpNameV2`
- `ApiPerfRegisterV2(kOpName`
- `CalcOpNameTmpSize`
- `OP(OpName)`
- `ascir.ops.OpName`
- Python API 函数名
- UT/ST 用例名与测试目标名

### Step 2: 判断兼容性

修改 API 名、OpType、Python API 或 dtype 时，必须明确：

- 是否只改内部 AscendC API 调用名。
- 是否同步修改 ASCIR `OpType` / 节点名。
- 是否需要保留旧 API 的 wrapper。
- 现有 UT/ST 是否需要改名或只改调用点。

### Step 3: 精准修改

只改必要位置：

- 内部 API 名变更：通常只改 Codegen `GetApiName()` 或测试中的直接调用。
- ASCIR type 名变更：需要同步 `REG_ASC_IR`、ATT/Codegen 类引用、perf 名称、Python ops 暴露、Python API、UT/ST。
- dtype 变更：改 `DataType`、Impl dtype map，必要时改 `GetConversionDtype()`。
- tmp buffer 变更：改声明、实现、Codegen `CalcTmpBufSize()`。
- 输入输出变更：同步改注册、Python API、Codegen ApiCall、UT/ST 图函数和期望内核。

### Step 4: 验证影响范围

修改后再次检索旧名称和新名称，确认：

- 旧符号没有残留在必须改名的位置。
- 新符号已覆盖所有调用点。
- 测试名、目标名是否按用户要求同步。

---

## dtype 扩展流程

### Step 1: 验证目标算子存在

在 ASCIR 注册表中查找目标 `ascir_name`。

### Step 2: 检查 dtype 是否已注册

在目标算子注册块中搜索目标 `DT_*`。

- 已注册：记录并跳过。
- 未注册：继续添加。

### Step 3: 添加 dtype

按现有注册块风格添加：

- `.DataType("T", TensorType{...})`
- `.Impl(..., {{"T", TensorType{...}}})`

如果该算子按多个 dtype 分拆注册，应遵循现有结构，不混入不一致模式。

注意同时检查 `.DataTypes()`、`.DataType()` 和 `.Impl()` dtype map 三类位置；当前工程可能通过 symbol `T`、`T1`、`T2` 绑定多个输入输出 dtype。

### Step 4: 添加类型转换

当 `requires_cast=true` 时，在 Codegen 实现类中添加或扩展 `GetConversionDtype()`。

```cpp
[[nodiscard]] std::pair<std::vector<ge::DataType>, std::vector<ge::DataType>> GetConversionDtype(const ge::AscNode &node) override {
  std::map<ge::DataType, ge::DataType> dtype_conversion_map = {
      {ge::DT_INT8, ge::DT_INT32},
  };
  return GetConversionFromDtypeMap(node, dtype_conversion_map);
}
```

转换关系必须严格来自用户配置，不要推断额外转换。

### Step 5: 生成 ST

当用户要求生成测试，或 dtype 需要 cast，优先生成 ST 覆盖新 dtype。

---

## UT/ST 生成流程

### UT

适用场景：注册解析、ATT、Codegen、dtype 转换、tmp buffer 大小计算等可在 host 侧验证的逻辑。

要求：

- 优先查找同目录同类测试文件复用 fixture 和命名风格。
- 测试命名体现算子名、dtype、版本和场景。
- 覆盖正常路径和关键异常路径。

当前工程常见参考：

- ASCIR 注册/图基础：`autofuse/tests/ut/ascir/test_ascir.cpp`、`test_ascir_ops.cpp`、`test_ascir_graph.cpp`。
- Codegen/API Call：`autofuse/tests/ut/codegen/api_call/`、`autofuse/tests/v35/ut/codegen/reg_api_call/`。
- V2 perf：`autofuse/tests/v35/ut/att/gen_model_info/api_perf_register/test_ascir_perf_v2.cpp`。
- Python API：`autofuse/tests/ut/python/test_python_ascir.py`。

### ST / E2E

#### 强制决策规则

- **每个新增 ASCIR 都必须考虑 backend E2E，默认应补充。** 如果判断不需要，方案中必须写明已有哪个 E2E 能覆盖其注册、Codegen 和执行路径，不能静默省略。
- **更新已有 ASCIR 必须先定位现有 E2E 并判断是否同步修改。** 以下任一变化通常需要更新：输入/输出数量或顺序、dtype、IR 属性、API/ApiCall 名、tmp buffer、shape/stride、生成 kernel 文本、参考值、容差、编译选项、target 名。
- regbase UT 只验证 API 实现，Codegen UT 只验证调用文本；它们都不能替代 `Data → Load → Op → Store → Output` 的完整 backend E2E。

#### BesselJ0 V2 参考结构

参考目录：`autofuse/tests/v35/st/backend_e2e_v2/bessel_j0_store_test/`。

1. `CMakeLists.txt` 使用 `backend_e2e_st_test` 定义 Codegen、生成 kernel/tiling 和运行测试：

```cmake
backend_e2e_st_test(bessel_j0_store_test
    CODEGEN bessel_j0_store_backend_generator.cpp
    KERNEL_SRC
        bessel_j0_store_test_kernel.cpp
        bessel_j0_store_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_bessel_j0_store_kernel.cpp)
target_include_directories(bessel_j0_store_test_e2e_v2 PRIVATE ${CODE_ROOT_DIR}/v35/ascendc/api_regbase)
```

只有测试直接依赖 regbase 源码头时才增加对应 include 目录，不要机械复制。

2. 后端生成器必须：

- 设置并在 TearDown 中重置 `RuntimeStubV2` 和平台上下文；
- 调用 `ShareGraph::{OpName}StoreFusedGraph(dims)`；
- 执行 `Optimizer::Optimize` 和 `Codegen::Generate`；
- 断言生成 kernel 包含实际 API 名，例如 `BesselJ0Extend`；
- 将去除子目录 include 后的 kernel、tiling、tiling data 写入 `KERNEL_SRC_LIST` 指定文件；
- 捕获异常并保证生成失败会使测试失败。

3. 运行 E2E 必须：

- 声明生成 kernel 和 `AutofuseTiling`；
- 为每个输入、输出和期望值分配/释放 GM；
- 生成覆盖定义域和分支的数据，不能只填随机值；
- 构造独立参考结果，不能调用被测 AscendC API 自己作为 golden；
- 调用 `AutofuseTiling`，再通过 `ICPU_RUN_KF` 执行 kernel；
- 按 dtype/数值特征设置合理绝对误差或相对误差，统计 `diff_count`；
- 参数化至少覆盖对齐、非对齐和较大 shape，例如 `{32,16}`、`{32,18}`、`{512,15}`。

4. share graph 必须同步：

- 在 `share_graph.h` 声明 `{OpName}StoreFusedGraph(size_t dims_size)`；
- 在 `share_graph.cc` 构造 `Data → Load → Op → Store → Output`；
- 设置 Data/Output index、输入输出 dtype、axis/repeats/strides 和 IR 属性；
- 根据接口选择一输入、二输入、多输出对应的外层 ComputeGraph helper，不能固定套用一元模板。

5. 构建和执行入口必须同步：

- 在 `autofuse/tests/v35/st/backend_e2e_v2/CMakeLists.txt` 增加 `add_subdirectory({case_name}_test)`；
- 在 `scripts/test/run_autofuse_test.sh` 加入 `{case_name}_test_e2e_v2`，使统一 E2E 流程执行它；
- 不加脚本项不一定影响 target 单独构建，但会造成统一测试遗漏。

#### 更新 ASCIR 时的 E2E 审计表

| 变更 | 必查 E2E 内容 |
|---|---|
| 输入/输出变化 | share graph、kernel extern ABI、GM 分配、`ICPU_RUN_KF` 参数 |
| dtype 变化 | tensor dtype、分配类型、参考计算、误差策略 |
| 新增/修改 Attr | share graph `SetXxx`、Codegen API 模板参数断言、属性边界 case |
| API/ApiCall 改名 | 生成器中的 kernel 字符串断言和期望 kernel |
| tmp buffer 变化 | tiling/workspace、执行参数和相关 shape case |
| 算法或定义域变化 | 输入数据、golden、NaN/Inf/边界处理、容差 |
| Device 编译限制 | target/JIT 的 Device 选项及真实 device 编译验证 |
| 仅内部重构 | 仍运行现有 E2E；确认输出和生成 API 不变后可不改文件 |

版本对应目标名：

- v1：`{case_name}_test_e2e`
- v2：`{case_name}_test_e2e_v2`

其他参考：

- V2 perf ST：`autofuse/tests/v35/st/att/gen_model_info/test_ascir_perf_v2.cpp`。
- ASCIR tool ST：`autofuse/tests/st/codegen/ascir_tool/`。
- Python ST：`autofuse/tests/st/python/test_python_ascir.py`、`test_python_ascir_st.py`。

---

## 完整修改面检查矩阵

新增或更新 ASCIR 时，不得只检查 `REG_ASC_IR`。先按下表逐项标记为“需要修改 / 已覆盖 / 不适用（说明理由）”，再给方案。

| 层级 | 必查内容 | 常见路径 |
|---|---|---|
| AscendC 实现 | 主 API、helper、输入输出、模板属性、支持 dtype、tmp buffer | `autofuse/v35/ascendc/api_regbase/` |
| regbase 构建 | 每个 `.h` 是否生成 `*_reg_base.h`，子目录输出目录是否创建 | `autofuse/v35/ascendc/api_regbase/CMakeLists.txt` |
| regbase 注册 | 每个生成头是否声明字符串并加入 `api_to_file` | `autofuse/v35/codegen/ascendc_reg_base_api_register.cpp` |
| ATT | ATT 类宏、perf 名称、输入输出语义 | `autofuse/v35/ascir/generator/v2_ascir_att_impl.h` |
| Codegen | ApiCall、API 名、Load/Include 头、tmp、dtype 转换、节点校验 | `autofuse/v35/ascir/generator/v2_ascir_codegen_impl.h` |
| ASCIR 注册 | Input/Output/Attr/ComputeType/Impl/dtype map 与顺序 | `autofuse/v35/ascir/generator/ascir_builtin_ops_v2.cpp` |
| 特殊 ApiCall | 多输出、模板属性或新签名是否需新增 ApiCall 类及 CMake 纳入 | `autofuse/v35/codegen/reg_api_call/` |
| tmp buffer | 声明、实现、系数、UT | `autofuse/v35/ascir/reg_func/` |
| 性能模型 | 常量和 V2 perf 注册 | `autofuse/att/base/att_const_values.h`、`autofuse/v35/att/api_perf_register/ascir_api_perf_v2.cpp` |
| Python 类型 | `REGISTERED_OPS` 中按 C++ 注册顺序加入 `OP(OpName)` | `autofuse/compiler/py_module/pyascir.h` |
| Python IR 属性 | Attr 常量、getter/setter、`attr_handlers` | `autofuse/compiler/py_module/pyascir.cpp` |
| Python 易用 API | 一元/二元、多输出、属性 wrapper 和返回顺序 | `autofuse/compiler/python/ascir_api.py` |
| regbase UT | AscendC API 行为、边界、dtype、多输出 | `autofuse/tests/v35/ut/ascendc/api_regbase/` |
| Codegen UT | API 调用名、模板参数、多输出、tmp buffer | `autofuse/tests/v35/ut/codegen/reg_api_call/` |
| share graph | 声明和实现；输入、输出、dtype、属性必须一致 | `autofuse/tests/framework/share_graph/` |
| backend E2E | 子目录三件套、父 CMake、生成 kernel 检查 | `autofuse/tests/v35/st/backend_e2e_v2/` |
| 测试入口 | 是否加入脚本目标；不加入不影响 target 构建，但不会被脚本执行 | `scripts/test/run_autofuse_test.sh` |
| JIT 编译参数 | Host 与 Device 参数必须分离，设备选项进入 bisheng device 命令 | `autofuse/compiler/python/compile_adapter.py`、`ascendc_compile.py` |
| 构建打包 | 新文件/生成文件是否被 CMake、install、package 收集；默认不得牵连 superkernel | 相关 `CMakeLists.txt`、`cmake/package.cmake`、`build.sh` |

### 按算子接口决定额外检查

- 普通一元单输出：`UnaryApiTmpCall` 等同类实现。
- 二元单输出：Igamma/Igammac/Zeta 等必须同步二输入图、Python wrapper 和 launch 参数。
- 双输出：Frexp 必须检查专用 ApiCall、两个输出 dtype/顺序、share graph、launch ABI。
- 模板属性：ShiftedChebyshev T/U/V/W 的 `n` 必须同时存在于 `.Attr<int64_t>("n")`、Python `IrAttr.n` 绑定、图构造、Codegen 模板参数和 UT/E2E。
- 位宽或 dtype 改变：SignBit/Frexp 等必须检查输出 dtype、ApiCall 类型和 Python tensor 描述。
- helper 依赖较多：必须建立依赖拓扑，按“helper 在前、主 API 在后”返回 `LoadApiHeaderFiles()`。

---

## regbase 与官方 API 头文件规则

### `LoadApiHeaderFiles()` 与 `IncludeApiHeaderFiles()` 不可混用

- `LoadApiHeaderFiles()`：加载项目内 `*_reg_base.h` 字符串，并把实现源码内联进生成 kernel。
- `IncludeApiHeaderFiles()`：在生成 kernel 顶部生成真实的 CANN 官方头文件 `#include`。
- 项目 helper 应走前者；`AscendC::Adds/Div/Power`、`Lgamma`、`ERFC` 等官方 API 声明应走后者。

### 禁止 regbase 内部残留项目相对 include

CMake 将 `.h` 原文包装成 raw string，内部 `#include "helper.h"` 不会自动消失。即使 helper 已通过 `LoadApiHeaderFiles()` 内联，编译器仍会去文件系统查找原始头并报错。

处理规则：

1. 搜索新增/修改的所有 regbase 文件中的 `#include "..."`。
2. 项目内部 helper include 从源码头删除。
3. 在 Codegen `LoadApiHeaderFiles()` 中显式按依赖顺序列出对应 `*_reg_base.h`。
4. 在 regbase CMake 中生成每个 helper 的 raw-string 头。
5. 在 `ascendc_reg_base_api_register.cpp` 中完成字符串声明和 map 注册。
6. 不用给 device 编译器临时增加源码目录 `-I` 绕过机制。

### 官方 API 依赖审计

对 regbase 主文件和所有 helper 搜索：

```text
AscendC::  Reg::  ERFC::  Lgamma  Power  Adds  Div  Duplicate  Cast
```

然后对照已有同类实现确定最小官方头集合。不得只根据函数名猜测，也不得因一个未声明错误无差别加入全部头文件。

---

## Python 暴露完整性规则

### 暴露 `ascir.ops.OpName`

- 在 `pyascir.h` 的 `REGISTERED_OPS` 增加 `OP(OpName)`。
- 顺序必须严格跟随 C++ `REG_ASC_IR` 注册顺序，不按字母重新排序。
- 仅增加 `ascir_api.py` wrapper 不能创建 `ascir.ops.OpName`；真正类型由 C++ 扩展生成。

### 暴露 IR 属性

`.Attr<T>("name")` 只生成 C++ `GetXxx/SetXxx`，不会自动成为 Python 属性。每个 Python 要设置/读取的 IR Attr 都检查三处：

1. 属性名常量，如 `kNAttr = "n"`。
2. `DEFINE_IR_ATTR_ACCESSORS`，完成 Python/C++ 类型检查和 `Get/Set` 转换。
3. `IrAttr<OpType>::attr_handlers`，把访问器挂到具体 OpType。

C++ E2E 可直接调用 `op.ir_attr.SetN()`，因此可能通过；Python 上板用例走 `pyascir.cpp`，缺绑定会在构图阶段报 `IrAttr object has no attribute`。这两类验证不可互相替代。

### 运行版本一致性

源码修复后，必须确认目标环境实际加载的是新构建的 Python 文件和 native `.so`：

- 打印 `autofuse`、`pyautofuse` 模块路径。
- 检查安装包/上板目录是否已同步。
- 检查生成 kernel 是否反映新 `Load/IncludeApiHeaderFiles()`。
- 日志参数与当前测试脚本不一致时，优先判断“部署旧文件”，不要重复修改已正确的源码。

---

## ASCIR reg API 性能基线校验（外抛 for 循环先行，vector bound 兜底）

注册或更新一个 ASCIR reg API 后，若需要判断“该 API 是否存在性能问题”，必须按以下**分支流程**执行，不能只凭功能正确就结束：

1. **第一道门 — 外抛 for 循环静态判定**（无需上板）：见「校验一」。
2. 若**外抛命中** → 性能问题成立，**跳过 vector bound 校验**（外抛循环会把计算集中在 AIV_VECTOR，大概率同时命中 vector bound，重复 profiling 无增量价值），直接进入「强制 ATT 性能建模」。
3. 若**外抛未命中** → 设计典型用例上板采集 profiling，执行「校验二：vector bound」。

### 校验一：外抛 for 循环（静态 codegen 分析 + 强制 ATT 性能建模）

适用对象：对轴的个数有要求的 ASCIR reg API，典型如 `IsNan`、`Load`/`Store`/`NDDMA`、`Transpose` 等对轴数有上限的算子。当输入轴数超过内部最大 loop 数，codegen 会把多余轴“外抛”成 `outer_for_*` 循环，带来额外迭代开销。

定位阈值与外抛点（修改前必须检索确认，不要只凭常量名）：

- 向量函数路径：`autofuse/v35/codegen/vec_func_call/vec_func_call.cpp` 中 `kVFMaxLoop = 4U`，`VfCall::Generate` 在 `loop_num > kVFMaxLoop` 时调用 `CreateOuterForVFCall` 外抛。
- DMA/transpose 路径：`autofuse/v35/codegen/reg_api_call/reg_api_call_utils.cpp` 中 `kFourAxisNum = 4U`，超过则填充 `api_param.outer_loop_axes` 并注释“超过四层 for 循环，需要外抛”。
- 外抛循环生成：`autofuse/codegen/codegen_kernel_loop.cpp` 的 `GenOuterLoopAxesPreProcess`/`GenOuterLoopAxesPostProcess`，生成 `for (int outer_for_i = 0; ...)`。

判定方法：构造覆盖高维（≥5 维）shape 的用例，跑 Codegen UT 或生成 kernel 文本，检索 `outer_for_` 是否出现。出现即外抛 for 循环 → 性能问题成立。

识别方法（哪些 ASCIR 涉及外抛）：不能只靠“典型算子”举例，必须按下列静态检索得到当前工程确定集合，并随注册表变化复核：

1. 检索外抛触发点：在 `autofuse/v35/codegen/reg_api_call/` 与 `autofuse/v35/codegen/vec_func_call/` 中搜索 `CreateComputeNodeOuterFor`、`CreateOuterForVFCall` 以及对 `outer_loop_axes` 的写入（`emplace_back`/`push_back`）。
2. 回溯 ApiCall 名：每个命中的 codegen 文件对应一个 `ApiCallName`（如 `CastV2ApiCall`、`CompareV2ApiCall`、`WhereRegApiCall`、`TransposeRegApiCall`、`UnaryBitWidthChangeApiCallV2`、`VfCall`，以及走共享 DataCopy 参数构建的 `LoadRegApiCall`/`StoreRegApiCall`/`NddmaApiCall`）。
3. 映射到 ASCIR：在 `autofuse/v35/ascir/generator/v2_ascir_codegen_impl.h` 中按 `GetApiCallName()` 返回值找到对应 Codegen impl 类，再到 `ascir_builtin_ops_v2.cpp` 确认注册的 `REG_ASC_IR` 算子。

当前工程外抛涉及 ASCIR（以源码实际检索为准，下表为基线参考）：

| 外抛路径 | ApiCallName | 涉及 ASCIR | 外抛 codegen 位置 |
|---|---|---|---|
| 向量融合 | `VfCall` | 任意被融合进 `VectorFunc` 的 elewise 算子（如 `Abs`/`Exp`/`Add`） | `vec_func_call.cpp` `CreateOuterForVFCall` |
| DMA | `LoadRegApiCall`/`StoreRegApiCall`/`NddmaApiCall` | `Load`/`Store`/`NDDMA` | `reg_api_call_utils.cpp` 写 `outer_loop_axes` |
| Transpose | `TransposeRegApiCall` | `Transpose` | `reg_transpose_api_call.cpp` 写 `outer_loop_axes` |
| 位宽变化 | `UnaryBitWidthChangeApiCallV2` | `IsNan`/`IsFinite`/`IsInf` | `unary_bitwidth_change_api_call_v2.cpp` `CreateComputeNodeOuterFor` |
| Cast | `CastV2ApiCall` | `Cast` | `cast_v2_api_call.cpp` `CreateComputeNodeOuterFor` |
| Compare | `CompareV2ApiCall` | `Compare` | `compare_v2_api_call.cpp` `CreateComputeNodeOuterForIfRequired` |
| Where | `WhereRegApiCall` | `Where` | `reg_where_api_call.cpp` `CreateComputeNodeOuterFor` |

注：并非所有 unary 类 ASCIR 都外抛。`SignBit`（`UnaryApiTmpCall`）、`Frexp`（`UnaryDoubleOutputApiTmpCall`）等走普通 unary ApiCall，其 `Generate` 无 `CreateComputeNodeOuterFor`/`outer_loop_axes` 路径，不外抛。识别时必须以「该 ApiCall 的 codegen `Generate` 是否含外抛分支」为准，不能按算子名归类。

新增/修改 ASCIR 时，若其 `GetApiCallName()` 落在表中外抛路径，或会被融合进 `VectorFunc`，即判定该 ASCIR 涉及外抛，必须进入本节判定与建模流程。

强制 ATT 性能建模：一旦判定该 ASCIR 存在外抛场景，**必须触发 af-perf-modeler 为其建模**，并确认 perf 公式把外抛的最外侧维度纳入循环次数（repeat_time/call_count），不能漏算外抛迭代开销。本仓已有约定：`autofuse/v35/att/api_perf_register/ascir_api_perf_v2.cpp` 的 `LoadApi`/`NddmaApi` 注释明确“外抛for循环：最外侧4个维度丢到循环次数里面去”。新增/更新此类 ASCIR 时若 perf 公式未覆盖外抛分支，即建模缺失 → 性能问题。

### 校验二：vector bound（profiling 统计，仅当外抛未命中时执行）

定义：在典型单 ASCIR 用例上，profiling 采集到的 AIV_VECTOR 大于 AIV_MTE2 **且** 大于 AIV_MTE3，即计算被“绑死”在向量核上、无法与 load/store pipe 重叠，属于性能瓶颈。仅当校验一确认无外抛时执行本节。

采集方法：

1. 复用 `autofuse/tests/st/device_validation/` 的 `InProcessProfiler` / `msprof` 流程，或直接用 `msprof` 采集目标算子 kernel。
2. 导出 `op_summary*.csv`，提取每个 kernel 的 AIV_VECTOR / AIV_MTE2 / AIV_MTE3 占比或 cycles。
   - 当前 `autofuse/tests/st/device_validation/python/profiler.py`、`kernel_activity.py` 只解析 `task_duration`/`duration_us`，**未提取** per-pipe 占比；必须直接从 `op_summary*.csv` 读取对应列，不要假设解析器已支持。
   - 具体列名以目标环境实际 msprof 导出为准（常见为 `aiv_vec_ratio`/`aiv_mte2_ratio`/`aiv_mte3_ratio`）；不确定时先用一行实际导出确认列名再批量统计。
3. 与本仓性能模型对照：`PipeType` 定义在 `autofuse/att/base/base_types.h` 的 `enum class PipeType`（含 `AIV_MTE2`/`AIV_MTE3`/`AIV_VEC`），perf 公式写入 `perf.pipe_res[PipeType::AIV_VEC]` 等（见 `ascendc_regbase_perf.cpp`、`ascir_api_perf_v2.cpp`）。若实测 AIV_VECTOR 主导而性能模型未体现 MTE2/MTE3 可重叠，说明模型与实测偏离，需按 af-perf-modeler 复核。

判定：`AIV_VECTOR > AIV_MTE2 AND AIV_VECTOR > AIV_MTE3` 命中即 vector bound → 性能问题。

### 基础用例构造要求

为被校验 ASCIR reg API 构造一批单算子典型用例（复用 share graph `Data → Load → Op → Store → Output`，见“UT/ST 生成流程 → BesselJ0 V2 参考结构”）：

- 外抛判定用例：必须覆盖高维（≥5 维）shape 才能触发外抛分支；命中即止，无需上板。
- vector bound 用例（仅外抛未命中时设计）：
  - **dtype 覆盖**：覆盖该 ASCIR 在 `ascir_builtin_ops_v2.cpp` 中 `.DataTypes()`/`.Impl()` dtype map 注册的全部 dtype；存在 cast 的算子（`GetConversionDtype()`）同时覆盖源 dtype 与转换目标 dtype。每个 dtype 至少一组用例。
  - **shape 覆盖**：对齐、非对齐、较大 shape 三类至少各一组（如 `{32,16}`、`{32,18}`、`{512,15}`）；DMA/带宽敏感算子额外覆盖小块边界（256B 临界、1 个 block、尾块）；用低维（≤4 维）典型 shape，避免外抛循环干扰占比统计。
- 数据要覆盖算子定义域和分支，不能只填随机值；参考结果必须独立构造，禁止用被测 AscendC API 自证。

### 输出结论格式

性能基线校验完成后，方案中必须补充：

```text
性能基线结论：
- 外抛 for 循环：[命中/未命中]，证据：<高维用例> 生成 kernel [含/不含] outer_for_*
- 校验路径：[外抛命中→跳过 vector bound / 外抛未命中→已做 vector bound]
- vector bound：[命中/未命中/跳过/未知]，证据：<用例>/<shape> 上 AIV_VECTOR=<..> vs AIV_MTE2=<..>、AIV_MTE3=<..>（未知=无上板环境，见「无上板环境处理」）
- ATT 性能建模：[已覆盖/缺失/不适用]，外抛场景是否触发 af-perf-modeler 建模，perf 公式是否把最外侧维度纳入循环次数
- 性能问题：[存在/不存在]，若存在说明最小改进方向（如降低轴数/调整 codegen/补 ATT 性能建模）
```

### 无上板环境处理（仅当「外抛未命中 + 需 vector bound 兜底」时适用）

若当前环境无法上板采集 profiling，**必须先给用户明确告警**，不得默认“无性能问题”，并按下列步骤为用户产出可自行上板执行的交付物。

告警格式：

```text
⚠️ 性能基线校验受限告警
- 状态：外抛 for 循环未命中，需上板 profiling 校验 vector bound，但当前无上板环境。
- 影响：无法确认是否存在 vector bound，性能结论暂为“未知”，不代表“无性能问题”。
- 已为你准备：典型用例清单 + 上板执行步骤 + vector bound 识别方法，请在有设备环境后按步骤执行并回填结论。
```

典型用例清单（为用户设计，按「基础用例构造要求」填充实例）：

```text
目标 ASCIR：<OpName>   支持 dtype：<从 .DataTypes()/.Impl() 取>   是否含 cast：<是/否>
| 用例名 | dtype | shape | 说明 |
|---|---|---|---|
| <op>_<dtype>_aligned | DT_FLOAT | {32,16} | 对齐，低维避免外抛 |
| <op>_<dtype>_unaligned | DT_FLOAT16 | {32,18} | 非对齐 |
| <op>_<dtype>_large | DT_FLOAT | {512,15} | 较大 shape |
| ... | ... | ... | 每个 dtype 至少一组；DMA 算子加 256B 临界/尾块 |
参考结果：独立构造（不可用被测 AscendC API 自证）；数据须覆盖算子定义域与分支。
```

上板执行方法（交付给用户按步执行）：

1. 复用 `autofuse/tests/st/device_validation/` 的单算子 backend E2E 用例（share graph `Data → Load → Op → Store → Output`），或自带单算子 kernel。
2. 用 `msprof` 采集（≥8.1 语法）：
   ```bash
   msprof --output=./prof_out --application="python run_<case>.py"
   ```
   容器内 `msprof --application=` 不可用时，改用进程内 ACL profiling（`aclprofInit/CreateConfig/Start/Stop`，参考 `autofuse/tests/st/device_validation/backend/inprocess_profiler.cpp`）。
3. 导出 summary：
   ```bash
   msprof --export=on --output=./prof_out
   # 产物：./prof_out/*/mindstudio_profiler_output/op_summary*.csv
   ```

vector bound 识别方法（交付给用户执行）：

1. 在 `op_summary*.csv` 中按 `Op Name`/`Kernel Name` 定位目标算子 kernel（当前 `profiler.py`/`kernel_activity.py` 不解析 per-pipe 列，必须手动读 CSV）。
2. 读取该 kernel 的 `aiv_vec_ratio`、`aiv_mte2_ratio`、`aiv_mte3_ratio`（列名以实际 msprof 导出为准，先打印表头确认）。
3. 判定：`aiv_vec_ratio > aiv_mte2_ratio 且 aiv_vec_ratio > aiv_mte3_ratio` → 命中 vector bound → 性能问题。
4. 同时与本仓性能模型对照 `perf.pipe_res[PipeType::AIV_VEC]`/`AIV_MTE2`/`AIV_MTE3`（见 `ascir_api_perf_v2.cpp`），若实测与模型偏离，按 af-perf-modeler 复核。
5. 将每个用例的 `AIV_VECTOR/AIV_MTE2/AIV_MTE3` 回填到「性能基线结论」对应项。

---

## 上板错误分阶段诊断

先找“首个有效错误”，再按阶段定位；后续大量错误通常只是级联诊断。

| 关键字 | 阶段 | 典型根因 | 最小方案 |
|---|---|---|---|
| `acl/acl.h: No such file`、`xaclfk` | 框架前置辅助库构建 | CANN include/lib 路径硬编码或环境混用 | 对齐 `$ASCEND_HOME_PATH`；若框架继续执行，单独记录但继续寻找用例主错误 |
| `module ... has no attribute tc_*` | 用例发现 | 文件名、函数名或部署内容不一致 | 同步正确脚本，保证顶层入口与 case_name 完全一致 |
| `'IrAttr' object has no attribute 'x'` | Python 构图 | `pyascir.cpp` 属性绑定遗漏或旧 `.so` | 补常量/访问器/handler；重编重装并检查加载路径 |
| `helper.h file not found` 且来自生成 kernel | Device C++ 编译 | regbase 内部项目 include 残留 | 删除内部 include，改为 Load + CMake 生成 + map 注册 |
| `ERFC/Lgamma undeclared`、`no member Adds/Div/Power` | Device C++ 编译 | `IncludeApiHeaderFiles()` 缺官方头 | 根据实际调用补最小官方头并检查生成 kernel 顶部 |
| `out of scbzi imm range` | Device 汇编 | 内联后控制流过大，短跳转越界 | 当前工程不新增通用 Device 编译参数；记录为编译器限制，并与用户确认是否调整实现或暂不支持该场景 |
| `symbol lookup error`、undefined symbol | 动态加载 | 新旧库或 `LD_LIBRARY_PATH` 混用 | 检查模块/so 路径和依赖版本，避免多个 CANN/autofuse 混载 |
| `scalar_pv.cc ... unknown instr` | ICPU/tikicpulib 执行 | toolkit、tikicpulib、系统包版本不匹配或不支持指令 | 先确认包/系统/版本一致，不误判为 ASCIR 注册错误 |

### 已验证问题样本

- `59error.log`：BesselJ0 内部 `bessel_j_utils.h` 残留。
- `61error.log`：BesselY0 `scbzi` 越界；选项必须传 Device。
- `63error.log`：测试模块缺同名入口函数，尚未进入 ASCIR 构图。
- `64error.log`：Igamma 缺 ERFC/Lgamma/Power/二元接口官方头。
- `70error.log`：ShiftedChebyshev 的 Python `IrAttr.n` 未暴露或旧 native 模块仍在加载。

---

## 提交历史提供的强制回归检查

以下提交体现了“初始适配后仍容易遗漏”的真实修改面，后续新增/更新 ASCIR 必须在方案中主动检查：

- `8c91f9e55d8658b58a3d1d2111b16a7a971b58cc`：ASCIR adapter、Python ops/API、share graph、backend E2E、ApiCall、perf、regbase 注册和测试入口的主体改动。
- `9c2dfd24fd01809d9ba9cf813fa6da97b5b26ad8`：修复 regbase 内部 include/raw-string 依赖和 helper 注册。
- `7c31dee94de9302411063e633a1cea6207ded9eb`：曾尝试增加 Device 专属编译选项；当前方案已撤销该通用配置，后续不得将其作为默认修复。
- `40ddca2b086c2261aedb5845656772b17c7e6136`：Igamma/Igammac 补 `IncludeApiHeaderFiles()`。
- `e01eeb568a22b925d3f38b20ac4215160731dda8`：ShiftedChebyshev T/U/V/W 补 Python `IrAttr.n`。

审计新提交时，使用同样的历史范围方法检查“初始提交 + 后续修复”，避免只复制初始提交中的不完整模式。

---

## 输出修改方案的固定格式

每次给用户方案必须包含：

1. **接口事实**：输入/输出、dtype、属性、ComputeType、tmp、实际 AscendC API。
2. **必改文件**：按上方矩阵列出文件和具体符号。
3. **条件修改**：说明为何适用或不适用，不把全部路径机械加入。
4. **风险扫描**：内部 include、官方 API 头、Python Attr、多输出/模板、Device 参数、打包同步。
5. **测试覆盖**：regbase UT、Codegen UT、share graph、backend E2E、脚本入口、上板测试各自验证什么。
6. **验证命令**：先 source 环境，增量编译，格式检查，再运行目标测试。
7. **部署确认**：Python/native `.so` 和测试文件的实际加载路径。
8. **性能基线结论**：外抛先行、vector bound 兜底的分支校验结果（外抛命中则跳过 vector bound 并补 ATT 建模）、证据和最小改进方向；无上板环境时说明退化策略。

---

## 验证要求

### 静态验证

每次修改后至少执行：

- 检索新增/修改的 `OpName`、dtype、tmp buffer 函数名。
- 检索旧名称，确认没有遗漏必须修改的引用。
- 检查新增文件是否被 CMake 或脚本纳入。

### 编译验证

涉及 C++ 注册、Codegen、ATT、tmp buffer、UT/ST 时，触发 af-build-runner 并优先增量编译：

```bash
cmake --build build --target aihac_codegen -j 8
cmake --build build --target <test_target> -j 8
```

### 测试验证

涉及 UT/ST 时，触发 af-test-developer 并运行相关测试：

```bash
bash scripts/test/run_autofuse_test.sh -u -m codegen -j 8
bash scripts/test/run_autofuse_test.sh -s -m e2e -j 8
```

如果无法完整运行，必须说明原因，并给出已完成的替代验证。

---

## 输出摘要

完成任务后向用户说明：

- 修改了哪些文件和核心符号。
- 新增或修改了哪些 ASCIR 注册、dtype、tmp buffer、API、UT/ST。
- 执行了哪些验证命令，结果如何。
- 是否存在需要用户人工确认的 tmp buffer 系数、期望值计算或兼容性决策。
- 性能基线校验：vector bound / 外抛 for 循环是否命中，上板环境是否可用，最小改进方向。
