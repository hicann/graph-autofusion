---
name: af-perf-modeler
description: Use when graph-autofusion 用户提到 Cast/Reduce/Compare 等算子的性能公式、ATT 性能建模、MicroAPI 成本、repeat_time/call_count、codegen 与性能公式对齐。
---

# graph-autofusion V2 节点性能公式建模

用于基于 `autofuse` 源码为 V2 节点建立性能公式。适用场景包括：Cast / Reduce / Compare 等算子建模、ATT 性能公式 review、MicroAPI 成本分析、repeat_time / call_count 计算，以及 codegen 与性能公式对齐。

核心原则：**性能公式必须从节点对应 AscendC/MicroAPI 源码和 codegen 实际传参反推，不能只套现有公式或凭经验估算。**

## 建模前提

建模前必须先定位并读取：

- codegen impl：`autofuse/v35/ascir/generator/v2_ascir_codegen_impl.h`。
- 节点对应的 AscendC API 源码：`autofuse/v35/ascendc/api_regbase/` 下由 `GetApiName()` 指向的实现。
- codegen API 调用生成逻辑：`autofuse/v35/codegen/` 下由 `GetApiCallName()` 指向的 `{ApiCallName}::Generate`。
- 性能表：`autofuse/v35/att/api_perf_register/perf_param_v2.cpp`。
- 当前性能注册/实现：`autofuse/v35/att/api_perf_register/ascir_api_perf_v2.cpp` 和 `autofuse/v35/att/api_perf_register/ascendc_regbase_perf.cpp`。
- 如需从 codegen 向 ATT 透传参数，参考 `VectorFuncNodeParams`、`EnrichAscirGraphNodeParams`、`FillSpecificParams`、`specific_params` 及其参数传递路径。

如果源码入口、API 参数、codegen 传参、分支含义或 MicroAPI 语义不确定，必须先向用户说明不确定点并询问，不能编造公式。

## 通用建模流程

1. **定位算子源码入口**
   - 从 `v2_ascir_codegen_impl.h` 找到目标算子对应的 `GetApiName()`。
   - 在 `autofuse/v35/ascendc/api_regbase/` 下查找该 API 名称的定义。
   - 读取 API 实现及其调用的 helper，不只读 perf 代码。
   - 若 API 名称和源码实现无法唯一对应，先询问用户确认。

2. **确认源码输入参数与 codegen 实际传参**
   - 先分析 AscendC API 源码的函数签名，确认参数含义、类型、顺序和模板参数。
   - 从 `v2_ascir_codegen_impl.h` 找到目标算子的 `GetApiCallName()`。
   - 根据 `ApiCallName` 在 `autofuse/v35/codegen/` 下找到 `{ApiCallName}::Generate`。
   - 从 `Generate` 中分析生成代码实际传入 AscendC API 的参数，包括 `inputs`、`outputs`、scalar、offset、stride、mask、loop 参数、tmp buf、axis 信息等。
   - 注意：codegen 的 `inputs/outputs` 是生成代码用的 `Tensor`，ATT 性能计算的 `input_shapes/output_shapes` 是性能建模用的 `TensorShapeInfo`，两者语义对应但不是同一对象。

3. **必要参数从 codegen 传递到 ATT**
   - 如果性能公式需要 codegen 阶段才能确定的参数，例如 loop merge 后的 `cal_count`、`outer_repeats`、stride、mask mode、特殊分支标记，不要在 ATT 中凭空假设。
   - 先确认 ATT 性能函数能访问什么对象：如果只能访问 `NodeInfo`，必须走 `EnrichAscirGraphNodeParams` 预注册空参数结构体 -> `{ApiCallName}::Generate` 填充具体值 -> `FillSpecificParams` 提取到 `NodeInfo` 的链路。
   - 参考 `VectorFuncNodeParams`、Reduce 参数和目标算子现有链路，将所需参数定义成独立结构体，并加入 `AscirNodeParams::specific_params` 可承载的类型。
   - 对只能在 `{ApiCallName}::Generate` 中确定的参数，例如 `ApiLoopParams`、合轴后实参、实际生成分支，`EnrichAscirGraphNodeParams` 只预注册空结构体，不要在预注册阶段猜具体值。
   - `{ApiCallName}::Generate` 填充给 ATT 的节点参数必须使用 codegen/merge 阶段的原始符号参数，例如 `merge_info`、`ApiLoopParams` 中的 repeat、stride、count；不要保存 `tpipe.tiler.ActualSize(...)`、`tpipe.tiler.Size(...)` 等用于生成 C++ 调用字符串的展开表达式，ATT 性能公式无法稳定识别这些 tiler 表达式。
   - 在 `FillSpecificParams` 或同等参数填充入口中解析目标算子类型，并将节点参数写入 `NodeInfo`；如果性能函数确实能直接访问节点对象，才允许绕过 `NodeInfo` 直接读 ext attr，并必须在设计中说明原因。
   - 在 ATT 性能建模入口从 `NodeInfo` 读取该参数，用于公式分支和循环次数计算；缺失时必须有明确回退或报错策略。
   - 参数传递属于设计变更，需说明兼容性、默认值和缺失参数时的处理方式。

4. **分析源码分支**
   - 保持与源码 `if constexpr`、`if`、模板参数、dtype 特化、dim 分支、broadcast 分支、scalar 分支、mask mode 分支一致。
   - 每个源码分支都要说明触发条件、codegen 参数来源和公式差异。
   - 不允许只按 dtype 或经验简化掉源码中的关键分支。

5. **枚举 MicroAPI**
   - 以源码中的每个 `MicroAPI::xxx` 或 AscendC 基础向量 API 作为最小建模粒度。
   - 将源码 MicroAPI 映射到 `perf_param_v2.cpp` 中的 `kVfInstructPerfTable` 类型，例如 `MicroAPI::Add` 映射为 `kAdd`。
   - 映射前必须查 `perf_param_v2.cpp`，不要假设表项存在。
   - 如果找不到精确匹配，使用 `kPlaceholder`，并在代码注释中说明真实 MicroAPI 名称和暂用原因。
   - `MicroAPI::DataCopy` 按 load/store 两类统一建模：load 使用 input dtype，store 使用 output dtype；`LoadDist`、`StoreDist`、pack/unpack 模式只作为注释说明，不因模式变化额外增加一条 DataCopy 成本。

## API 参数对齐原则

性能模型透传的参数必须与实际 AscendC API 调用保持一一对应：

- 排除输入、输出 Tensor 后，按 API 签名的原始顺序保存其余参数；不要用 `repeat_count`、`flattened_count` 或分支标志替换原始参数。
- 参数的维度、stride、mask、offset 和 loop 数组要保持与生成代码相同的语义粒度；性能模型只在计算公式内部派生临时量。
- `CastExtend(dst, src, output_dims, output_stride, input_stride)` 的节点参数应保存 `output_dims`、`output_strides` 和 `input_strides`，仅排除 `dst`、`src`。
- `CastExtend` 节点参数中的 `output_dims`、`output_strides`、`input_strides` 应来自 `merge_info`/`ApiLoopParams` 的原始表达式；实际生成 API 调用时可以继续使用 `tpipe.tiler`，但传给性能公式的节点参数不能使用 tiler 展开结果。
- 参数链路必须可追溯：`EnrichAscirGraphNodeParams` 预注册 -> `{ApiCallName}::Generate` 按实际调用填充 -> `FillSpecificParams` 提取 -> `NodeInfo` -> perf 函数。
- 如果原始 API 参数缺失，必须明确回退到 shape 信息的条件和精度影响；禁止在 ATT 中凭经验重建 codegen 已经确定的参数。

6. **计算 MicroAPI 执行次数**
   - 每个源码中的 MicroAPI 都应能追溯到性能公式中的一次 `VfPerfUtils::AddVfInstructPerf` 计算。
   - `AddVfInstructPerf` 最后一个参数必须来自源码循环次数或 codegen 透传参数。
   - 对嵌套循环计算总次数，例如 `outer_count * repeat_time`。
   - 对只在循环外执行一次的 `Duplicate`、mask 创建、scalar 初始化等，次数按源码实际执行次数建模，不要误乘 `repeat_time`。
   - 如果循环次数依赖运行时数据值，必须说明无法精确获取的原因，并采用保守估计或要求透传参数。

7. **合并同类 MicroAPI**
   - 同一种 MicroAPI 类型可以只调用一次 `VfPerfUtils::AddVfInstructPerf`。
   - 最后一个参数使用该 MicroAPI 在当前分支中的总执行次数。
   - 合并时必须保留注释或表格，能追溯每一部分次数来自哪个源码位置。
   - 对 `kPlaceholder`，合并键必须包含真实 MicroAPI、操作方向/模式和 dtype。只有三者都相同才能合并。
   - 即使性能表类型相同，也禁止合并不同真实 MicroAPI，例如 `Not` 与 `ShiftRights`、`Pack` 与 `UnPack`。
   - `MicroAPI::DataCopy` 的 load 与 store 必须分开，不能因为都映射为 `kPlaceholder` 而汇总。
   - `kPlaceholder` 直接调用 `VfPerfUtils::AddVfInstructPerf`，不要增加 `AddPlaceholderPerf` 一类包装函数。

8. **套用统一 cost 结构**
   - 每个 MicroAPI 通过 `VfPerfUtils::AddVfInstructPerf(type, dtype, max_latency, all_vf_instruct_cost, count)` 累加。
   - `AddVfInstructPerf` 内部查询 latency 和 throughput。
   - latency 取所有 MicroAPI 的最大值。
   - `throughput * count` 累加到 `all_vf_instruct_cost`。
   - 最终公式固定为：

```cpp
Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
res.Simplify();
perf.pipe_res[PipeType::AIV_VEC] = res;
```

## MicroAPI 映射规则

一般情况下，性能表类型为 MicroAPI 指令名增加 `k` 前缀，例如 `MicroAPI::Add` 对应 `kAdd`、
`MicroAPI::UpdateMask` 对应 `kUpdateMask`。不要在 skill 中维护完整的一一映射表，建模时按以下顺序确认：

1. 提取真实 MicroAPI 指令名，以 `k + 指令名` 作为候选性能表类型。
2. 在 `perf_param_v2.cpp` 和常量定义中确认候选类型真实存在，并确认目标 dtype 受支持；不能只根据命名猜测。
3. 对带模式语义的模板 API，使用性能表中对应的语义后缀，例如
   `CompareScalar<..., CMPMODE::NE>` 对应 `kCompareScalarNE`。
4. 如果没有精确表项，使用 `kPlaceholder`，并按真实 MicroAPI、操作方向/模式和 dtype 分开调用。
5. `DataCopy` 的 load 与 store 即使都没有精确表项，也必须分开建模并分别注释；同一次 load 或 store 的 `LoadDist`/`StoreDist` 模式不额外重复建模。
6. 如果性能表已有对应 MicroAPI 类型但暂不支持源码中的真实 dtype，仍按源码真实 dtype 调用该 MicroAPI 类型；不要为了命中现有表项改成中间 dtype、输出 dtype 或 `kPlaceholder`。性能表缺项应后续扩展，公式必须先保持源码语义正确。

## Placeholder 调用示例

错误：不同真实操作或不同方向被汇总，后续无法替换为精确表项。

```cpp
VfPerfUtils::AddVfInstructPerf(kPlaceholder, dtype, max_latency, all_vf_instruct_cost, repeat_time * 4);
```

正确：直接调用 `AddVfInstructPerf`，每个占位项都能追溯到唯一真实操作。

```cpp
// MicroAPI::DataCopy (load).
VfPerfUtils::AddVfInstructPerf(kPlaceholder, input_dtype, max_latency, all_vf_instruct_cost, repeat_time);
// MicroAPI::DataCopy (store).
VfPerfUtils::AddVfInstructPerf(kPlaceholder, output_dtype, max_latency, all_vf_instruct_cost, repeat_time);
// MicroAPI::Pack<uint32_t, int64_t>, two calls with the same mode and dtype.
VfPerfUtils::AddVfInstructPerf(kPlaceholder, int64_dtype, max_latency, all_vf_instruct_cost, repeat_time * 2);
// MicroAPI::UnPack<uint64_t, uint32_t>.
VfPerfUtils::AddVfInstructPerf(kPlaceholder, uint32_dtype, max_latency, all_vf_instruct_cost, repeat_time);
```

## 建模输出模板

~~~markdown
### 源码依据
- codegen impl：`...`
- API 名称：`...`
- API 源码：`...`
- API 调用生成：`...::{ApiCallName}::Generate`
- 性能表：`.../perf_param_v2.cpp`
- 当前 perf 实现：`.../ascendc_regbase_perf.cpp`

### 参数来源
| 源码参数 | codegen 传参来源 | ATT 是否可直接获取 | 处理方式 |
| --- | --- | --- | --- |
| `dst` | `outputs[0]` | 是/否 | ... |
| `src` | `inputs[0]` | 是/否 | ... |
| `loop_param` | `ApiLoopParams` | 否 | 需通过节点参数透传 |

### 分支分析
- 分支 A：触发条件，参数来源，源码行，公式差异
- 分支 B：触发条件，参数来源，源码行，公式差异

### MicroAPI 计数
| MicroAPI | perf 类型 | 执行次数 | dtype | 说明 |
| --- | --- | --- | --- | --- |

### 公式
```cpp
Expr max_latency = CreateExpr(0);
Expr all_vf_instruct_cost = CreateExpr(0);
// AddVfInstructPerf(...)
Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
```

### 不确定点
- 无；或列出需要用户确认/源码待确认的问题。
~~~

## 常见错误

| 错误 | 正确做法 |
| --- | --- |
| 直接复述现有 perf 公式 | 先读 API 源码和 codegen 传参，再解释现有公式是否简化 |
| 只看 `input_shapes/output_shapes` | 同时看 `{ApiCallName}::Generate`，确认实际传入 API 的参数 |
| 需要 loop 参数却在 ATT 中猜测 | 通过节点参数从 codegen 透传，或说明无法精确获取 |
| Generate 才能确定的参数却要求 `EnrichAscirGraphNodeParams` 计算具体值 | `EnrichAscirGraphNodeParams` 只预注册空结构，`Generate` 填具体值 |
| 性能函数只能访问 `NodeInfo` 却让 ATT 直接读 Node | 补 `FillSpecificParams`，把节点参数提取到 `NodeInfo` |
| 只写入节点参数但 ATT 读取路径不明确 | 明确链路：预注册空结构 -> Generate 填值 -> FillSpecificParams -> NodeInfo -> perf |
| 给 ATT 的节点参数保存 `tpipe.tiler.Size/ActualSize` 展开值 | 保存 `merge_info`/`ApiLoopParams` 原始符号表达式，tiler 表达式只用于生成 C++ 调用字符串 |
| 忽略 `if constexpr` 或 scalar/mask 分支 | 每个源码分支单独列触发条件和计数 |
| 把所有 MicroAPI 都乘 `repeat_time` | 按源码实际循环层级计算次数 |
| 找不到性能表项就跳过 | 使用 `kPlaceholder` 并注释真实 MicroAPI |
| 性能表暂不支持真实 dtype 就换成支持的 dtype | 保持源码真实 MicroAPI 类型和 dtype，后续扩展性能表 |
| 多个同类 MicroAPI 重复调用多次 | 合并为一次 `AddVfInstructPerf`，次数求和 |
| 把不同真实 MicroAPI 汇总到一个 `kPlaceholder` | 按真实 MicroAPI、方向/模式和 dtype 分开调用 |
| 把 `DataCopy` load 和 store 合并 | load 与 store 分开调用并分别注释 |
| 因 `DataCopy` 的 pack/unpack 模式再额外补一条成本 | DataCopy 只按 load/store 建模，dtype 分别使用 input/output |
| 用辅助函数包装 `kPlaceholder` | 直接调用 `VfPerfUtils::AddVfInstructPerf(kPlaceholder, ...)` |
| `UpdateMask` 使用 `kPlaceholder` | 使用性能表已有的 `kUpdateMask` |
| 未说明不确定点 | 不确定时先提问或列入“不确定点” |

## 验证要求

完成公式后至少检查：

- API 源码入口能从 `GetApiName()` 追溯。
- API 调用参数能从 `{ApiCallName}::Generate` 追溯。
- ATT 使用的额外参数若来自 codegen，已通过节点参数传递；若性能函数只能访问 `NodeInfo`，必须确认 `EnrichAscirGraphNodeParams` 已预注册空结构、`Generate` 已填值、`FillSpecificParams` 已提取到 `NodeInfo`。
- 公式中的每个 `AddVfInstructPerf` 都能追溯到源码 MicroAPI。
- 每个循环次数都能追溯到源码循环、codegen 参数或明确的保守估计。
- 每个 MicroAPI 类型都在 `perf_param_v2.cpp` 中存在；不存在时已用 `kPlaceholder` 并说明。
- 每个 `kPlaceholder` 都有唯一的真实 MicroAPI、操作方向/模式和 dtype；load/store 及不同真实类型未合并。
- `kPlaceholder` 均直接调用 `VfPerfUtils::AddVfInstructPerf`，未增加包装函数。
- `MicroAPI::UpdateMask` 使用 `kUpdateMask`。
- 最终 cost 使用 `VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost`。
