# Transpose 融合

## 1. 特性背景

### 1.1 Transpose 融合的特殊性

Transpose 是 View 类算子（不产生新的计算，只改变数据的访问方式），在融合场景中的核心问题是：**数据重排应该在哪里完成**：

- **UB 内重排**：Transpose 作为独立节点保留在融合链中，由专用转置 API 在 UB 内完成重排，前后的 elewise 计算都按转置后的轴序访问数据；
- **折入搬运**：将转置语义折算成非连续的 stride 表示，下沉到 Load/Store 搬运中完成（NDDMA 非连续 DMA 或非连续向量访问），融合链中不再有 Transpose 节点。

两条路线各有适用场景：UB 内重排有额外指令开销，但保持了搬运的连续性；折入搬运省去重排指令，但搬运变为非连续。Schedule 阶段无法确定哪种更优时，会同时生成多个模板，由 ATT 在 tiling 阶段结合性能公式选优。

### 1.2 设计目标

- **转置不单独成算子**：Transpose 尽量融合进 kernel，避免"独立转置算子 + 两次搬运"的开销；
- **双模板择优**：保留（UB 重排）与消除（折入搬运）两条路线生成多模板，由 ATT 打分选择；
- **按平台选择实现**：V1 平台使用 5HD 分形指令风格的 ConfusionTranspose，v35 平台使用 MicroAPI gather 风格的 TransposeExtend，并通过 NDDMA 支持转置折入搬运。

## 2. 用户使用场景

| 场景 | 说明 |
| --- | --- |
| 融合链中的轴交换 | elewise 链中夹带 Transpose（如 `abs(transpose(x))`），Transpose 保留或消除由打分决定 |
| 尾轴转置 | 输入尾轴与输出尾轴不同（如 [H,W]→[W,H]），必须 UB 内重排，Transpose 节点保留 |
| 非尾轴转置 | 尾轴保持不变、只交换高维轴（如 NCHW→NHWC 的 [N,C,H,W]→[N,H,W,C]），尾轴较大时倾向折入搬运消除 |
| 多 Transpose 图 | 图中含多个 Transpose 时不生成保留模板，全部消除折入 Load |
| Concat 的转置实现 | v1/v35 默认 `concat_alg=kConcatAlgTranspose`，Concat 以 transpose 方式实现 |

## 3. 特殊背景及限制

本节描述 Transpose 融合在三个模块间的隐式约束，大部分无法从单个模块代码直接看出。

### 3.1 转置语义的隐式表达

Transpose 节点上**没有显式的 perm/axes 属性**。转置语义通过"输入 tensor 的轴序 vs 输出 tensor 的轴序差异"表达：输入 `attr.axis` 与输出 `attr.axis` 是同一批轴（AxisId）的不同排列，repeats/strides/vectorized_axis 随之重排。Python 前端接口 `Transpose(x, axis=..., size=..., stride=...)` 由用户直接给出转置后的轴序。这意味着：

- Codegen 和 ATT 都必须通过对比输入/输出轴序来还原 permute 向量，任何一侧单独读取"输出轴序"都会丢失转置信息；
- 中间节点的 axis/sched.axis/vectorized_axis 在转置前后必须保持一致的轴序约定，否则轴对不上。

### 3.2 与 Reduce 的互斥约束

**Reduce 不支持与 Transpose 融合**（View 类算子中唯一的例外项）：

- 调度阶段检测到图同时含 Reduce 和 Transpose 时，直接在原图上消除全部 Transpose；
- 含 Reduce 时向量化轴重排被跳过（tiling 策略暂不支持 reduce 与 transpose 融合的场景）；
- Reduce 下游 + 尾轴转置的 Load 需要额外做对齐处理。

### 3.3 模板选择与消除的触发条件

- **打分阈值**：尾轴转置必须保留（UB 重排）；非尾轴转置按尾轴字节数与阈值 512B 比较，尾轴小于 512B 时 UB 重排占优（保留），否则消除占优（折入非连续搬运）；
- **多 Transpose 限制**：图中含多个 Transpose 时只生成消除模板，不生成保留模板；
- **消除的标记约定**：消除路线会把上游 Load 改名并加 `transpose_` 前缀，该前缀同时是后续 NDDMA 判断的标记，属于 Schedule 与 v35 优化 pass 之间的隐式协议；
- **调度轴同步**：单 Transpose 场景，输入链上所有节点的 sched.axis 会被同步为 Transpose 的调度轴（转置轴必须 x/y 双切分）；多 Transpose 场景依赖提前消除，否则轴同步无法进行。

### 3.4 Codegen 的轴数与形态限制

- **V1（ConfusionTranspose）**：仅支持 7 种固定 permute 模式（ND2ND_ONLY、102、0213、2103、021、210、0321），permute 向量查 `kPermutationTable` 命中后才可生成，未命中报 `TRANSPOSE_INVALID` 编译失败；
- **V2（TransposeExtend）**：参与转置的轴最多 4 个（`total_dim ≤ 4`），其中内层连续轴最多 3 个（`inner_dim ≤ 3`，按 `vectorized_strides[i] == repeats[i+1]*strides[i+1]` 判定连续性）；超出部分由 Codegen 外抛 `outer_for_i` for 循环；
- **dtype**：V2 转置 base case 仅支持 2/4 字节 dtype；V1 的 API tiling 字段（param0~param17）按 dtype size 和 32B block/16 对齐约束预计算；
- **tmp buffer**：V1/V2 转置是唯一使用"全生命周期 TBuf"（life_time_axis_id = -1）的 API，tmp 用途不同（V1 存分形重排中间数据，V2 存 gather 索引），大小计算方式也不同；
- **同步**：V1 生成调用后追加 `PipeBarrier<PIPE_ALL>()`，V2 无此要求。

### 3.5 与其他机制交叉的隐式约束

- **API 缓存禁用**：图内含 Transpose 则全图不做 API 结果缓存（NodeCacheMarker），避免转置双切分场景下的缓存正确性问题；
- **对齐宽度**：存在 fp32 Transpose 时对齐宽度取 64B，否则 32B；
- **Load/Store 最内轴不同**：Transpose 场景下 Load 和 Store 各有不同的最内轴（这是 Transpose 的本质特性），影响两处：非对齐策略需分别按 Load 侧/Store 侧连续尾轴数重算 vectorized_strides；ATT 轴排序求解器在轴数超限时保持 Transpose 的同等切分顺序而放弃 Reduce 平衡优化（equal-order tiling）；
- **NDDMA 前置条件**：v35 将 Transpose 合并进 Load 生成 NDDMA 节点时，要求最内连续向量化轴数不超过 3，且 broadcast 融合要求输入内存连续（双切分且输入非 scalar 时丢弃该融合）。

## 4. 对外接口

### 4.1 算子定义接口

- **IR 注册**：`REG_ASC_IR(Transpose).Input("x","T").Output("y","T").ComputeType(kComputeTranspose)`，支持 dtype：INT16/UINT16/INT32/UINT32/FLOAT16/FLOAT（以各平台注册为准）；
- **Python 前端**：`Transpose(owner_graph, x, axis=..., size=..., stride=...)`，用户直接指定转置后的轴序、size 与 stride。

### 4.2 生成代码接口（Device 侧）

| 平台 | API | 形态 |
| --- | --- | --- |
| V1 | `ConfusionTranspose<T>(dst, src, tmp_buf, transposeType, apiTilingData)` | 5HD 分形指令，按 `AutoFuseTransposeType` 枚举分派，需配套 `ConfusionTransposeTiling`（param0~param17） |
| V2（v35） | `TransposeExtend<inner_dim, total_dim, T>(dst, src, tmp_buf, dst_dims[], src_strides[], dst_strides[])` | MicroAPI gather，dims/strides 全参数化，索引类型按 dtype 大小选 int16_t/int32_t |
| v35（搬运路线） | `DataCopyNddma<T, dim>(ub, gm[offset], dst_dims, dst_strides, src_strides)` | 非连续 DMA，转置折入 Load 时使用，维度上限 5 |

### 4.3 ATT 侧接口

- **节点参数结构**：`TransposeNodeParams`（valid/inner_dim/total_dim/outer_loop_axes/output_dims/input_strides/output_strides），由 Codegen 填充、经节点扩展属性透传给 ATT 性能建模；
- **专用 API tiling 生成**：`REGISTER_API_TILING_FUNC("Transpose", ...)`，为 V1 ConfusionTranspose 各 scene 生成 host 侧 tiling 计算代码；
- **性能公式注册**：V2 求值函数 `ascir_v2::TransposeApi`，工厂键 `"TransposeV2"`；V1 无专用公式（占位为 kUnitVector）。

## 5. 整体架构

Transpose 融合由三个模块协作完成，核心流程如下：

```text
┌────────────────────────────────────────────────────────────────────┐
│ Schedule（决策层）                                                   │
│   TransposeFusionCaseGenerator                                      │
│     ├─ 无 Transpose ──────────────► 直接返回                         │
│     ├─ 含 Reduce ─────────────────► 原图消除全部 Transpose            │
│     ├─ 单 Transpose ──────────────► 双模板：                          │
│     │      模板A：保留（UB 重排）  +  模板B：消除（折入 Load）          │
│     └─ 多 Transpose ──────────────► 仅消除模板                        │
│   GenTransposeTilingGroup：转置轴 x/y 双切分                          │
│   v35 NddmaTemplate：Load+Transpose 合并为 NDDMA 节点                 │
├────────────────────────────────────────────────────────────────────┤
│ Codegen（生成层）                                                    │
│   保留路线：V1 TransposeApiCall（ConfusionTranspose）                 │
│             V2 TransposeRegApiCall（TransposeExtend，gather）        │
│   消除路线：NddmaApiCall（非连续 DMA）/ 非连续向量访问                 │
│   填充 TransposeNodeParams 透传 ATT                                  │
├────────────────────────────────────────────────────────────────────┤
│ ATT（评估层）                                                        │
│   双模板打分：尾轴转置 score=1（保留）；非尾轴按尾轴 512B 阈值比较        │
│   V2 性能公式：latency 分档 + repeat_time + gather_count              │
│   V1 专用 API tiling：32B block / 16 对齐预计算                       │
│   轴排序：Transpose 场景保持 Load/Store equal-order                   │
└────────────────────────────────────────────────────────────────────┘
```

模块间契约：Schedule 产出的 ImplGraph 决定了转置以"节点"还是"非连续 stride"形式存在；Codegen 忠实翻译并在生成 API 参数时计算 inner_dim/total_dim/outer_loop_axes 等透传参数；ATT 以这些参数评估性能公式并决定最终模板与 tiling 取值。

## 6. 核心实现

### 6.1 Schedule：模板决策与轴切分

#### 6.1.1 保留与消除（TransposeFusionCaseGenerator）

`autofuse/optimize/task_generator/transpose_schedule_case_generator.cpp`，在 Split、Concat 之后、Reduce 之前执行：

- **消除过程**：`TransposeConvertProcess` 沿 Transpose 输入链上推（`UpdateAxis`/`UpdateAxisByPath`），对路径上每个节点重排输出 axis 和调度轴，把转置语义折进 Load 的非连续 stride 表示，Load 改名加 `transpose_` 前缀后删除 Transpose 节点；
- **打分函数**（`TransposeScoreFunctionGenerator`）：
  - 尾轴转置（输入尾轴 ≠ 输出尾轴）：score=1，必须 UB 重排；
  - 非尾轴转置：尾轴字节数 < 512B（`kTransposeNoNeedUBConvertSize`）时 score=1（UB 重排），否则 score=-1（消除）。

#### 6.1.2 轴切分（GenTransposeTilingGroup）

`autofuse/optimize/autoschedule/tiling_group.cpp`：比较输入/输出轴序，从尾轴向前相同的轴放入 `n_group`（无需双切分）；从第一个不同的轴开始，输入轴放入 `x_group`、输出轴放入 `y_group`（转置轴拆成两组，形成"双切分"），其余轴归入 `y_group`。因此**转置轴可以作为切分轴，但必须 x/y 双切分**，保证 Load 侧与 Store 侧各自按自己的最内轴切分。

#### 6.1.3 调度轴同步与 NDDMA 模板

- `Scheduler::SynchronizeTransposeInputSchedAxis`：单 Transpose 场景，把输入链上所有节点的 sched.axis 同步为 Transpose 的调度轴；多 Transpose 场景要求提前消除；
- v35 `NddmaTemplate`（`autofuse/v35/optimize/template/nddma_template.cpp`）：`TransposeToNddmaNode` 从 Transpose 向上遍历，在 Load 后插入新 Transpose 节点并同步中间节点轴序，最后 `MergeLoadAndTranspose` 将 Load 节点类型改为 Nddma 并删除插入节点，即 **Load+Transpose 合成一个 NDDMA 节点**；生成后调用 `UnAlignmentStrategy::ModifyTransposeFusionVectorizedStrides` 修正向量化 stride。

### 6.2 Codegen：两代转置 API

#### 6.2.1 V1：TransposeApiCall（ConfusionTranspose）

`autofuse/codegen/api_call/transpose/transpose_api_call.cpp`：

- **类型识别**：`CodeGenGetTransposeType` 把输出 vectorized_axis 在输入中的位置序列当作 permute 向量查 `kPermutationTable`（覆盖 2D/3D/4D 共 21 种 permute），归约到 8 个 `AutoFuseTransposeType`（含 INVALID）；
- **生成代码示例**：

```cpp
AutoFuseTransposeType transposeType = AutoFuseTransposeType::TRANSPOSE_ND2ND_021;
auto apiTilingData = t->transpose_tilingData_<tiling_case_id>;
codegen::ConfusionTranspose<float>(y[offset_y], x[offset_x], tmp_buf_<id>, transposeType, apiTilingData);
AscendC::PipeBarrier<PIPE_ALL>();
```

- **API tiling**：`ConfusionTransposeTiling` 固定 18 个字段（param0~param17），各 scene 语义不同（如 scene10 用 param0=height、param1=width、param2=highBlock、param3=stride），host 侧取值由 ATT 专用 tiling 生成器计算；
- **tmp buffer**：通过 `tmp_buf_id.find(-1L)` 取全生命周期 TBuf 复用 id。

底层 API 模板 `autofuse/ascendc/api/transpose.h` 按 transposeType 分派到 `ConfusionTransposeNd2Nd10/102/0213/2103/021/210/0321`，底层为 5HD 分形指令风格的 ConfigMatrix 实现。

#### 6.2.2 V2：TransposeRegApiCall（TransposeExtend）

`autofuse/v35/codegen/reg_api_call/reg_transpose_api_call.cpp`：

- **通用参数化**：不再查表，任意 permute 拆为"外层循环轴 + 最多 4 个内层转置轴"，模板参数为 `[inner_dim, total_dim]`；
- **连续性判定**：`GetContinuousInnerAxisNum` 从倒数第二轴向前按 stride 连续性统计内层轴数，上限 3；`transpose_total_axis_num = min(vectorized_axis 数, 4)`；
- **超维外抛**：超过 4 个轴的外层轴 repeats 放入 `api_param->outer_loop_axes`，由生成器外抛 `for (int outer_for_i = ...)` 循环，其 stride 折入 inner_offset；
- **生成代码示例**：

```cpp
TransposeExtend<inner_dim, total_dim, T>(y[off], x[off], tmp_buf_0,
    {static_cast<int16_t>(d0), ...},   // output_dims
    {static_cast<int16_t>(s0), ...},   // input_strides（已按输出轴序重排）
    {static_cast<int16_t>(os0), ...}); // output_strides
```

底层 API 模板 `autofuse/v35/ascendc/api_regbase/transpose.h` 为纯 MicroAPI gather 方案：`GenOne/Two/ThreeInnerDimTransposeIndex` 用 `MicroAPI::Arange/Div/Muls/Add` 生成每元素源偏移索引（存入 tmp_buf），数据搬移用 `MicroAPI::Gather` + `StoreAlign` 完成。

#### 6.2.3 V1/V2 对比

| 维度 | V1 `TransposeApiCall` | V2 `TransposeRegApiCall` |
| --- | --- | --- |
| 类型识别 | `kPermutationTable` 查表，仅 7 种固定 permute | 任意 permute，inner_dim/total_dim 参数化 |
| 底层 API | `ConfusionTranspose`（5HD 分形指令） | `TransposeExtend`（MicroAPI gather） |
| tiling | 专用 `ConfusionTransposeTiling`（param0~17） | 无专用 tiling，dims/strides 直接入参 |
| 超维处理 | 查表失败报 INVALID 编译失败 | 外抛 `outer_for_i` for 循环 |
| tmp buffer | 分形重排中间数据 | gather 索引 buffer |
| 同步 | 追加 `PipeBarrier<PIPE_ALL>` | 无 |
| ATT 参数 | API tiling data | `TransposeNodeParams` |

### 6.3 ATT：性能公式与 tiling 计算

#### 6.3.1 参数透传链路

```text
TransposeRegApiCall::Generate 填充 TransposeNodeParams
  → 节点扩展属性（ascir_param::TransposeNodeParams）
  → ATT parser（specific_params_builder）提取到 NodeInfo
  → TransposePerf 消费
```

透传参数为 merge 阶段的原始符号表达式（inner_dim/total_dim/outer_loop_axes/output_dims/input_strides/output_strides），不使用 tiler 展开后的 C++ 表达式，保证性能公式可稳定求值。

#### 6.3.2 V2 性能公式（TransposePerf）

`autofuse/v35/att/api_perf_register/ascendc_regbase_perf.cpp`，公式结构：

```text
perf(AIV_VEC) = (VFHeadCost + MaxLatency + all_vf_instruct_cost) × outer_count
```

- **cal_count**（内层计算量）= 连续内层轴各维乘积 `∏ output_dims[total_dim - inner_dim ...]`；
- **repeat_time** = `ceil(cal_count / repeat_elm)`，其中 repeat_elm 按 dtype 取 64（fp32 口径）/128（fp16）；
- **outer_count** = `outer_loop_axes` 各轴 repeats 的乘积（空则为 1）；
- **MaxLatency**（单次固定耗时，按 total_dim × inner_dim 分档）：

| 场景 | latency |
| --- | --- |
| inner_dim=1（dim2/3/4） | 69 |
| inner_dim=2（dim3/4） | 167 |
| inner_dim=3（dim4） | 386 |

- **all_vf_instruct_cost**：按 MicroAPI 实现逐指令累加 `throughput × 次数`——索引生成的 `Muls × repeat_time`，数据搬移的 `Load × repeat_time + Gather2 × (repeat_time × gather_count) + Store × repeat_time`；其中 gather_count 引入动态符号 `transpose_stride_count`：

```text
transpose_stride_count = (input_strides.back() % (128/dataSize) == 0) ? 128/dataSize
                                                                      : input_strides.back() % (128/dataSize)
gather_count = inner_dim ≤ 2 ? 2 × stride_count : stride_count + 1
```

公式快照由单测锁定（`autofuse/tests/v35/ut/att/gen_model_info/api_perf_register/test_ascir_perf_v2.cpp`），如 dim2/inner1 场景为 `(((4 × transpose_stride_count) + 94) × 3)`。

#### 6.3.3 V1 专用 API tiling 生成

`autofuse/att/gen_model_info/api_tiling_gen/api/confusion_transpose.cpp`：用与 Codegen 相同的 `kPermutationTable` 把 permute 匹配到 7 种模式，为每种模式生成专用 tiling 计算代码。tiling 参数本质是按 32B block（`blockSize = 32/typeSize`）和 16 对齐（`firstAxisAlign = ALIGN_UP(height, 16)`，`repeat = width / blockSize`，`highBlock = height / 16`）的 cube 约束预计算的 stride/repeat。

#### 6.3.4 tiling 候选评估的特殊逻辑

- 转置节点公式通过统一的 `EvaluateModeledPerf(tiling_data)` 参与候选评估，无独立的 transpose 求解分支；
- 特殊点在轴排序层面：`arg_list_reorder.cpp` 的 `MakeSureLoadStoreInnerestSameOrder` 在 Load/Store 最内轴与 Reduce 切分轴重叠且合并后超过 2 根轴时，**保持 Transpose 的同等切分顺序而放弃 Reduce 平衡优化**（equal-order tiling），这利用了 Transpose 场景 Load/Store 各有不同最内轴、必须同等优先级切分的特性。

## 7. 关键文件索引

| 文件路径 | 职责 |
| --- | --- |
| `autofuse/ascir/generator/ascir_builtin_ops_v1.cpp` | Transpose IR 注册 |
| `autofuse/optimize/task_generator/transpose_schedule_case_generator.cpp` | 保留/消除双模板生成与打分函数 |
| `autofuse/optimize/autoschedule/tiling_group.cpp` | GenTransposeTilingGroup 转置轴双切分 |
| `autofuse/optimize/autoschedule/schedule.cpp` | SynchronizeTransposeInputSchedAxis 调度轴同步 |
| `autofuse/optimize/autoschedule/node_cache_marker.cpp` | 含 Transpose 全图禁用 API 缓存 |
| `autofuse/v35/optimize/template/nddma_template.cpp` | Load+Transpose 合并为 NDDMA 节点 |
| `autofuse/v35/optimize/un_alignment_strategy.cpp` | Transpose 前后节点向量化 stride 修正 |
| `autofuse/codegen/api_call/transpose/transpose_api_call.cpp` | V1 TransposeApiCall（ConfusionTranspose） |
| `autofuse/v35/codegen/reg_api_call/reg_transpose_api_call.cpp` | V2 TransposeRegApiCall（TransposeExtend）与节点参数填充 |
| `autofuse/ascendc/api/transpose.h` / `transpose_base_type.h` | V1 底层 API 模板与转置类型定义 |
| `autofuse/v35/ascendc/api_regbase/transpose.h` | V2 MicroAPI gather 实现 |
| `autofuse/v35/codegen/reg_api_call/reg_nddma_api_call.cpp` | 转置折入搬运的 NDDMA API |
| `autofuse/v35/ascir/reg_func/transpose_v2.cpp` | V2 tmp buffer 大小计算 |
| `autofuse/common/ascir_node_param/ascir_node_param.h` | TransposeNodeParams 节点参数结构 |
| `autofuse/v35/att/api_perf_register/ascendc_regbase_perf.cpp` | V2 TransposePerf 性能公式 |
| `autofuse/v35/att/api_perf_register/ascir_api_perf_v2.cpp` | V2 性能公式注册 |
| `autofuse/v35/att/api_perf_register/perf_param_v2.cpp` | MicroAPI 指令系数表 |
| `autofuse/att/gen_model_info/api_tiling_gen/api/confusion_transpose.cpp` | V1 ConfusionTranspose 专用 tiling 生成 |
| `autofuse/att/gen_model_info/expr_gen/arg_list_reorder.cpp` | Transpose equal-order 轴排序 |
| `autofuse/tests/v35/ut/att/gen_model_info/api_perf_register/test_ascir_perf_v2.cpp` | 性能公式快照单测 |
