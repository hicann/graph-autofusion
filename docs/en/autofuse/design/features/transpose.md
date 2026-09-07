# Transpose Fusion

## 1. Feature Background

### 1.1 The Special Nature of Transpose Fusion

Transpose is a View-type operator (it produces no new computation and only changes how data is accessed). The core question in fusion scenarios is: **where should the data rearrangement be done**:

- **UB rearrangement**: The Transpose node is kept in the fusion chain, and a dedicated transpose API performs the rearrangement within UB; the preceding and following elewise computations access data in the transposed axis order;
- **Folding into data transfer**: The transpose semantics are folded into a non-contiguous stride representation and sunk into the Load/Store transfer (NDDMA non-contiguous DMA or non-contiguous vector access), so the fusion chain no longer contains a Transpose node.

Each route has its applicable scenarios: UB rearrangement has extra instruction overhead but keeps transfers contiguous; folding into transfer eliminates rearrangement instructions but makes transfers non-contiguous. When the Schedule phase cannot determine which is better, it generates multiple templates, and ATT selects the best one at the tiling phase using performance formulas.

### 1.2 Design Goals

- **Transpose never becomes a standalone operator**: Transpose is fused into the kernel as much as possible, avoiding the overhead of "a standalone transpose operator plus two transfers";
- **Dual-template selection**: Both routes (keep for UB rearrangement, eliminate by folding into transfer) are generated as multiple templates, and ATT scores and selects between them;
- **Platform-specific implementation**: The V1 platform uses 5HD fractal-instruction-style ConfusionTranspose; the v35 platform uses MicroAPI gather-style TransposeExtend, and supports folding transpose into transfer via NDDMA.

## 2. User Scenarios

| Scenario | Description |
| --- | --- |
| Axis swap in a fusion chain | A Transpose embedded in an elewise chain (e.g., `abs(transpose(x))`); whether to keep or eliminate it is decided by scoring |
| Tail-axis transpose | The input tail axis differs from the output tail axis (e.g., [H,W]→[W,H]); UB rearrangement is mandatory and the Transpose node is kept |
| Non-tail-axis transpose | The tail axis is unchanged and only higher dimensions are swapped (e.g., [N,C,H,W]→[N,H,W,C] for NCHW→NHWC); a large tail axis favors elimination by folding into transfer |
| Multi-Transpose graphs | When a graph contains multiple Transposes, no keep template is generated; all are eliminated and folded into Load |
| Concat via transpose | v1/v35 default to `concat_alg=kConcatAlgTranspose`; Concat is implemented via the transpose algorithm |

## 3. Special Background and Limitations

This section describes the implicit constraints of Transpose fusion across the three modules; most of them cannot be directly derived from the code of a single module.

### 3.1 Implicit Representation of Transpose Semantics

A Transpose node has **no explicit perm/axes attribute**. The transpose semantics are expressed through the difference between the input tensor's axis order and the output tensor's axis order: the input `attr.axis` and the output `attr.axis` are different permutations of the same set of axes (AxisId), with repeats/strides/vectorized_axis permuted accordingly. The Python frontend interface `Transpose(x, axis=..., size=..., stride=...)` takes the transposed axis order directly from the user. This means:

- Both Codegen and ATT must reconstruct the permute vector by comparing input/output axis orders; reading only the "output axis order" on either side loses the transpose information;
- The axis order conventions of intermediate nodes' axis/sched.axis/vectorized_axis must stay consistent before and after the transpose; otherwise the axes will not line up.

### 3.2 Mutual Exclusion with Reduce

**Reduce does not support fusion with Transpose** (the only exception among View-type operators):

- When the scheduling phase detects both Reduce and Transpose in the graph, all Transposes are eliminated directly on the original graph;
- Vectorized-axis reordering is skipped when Reduce is present (the tiling strategy does not yet support fusion scenarios with both reduce and transpose);
- Loads with a tail-axis transpose downstream of a Reduce require extra alignment handling.

### 3.3 Trigger Conditions for Template Selection and Elimination

- **Scoring threshold**: A tail-axis transpose must be kept (UB rearrangement); a non-tail-axis transpose is compared against the threshold of 512B by the tail-axis byte size. A tail axis smaller than 512B favors UB rearrangement (keep); otherwise elimination wins (fold into non-contiguous transfer);
- **Multi-Transpose limit**: When the graph contains multiple Transposes, only the elimination template is generated; no keep template;
- **Naming convention for elimination**: The elimination route renames the upstream Load with a `transpose_` prefix, which also serves as the marker for subsequent NDDMA detection — an implicit protocol between Schedule and the v35 optimization passes;
- **Scheduling axis synchronization**: In the single-Transpose scenario, the sched.axis of all nodes on the input chain is synchronized to the Transpose's scheduling axis (transposed axes must be split into the x/y double split); the multi-Transpose scenario relies on prior elimination; otherwise axis synchronization cannot proceed.

### 3.4 Axis-Count and Form Restrictions in Codegen

- **V1 (ConfusionTranspose)**: Only 7 fixed permute patterns are supported (ND2ND_ONLY, 102, 0213, 2103, 021, 210, 0321). The permute vector must hit `kPermutationTable` to be generated; a miss reports `TRANSPOSE_INVALID` and fails compilation;
- **V2 (TransposeExtend)**: At most 4 axes participate in the transpose (`total_dim ≤ 4`), of which at most 3 are contiguous inner axes (`inner_dim ≤ 3`, contiguity determined by `vectorized_strides[i] == repeats[i+1]*strides[i+1]`); the excess axes are hoisted into `outer_for_i` for loops by Codegen;
- **dtype**: The V2 transpose base case supports only 2/4-byte dtypes; the V1 API tiling fields (param0~param17) are pre-computed from dtype size and the 32B block / 16-alignment constraints;
- **tmp buffer**: V1/V2 transposes are the only APIs using the "full-lifetime TBuf" (life_time_axis_id = -1); the tmp purposes differ (V1 stores fractal rearrangement intermediates, V2 stores gather indices), and so do the size calculations;
- **Synchronization**: V1 appends `PipeBarrier<PIPE_ALL>()` after the call; V2 has no such requirement.

### 3.5 Implicit Constraints Crossing Other Mechanisms

- **API cache disabling**: If the graph contains a Transpose, API result caching is disabled for the entire graph (NodeCacheMarker) to avoid cache correctness issues in the transpose double-split scenario;
- **Alignment width**: The alignment width is 64B when an fp32 Transpose exists, otherwise 32B;
- **Different innermost axes of Load/Store**: In the Transpose scenario, Load and Store each have different innermost axes (an essential property of Transpose). This affects two places: the un-alignment strategy recomputes vectorized_strides by the contiguous tail axis count on the Load side and the Store side separately; and the ATT axis-ordering solver keeps the Transpose's equal-order splitting and gives up the Reduce balance optimization (equal-order tiling) when the merged axis count exceeds the limit;
- **NDDMA preconditions**: When v35 merges a Transpose into a Load to form an NDDMA node, the innermost contiguous vectorized axis count must not exceed 3, and broadcast fusion requires the input memory to be contiguous (dropped in the double-split case with non-scalar input).

## 4. External Interfaces

### 4.1 Operator Definition Interfaces

- **IR registration**: `REG_ASC_IR(Transpose).Input("x","T").Output("y","T").ComputeType(kComputeTranspose)`; supported dtypes: INT16/UINT16/INT32/UINT32/FLOAT16/FLOAT (subject to each platform's registration);
- **Python frontend**: `Transpose(owner_graph, x, axis=..., size=..., stride=...)`, where the user specifies the transposed axis order, sizes, and strides directly.

### 4.2 Generated Code Interfaces (Device Side)

| Platform | API | Form |
| --- | --- | --- |
| V1 | `ConfusionTranspose<T>(dst, src, tmp_buf, transposeType, apiTilingData)` | 5HD fractal instructions, dispatched by the `AutoFuseTransposeType` enum, requiring the accompanying `ConfusionTransposeTiling` (param0~param17) |
| V2 (v35) | `TransposeExtend<inner_dim, total_dim, T>(dst, src, tmp_buf, dst_dims[], src_strides[], dst_strides[])` | MicroAPI gather, fully parameterized by dims/strides; index type selected as int16_t/int32_t by dtype size |
| v35 (transfer route) | `DataCopyNddma<T, dim>(ub, gm[offset], dst_dims, dst_strides, src_strides)` | Non-contiguous DMA, used when the transpose is folded into Load; dimension limit 5 |

### 4.3 ATT-Side Interfaces

- **Node parameter structure**: `TransposeNodeParams` (valid/inner_dim/total_dim/outer_loop_axes/output_dims/input_strides/output_strides), filled by Codegen and passed to ATT performance modeling via node extended attributes;
- **Dedicated API tiling generation**: `REGISTER_API_TILING_FUNC("Transpose", ...)` generates Host-side tiling computation code for each V1 ConfusionTranspose scene;
- **Performance formula registration**: The V2 evaluation function `ascir_v2::TransposeApi` with the factory key `"TransposeV2"`; V1 has no dedicated formula (a kUnitVector placeholder).

## 5. Overall Architecture

Transpose fusion is accomplished through the cooperation of three modules. The core flow is as follows:

```text
┌────────────────────────────────────────────────────────────────────┐
│ Schedule (Decision Layer)                                           │
│   TransposeFusionCaseGenerator                                      │
│     ├─ No Transpose ─────────────────► return directly              │
│     ├─ Contains Reduce ──────────────► eliminate all Transposes     │
│     │                                   on the original graph       │
│     ├─ Single Transpose ─────────────► dual templates:              │
│     │      A: keep (UB rearrangement) + B: eliminate (fold into     │
│     │                                   Load)                       │
│     └─ Multiple Transposes ──────────► elimination template only    │
│   GenTransposeTilingGroup: x/y double split of transposed axes      │
│   v35 NddmaTemplate: merge Load+Transpose into an NDDMA node        │
├────────────────────────────────────────────────────────────────────┤
│ Codegen (Generation Layer)                                          │
│   Keep route:  V1 TransposeApiCall (ConfusionTranspose)             │
│                V2 TransposeRegApiCall (TransposeExtend, gather)     │
│   Eliminate route: NddmaApiCall (non-contiguous DMA) /              │
│                non-contiguous vector access                         │
│   Fill TransposeNodeParams and pass them to ATT                     │
├────────────────────────────────────────────────────────────────────┤
│ ATT (Evaluation Layer)                                              │
│   Dual-template scoring: tail-axis transpose score=1 (keep);        │
│     non-tail-axis compared against the 512B tail threshold          │
│   V2 performance formula: latency tiers + repeat_time +             │
│     gather_count                                                    │
│   V1 dedicated API tiling: 32B block / 16-alignment pre-computation │
│   Axis ordering: keep Load/Store equal-order for Transpose          │
└────────────────────────────────────────────────────────────────────┘
```

Inter-module contract: the ImplGraph produced by Schedule determines whether the transpose exists as a "node" or as "non-contiguous strides"; Codegen translates it faithfully and computes pass-through parameters such as inner_dim/total_dim/outer_loop_axes when generating API parameters; ATT evaluates the performance formulas with these parameters and decides the final template and tiling values.

## 6. Core Implementation

### 6.1 Schedule: Template Decision and Axis Splitting

#### 6.1.1 Keep and Eliminate (TransposeFusionCaseGenerator)

`autofuse/optimize/task_generator/transpose_schedule_case_generator.cpp`, executed after Split/Concat and before Reduce:

- **Elimination process**: `TransposeConvertProcess` pushes up along the Transpose input chain (`UpdateAxis`/`UpdateAxisByPath`), reordering each node's output axes and scheduling axes on the path, folding the transpose semantics into the Load's non-contiguous stride representation. The Load is renamed with the `transpose_` prefix and the Transpose node is deleted;
- **Scoring function** (`TransposeScoreFunctionGenerator`):
  - Tail-axis transpose (input tail axis ≠ output tail axis): score=1, UB rearrangement is mandatory;
  - Non-tail-axis transpose: score=1 (UB rearrangement) when the tail-axis byte size < 512B (`kTransposeNoNeedUBConvertSize`); otherwise score=-1 (eliminate).

#### 6.1.2 Axis Splitting (GenTransposeTilingGroup)

`autofuse/optimize/autoschedule/tiling_group.cpp`: compares input/output axis orders. Axes identical from the tail axis forward go into `n_group` (no double split needed); starting from the first differing axis, input axes go into `x_group` and output axes into `y_group` (the transposed axes are split into two groups, forming the "double split"); the remaining axes go into `y_group`. Therefore **a transposed axis can be a split axis, but must be x/y double-split**, ensuring that the Load side and the Store side each split along their own innermost axis.

#### 6.1.3 Scheduling Axis Synchronization and the NDDMA Template

- `Scheduler::SynchronizeTransposeInputSchedAxis`: in the single-Transpose scenario, the sched.axis of all nodes on the input chain is synchronized to the Transpose's scheduling axis; the multi-Transpose scenario requires prior elimination;
- v35 `NddmaTemplate` (`autofuse/v35/optimize/template/nddma_template.cpp`): `TransposeToNddmaNode` traverses upward from the Transpose, inserts a new Transpose node after the Load and synchronizes intermediate nodes' axis orders, and finally `MergeLoadAndTranspose` changes the Load node type to Nddma and deletes the inserted node — that is, **Load+Transpose merge into a single NDDMA node**; after generation, `UnAlignmentStrategy::ModifyTransposeFusionVectorizedStrides` fixes the vectorized strides.

### 6.2 Codegen: Two Generations of Transpose APIs

#### 6.2.1 V1: TransposeApiCall (ConfusionTranspose)

`autofuse/codegen/api_call/transpose/transpose_api_call.cpp`:

- **Type recognition**: `CodeGenGetTransposeType` treats the position sequence of the output vectorized_axis within the input as the permute vector and looks it up in `kPermutationTable` (covering 21 permutes over 2D/3D/4D), reducing to 8 `AutoFuseTransposeType` values (including INVALID);
- **Generated code example**:

```cpp
AutoFuseTransposeType transposeType = AutoFuseTransposeType::TRANSPOSE_ND2ND_021;
auto apiTilingData = t->transpose_tilingData_<tiling_case_id>;
codegen::ConfusionTranspose<float>(y[offset_y], x[offset_x], tmp_buf_<id>, transposeType, apiTilingData);
AscendC::PipeBarrier<PIPE_ALL>();
```

- **API tiling**: `ConfusionTransposeTiling` has 18 fixed fields (param0~param17) with per-scene semantics (e.g., scene10 uses param0=height, param1=width, param2=highBlock, param3=stride); Host-side values are computed by ATT's dedicated tiling generator;
- **tmp buffer**: the full-lifetime TBuf reuse id is obtained via `tmp_buf_id.find(-1L)`.

The underlying API template `autofuse/ascendc/api/transpose.h` dispatches by transposeType to `ConfusionTransposeNd2Nd10/102/0213/2103/021/210/0321`, implemented with 5HD fractal-instruction-style ConfigMatrix variants.

#### 6.2.2 V2: TransposeRegApiCall (TransposeExtend)

`autofuse/v35/codegen/reg_api_call/reg_transpose_api_call.cpp`:

- **General parameterization**: no table lookup; any permute is decomposed into "outer loop axes + at most 4 inner transpose axes", with template parameters `[inner_dim, total_dim]`;
- **Contiguity determination**: `GetContinuousInnerAxisNum` counts inner axes from the second-to-last axis backward by stride contiguity, with a limit of 3; `transpose_total_axis_num = min(vectorized_axis count, 4)`;
- **Hoisting excess axes**: the repeats of outer axes beyond 4 go into `api_param->outer_loop_axes`; the generator hoists them into `for (int outer_for_i = ...)` loops and folds their strides into the inner offset;
- **Generated code example**:

```cpp
TransposeExtend<inner_dim, total_dim, T>(y[off], x[off], tmp_buf_0,
    {static_cast<int16_t>(d0), ...},   // output_dims
    {static_cast<int16_t>(s0), ...},   // input_strides (already reordered by output axis order)
    {static_cast<int16_t>(os0), ...}); // output_strides
```

The underlying API template `autofuse/v35/ascendc/api_regbase/transpose.h` is a pure MicroAPI gather scheme: `GenOne/Two/ThreeInnerDimTransposeIndex` generates per-element source offset indices via `MicroAPI::Arange/Div/Muls/Add` (stored in tmp_buf), and data movement is done by `MicroAPI::Gather` + `StoreAlign`.

#### 6.2.3 V1/V2 Comparison

| Dimension | V1 `TransposeApiCall` | V2 `TransposeRegApiCall` |
| --- | --- | --- |
| Type recognition | `kPermutationTable` lookup; only 7 fixed permutes | Any permute, parameterized by inner_dim/total_dim |
| Underlying API | `ConfusionTranspose` (5HD fractal instructions) | `TransposeExtend` (MicroAPI gather) |
| Tiling | Dedicated `ConfusionTransposeTiling` (param0~17) | No dedicated tiling; dims/strides passed directly |
| Excess axes | Table miss reports INVALID and fails compilation | Hoisted into `outer_for_i` for loops |
| tmp buffer | Fractal rearrangement intermediates | Gather index buffer |
| Synchronization | Appends `PipeBarrier<PIPE_ALL>` | None |
| ATT parameters | API tiling data | `TransposeNodeParams` |

### 6.3 ATT: Performance Formulas and Tiling Computation

#### 6.3.1 Parameter Pass-Through Chain

```text
TransposeRegApiCall::Generate fills TransposeNodeParams
  → node extended attributes (ascir_param::TransposeNodeParams)
  → ATT parser (specific_params_builder) extracts into NodeInfo
  → consumed by TransposePerf
```

The pass-through parameters are the original symbolic expressions from the merge phase (inner_dim/total_dim/outer_loop_axes/output_dims/input_strides/output_strides), not the tiler-expanded C++ expressions, so that the performance formulas can be evaluated stably.

#### 6.3.2 V2 Performance Formula (TransposePerf)

`autofuse/v35/att/api_perf_register/ascendc_regbase_perf.cpp`, formula structure:

```text
perf(AIV_VEC) = (VFHeadCost + MaxLatency + all_vf_instruct_cost) × outer_count
```

- **cal_count** (inner computation amount) = the product of the contiguous inner axis dimensions `∏ output_dims[total_dim - inner_dim ...]`;
- **repeat_time** = `ceil(cal_count / repeat_elm)`, where repeat_elm is 64 (fp32 scale) / 128 (fp16) by dtype;
- **outer_count** = the product of the repeats of `outer_loop_axes` (1 when empty);
- **MaxLatency** (fixed per-call latency, tiered by total_dim × inner_dim):

| Scenario | latency |
| --- | --- |
| inner_dim=1 (dim2/3/4) | 69 |
| inner_dim=2 (dim3/4) | 167 |
| inner_dim=3 (dim4) | 386 |

- **all_vf_instruct_cost**: accumulated instruction by instruction as `throughput × count` following the MicroAPI implementation — `Muls × repeat_time` for index generation, and `Load × repeat_time + Gather2 × (repeat_time × gather_count) + Store × repeat_time` for data movement, where gather_count introduces the dynamic symbol `transpose_stride_count`:

```text
transpose_stride_count = (input_strides.back() % (128/dataSize) == 0) ? 128/dataSize
                                                                      : input_strides.back() % (128/dataSize)
gather_count = inner_dim ≤ 2 ? 2 × stride_count : stride_count + 1
```

Formula snapshots are locked by unit tests (`autofuse/tests/v35/ut/att/gen_model_info/api_perf_register/test_ascir_perf_v2.cpp`); for example, the dim2/inner1 scenario is `(((4 × transpose_stride_count) + 94) × 3)`.

#### 6.3.3 V1 Dedicated API Tiling Generation

`autofuse/att/gen_model_info/api_tiling_gen/api/confusion_transpose.cpp`: matches the permute to one of the 7 patterns using the same `kPermutationTable` as Codegen, and generates dedicated tiling computation code for each pattern. The tiling parameters are essentially strides/repeats pre-computed under the cube constraints of the 32B block (`blockSize = 32/typeSize`) and 16-alignment (`firstAxisAlign = ALIGN_UP(height, 16)`, `repeat = width / blockSize`, `highBlock = height / 16`).

#### 6.3.4 Special Logic in Tiling Candidate Evaluation

- The Transpose node formula participates in candidate evaluation through the unified `EvaluateModeledPerf(tiling_data)`; there is no separate transpose solving branch;
- The special point is at the axis-ordering level: `MakeSureLoadStoreInnerestSameOrder` in `arg_list_reorder.cpp` — when the Load/Store innermost axes overlap the Reduce split axes and the merged axis count exceeds 2, it **keeps the Transpose's equal-order splitting and gives up the Reduce balance optimization** (equal-order tiling), exploiting the property that Load/Store have different innermost axes in the Transpose scenario and must be split at equal priority.

## 7. Key File Index

| File Path | Responsibility |
| --- | --- |
| `autofuse/ascir/generator/ascir_builtin_ops_v1.cpp` | Transpose IR registration |
| `autofuse/optimize/task_generator/transpose_schedule_case_generator.cpp` | Keep/eliminate dual-template generation and scoring functions |
| `autofuse/optimize/autoschedule/tiling_group.cpp` | GenTransposeTilingGroup double split of transposed axes |
| `autofuse/optimize/autoschedule/schedule.cpp` | SynchronizeTransposeInputSchedAxis scheduling axis synchronization |
| `autofuse/optimize/autoschedule/node_cache_marker.cpp` | Disabling API cache graph-wide when Transpose exists |
| `autofuse/v35/optimize/template/nddma_template.cpp` | Merging Load+Transpose into an NDDMA node |
| `autofuse/v35/optimize/un_alignment_strategy.cpp` | Vectorized stride fixes for nodes around Transpose |
| `autofuse/codegen/api_call/transpose/transpose_api_call.cpp` | V1 TransposeApiCall (ConfusionTranspose) |
| `autofuse/v35/codegen/reg_api_call/reg_transpose_api_call.cpp` | V2 TransposeRegApiCall (TransposeExtend) and node parameter filling |
| `autofuse/ascendc/api/transpose.h` / `transpose_base_type.h` | V1 underlying API templates and transpose type definitions |
| `autofuse/v35/ascendc/api_regbase/transpose.h` | V2 MicroAPI gather implementation |
| `autofuse/v35/codegen/reg_api_call/reg_nddma_api_call.cpp` | NDDMA API for folding transpose into transfer |
| `autofuse/v35/ascir/reg_func/transpose_v2.cpp` | V2 tmp buffer size computation |
| `autofuse/common/ascir_node_param/ascir_node_param.h` | TransposeNodeParams node parameter structure |
| `autofuse/v35/att/api_perf_register/ascendc_regbase_perf.cpp` | V2 TransposePerf performance formula |
| `autofuse/v35/att/api_perf_register/ascir_api_perf_v2.cpp` | V2 performance formula registration |
| `autofuse/v35/att/api_perf_register/perf_param_v2.cpp` | MicroAPI instruction coefficient table |
| `autofuse/att/gen_model_info/api_tiling_gen/api/confusion_transpose.cpp` | V1 ConfusionTranspose dedicated tiling generation |
| `autofuse/att/gen_model_info/expr_gen/arg_list_reorder.cpp` | Transpose equal-order axis ordering |
| `autofuse/tests/v35/ut/att/gen_model_info/api_perf_register/test_ascir_perf_v2.cpp` | Performance formula snapshot unit tests |
