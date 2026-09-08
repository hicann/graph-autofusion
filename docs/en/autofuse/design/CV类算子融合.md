### Feature Description

Matmul+Elemwise fusion is one of the key features of the GE automatic fusion subsystem (`compiler/graph/optimize/autofuse/`). It belongs to the Cube+Vector (CV) fusion category. This feature fuses a Matmul operator (Cube type) with its following Elemwise operator (Vector type), enabling mixed execution on AIC (Cube cores) and AIV (Vector cores), reducing intermediate data movement overhead and improving overall execution performance.

This feature is an important part of the automatic fusion capability. It replaces the earlier manual fusion approach and provides users with automated operator fusion optimization.

### Feature Highlights

Main capabilities of Matmul+Elemwise fusion:

1. **Automatic fusion opportunity detection**: Identifies Matmul operators and their following Elemwise operators in the graph, and determines whether they satisfy fusion conditions.

2. **Template selection and generation**:
   - **UB fusion template**: When UB reuse conditions are satisfied, generates a UB full-load or non-full-load fusion template. The Cube output is written directly to UB, and the Vector side reads from UB for execution.
   - **Fallback template**: When UB fusion conditions are not satisfied, for example when the ACT template is not adapted or tiling has no valid solution, generates a general fallback template. The Cube output is written to GM, and the Vector side reads from GM for execution.

3. **Tiling strategy inheritance**: The tiling strategy of the fused operator inherits the tiling strategy of the standalone Matmul operator, ensuring that fusion does not affect Cube computation efficiency.

4. **ACT code generation**: Uses the ACT template library to generate AscendC kernel code for the fused operator and supports MIX operator mixed execution.

5. **Parallel execution benefits**: Implements Cube and Vector parallel execution through CV fusion, reduces intermediate data movement, and improves overall performance.

### Technical Solution

# Matmul+Elemwise Fusion Requirement Analysis and Design

## Overview

Matmul+Elemwise fusion is a type of CV (Cube+Vector) fusion. It fuses a Cube-type operator (Matmul) with a Vector-type operator (Elemwise), enabling mixed execution on AIC and AIV. The core ideas are:

- **Reducing data movement**: In the traditional execution mode, the Matmul output must be written to GM (Global Memory), and then the Elemwise operator reads the data from GM, which introduces additional data movement overhead. After fusion, the Matmul output can be written directly to UB (Unified Buffer), and the Elemwise operator reads from UB, reducing GM-UB data movement.

- **Parallel execution**: CV fusion supports parallel execution between AIC (Cube cores) and AIV (Vector cores). While Matmul is computing the L0C result, the Vector core can execute Elemwise operations on the previous data tile in parallel, improving overall execution efficiency.

- **Tiling strategy inheritance**: The tiling strategy of the fused operator inherits the standalone Matmul strategy, ensuring that fusion does not affect Cube computation efficiency. This is the key design principle of Matmul+Elemwise fusion.

## Functional Requirements

### Functional Requirement 1: Automatic Fusion Opportunity Detection

#### 1. Introduction

Automatically identifies Matmul operators and their following Elemwise operators in the graph, and determines whether they satisfy fusion conditions. Fusion condition checks include:
- The number of consumers of the Matmul output
- The type of the following operator, which must be Elemwise
- Whether other operators, such as Reduce or Gather, exist in between
- Whether the graph structure satisfies UB fusion or fallback fusion conditions

#### 2. Input

- **Input data**: The optimized GE computation graph (`ascir::HintGraph`), containing Matmul operators and Elemwise operators.
- **Input source**: The input graph of the AutofuseOptimize phase.
- **Quantity**: One complete computation graph.
- **Measurement unit**: Number of graph nodes.
- **Timing requirement**: Executed in the AutofuseOptimize phase and should not significantly affect compilation time.
- **Valid input range**: The graph must contain at least one Matmul operator and at least one Elemwise operator, and the Matmul output must connect to the Elemwise operator.

#### 3. Processing

Fusion opportunity detection flow:

1. **Traverse graph nodes**: Traverse all nodes in the computation graph and identify Cube-type nodes (Matmul).

2. **Check following operators**: For each Matmul node, check whether its following operator is an Elemwise operator.

3. **Determine fusion conditions**:
   - Check the number of consumers of the Matmul output, whether it is unique
   - Check the type of the following operator, whether it is Elemwise
   - Check whether other operators, such as Cast or Broadcast, exist in between

4. **Generate fusion cases**: Generate the corresponding fusion case according to the fusion conditions, either UB fusion or fallback fusion.

#### 4. Output

- **Output data**: A fusion task list (`std::vector<ScheduleTask>`), containing UB fusion tasks and fallback fusion tasks.
- **Output destination**: Passed to subsequent Schedule and Codegen modules.
- **Quantity**: Multiple fusion tasks generated according to the number of fusion opportunities.
- **Measurement unit**: Number of fusion tasks.
- **Valid output range**: Each fusion task contains the fused graph structure, template type, tiling strategy, and related information.

### Functional Requirement 2: UB Fusion Template Generation

#### 1. Introduction

When UB fusion conditions are satisfied, a UB full-load or non-full-load fusion template is generated. The core of the UB fusion template is to write the Matmul output to UB and let the Elemwise operator read from UB, reducing GM-UB data movement.

#### 2. Input

- **Input data**:
  - Split graph structure (`task.grouped_graphs`)
  - Matmul node and following node information
  - Tiling strategy parameters, inherited from the standalone Matmul operator
- **Input source**: Fusion tasks generated by `CubeScheduleCaseGenerator`.
- **Quantity**: Multiple split subgraphs.
- **Measurement unit**: Number of subgraphs.
- **Timing requirement**: Executed during the fusion task generation phase.
- **Valid input range**: The subgraph must satisfy UB fusion conditions.

#### 3. Processing

UB fusion template generation flow:

1. **Check subgraph structure**: Check whether each subgraph contains a Cube-type node (Matmul) and a Vector-type node (Elemwise).

2. **Generate Nddma nodes**: Convert the Load-Broadcast pattern into an Nddma node, enabling Load and Broadcast fusion. An Nddma node can load from GM to UB directly and supports broadcast operations.

3. **Select full-load or non-full-load mode**: Select UB full-load or non-full-load mode according to the tiling strategy and UB size:
   - **UB full-load**: One tile of the Matmul output can be fully loaded into UB, and Elemwise can process the complete tile at once.
   - **UB non-full-load**: One tile of the Matmul output exceeds UB capacity and must be loaded into UB in multiple batches, so Elemwise processes it in batches.

4. **Generate fusion task**: Generate a UB fusion task (`ScheduleTask`) and set the template type to `kUBFuse`.

#### 4. Output

- **Output data**: A UB fusion task containing the fused graph structure, template type (`kUBFuse`), and tiling strategy.
- **Output destination**: Passed to the Schedule and Codegen modules.
- **Quantity**: One UB fusion task.
- **Measurement unit**: Fusion task.
- **Timing**: Generated during the fusion task generation phase and used by subsequent phases.
- **Valid output range**: The `cube_type` of the UB fusion task must be `ascir::CubeTemplateType::kUBFuse`.

### Functional Requirement 3: Fallback Fusion Template Generation

#### 1. Introduction

When UB fusion conditions are not satisfied, a fallback fusion template is generated. The core of the fallback fusion template is to write the Matmul output to GM and let the Elemwise operator read from GM. This reduces operator invocation overhead, but GM-UB data movement is still required.

#### 2. Input

- **Input data**:
  - Split graph structure (`task.grouped_graphs`)
  - Matmul node and following node information
  - Tiling strategy parameters, inherited from the standalone Matmul operator
- **Input source**: Fusion tasks generated by `CubeScheduleCaseGenerator`.
- **Quantity**: Multiple split subgraphs.
- **Measurement unit**: Number of subgraphs.
- **Timing requirement**: Executed during the fusion task generation phase.
- **Valid input range**: The subgraph does not satisfy UB fusion conditions but can still be fused.

#### 3. Processing

Fallback fusion template generation flow:

1. **Check subgraph structure**: Check whether each subgraph contains a Cube-type node (Matmul) and a Vector-type node (Elemwise).

2. **Graph splitting**: When the graph contains multiple Cube nodes or complex Vector nodes, split it into multiple subgraphs, each of which is executed independently.

3. **Move Cube subgraphs**: Move Cube-type subgraphs to the end of the execution sequence, ensuring that Vector execution can immediately follow after Cube computation completes.

4. **Generate fusion task**: Generate a fallback fusion task (`ScheduleTask`) and set the template type to `kCommon`.

#### 4. Output

- **Output data**: A fallback fusion task containing the fused graph structure, template type (`kCommon`), and tiling strategy.
- **Output destination**: Passed to the Schedule and Codegen modules.
- **Quantity**: One fallback fusion task.
- **Measurement unit**: Fusion task.
- **Timing**: Generated during the fusion task generation phase and used by subsequent phases.
- **Valid output range**: The `cube_type` of the fallback fusion task must be `ascir::CubeTemplateType::kCommon`.

### Functional Requirement 4: Tiling Strategy

#### 1. Introduction

The tiling strategy of the fused operator inherits the tiling strategy of the standalone Matmul operator, ensuring that fusion does not affect Cube computation efficiency. Tiling strategy inheritance includes:
- Tile size (L1Tile and L0Tile)
- Full-load mode selection (A full-load, B full-load, AB full-load)
- L0C2OUT mode selection (OnTheFly, Fixpipe)

#### 2. Input

- **Input data**:
  - Tiling strategy parameters of the standalone Matmul operator, obtained from the tiling key
  - Fused graph structure
- **Input source**: Matmul operator tiling strategy generation module.
- **Quantity**: One set of tiling parameters.
- **Measurement unit**: Number of parameters, such as m, n, k, and batch.
- **Timing requirement**: Executed during the fusion task generation phase.
- **Valid input range**: Tiling parameters must comply with the Matmul operator tiling strategy specification.

#### 3. Processing

Tiling strategy inheritance flow:

##### 1. Tiling Inheritance and Alignment Principles

| Principle | Description |
|------|------|
| **Cube Tiling Inheritance** | Vector tiling is strictly based on Cube `baseM`/`baseN` |
| **Mandatory UB Alignment** | Data processed by Vector cores must be aligned to 32 bytes |
| **Round-up Alignment Strategy** | When `baseN` does not satisfy alignment requirements, it is rounded up to ensure hardware compatibility |

##### 2. Vector Tiling Calculation Formula

| Parameter | Formula | Description |
|------|---------|------|
| **ub_align_value** | `32 / cube_output_type_size` | UB alignment granularity, measured in number of elements |
| **basen_align** | `ceil(baseN / ub_align_value) * ub_align_value` | Aligns `baseN` to the UB granularity |
| **basen_basem_align** | `(baseM * basen_align) / 2 + basen_align` | Single Vector-core processing size, used to estimate UB/workspace requirements |

##### 3. UB Fusion Feasibility Check

| Vector Tiling Key | Condition | Fusion Mode |
|-------------------|---------|---------|
| **0 (No DB)** | `basen_basem_align * dtype_size <= ub_size` | UB full-load, single buffer |
| **1 (DB)** | `basen_basem_align * dtype_size * 2 <= ub_size` | UB non-full-load loop, double buffer |
| **-1 (Safety)** | Not satisfying the above conditions | Safety fusion, Cube output to GM |


#### 4. Output

- **Output data**: Fused operator tiling data structure, containing Matmul tiling parameters.
- **Output destination**: Passed to the Codegen module and runtime.
- **Quantity**: One tiling data structure.
- **Measurement unit**: Structure.
- **Timing**: Generated during the fusion task generation phase and used at runtime.
- **Valid output range**: The tiling data structure must comply with the CMCT template library specification.

### Functional Requirement 5: ACT Code Generation

#### 1. Introduction

Uses the ACT template library to generate AscendC kernel code for the fused operator. The CMCT template library provides Matmul fusion kernel templates and epilogue templates, supporting mixed execution.

#### 2. Input

- **Input data**:
  - Fused graph structure
  - Tiling strategy parameters
  - ACT template configuration, such as kernel type and epilogue type
- **Input source**: Fusion task and tiling strategy.
- **Quantity**: One fusion task.
- **Measurement unit**: Fusion task.
- **Timing requirement**: Executed during the Codegen phase.
- **Valid input range**: The fusion task must contain a complete graph structure and tiling strategy.

#### 3. Processing

ACT code generation flow:

1. **Select kernel template**: Select the corresponding kernel template according to the fusion type, UB fusion or fallback fusion:
   - **UB fusion**: Uses the `KernelMatmulMixWithoutQue` template, supporting AIC+AIV mixed execution.
   - **Fallback fusion**: Uses the `KernelMatmulWithoutQue` template, pure AIC execution.

2. **Configure epilogue template**:
   - **UB fusion**: Uses the `BlockEpilogueCV` template, supporting Cube output to UB and Vector reading from UB.
   - **Fallback fusion**: Uses the `BlockEpilogueEmpty` template, with Cube output to GM.

3. **Generate AscendC code**: Generate AscendC kernel code according to the template and tiling parameters, including:
    - Kernel function definition, such as `MatMulActKernelFusion`
    - Tiling data structure definition
    - Epilogue operation code, such as Elemwise operations

4. **Compile fused operator**: Compile the generated AscendC code into an executable operator kernel and package it into the OM model.

#### 4. Output

- **Output data**: AscendC kernel code (`.cpp` and `.h` files) of the fused operator and the compiled operator binary (`.so` file).
- **Output destination**: Packaged into the OM model and loaded at runtime.
- **Quantity**: A set of code files and one binary file.
- **Measurement unit**: Number of files.
- **Timing**: Generated during the Codegen and Build phases and used at runtime.
- **Valid output range**: The generated code must comply with the AscendC specification, and the compiled binary must be executable on Ascend chips.


### Key Technologies and Algorithms

#### 1. CV Fusion Parallel Flow

The core of CV fusion is to enable parallel execution between Cube (AIC) and Vector (AI1/AI2). The detailed flow is as follows:

```
Timeline:
T0: AIC executes Matmul Tile 1
    └─ Computes L0C result and writes it to UB (UB fusion) or GM (fallback fusion)
T1: AIC executes Matmul Tile 2
    └─ Computes L0C result and writes it to UB or GM
    └─ Meanwhile, AI1/AI2 executes the Elemwise operation of Tile 1, reading from UB or GM
T2: AIC executes Matmul Tile 3
    └─ Computes L0C result and writes it to UB or GM
    └─ Meanwhile, AI1/AI2 executes the Elemwise operation of Tile 2
...
```

Benefit analysis:
- **Reduced data movement**: UB fusion reduces GM-UB data movement and saves about 20-40% of data movement time.
- **Parallel execution**: AIC and AI1/AI2 execute in parallel, improving overall performance by about 15-30%.
- **Memory reuse**: UB memory is reused, reducing memory consumption.

#### 2. Nddma Node Generation

The Nddma node is a key technology for fusing the Load-Broadcast pattern into a single node. The detailed implementation includes:

- **Load-Broadcast fusion**: Fuses Load and Broadcast operators into an Nddma operator, which supports loading from GM to UB and performing broadcast at the same time.
- **Load-Cast-Broadcast fusion**: Handles more complex patterns by fusing Load, Cast, and Broadcast into an Nddma operator.


#### 3. ACT Template Library Usage

The ACT template library provides Matmul fusion kernel and epilogue templates. Key templates include:

- **KernelMatmulMixWithoutQue**: Kernel template that supports AIC+AI1/AI2 mixed execution, used for UB fusion.
- **KernelMatmulWithoutQue**: Kernel template for pure AIC execution, used for fallback fusion.
- **BlockEpilogueCV**: Epilogue template where Cube output is written to UB, used for UB fusion.
- **BlockEpilogueEmpty**: Epilogue template where Cube output is written to GM, used for fallback fusion.

Example usage:

```cpp
// UB fusion template
using FusionOp = AutoFusionVector;
using BlockEpilogue = Block::BlockEpilogueCV<L0TileShape, OutType, OutType, FusionOp>;
using MatmulKernel = Kernel::KernelMatmulMixWithoutQue<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

// Fallback fusion template
using FusionOp = Block::DefaultFusion<OutType, OutType>;
using BlockEpilogue = Block::BlockEpilogueEmpty;
using MatmulKernel = Kernel::KernelMatmulWithoutQue<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
```

Implementation location: `compiler/graph/optimize/autofuse/v35/ascendc/api_cube/matmul/mat_mul_pingpong_basic_cmct.h`

#### 4. Tiling Key Generation

The tiling key is used to generate the tiling strategy of the Matmul operator. Key parameters include:

- `API_LEVEL`: API level, High Level or Basic Level
- `A_TRANS`, `B_TRANS`: Matrix transpose flags
- `BATCH_MODEL`: Batch model
- `MODEL`: Computation model, such as Basic, StreamK, and KEqualZero
- `FULL_LOAD`: Full-load mode, such as No Full Load, A Full Load, B Full Load, and AB Full Load
- `L0C2OUT_MODEL`: L0C output mode, such as OnTheFly and Fixpipe

Full-load mode descriptions:
- **No Full Load**: Non-full-load mode. Matmul inputs are loaded into L1 in multiple steps.
- **A Full Load**: A matrix full-load. The A matrix is loaded into L1 at once.
- **B Full Load**: B matrix full-load. The B matrix is loaded into L1 at once.
- **AB Full Load**: AB matrix full-load. Both A and B matrices are loaded into L1 at once.

Implementation location: `compiler/graph/optimize/autofuse/v35/ascendc/api_cube/matmul/mat_mul_tiling_key.h`

### Flow Design

#### 1. Overall Matmul+Elemwise Fusion Flow

```
AutofuseOptimize phase:
├─ 1. Traverse graph nodes and identify Matmul operators
├─ 2. Check following operators and determine fusion conditions
├─ 3. CubeScheduleCaseGenerator.GenerateGeneralCase
│   ├─ Split graph structure, inserting Load, Store, and Workspace
│   ├─ Generate a general fusion case
│   └─ Determine whether graph splitting is required
├─ 4. CubeScheduleCaseGenerator.GeneratorTask
│   ├─ ScheduleGroupGraphPartitioner.PartitionByConnectivity
│   │   ├─ Split into multiple subgraphs (grouped_graphs)
│   │   └─ Check the number of subgraphs, whether it is greater than 1
│   ├─ If grouped_graphs.size() > 1:
│   │   ├─ task.cube_type = kCommon (fallback template)
│   │   ├─ MoveCubeGraphsToEnd (move Cube subgraphs to the end)
│   │   ├─ GeneratorUbTask (generate UB fusion task)
│   │   │   ├─ ub_task.cube_type = kUBFuse
│   │   │   ├─ Check subgraph structure (Load-Broadcast pattern)
│   │   │   ├─ GenNddmaNode (generate Nddma node)
│   │   │   └─ Add ub_task to tasks
│   │   └─ Add task to tasks
│   └─ If grouped_graphs.size() == 1:
│       └─ task.cube_type = kFixpip (single Cube subgraph)
│       └─ Add task to tasks
└─ 5. Return fusion task list (tasks)
```

#### 2. UB Fusion Task Generation Flow (GeneratorUbTask)

```
GeneratorUbTask:
├─ 1. Traverse grouped_graphs and check subgraph structure
├─ 2. For non-Cube subgraphs:
│   ├─ Check Load node
│   ├─ Check following nodes of Load (Broadcast or Cast)
│   ├─ If it is Load-Broadcast:
│   │   └─ GenNddmaNode (Load + Broadcast -> Nddma)
│   ├─ If it is Load-Cast-Broadcast:
│   │   └─ SwapCastBrcAndGenNddma (swap Cast and Broadcast, then generate Nddma)
│   └─ Check whether Broadcast nodes still exist; if yes, return failure
├─ 3. Add the processed subgraphs to ub_task.grouped_graphs
└─ 4. Add ub_task to tasks
```

#### 3. ACT Code Generation Flow

```
Codegen phase:
├─ 1. Select template according to the cube_type of the fusion task
│   ├─ If cube_type == kUBFuse:
│   │   ├─ Use KernelMatmulMixWithoutQue (mixed kernel)
│   │   ├─ Use BlockEpilogueCV (UB epilogue)
│   │   ├─ Kernel Type = ASCENDC_TPL_MIX_AIC_1_2
│   │   └─ Generate AscendC code
│   └─ If cube_type == kCommon:
│       ├─ Use KernelMatmulWithoutQue (pure AIC kernel)
│       ├─ Use BlockEpilogueEmpty (GM epilogue)
│       ├─ Kernel Type = ASCENDC_TPL_AIC_ONLY
│       └─ Generate AscendC code
├─ 2. Generate tiling data structure (MatMulV3BasicTilingData)
├─ 3. Generate kernel function (MatMulActKernelFusion)
└─ 4. Compile AscendC code into operator binary (.so)
```

### Related Documents


### Test Plan

| **Test Category** | **Key Test Item** | **Test Method** | **Case Type** |
| ------------ | -------------- | ------------ | ------------ |
| **Function** | UB fusion task generation | Construct a Load-Broadcast pattern graph and verify whether the generated fusion task is `kUBFuse` | UT |
| **Function** | Fallback fusion task generation | Construct a complex graph structure, such as multiple Cube nodes, and verify whether the generated fusion task is `kCommon` | UT |
| **Function** | Nddma node generation | Verify whether `GenNddmaNode` correctly fuses Load and Broadcast | UT |
| **Function** | Tiling strategy inheritance | Verify whether fused operator tiling parameters inherit Matmul parameters | UT |
| **Performance** | CV fusion parallel execution | Use profiling tools to measure AIC and AIV parallel execution time | ST |
| **Performance** | Data movement overhead | Measure data movement time differences between UB fusion and fallback fusion | ST |
| **Precision** | Fused operator precision | Compare outputs before and after fusion to ensure no precision loss | ST |
| **Compatibility** | Old OM execution on the new version | Execute an OM compiled by an old version, without fused operators, on the new runtime version | ST |

### Acceptance Criteria

1. **Functional acceptance**:
   - UB fusion task generation is correct (`cube_type == kUBFuse`, containing Nddma nodes)
   - Fallback fusion task generation is correct (`cube_type == kCommon`)
   - Tiling strategy inheritance is correct, and parameters match
   - ACT code generation is correct, compilation succeeds, and execution is successful

2. **Performance acceptance**:
   - UB fusion performance improves by more than 15% compared with the unfused version
   - Fallback fusion performance does not regress compared with the unfused version
   - Compilation time increases by less than 5%

3. **Precision acceptance**:
   - Operator outputs before and after fusion are precision-consistent, with error less than 1e-5
   - Typical recommended models have no precision loss

### Notes

Supplement the benefit evaluation model for the UB full-load template.

![cv_fusion_diagram_v2.drawio.png](https://raw.gitcode.com/user-images/assets/8824148/5c7bc09a-2066-4457-914f-4223f69171af/cv_fusion_diagram_v2.drawio.png 'cv_fusion_diagram_v2.drawio.png')
