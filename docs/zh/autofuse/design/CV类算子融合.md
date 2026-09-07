### 功能描述

Matmul+Elemwise 融合是 GE 自动融合子系统（compiler/graph/optimize/autofuse/）的关键特性之一，属于 Cube+Vector（CV）融合范畴。该特性通过将 Matmul 算子（Cube 类型）与后继的 Elemwise 算子（Vector 类型）融合，实现 AIC（Cube 核）与 AIV（Vector 核）的混合执行，减少中间数据的搬运开销，提升整体执行性能。

本特性是自动融合能力的重要组成部分，替代了早期的手动融合方式，为用户提供自动化的算子融合优化能力。
### 功能要点

Matmul+Elemwise 融合的主要功能：

1. **自动识别融合机会**：识别图中的 Matmul 算子及其后继的 Elemwise 算子，判断是否满足融合条件。

2. **模板选择与生成**：
   - **UB 融合模板**：当满足 UB 复用条件时，生成 UB 全载或非全载融合模板，实现 Cube 输出直接写入 UB，Vector 从 UB 读取执行。
   - **兜底模板**：当 UB 融合条件不满足时（ACT模板未适配或tiling 无解时），生成通用兜底模板，Cube 输出写入 GM，Vector 从 GM 读取执行。

3. **Tiling 策略继承**：融合算子的 Tiling 策略继承 Matmul 单算子的 Tiling 策略，确保融合后不影响 Cube 的计算效率。

4. **ACT 代码生成**：使用 ACT模板库生成融合算子的 AscendC Kernel 代码，支持 MIX算子混合执行。

5. **并行执行收益**：通过 CV 融合实现 Cube 和 Vector 的并行执行，减少中间数据搬运，提升整体性能。

### 技术方案

# Matmul+Elemwise 融合需求分析&设计

## 整体介绍

Matmul+Elemwise 融合属于 CV（Cube+Vector）融合的一种，通过将 Cube 类型算子（Matmul）与 Vector 类型算子（Elemwise）融合，实现 AIC 和AIV 的混合执行。融合的核心思想是：

- **减少数据搬运**：传统方式下，Matmul 输出需要写入 GM（Global Memory），然后 Elemwise 从 GM 读取数据执行，增加了数据搬运开销。融合后，Matmul 输出可以直接写入 UB（Unified Buffer），Elemwise 从 UB 读取数据执行，减少了 GM-UB 的搬运。

- **并行执行**：CV 融合支持 AIC（Cube 核）和 AIV（Vector 核）的并行执行。在 Matmul 计算 L0C 结果的同时，Vector 核可以并行执行前一批数据的 Elemwise 操作，提升了整体执行效率。

- **Tiling 策略继承**：融合算子的 Tiling 策略继承 Matmul 单算子的策略，确保融合不影响 Cube 的计算效率。这是 Matmul+Elemwise 融合的关键设计原则。

## 功能需求

### 功能需求 1：自动识别融合机会

#### 1. 介绍

自动识别图中的 Matmul 算子及其后继的 Elemwise 算子，判断是否满足融合条件。融合条件的判断包括：
- Matmul 输出的消费者数量
- 后继算子的类型（必须是 Elemwise 类型）
- 中间是否存在其他算子（如 Reduce、Gather 等）
- 图结构是否满足 UB 融合或兜底融合的条件

#### 2. 输入

- **输入数据**：优化后的 GE 计算图（`ascir::HintGraph`），包含 Matmul 算子和 Elemwise 算子。
- **输入来源**：AutofuseOptimize 阶段的输入图。
- **数量**：一个完整的计算图。
- **度量单位**：图节点数量。
- **时间要求**：在 AutofuseOptimize 阶段执行，不影响编译时长。
- **有效输入范围**：图中必须包含至少一个 Matmul 算子和至少一个 Elemwise 算子，且 Matmul 的输出连接到 Elemwise 算子。

#### 3. 处理

融合机会识别的处理流程：

1. **遍历图节点**：遍历计算图中的所有节点，识别 Cube 类型节点（Matmul）。

2. **检查后继算子**：对于每个 Matmul 节点，检查其后继算子是否为 Elemwise 类型。

3. **判断融合条件**：
   - 检查 Matmul 输出的消费者数量（是否唯一）
   - 检查后继算子的类型（是否为 Elemwise）
   - 检查中间是否存在其他算子（如 Cast、Broadcast 等）

4. **生成融合 Case**：根据融合条件，生成相应的融合 Case（UB 融合或兜底融合）。

#### 4. 输出

- **输出数据**：融合任务列表（`std::vector<ScheduleTask>`），包含 UB 融合任务和兜底融合任务。
- **输出到何处**：传递给后续的 Schedule 和 Codegen 模块。
- **数量**：根据融合机会的数量生成多个融合任务。
- **度量单位**：融合任务数量。
- **有效输出范围**：每个融合任务包含融合后的图结构、模板类型、Tiling 策略等信息。

### 功能需求 2：UB 融合模板生成

#### 1. 介绍

当满足 UB 融合条件时，生成 UB 全载或非全载融合模板。UB 融合模板的核心是将 Matmul 输出写入 UB，Elemwise 从 UB 读取数据执行，减少 GM-UB 的数据搬运。

#### 2. 输入

- **输入数据**：
  - 拆分后的图结构（`task.grouped_graphs`）
  - Matmul 节点及其后继节点信息
  - Tiling 策略参数（继承 Matmul 单算子的 Tiling 策略）
- **输入来源**：CubeScheduleCaseGenerator 生成的融合任务。
- **数量**：多个拆分后的子图。
- **度量单位**：子图数量。
- **时间要求**：在融合任务生成阶段执行。
- **有效输入范围**：子图必须满足 UB 融合条件。

#### 3. 处理

UB 融合模板生成的处理流程：

1. **检查子图结构**：检查每个子图是否包含 Cube 类型节点（Matmul）和 Vector 类型节点（Elemwise）。

2. **生成 Nddma 节点**：将 Load-Broadcast 模式转换为 Nddma 节点，实现 Load 和 Broadcast 的融合。Nddma 节点可以直接从 GM 加载到 UB，并支持广播操作。

3. **选择全载/非全载模式**：根据 Tiling 策略和 UB 大小，选择 UB 全载或非全载模式：
   - **UB 全载**：Matmul 的一个 Tile 输出可以完全载入 UB，Elemwise 可以一次性处理完整 Tile。
   - **UB 非全载**：Matmul 的一个 Tile 输出超过 UB 大小，需要分多次载入 UB，Elemwise 需要分批处理。

4. **生成融合任务**：生成 UB 融合任务（`ScheduleTask`），设置模板类型为 `kUBFuse`。

#### 4. 输出

- **输出数据**：UB 融合任务，包含融合后的图结构、模板类型（`kUBFuse`）、Tiling 策略。
- **输出到何处**：传递给 Schedule 和 Codegen 模块。
- **数量**：一个 UB 融合任务。
- **度量单位**：融合任务。
- **时序**：在融合任务生成阶段生成，后续阶段使用。
- **有效输出范围**：UB 融合任务的 `cube_type` 必须为 `ascir::CubeTemplateType::kUBFuse`。

### 功能需求 3：兜底融合模板生成

#### 1. 介绍

当 UB 融合条件不满足时，生成兜底融合模板。兜底融合模板的核心是将 Matmul 输出写入 GM，Elemwise 从 GM 读取数据执行，虽然减少了算子调用开销，但仍需 GM-UB 的数据搬运。

#### 2. 输入

- **输入数据**：
  - 拆分后的图结构（`task.grouped_graphs`）
  - Matmul 节点及其后继节点信息
  - Tiling 策略参数（继承 Matmul 单算子的 Tiling 策略）
- **输入来源**：CubeScheduleCaseGenerator 生成的融合任务。
- **数量**：多个拆分后的子图。
- **度量单位**：子图数量。
- **时间要求**：在融合任务生成阶段执行。
- **有效输入范围**：子图不满足 UB 融合条件，但仍可进行融合。

#### 3. 处理

兜底融合模板生成的处理流程：

1. **检查子图结构**：检查每个子图是否包含 Cube 类型节点（Matmul）和 Vector 类型节点（Elemwise）。

2. **图拆分处理**：当图中包含多个 Cube 节点或复杂的 Vector 节点时，需要拆分为多个子图，每个子图独立执行。

3. **移动 Cube 子图**：将 Cube 类型的子图移动到执行序列的末尾，确保 Cube 计算完成后，Vector 可以立即执行后续操作。

4. **生成融合任务**：生成兜底融合任务（`ScheduleTask`），设置模板类型为 `kCommon`。

#### 4. 输出

- **输出数据**：兜底融合任务，包含融合后的图结构、模板类型（`kCommon`）、Tiling 策略。
- **输出到何处**：传递给 Schedule 和 Codegen 模块。
- **数量**：一个兜底融合任务。
- **度量单位**：融合任务。
- **时序**：在融合任务生成阶段生成，后续阶段使用。
- **有效输出范围**：兜底融合任务的 `cube_type` 必须为 `ascir::CubeTemplateType::kCommon`。

### 功能需求 4：Tiling 策略

#### 1. 介绍

融合算子的 Tiling 策略继承 Matmul 单算子的 Tiling 策略，确保融合不影响 Cube 的计算效率。Tiling 策略继承包括：
- Tile 大小（L1Tile、L0Tile）
- 全载模式选择（A 全载、B 全载、AB 全载）
- L0C2OUT 模式选择（OnTheFly、Fixpipe）

#### 2. 输入

- **输入数据**：
  - Matmul 单算子的 Tiling 策略参数（从 Tiling Key 中获取）
  - 融合后的图结构
- **输入来源**：Matmul 算子的 Tiling 策略生成模块。
- **数量**：一组 Tiling 参数。
- **度量单位**：参数数量（如 m、n、k、batch 等）。
- **时间要求**：在融合任务生成阶段执行。
- **有效输入范围**：Tiling 参数必须符合 Matmul 算子的 Tiling 策略规范。

#### 3. 处理

Tiling 策略继承的处理流程：
##### 1. Tiling 继承与对齐原则

| 原则 | 说明 |
|------|------|
| **Cube Tiling 继承** | Vector Tiling 严格基于 Cube 的 baseM/baseN |
| **UB 对齐强制** | Vector 核处理的数据必须对齐到 32 字节 |
| **向上对齐策略** | baseN 不满足对齐时，向上对齐（增加元素，确保硬件兼容） |

##### 2. Vector Tiling 计算公式

| 参数 | 计算公式 | 说明 |
|------|---------|------|
| **ub_align_value** | `32 / cube_output_type_size` | UB 对齐粒度（元素数量） |
| **basen_align** | `ceil(baseN / ub_align_value) * ub_align_value` | baseN 对齐到 UB 粒度 |
| **basen_basem_align** | `(baseM * basen_align) / 2 + basen_align` | Vector 核单次处理大小（估算 UB/workspace） |

##### 3. UB 融合可行性判断

| Vector Tiling Key | 判断条件 | 融合模式 |
|-------------------|---------|---------|
| **0（No DB）** | `basen_basem_align * dtype_size <= ub_size` | UB 全载，单缓冲 |
| **1（DB）** | `basen_basem_align * dtype_size * 2 <= ub_size` | UB 非全载循环，双缓冲 |
| **-1（Safety）** | 不满足上述条件 | Safety 融合，Cube 输出到 GM |


#### 4. 输出

- **输出数据**：融合算子的 Tiling Data 结构，包含 Matmul 的 Tiling 参数。
- **输出到何处**：传递给 Codegen 模块和运行时。
- **数量**：一个 Tiling Data 结构。
- **度量单位**：结构体。
- **时序**：在融合任务生成阶段生成，运行时使用。
- **有效输出范围**：Tiling Data 结构必须符合 CMCT 模板库的规范。

### 功能需求 5：ACT 代码生成

#### 1. 介绍

使用 ACT 模板库生成融合算子的 AscendC Kernel 代码。CMCT 模板库提供了 Matmul 融合的 Kernel 模板和 Epilogue 模板，支持混合执行。

#### 2. 输入

- **输入数据**：
  - 融合后的图结构
  - Tiling 策略参数
  - ACT模板配置（如 Kernel 类型、Epilogue 类型）
- **输入来源**：融合任务和 Tiling 策略。
- **数量**：一个融合任务。
- **度量单位**：融合任务。
- **时间要求**：在 Codegen 阶段执行。
- **有效输入范围**：融合任务必须包含完整的图结构和 Tiling 策略。

#### 3. 处理

ACT代码生成的处理流程：

1. **选择 Kernel 模板**：根据融合类型（UB 融合或兜底融合）选择对应的 Kernel 模板：
   - **UB 融合**：使用 `KernelMatmulMixWithoutQue` 模板，支持 AIC+AIV 混合执行。
   - **兜底融合**：使用 `KernelMatmulWithoutQue` 模板，纯 AIC 执行。

2. **配置 Epilogue 模板**：
   - **UB 融合**：使用 `BlockEpilogueCV` 模板，支持 Cube 输出到 UB，Vector 从 UB 读取执行。
   - **兜底融合**：使用 `BlockEpilogueEmpty` 模板，Cube 输出到 GM。

3. **生成 AscendC 代码**：根据模板和 Tiling 参数，生成 AscendC Kernel 代码，包括：
    - Kernel 函数定义（如 `MatMulActKernelFusion`）
    - Tiling Data 结构定义
    - Epilogue 操作代码（Elemwise 操作）

4. **编译融合算子**：将生成的 AscendC 代码编译为可执行的算子 Kernel，打包到 OM 模型中。

#### 4. 输出

- **输出数据**：融合算子的 AscendC Kernel 代码（`.cpp`、`.h` 文件）和编译后的算子二进制（`.so` 文件）。
- **输出到何处**：打包到 OM 模型中，运行时加载执行。
- **数量**：一组代码文件和一个二进制文件。
- **度量单位**：文件数量。
- **时序**：在 Codegen 和 Build 阶段生成，运行时使用。
- **有效输出范围**：生成的代码必须符合 AscendC 规范，编译后的二进制必须能在昇腾芯片上执行。


### 关键技术/算法

#### 1. CV 融合并行流程

CV 融合的核心是实现 Cube（AIC）和 Vector（AI1/AI2）的并行执行，具体流程如下：

```
时间轴：
T0: AIC 执行 Matmul Tile 1
    └─ 计算 L0C 结果，写入 UB（UB 融合）或 GM（兜底融合）
T1: AIC 执行 Matmul Tile 2
    └─ 计算 L0C 结果，写入 UB 或 GM
    └─ 同时，AI1/AI2 执行 Tile 1 的 Elemwise 操作（从 UB 或 GM 读取）
T2: AIC 执行 Matmul Tile 3
    └─ 计算 L0C 结果，写入 UB 或 GM
    └─ 同时，AI1/AI2 执行 Tile 2 的 Elemwise 操作
...
```

收益分析：
- **减少数据搬运**：UB 融合减少了 GM-UB 的数据搬运，节省约 20-40% 的搬运时间。
- **并行执行**：AIC 和 AI1/AI2 并行执行，整体性能提升约 15-30%。
- **内存复用**：UB 内存复用，减少内存占用。

#### 2. Nddma 节点生成

Nddma 节点是将 Load-Broadcast 模式融合为单一节点的关键技术。具体实现：

- **Load-Broadcast 融合**：将 Load 算子和 Broadcast 算子融合为 Nddma 算子，支持从 GM 加载到 UB 并同时执行广播操作。
- **Load-Cast-Broadcast 融合**：处理更复杂的模式，将 Load、Cast、Broadcast 融合为 Nddma 算子。


#### 3. ACT 模板库使用

ACT 模板库提供了 Matmul 融合的 Kernel 和 Epilogue 模板，关键模板：

- **KernelMatmulMixWithoutQue**：支持 AIC+AI1/AI2 混合执行的 Kernel 模板（UB 融合）。
- **KernelMatmulWithoutQue**：纯 AIC 执行的 Kernel 模板（兜底融合）。
- **BlockEpilogueCV**：Cube 输出到 UB 的 Epilogue 模板（UB 融合）。
- **BlockEpilogueEmpty**：Cube 输出到 GM 的 Epilogue 模板（兜底融合）。

使用示例：

```cpp
// UB 融合模板
using FusionOp = AutoFusionVector;
using BlockEpilogue = Block::BlockEpilogueCV<L0TileShape, OutType, OutType, FusionOp>;
using MatmulKernel = Kernel::KernelMatmulMixWithoutQue<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

// 兜底融合模板
using FusionOp = Block::DefaultFusion<OutType, OutType>;
using BlockEpilogue = Block::BlockEpilogueEmpty;
using MatmulKernel = Kernel::KernelMatmulWithoutQue<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
```

实现位置：`compiler/graph/optimize/autofuse/v35/ascendc/api_cube/matmul/mat_mul_pingpong_basic_cmct.h`

#### 4. Tiling Key 生成

Tiling Key 用于生成 Matmul 算子的 Tiling 策略，关键参数：

- `API_LEVEL`：API 级别（High Level 或 Basic Level）
- `A_TRANS`、`B_TRANS`：矩阵转置标志
- `BATCH_MODEL`：Batch 模型
- `MODEL`：计算模型（Basic、StreamK、KEqualZero）
- `FULL_LOAD`：全载模式（No Full Load、A Full Load、B Full Load、AB Full Load）
- `L0C2OUT_MODEL`：L0C 输出模式（OnTheFly、Fixpipe）

全载模式说明：
- **No Full Load**：非全载模式，Matmul 的输入分多次载入 L1。
- **A Full Load**：A 矩阵全载，A 矩阵一次性载入 L1。
- **B Full Load**：B 矩阵全载，B 矩阵一次性载入 L1。
- **AB Full Load**：AB 矩阵全载，A 和 B 矩阵都一次性载入 L1。

实现位置：`compiler/graph/optimize/autofuse/v35/ascendc/api_cube/matmul/mat_mul_tiling_key.h`

### 流程设计

#### 1. Matmul+Elemwise 融合总体流程

```
AutofuseOptimize 阶段:
├─ 1. 遍历图节点，识别 Matmul 算子
├─ 2. 检查后继算子，判断融合条件
├─ 3. CubeScheduleCaseGenerator.GenerateGeneralCase
│   ├─ 拆分图结构（插入 Load、Store、Workspace）
│   ├─ 生成通用融合 Case
│   └─ 判断是否需要图拆分
├─ 4. CubeScheduleCaseGenerator.GeneratorTask
│   ├─ ScheduleGroupGraphPartitioner.PartitionByConnectivity
│   │   ├─ 拆分为多个子图（grouped_graphs）
│   │   └─ 判断子图数量（是否 > 1）
│   ├─ 如果 grouped_graphs.size() > 1:
│   │   ├─ task.cube_type = kCommon（兜底模板）
│   │   ├─ MoveCubeGraphsToEnd（移动 Cube 子图到末尾）
│   │   ├─ GeneratorUbTask（生成 UB 融合任务）
│   │   │   ├─ ub_task.cube_type = kUBFuse
│   │   │   ├─ 检查子图结构（Load-Broadcast 模式）
│   │   │   ├─ GenNddmaNode（生成 Nddma 节点）
│   │   │   └─ 添加 ub_task 到 tasks
│   │   └─ 添加 task 到 tasks
│   └─ 如果 grouped_graphs.size() == 1:
│       └─ task.cube_type = kFixpip（单 Cube 子图）
│       └─ 添加 task 到 tasks
└─ 5. 返回融合任务列表（tasks）
```

#### 2. UB 融合任务生成流程（GeneratorUbTask）

```
GeneratorUbTask:
├─ 1. 遍历 grouped_graphs，检查子图结构
├─ 2. 对于非 Cube 子图：
│   ├─ 检查 Load 节点
│   ├─ 检查 Load 的后继节点（Broadcast 或 Cast）
│   ├─ 如果是 Load-Broadcast:
│   │   └─ GenNddmaNode（Load + Broadcast -> Nddma）
│   ├─ 如果是 Load-Cast-Broadcast:
│   │   └─ SwapCastBrcAndGenNddma（交换 Cast 和 Broadcast，生成 Nddma）
│   └─ 检查是否仍包含 Broadcast 节点（如果有，返回失败）
├─ 3. 将处理后的子图添加到 ub_task.grouped_graphs
└─ 4. 添加 ub_task 到 tasks
```

#### 3. ACT 代码生成流程

```
Codegen 阶段:
├─ 1. 根据融合任务（task）的 cube_type 选择模板
│   ├─ 如果 cube_type == kUBFuse:
│   │   ├─ 使用 KernelMatmulMixWithoutQue（混合 Kernel）
│   │   ├─ 使用 BlockEpilogueCV（UB Epilogue）
│   │   ├─ Kernel Type = ASCENDC_TPL_MIX_AIC_1_2
│   │   └─ 生成 AscendC 代码
│   └─ 如果 cube_type == kCommon:
│       ├─ 使用 KernelMatmulWithoutQue（纯 AIC Kernel）
│       ├─ 使用 BlockEpilogueEmpty（GM Epilogue）
│       ├─ Kernel Type = ASCENDC_TPL_AIC_ONLY
│       └─ 生成 AscendC 代码
├─ 2. 生成 Tiling Data 结构（MatMulV3BasicTilingData）
├─ 3. 生成 Kernel 函数（MatMulActKernelFusion）
└─ 4. 编译 AscendC 代码为算子二进制（.so）
```

### 相关文档


### 测试方案

| **测试类别** | **关键测试项** | **测试方法** | **用例类型** |
| ------------ | -------------- | ------------ | ------------ |
| **功能** | UB 融合任务生成 | 构造 Load-Broadcast 模式图，验证生成的融合任务是否为 kUBFuse | UT |
| **功能** | 兜底融合任务生成 | 构造复杂图结构（如多个 Cube 节点），验证生成的融合任务是否为 kCommon | UT |
| **功能** | Nddma 节点生成 | 验证 GenNddmaNode 是否正确融合 Load 和 Broadcast | UT |
| **功能** | Tiling 策略继承 | 验证融合算子的 Tiling 参数是否继承 Matmul 的参数 | UT |
| **性能** | CV 融合并行执行 | 使用 Profiling 工具测量 AIC 和 AIV 的并行执行时间 | ST |
| **性能** | 数据搬运开销 | 测量 UB 融合和兜底融合的数据搬运时间差异 | ST |
| **精度** | 融合算子精度 | 对比融合前后算子的输出精度，确保无精度损失 | ST |
| **兼容性** | 老 OM 在新版本执行 | 使用老版本编译的 OM（无融合算子），在新版本运行时执行 | ST |
### 验收标准

1. **功能验收**：
   - UB 融合任务生成正确（cube_type == kUBFuse，包含 Nddma 节点）
   - 兜底融合任务生成正确（cube_type == kCommon）
   - Tiling 策略继承正确（参数匹配）
   - ACT 代码生成正确（编译成功，执行无误）

2. **性能验收**：
   - UB 融合性能提升 > 15%（对比未融合版本）
   - 兜底融合性能不裂化（对比未融合版本）
   - 编译时长增加 < 5%

3. **精度验收**：
   - 融合前后算子输出精度一致（误差 < 1e-5）
   - 典型推荐模型精度无损失

### 备注
补充UB全载模板收益评估模型
![cv_fusion_diagram_v2.drawio.png](https://raw.gitcode.com/user-images/assets/8824148/5c7bc09a-2066-4457-914f-4223f69171af/cv_fusion_diagram_v2.drawio.png 'cv_fusion_diagram_v2.drawio.png')
