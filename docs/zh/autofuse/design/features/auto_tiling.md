# Auto Tiling

在 Schedule 阶段，系统会根据前端输入的 AscIR 图生成多种优化后的切分图，Auto Tiling 需要评估这些图的最优切分方案及其对应的 Kernel 性能，从而选出最优模板。

Tiling 策略决定了 Kernel 执行时需要使用的核的数量，每次从 GM 中搬运多少数据到 UB 内，在 UB 内需要循环多少次，每次在 UB 内搬运多少次到 GM 中等，这些因素共同影响着 Kernel 的执行性能。Auto Tiling 的目标是找到能够实现 Kernel 最佳执行性能的 Tiling 策略。

## 核心流程

<div style="text-align: center;">
<img src="../../figures/autotiling_core_flow.png" alt="Auto Tiling 核心流程" style="width: 60%; max-width: 800px;">
</div>

Kernel 的执行逻辑会在 AscIR 中表达，Auto Tiling 会根据 AscIR 图的表达提取关键信息，包括：

### 1. LocalBuffer 占用建模

Auto Tiling 求解需确保每一级 LocalBuffer 的占用在硬件允许的范围内，比如，Kernel 申请的 TQue/TBuf 及临时 Buf 申请大小之和不能超过硬件的 UB 大小限制，AscIR 会表达出每个 Tensor 的 Location 是在 GM/UB 上，Tensor 间的复用关系，自动 Tiling 根据这些信息将各级 LocalBuffer 的约束进行符号化表达。

### 2. 耗时公式建模

Auto Tiling 会对各个 API 进行性能建模，通过符号化的形式表达这些 API 在各个流水上的性能，根据 AscIR 表达的循环轴来确定表达 API 的调用次数，从而推导出该 AscIR 的所有 API 在各个流水上的总耗时，此处假设瓶颈流水线的执行可以较好地掩盖非瓶颈流水线的执行，因此任务执行的总耗时主要体现在瓶颈流水线的时间上，如下图：

下图表达了三个 pipe 流水，瓶颈流水为 AIV_MTE2，AIV_MTE2 在执行时会掩盖 AIV_MTE3 和 AIV_VEC 的执行，任务执行的总耗时体现在 AIV_MTE2 的执行耗时上。

<div style="text-align: center;">
<img src="../../figures/autotiling_pipe_pipeline.png" alt="pipe 流水示意图" style="width: 45%; max-width: 800px;">
</div>

### 3. 求解器求解 Tiling

Auto Tiling 以算子实现过程中的存储占用不超过 NPU 各级物理存储大小为约束条件，以最小化瓶颈执行单元的耗时为优化目标，通过建模数据优化模型进行求解。例如，通过启发式求解方法，从初始解开始，在满足约束条件的可行域内，根据性能公式建模的梯度下降方向搜索满足内存要求的解空间，直到公式建模的值达到最小，从而退出寻优流程，返回最优 Tiling。

## 关键技术

### 性能公式建模

- **目标**：性能公式的仿真精度决定了模板选择的准确性及启发式求解的准确性，因此需要对 API 进行较为精确的性能建模。
- **实现**：
  - 针对 AscendC 稳定的基础 API，自动 Tiling 模块会根据不同的输入采集性能数据，得到性能与输入的关系，并建立性能模型，通过符号化的技术表达出来。
  - 针对 AscendC 易变的 API，自动 Tiling 模块会基于 API 调用的逻辑，计算其调用入参和调用次数，生成该 API 的完整性能模型，从而得到相对准确的性能建模。

### 轴排序求解器

- **目标**：基于性能公式梯度下降的求解方式高度依赖性能公式的精确建模，并且 Tiling 求解时间存在不确定性，容易出现 host bound 问题，而 API 本身对轴的切分有特定的偏好，因此设计了一种轴排序求解器作为迭代求解的替代方案。
- **实现**：
  1. 首先需要基于 API 来确定切分轴的优先级，顺序如下：
     1. 父轴优先级高于子轴。
     2. 规约化类轴高于非规约化类轴。
     3. 广播轴高于非广播轴。
     4. 非最内轴高于最内轴。
  2. 其次再分成两个部分切分，包括：
     - **核内 Tiling**：按照轴排序的逆序依次遍历，优先将变量调整至最大值，判断是否符合硬件约束条件，若不满足，则通过二分法调整该变量，直到符合硬件约束条件为止，随后调整下一个核内 Tiling 变量，直至所有的变量均满足硬件约束条件，比如 s1tt2、s1tt、s1t、s2t 是 Tiling 相关轴，其切分流程如下：

       优先遍历 s2t，调至最大值 1024，符合硬件约束条件；然后遍历 s1t，调至最大值 256，符合硬件约束条件；然后依次调整下个变量 s1tt > s1tt2，直到所有的变量均满足硬件约束条件。

       <div style="text-align: center;">
       <img src="../../figures/autotiling_intra_core_tiling.png" alt="核内 Tiling 示意图" style="width: 40%; max-width: 800px;">
       </div>

     - **多核 Tiling**：识别与多核相关的变量，按从大到小的顺序遍历这些变量，找到更大核数占用的记录，若超出物理核数（以 Atlas A2 训练系列产品 / Atlas A2 推理系列产品为例，NPU 核数为 48）则返回。

       如下图所示，bngs1T 是多核切分轴，其切分流程如下。选择策略为：核数占用不同时，优先选择占用核数更大的记录，根据上述策略，最终选定的占用核数为 47。

       <div style="text-align: center;">
       <img src="../../figures/autotiling_multi_core_tiling.png" alt="多核 Tiling 示意图" style="width: 60%; max-width: 800px;">
       </div>
