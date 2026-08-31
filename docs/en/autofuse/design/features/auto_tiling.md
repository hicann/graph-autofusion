# Auto Tiling

During the Schedule phase, the system generates multiple optimized split graphs based on the AscIR graph input from the frontend. Auto Tiling needs to evaluate the optimal splitting scheme and corresponding Kernel performance of these graphs, thereby selecting the optimal template.

The Tiling strategy determines the number of cores to be used when the Kernel executes, how much data is moved from GM to UB each time, how many loops are needed in UB, and how many times data is moved from UB to GM each time. These factors collectively affect the execution performance of the Kernel. The goal of Auto Tiling is to find the Tiling strategy that achieves the best execution performance for the Kernel.

## Core Process

<div style="text-align: center;">
<img src="../../figures/autotiling_core_flow.png" alt="Auto Tiling Core Process" style="width: 60%; max-width: 800px;">
</div>

The execution logic of the Kernel is expressed in AscIR. Auto Tiling extracts key information based on the expression of the AscIR graph, including:

### 1. LocalBuffer Occupancy Modeling

Auto Tiling solving needs to ensure that the occupancy of each level of LocalBuffer is within the range allowed by the hardware. For example, the sum of the sizes of TQue/TBuf and temporary Buf applied by the Kernel cannot exceed the UB size limit of the hardware. AscIR expresses whether the Location of each Tensor is on GM/UB and the reuse relationship between Tensors. Auto Tiling symbolically expresses the constraints of each level of LocalBuffer based on this information.

### 2. Time Consumption Formula Modeling

Auto Tiling performs performance modeling for each API, expresses the performance of these APIs on each pipeline in a symbolic form, determines the number of calls to the API based on the loop axis expressed by AscIR, and thereby derives the total time consumption of all APIs of this AscIR on each pipeline. It is assumed here that the execution of the bottleneck pipeline can better cover the execution of non-bottleneck pipelines, so the total time consumption of task execution is mainly reflected in the time of the bottleneck pipeline, as shown in the figure below:

The figure below shows three pipe pipelines, with AIV_MTE2 being the bottleneck pipeline. The execution of AIV_MTE2 will cover the execution of AIV_MTE3 and AIV_VEC, and the total time consumption of task execution is reflected in the execution time of AIV_MTE2.

<div style="text-align: center;">
<img src="../../figures/autotiling_pipe_pipeline.png" alt="Pipe Pipeline Diagram" style="width: 45%; max-width: 800px;">
</div>

### 3. Solver Solving Tiling

Auto Tiling takes the storage occupancy during operator implementation not exceeding the physical storage size of each level of NPU as the constraint condition, and minimizes the time consumption of the bottleneck execution unit as the optimization objective, and solves by modeling the data optimization model. For example, through a heuristic solving method, starting from the initial solution, within the feasible domain that satisfies the constraint conditions, search the solution space that meets the memory requirements according to the gradient descent direction modeled by the performance formula, until the value modeled by the formula reaches the minimum, then exit the optimization process and return the optimal Tiling.

## Key Technologies

### Performance Formula Modeling

- **Objective**: The simulation accuracy of the performance formula determines the accuracy of template selection and the accuracy of heuristic solving, so it is necessary to perform relatively accurate performance modeling for APIs.
- **Implementation**:
  - For stable basic APIs of Ascend C, the Auto Tiling module collects performance data according to different inputs, obtains the relationship between performance and input, establishes a performance model, and expresses it through symbolic technology.
  - For volatile APIs of Ascend C, the Auto Tiling module calculates the call parameters and call times based on the API calling logic, generates a complete performance model for the API, thereby obtaining a relatively accurate performance model.

### Axis Sorting Solver

- **Objective**: The solving method based on performance formula gradient descent highly depends on the accurate modeling of the performance formula, and the Tiling solving time is uncertain, which is prone to host bound problems. Moreover, the API itself has specific preferences for axis splitting, so an axis sorting solver is designed as an alternative to iterative solving.
- **Implementation**:
  1. First, the priority of the splitting axis needs to be determined based on the API, the order is as follows:
     1. The parent axis has higher priority than the child axis.
     2. Reduction axes have higher priority than non-reduction axes.
     3. Broadcast axes have higher priority than non-broadcast axes.
     4. Non-innermost axes have higher priority than innermost axes.
  2. Secondly, split into two parts, including:
     - **Intra-core Tiling**: Traverse in reverse order of axis sorting, preferentially adjust the variable to the maximum value, determine whether it meets the hardware constraint conditions. If not, adjust the variable through the dichotomy until it meets the hardware constraint conditions, then adjust the next intra-core Tiling variable until all variables meet the hardware constraint conditions. For example, s1tt2, s1tt, s1t, s2t are Tiling-related axes, the splitting process is as follows:

       Preferentially traverse s2t, adjust to the maximum value of 1024, which meets the hardware constraint conditions; then traverse s1t, adjust to the maximum value of 256, which meets the hardware constraint conditions; then adjust the next variable s1tt > s1tt2 in sequence, until all variables meet the hardware constraint conditions.

       <div style="text-align: center;">
       <img src="../../figures/autotiling_intra_core_tiling.png" alt="Intra-core Tiling Diagram" style="width: 40%; max-width: 800px;">
       </div>
     - **Multi-core Tiling**: Identify variables related to multi-core, traverse these variables in descending order, and find records with larger core occupancy. If it exceeds the number of physical cores (taking Atlas A2 training series products / Atlas A2 inference series products as an example, the number of NPU cores is 48), return.

       As shown in the figure below, bngs1T is the multi-core splitting axis, and the splitting process is as follows. The selection strategy is: when the core occupancy is different, preferentially select the record with larger core occupancy. According to the above strategy, the finally selected core occupancy is 47.

       <div style="text-align: center;">
       <img src="../../figures/autotiling_multi_core_tiling.png" alt="Multi-core Tiling Diagram" style="width: 60%; max-width: 800px;">
       </div>
