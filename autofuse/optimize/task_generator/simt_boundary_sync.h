/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.hiascend.com/software/licensedistributionexception
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef AUTOFUSE_OPTIMIZE_TASK_GENERATOR_SIMT_BOUNDARY_SYNC_H_
#define AUTOFUSE_OPTIMIZE_TASK_GENERATOR_SIMT_BOUNDARY_SYNC_H_

#include <vector>

#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "indirect_load_utils.h"

namespace optimize::task_generator {

// IndirectLoad SIMT 模板节点与普通调度节点的 view 边界适配，全部收敛在此。
// 背景：SIMT 角色节点（kSimtInlineTransform/kSimtFanoutBoundary 等）跳过主调度
// tiling（TileSplit/BlockSplit 不改写其 tensor view），但其输出会被 normal
// schedule 的 VF 子图消费；边界 Load 拷贝的生产者 view 必须与消费者（被完整
// 调度的普通节点）一致，否则 ValidateInputTensorLoopAxis /
// BufQueAllocator 报 view 不一致。此函数在调度流水线（含 block split）全部
// 完成后调用，一次性将 SIMT 节点 tensor view 对齐到等价状态。
// tiled_axes_list: 调度期间实际执行过的 (outer, inner) split 轴对（ub tiling +
// block tiling），由调用方收集；不改写 sched 轴，避免影响 scalar evaluator
// 等依赖 "SIMT 节点 sched 轴未被 tiling 改写" 假设的路径。
af::Status SyncSimtBoundaryViews(af::AscGraph &graph,
                                 const std::vector<std::pair<af::AxisPtr, af::AxisPtr>> &tiled_axes_list);

}  // namespace optimize::task_generator

#endif  // AUTOFUSE_OPTIMIZE_TASK_GENERATOR_SIMT_BOUNDARY_SYNC_H_
