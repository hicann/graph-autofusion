/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_GRAPH_PASS_REDUNDANT_BROADCAST_ELIMINATE_PASS_H
#define OPTIMIZE_GRAPH_PASS_REDUNDANT_BROADCAST_ELIMINATE_PASS_H

#include "optimize/graph_pass/base_graph_pass.h"

namespace optimize {
// 剔除冗余 Broadcast 节点：当 Broadcast 输出与前驱节点输出各轴 repeat 一致时，
// Broadcast 未产生任何扩维，属冗余节点，直接删除并将前驱节点与后继节点相连。
class RedundantBroadcastEliminatePass final : public BaseGraphPass {
 public:
  RedundantBroadcastEliminatePass() = default;
  Status RunPass(af::AscGraph &graph) override;
  ~RedundantBroadcastEliminatePass() override = default;
};
}  // namespace optimize

#endif  // OPTIMIZE_GRAPH_PASS_REDUNDANT_BROADCAST_ELIMINATE_PASS_H
