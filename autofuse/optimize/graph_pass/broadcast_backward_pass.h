/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License in the root of the software repository for the full text of the License.
 * THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */
#ifndef OPTIMIZE_GRAPH_PASS_BROADCAST_BACKWARD_PASS_H
#define OPTIMIZE_GRAPH_PASS_BROADCAST_BACKWARD_PASS_H

#include "base_graph_pass.h"

namespace optimize {
class BroadcastBackwardPass final : public BaseGraphPass {
 public:
  BroadcastBackwardPass() = default;
  ~BroadcastBackwardPass() override = default;
  Status RunPass(af::AscGraph &graph) override;
};
}  // namespace optimize

#endif  // OPTIMIZE_GRAPH_PASS_BROADCAST_BACKWARD_PASS_H
