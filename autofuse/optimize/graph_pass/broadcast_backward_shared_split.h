/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0.
 */
#ifndef OPTIMIZE_GRAPH_PASS_BROADCAST_BACKWARD_SHARED_SPLIT_H
#define OPTIMIZE_GRAPH_PASS_BROADCAST_BACKWARD_SHARED_SPLIT_H

#include "base_graph_pass.h"

namespace optimize {
namespace broadcast_backward_shared_split {

Status SplitSharedBroadcastBranches(af::AscGraph &graph);
Status SplitSharedBroadcastConsumers(af::AscGraph &graph);

}  // namespace broadcast_backward_shared_split
}  // namespace optimize

#endif  // OPTIMIZE_GRAPH_PASS_BROADCAST_BACKWARD_SHARED_SPLIT_H
