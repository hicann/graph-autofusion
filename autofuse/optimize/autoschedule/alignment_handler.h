/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_ALIGNMENT_HANDLE_H_
#define OPTIMIZE_ALIGNMENT_HANDLE_H_

#include "platform/platform_factory.h"

namespace optimize::autoschedule {
class AlignmentHandler {
 public:
  // 只允许load出现尾轴非连续
  static af::Status AlignVectorizedStrides(ascir::ImplGraph &impl_graph);
  // 优化节点的vectorized stride
  static af::Status ModifyVectorizedStrides(af::AscGraph &impl_graph);
};
}  // namespace optimize::autoschedule

#endif  // OPTIMIZE_ALIGNMENT_HANDLE_H_
