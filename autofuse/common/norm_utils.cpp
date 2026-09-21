/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.huawei.com
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "norm_utils.h"

namespace ascgen_utils::norm {

af::Status SetNormInfo(const af::AscNodePtr &node, const NormInfo &info) {
  GE_ASSERT_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kNormInfoAttr, info), "Set NormInfo failed, node = %s", node->GetNamePtr());
  return af::SUCCESS;
}

af::Status TryGetNormInfo(const af::AscNodePtr &node, NormInfo &info) {
  info = NormInfo{};
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    return af::SUCCESS;
  }
  info = node->GetOpDesc()->TryGetExtAttr(kNormInfoAttr, NormInfo{});
  return af::SUCCESS;
}

bool HasNormInfo(const af::AscNodePtr &node) {
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    return false;
  }
  const NormInfo stored = node->GetOpDesc()->TryGetExtAttr(kNormInfoAttr, NormInfo{});
  return stored.kind != NormInfo::Kind::kNone;
}

}  // namespace ascgen_utils::norm
