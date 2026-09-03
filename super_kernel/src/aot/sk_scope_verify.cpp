/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unordered_map>
#include <vector>

#include "sk_scope_verify.h"
#include "sk_graph.h"
#include "sk_lock_detector.h"
#include "sk_log.h"
#include "sk_options_manager.h"
#include "sk_verify_graph.h"
#include "sk_verify_scope_split.h"

namespace {

class ScopeVerifyRunner {
 public:
  ScopeVerifyRunner(const aclskScopeVerifyGraphInfo *verifyGraph, size_t maxSplitResultCount,
                    aclskScopeVerifySplitResult *splitResults, size_t *realSplitResultCount)
      : verifyGraph_(verifyGraph),
        maxSplitResultCount_(maxSplitResultCount),
        splitResults_(splitResults),
        realSplitResultCount_(realSplitResultCount) {}

  aclError Run() {
    // realSplitResultCount is always reset before running verification, so callers can rely on it after failure.
    if (realSplitResultCount_ == nullptr) {
      SK_DLOGE("aclskScopeVerify realSplitResultCount is null");
      return ACL_ERROR_INVALID_PARAM;
    }
    *realSplitResultCount_ = 0;
    if (maxSplitResultCount_ > 0 && splitResults_ == nullptr) {
      SK_DLOGE("aclskScopeVerify splitResults is null, maxSplitResultCount=%zu", maxSplitResultCount_);
      return ACL_ERROR_INVALID_PARAM;
    }
    splitRecords_.clear();

    // Adapt GE scope verify graph to the internal SuperKernelGraph and initial scope groups.
    ScopeVerifyGraphBuilder builder;
    aclError ret = builder.Build(verifyGraph_, graph_, scopes_, verifyNodesById_);
    if (ret != ACL_SUCCESS) {
      SK_DLOGE("aclskScopeVerify ScopeVerifyGraphBuilder failed, ret=%d", ret);
      return ret;
    }

    // Match the original SuperKernel scope refinement order: run full-core ScheMode first,
    // then deadlock on refined scopes.
    if (!RunScopeVerifyScheModeSplitPass(graph_, scopes_, splitRecords_)) {
      SK_DLOGE("aclskScopeVerify ScheModeKernelSplitPass failed");
      return ACL_ERROR_FAILURE;
    }

    ret = LockDetector::GetDeviceCores();
    if (ret != ACL_SUCCESS) {
      SK_DLOGE("aclskScopeVerify GetDeviceCores failed, ret=%d", ret);
      return ret;
    }

    SuperKernelOptionsManager opts;
    if (!RunScopeVerifyDeadlockRefinePass(graph_, opts, scopes_, splitRecords_)) {
      SK_DLOGE("aclskScopeVerify DeadlockRefinePass failed");
      return ACL_ERROR_FAILURE;
    }

    // Map internal split records back to caller-owned aclskScopeVerifyNodeInfo pointers.
    return FillScopeVerifySplitResults();
  }

 private:
  aclError FillScopeVerifySplitResults() {
    *realSplitResultCount_ = splitRecords_.size();
    if (splitRecords_.size() > maxSplitResultCount_) {
      SK_DLOGE("aclskScopeVerify splitResults buffer is too small, maxSplitResultCount=%zu, realSplitResultCount=%zu",
               maxSplitResultCount_, splitRecords_.size());
      return ACL_ERROR_INVALID_PARAM;
    }
    for (size_t i = 0; i < splitRecords_.size(); ++i) {
      auto it = verifyNodesById_.find(splitRecords_[i].nodeId);
      if (it == verifyNodesById_.end()) {
        SK_DLOGE("aclskScopeVerify split node %lu is not found in verify node map", splitRecords_[i].nodeId);
        return ACL_ERROR_FAILURE;
      }
      splitResults_[i].splitNode = it->second;
      splitResults_[i].splitType = splitRecords_[i].splitType;
      splitResults_[i].splitReason = splitRecords_[i].splitReason;
      splitResults_[i].extendType = 0;
      splitResults_[i].extendInfo = nullptr;
    }
    return ACL_SUCCESS;
  }

  const aclskScopeVerifyGraphInfo *verifyGraph_ = nullptr;
  size_t maxSplitResultCount_ = 0;
  aclskScopeVerifySplitResult *splitResults_ = nullptr;
  size_t *realSplitResultCount_ = nullptr;
  SuperKernelGraph graph_;
  std::vector<SuperKernelScopeInfo> scopes_;
  std::unordered_map<uint64_t, aclskScopeVerifyNodeInfo *> verifyNodesById_;
  std::vector<ScopeVerifySplitRecord> splitRecords_;
};

}  // namespace

aclError RunScopeVerify(const aclskScopeVerifyGraphInfo *verifyGraph, size_t maxSplitResultCount,
                        aclskScopeVerifySplitResult *splitResults, size_t *realSplitResultCount) {
  ScopeVerifyRunner runner(verifyGraph, maxSplitResultCount, splitResults, realSplitResultCount);
  return runner.Run();
}
