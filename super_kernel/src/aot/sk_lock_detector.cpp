/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sk_lock_detector.cpp
 * \brief
 */

#include "sk_lock_detector.h"
#include "sk_log.h"
#include "sk_options_manager.h"
#include "sk_common.h"
#include "super_kernel.h"

int64_t LockDetector::deviceRealCubeNum = 0;
int64_t LockDetector::deviceRealVecNum = 0;

void LockDetector::Init(SuperKernelGraph &graph) {
  SK_LOGD("[lock detector] LockDetector::Init: Initializing lock detector");
  nodes.clear();
  depOpCubeNum = 0;
  depOpVecNum = 0;
  superKernelCubeNum = 0;
  superKernelVecNum = 0;
  skCoreInfoChanged_ = false;
  nodeNum = 0;
  kernelNodeNum = 0;
  skStreamIds.clear();
  graph_ = &graph;

  if (deviceRealCubeNum == 0 && deviceRealVecNum == 0) {
    GetDeviceCores();
  } else {
    SK_LOGD("[lock detector] LockDetector::Init: Using cached device cores: cube=%lu, vec=%lu", deviceRealCubeNum,
            deviceRealVecNum);
  }
}

aclError LockDetector::GetDeviceCores() {
  if (deviceRealCubeNum != 0 && deviceRealVecNum != 0) {
    return ACL_SUCCESS;
  }

  aclError ret = GetDeviceCoreNums(deviceRealCubeNum, deviceRealVecNum);
  if (ret != ACL_SUCCESS) {
    SK_LOGE("[lock detector] GetDeviceCores failed, ret=%d", ret);
    return ret;
  }

  SK_LOGI("[lock detector] GetDeviceCores success, cube=%u, vec=%u", deviceRealCubeNum, deviceRealVecNum);
  return ACL_SUCCESS;
}

std::pair<uint64_t, uint64_t> LockDetector::GetAvailableCores(bool isSuperKernel) const {
  if (isSuperKernel) {
    return {deviceRealCubeNum - depOpCubeNum, deviceRealVecNum - depOpVecNum};
  } else {
    return {deviceRealCubeNum - superKernelCubeNum, deviceRealVecNum - superKernelVecNum};
  }
}

bool LockDetector::IsInSKStream(const SuperKernelBaseNode &node) {
  return std::find(skStreamIds.begin(), skStreamIds.end(), node.GetStreamIdxInGraph()) != skStreamIds.end();
}

bool LockDetector::HasDeadlock(SuperKernelBaseNode *curNode) {
  curNode->SetVisited(true);
  tempVisitedNodes.emplace_back(curNode->GetNodeId());
  if (curNode->GetPreNodeId() == INVALID_TASK_ID) {
    return false;
  }

  const uint64_t preNodeId = curNode->GetPreNodeId();
  SuperKernelBaseNode *preNode = graph_->GetNodeById(preNodeId);
  if (preNode == nullptr) {
    SK_LOGE("[lock detector] HasDeadlock: preNode %lu not found for curNode %lu", preNodeId, curNode->GetNodeId());
    return false;
  }
  // if prenode is visited, means that wait node has already entered the detection process before
  // when finish in this condition, there are two situations:
  //      Case 1: one branch is checked forward
  //      Case 2: two branches (from wait node in graph) are checked forward, which ensure both branches return false in
  //      wait
  if (preNode->IsVisited()) {
    return false;
  }

  // when prenode has been fused in sk
  // if streams of sk with prenode intersects with the stream of current sk, means prenode has been executed
  if (HasIntersection(preNode->GetScopeStreamIds(), skStreamIds)) {
    return false;
  }

  bool hasDeadlock = true;
  switch (preNode->GetNodeType()) {
    case SkNodeType::NODE_KERNEL:
      hasDeadlock = preNode->IsScopeNode() ? HasDeadlock(preNode) : CheckKernelNodeDeadlock(preNode);
      break;
    case SkNodeType::NODE_WAIT:
      hasDeadlock = CheckWaitNodeDeadlock(preNode);
      break;
    case SkNodeType::NODE_NOTIFY:
      hasDeadlock = CheckNotifyNodeDeadlock(preNode);
      break;
    case SkNodeType::NODE_DEFAULT:
      hasDeadlock = HasDeadlock(preNode);
      break;
    case SkNodeType::NODE_RESET:
      hasDeadlock = HasDeadlock(preNode);
      break;
    default:
      SK_LOGD("nodeId: %u, unsupported node type %u in HasDeadlock", preNode->GetNodeId(), preNode->GetNodeType());
      deadlockReason_ = DeadlockFailReason::NO_SUPPORT_NODE;
      break;
  }

  return hasDeadlock;
}

bool LockDetector::CheckKernelNodeDeadlock(SuperKernelBaseNode *preNode) {
  if (!HasEnoughCores(preNode, false)) {
    SK_LOGI("Not enough cores for kernel node, nodeId=%lu, requiredCube=%u, requiredVec=%u", preNode->GetNodeId(),
            preNode->GetCubeNum(), preNode->GetVecNum());
    deadlockReason_ = DeadlockFailReason::KERNEL_INSUFFICIENT_CORES;
    return true;
  }
  if (HasDeadlock(preNode)) {
    SK_LOGI("Deadlock detected in kernel node, nodeId=%lu", preNode->GetNodeId());
    return true;
  }
  return false;
}

bool LockDetector::CheckWaitNodeDeadlock(SuperKernelBaseNode *preNode) {
  const uint64_t notifyId = preNode->GetCorrespondingNotifyNodeId();
  // Case 1: notify node not in modelRI
  if (notifyId == INVALID_TASK_ID) {
    SK_LOGI("Deadlock detected in wait node, waitNodeId=%lu, notifyNodeId=%lu is not in graph", preNode->GetNodeId(),
            notifyId);
    deadlockReason_ = DeadlockFailReason::NOTIFY_NOT_IN_GRAPH;
    return true;
  }
  SuperKernelBaseNode *notifyNode = graph_->GetNodeById(notifyId);
  // abnormal case, notify node not found
  if (notifyNode == nullptr) {
    SK_LOGE("[lock detector] CheckWaitNodeDeadlock: notifyNode %lu not found for waitNode %lu", notifyId,
            preNode->GetNodeId());
    deadlockReason_ = DeadlockFailReason::NOTIFY_INVALID;
    return true;
  }
  // Case 2: notify node is after sk range,
  if (IsAfterSKRange(*notifyNode)) {
    deadlockReason_ = DeadlockFailReason::NOTIFY_AFTER_SK_RANGE;
    return true;
  }
  // Case 3: check node before wait node in current stream
  if (HasDeadlock(preNode)) {
    SK_LOGI("Deadlock detected in wait node pre-path, waitNodeId=%lu", preNode->GetNodeId());
    return true;
  }
  // Case 4: check node before notify node (notify node not in sk stream) in different stream
  if (notifyNode->GetStreamIdxInGraph() != preNode->GetStreamIdxInGraph() && !IsInSKStream(*notifyNode)) {
    if (HasDeadlock(notifyNode)) {
      SK_LOGI("Deadlock detected in wait node cross-stream path, waitNodeId=%lu, notifyNodeId=%lu",
              preNode->GetNodeId(), notifyNode->GetNodeId());
      return true;
    }
  }
  return false;
}

bool LockDetector::CheckNotifyNodeDeadlock(SuperKernelBaseNode *preNode) {
  const uint32_t cubeNum = preNode->GetCubeNum();
  const uint32_t vecNum = preNode->GetVecNum();
  if ((cubeNum > 0 || vecNum > 0) && !HasEnoughCores(preNode, false)) {
    SK_LOGI("Not enough cores for notify node, nodeId=%lu, requiredCube=%u, requiredVec=%u", preNode->GetNodeId(),
            cubeNum, vecNum);
    deadlockReason_ = DeadlockFailReason::NOTIFY_INSUFFICIENT_CORES;
    return true;
  }
  std::vector<uint64_t> waitIds = preNode->GetCorrespondingWaitNodeIds();
  for (uint64_t waitId : waitIds) {
    SuperKernelBaseNode *waitNode = graph_->GetNodeById(waitId);
    // if exist wait node before sk, means notify has been executed, no need to check
    if (IsBeforeSKRange(*waitNode)) {
      break;
    } else {
      if (HasDeadlock(preNode)) {
        SK_LOGI("Deadlock detected in notify node path, notifyNodeId=%lu", preNode->GetNodeId());
        return true;
      }
    }
  }
  return false;
}

bool LockDetector::HasEnoughCores(const SuperKernelBaseNode *curNode, bool isSuperKernel) {
  const uint32_t curNodeCubeNum = curNode->GetCubeNum();
  const uint32_t curNodeVecNum = curNode->GetVecNum();

  if (isSuperKernel) {
    if (!skCoreInfoChanged_) {
      SK_LOGD("[lock detector] Candidate SK core requirement is unchanged for node %s, skip resource recheck",
              curNode->Format().c_str());
      return true;
    }
    // depOp cores retain the resource requirements discovered from preceding Wait/Notify paths.
    const auto availableCores = GetAvailableCores(true);
    if (superKernelCubeNum <= availableCores.first && superKernelVecNum <= availableCores.second) {
      SK_LOGD("[lock detector] Candidate SK for node %s fits available cores (cube %u<=%u, vec %u<=%u)",
              curNode->Format().c_str(), superKernelCubeNum, availableCores.first, superKernelVecNum,
              availableCores.second);
      return true;
    }
    SK_LOGD("[lock detector] Candidate SK for node %s conflicts with dependent ops (cube %u>%u or vec %u>%u)",
            curNode->Format().c_str(), superKernelCubeNum, availableCores.first, superKernelVecNum,
            availableCores.second);
    return false;
  } else {
    if (curNodeCubeNum <= depOpCubeNum && curNodeVecNum <= depOpVecNum) {
      SK_LOGD("[lock detector] Node %s: within current depOp limits (cube %u<=%u, vec %u<=%u)",
              curNode->Format().c_str(), curNodeCubeNum, depOpCubeNum, curNodeVecNum, depOpVecNum);
      return true;
    }

    std::pair<uint32_t, uint32_t> availableCores = GetAvailableCores(false);
    if (curNodeCubeNum <= availableCores.first && curNodeVecNum <= availableCores.second) {
      depOpCubeNum = std::max(depOpCubeNum, curNodeCubeNum);
      depOpVecNum = std::max(depOpVecNum, curNodeVecNum);
      SK_LOGD("[lock detector] Node %s: allocated from device (cube %u, vec %u), new depOp limits: cube %u, vec %u",
              curNode->Format().c_str(), curNodeCubeNum, curNodeVecNum, depOpCubeNum, depOpVecNum);
      return true;
    }
    SK_LOGD(
        "[lock detector] Node %s: insufficient cores for depOp (required: cube %u, vec %u, available: cube %u, "
        "vec %u)",
        curNode->Format().c_str(), curNodeCubeNum, curNodeVecNum, availableCores.first, availableCores.second);
    return false;
  }
}

void LockDetector::RollbackVisitedState(std::vector<uint64_t> &visitedNodes) {
  for (auto nodeId : visitedNodes) {
    SuperKernelBaseNode *node = graph_->GetNodeById(nodeId);
    if (node != nullptr) {
      node->SetVisited(false);
    }
  }
  visitedNodes.clear();
}

void LockDetector::Reset() {
  SK_LOGD("[lock detector] LockDetector::Reset: Resetting lock detector state");
  SK_LOGD(
      "[lock detector] Previous state: depOpCubeNum=%u, depOpVecNum=%u, superKernelCubeNum=%u, superKernelVecNum=%u, "
      "nodeNum=%u, kernelNodeNum=%u",
      depOpCubeNum, depOpVecNum, superKernelCubeNum, superKernelVecNum, nodeNum, kernelNodeNum);

  const size_t nodesCount = nodes.size();
  const size_t tempVisitedNodesCount = tempVisitedNodes.size();
  const size_t streamIdsCount = skStreamIds.size();
  const size_t streamRangesCount = skRangeInStream.size();

  depOpCubeNum = 0;
  depOpVecNum = 0;
  superKernelCubeNum = 0;
  superKernelVecNum = 0;
  skCoreInfoChanged_ = false;
  nodeNum = 0;
  kernelNodeNum = 0;
  deadlockReason_ = DeadlockFailReason::NOT_FIND_DEADLOCK;

  RollbackVisitedState(nodes);
  RollbackVisitedState(tempVisitedNodes);
  skStreamIds.clear();
  skRangeInStream.clear();
  SK_LOGD(
      "[lock detector] Reset: Completed, cleared %zu nodes, %zu tempVisitedNodes, %zu stream IDs, %zu stream ranges",
      nodesCount, tempVisitedNodesCount, streamIdsCount, streamRangesCount);
}

void LockDetector::UpdateSKRangeInStream(const SuperKernelBaseNode &curNode) {
  uint64_t nodeId = curNode.GetNodeIdxInStream();
  uint32_t streamId = curNode.GetStreamIdxInGraph();
  if (skRangeInStream.find(streamId) == skRangeInStream.end()) {
    skRangeInStream[streamId].first = nodeId;
    skRangeInStream[streamId].second = nodeId;
  } else {
    skRangeInStream[streamId].first = std::min(skRangeInStream[streamId].first, nodeId);
    skRangeInStream[streamId].second = std::max(skRangeInStream[streamId].second, nodeId);
  }
}

bool LockDetector::IsBeforeSKRange(const SuperKernelBaseNode &curNode) {
  uint64_t nodeId = curNode.GetNodeIdxInStream();
  uint32_t streamId = curNode.GetStreamIdxInGraph();
  if (skRangeInStream.find(streamId) == skRangeInStream.end()) {
    return false;
  }
  return nodeId < skRangeInStream[streamId].first;
}

bool LockDetector::IsAfterSKRange(const SuperKernelBaseNode &curNode) {
  uint64_t nodeId = curNode.GetNodeIdxInStream();
  uint32_t streamId = curNode.GetStreamIdxInGraph();
  if (skRangeInStream.find(streamId) == skRangeInStream.end()) {
    return false;
  }
  return nodeId > skRangeInStream[streamId].second;
}

bool LockDetector::HasIntersection(const std::unordered_set<uint32_t> &lhsStreams,
                                   const std::unordered_set<uint32_t> &rhsStreams) {
  for (const auto &streamId : lhsStreams) {
    if (rhsStreams.count(streamId) > 0) {
      return true;
    }
  }
  return false;
}

bool LockDetector::GetWaitNodeFusibleStatus(SuperKernelBaseNode &curNode) {
  uint64_t notifyId = curNode.GetCorrespondingNotifyNodeId();
  // Case 1: notify node not in modelRI
  if (notifyId == INVALID_TASK_ID) {
    SK_LOGD("[lock detector] Wait node %s: notify node %lu not found in graph", curNode.Format().c_str(), notifyId);
    deadlockReason_ = DeadlockFailReason::NOTIFY_NOT_IN_GRAPH;
    return false;
  }
  SuperKernelBaseNode *notifyNode = graph_->GetNodeById(notifyId);
  // abnormal case: notify node not found
  if (notifyNode == nullptr) {
    SK_LOGE("[lock detector] Wait node %s: notify node %lu not found in graph", curNode.Format().c_str(), notifyId);
    deadlockReason_ = DeadlockFailReason::NOTIFY_INVALID;
    return false;
  }
  // Case 2: first wait
  if (nodeNum == 0) {
    SK_LOGD("[lock detector] Wait node %s: first node in scope, cannot fuse", curNode.Format().c_str());
    deadlockReason_ = DeadlockFailReason::FIRST_WAIT;
    return false;
  }
  // Case 3: notify node is in the same SK stream
  if (IsInSKStream(*notifyNode)) {
    return CheckNotifyInSKStream(curNode, *notifyNode);
  }

  // Case 4: notify node is in other sk, which cover multi stream. these stream intersect with the stream of wait node
  //         Note: will not receive wait node which notify after it (in scope)
  if (HasIntersection(skStreamIds, notifyNode->GetScopeStreamIds())) {
    return true;
  }

  // Case 5: notify node has core resource requirement
  if (notifyNode->GetCubeNum() > 0 || notifyNode->GetVecNum() > 0) {
    bool canFuse = HasEnoughCores(notifyNode, false);
    SK_LOGD("[lock detector] Wait node %s: notify %s has cores, canFuse=%d", curNode.Format().c_str(),
            notifyNode->Format().c_str(), canFuse);
    if (canFuse) {
      tempVisitedNodes.emplace_back(notifyNode->GetNodeId());
    } else {
      deadlockReason_ = DeadlockFailReason::NOTIFY_INSUFFICIENT_CORES;
      return canFuse;
    }
  }

  // Case 6: notify node is in different stream, check for deadlock
  bool hasDeadlock = HasDeadlock(notifyNode);
  SK_LOGD("[lock detector] Wait node %s: notify %s HasDeadlock=%d", curNode.Format().c_str(),
          notifyNode->Format().c_str(), hasDeadlock);
  return !hasDeadlock;
}

bool LockDetector::CheckNotifyInSKStream(SuperKernelBaseNode &curNode, SuperKernelBaseNode &notifyNode) {
  if (IsAfterSKRange(notifyNode)) {
    SK_LOGE("[lock detector] Wait node %s: notify %s is after SK range, cannot fuse", curNode.Format().c_str(),
            notifyNode.Format().c_str());
    deadlockReason_ = DeadlockFailReason::NOTIFY_AFTER_SK_RANGE;
    return false;
  }
  SK_LOGD("[lock detector] Wait node %s: notify %s is before SK range, can fuse", curNode.Format().c_str(),
          notifyNode.Format().c_str());
  return true;
}

bool LockDetector::ShouldBypassValueWaitDeadlock(const SuperKernelBaseNode &curNode) const {
  if (curNode.GetNodeType() != SkNodeType::NODE_WAIT) {
    return false;
  }

  const auto &syncInfos = curNode.GetNodeInfos().syncInfos;
  const auto *option = opts_ == nullptr ? nullptr
                                        : static_cast<const AggressiveOptStrategiesOption *>(
                                              opts_->GetOption(aclskOptionType::AGGRESSIVE_OPT_STRATEGIES));
  const uint32_t valueBreakerBypass =
      option == nullptr ? ACLSK_VALUE_BREAKER_BYPASS_NONE : option->GetValue().valueBreakerBypass;
  // Only 0b10 keeps unpaired value waits alive through deadlock refine.
  return syncInfos.addrValue != nullptr && curNode.GetCorrespondingNotifyNodeId() == INVALID_TASK_ID &&
         (valueBreakerBypass & ACLSK_VALUE_BREAKER_BYPASS_UNPAIRED_WAIT) != 0;
}

bool LockDetector::GetFusibleStatus(SuperKernelBaseNode &curNode) {
  if (curNode.GetNodeType() == SkNodeType::NODE_NOTIFY) {
    if (curNode.GetCubeNum() == 0 && curNode.GetVecNum() == 0) {
      SK_LOGD("[lock detector] Notify node %s: not needed core resource, can fuse", curNode.Format().c_str());
      return true;
    } else {
      SK_LOGE("[lock detector] Notify node %s: in SK range with coreNum>0 (cube %u, vec %u), which not allowed",
              curNode.Format().c_str(), curNode.GetCubeNum(), curNode.GetVecNum());
      deadlockReason_ = DeadlockFailReason::NOTIFY_INVALID;
      return false;
    }
  } else if (curNode.GetNodeType() == SkNodeType::NODE_WAIT) {
    if (ShouldBypassValueWaitDeadlock(curNode)) {
      SK_LOGI("[lock detector] Wait node %s bypassed deadlock detection by value breaker policy",
              curNode.Format().c_str());
      return true;
    }
    tempVisitedNodes.clear();
    bool canFuse = GetWaitNodeFusibleStatus(curNode);
    if (canFuse) {
      nodes.insert(nodes.end(), tempVisitedNodes.begin(), tempVisitedNodes.end());
    } else {
      RollbackVisitedState(tempVisitedNodes);
    }
    return canFuse;
  } else if (curNode.GetNodeType() == SkNodeType::NODE_KERNEL) {
    if (curNode.IsScopeNode()) {
      SK_LOGD("[lock detector] Scope marker %s: no core resource, can fuse", curNode.Format().c_str());
      return true;
    }
    uint32_t cubeNum = curNode.GetCubeNum();
    uint32_t vecNum = curNode.GetVecNum();
    SK_LOGD("[lock detector] Kernel node %s: coreNum={%u, %u}, superKernelCubeNum=%u, superKernelVecNum=%u",
            curNode.Format().c_str(), cubeNum, vecNum, superKernelCubeNum, superKernelVecNum);
    return HasEnoughCores(&curNode, true);
  } else if (curNode.GetNodeType() == SkNodeType::NODE_RESET) {
    SK_LOGD("[lock detector] Reset node %s: no core resource, can fuse", curNode.Format().c_str());
    return true;
  } else if (curNode.GetNodeType() == SkNodeType::NODE_DEFAULT) {
    SK_LOGD("[lock detector] Default node %s: no core resource, can fuse", curNode.Format().c_str());
    return true;
  } else {
    SK_LOGW("[lock detector] Node %s: unsupported taskType=%u", curNode.Format().c_str(), curNode.GetNodeType());
    deadlockReason_ = DeadlockFailReason::NO_SUPPORT_NODE;
    return false;
  }
}

bool LockDetector::IsFusible(SuperKernelBaseNode &curNode) {
  // If node already visited (already checked or fused), return true directly without modifying state
  if (curNode.IsVisited()) {
    SK_LOGD("[lock detector] Node %s: already visited, can fuse", curNode.Format().c_str());
    return true;
  }

  SK_LOGI("[lock detector] IsFusible: Checking node %s, current state: nodeNum=%u, kernelNodeNum=%u",
          curNode.Format().c_str(), nodeNum, kernelNodeNum);
  deadlockReason_ = DeadlockFailReason::NOT_FIND_DEADLOCK;  // Reset before checking
  bool canFuse = GetFusibleStatus(curNode);
  // Only modify state if node can be fused
  if (canFuse) {
    skStreamIds.insert(curNode.GetStreamIdxInGraph());
    UpdateSKRangeInStream(curNode);
    nodeNum++;
    curNode.SetVisited(true);
    nodes.emplace_back(curNode.GetNodeId());

    if (curNode.GetNodeType() == SkNodeType::NODE_KERNEL && !curNode.IsScopeNode()) {
      kernelNodeNum++;
    }
    SK_LOGD(
        "[lock detector] fused nodeId=%s, nodeType=%u, nodeNum=%u, SuperKernelCubeNum=%u, SuperKernelVecNum=%u, "
        "depOpCubeNum=%u, depOpVecNum=%u",
        curNode.Format().c_str(), curNode.GetNodeType(), nodeNum, superKernelCubeNum, superKernelVecNum, depOpCubeNum,
        depOpVecNum);
  } else {
    SK_LOGI("[lock detector] Node %s: cannot be fused", curNode.Format().c_str());
    // If deadlock was detected, set the failure reason with detail
    if (deadlockReason_ != DeadlockFailReason::NOT_FIND_DEADLOCK) {
      curNode.SetFusionFailReason(deadlockReason_);
    }
  }

  return canFuse;
}

void LockDetector::SetNotifyNodesExpandNumForScope(SuperKernelScopeInfo &scope) {
  const uint32_t maxExpandVecNum = scope.GetSkCoreInfo().GetVectorNum();
  const uint32_t maxExpandCubeNum = scope.GetSkCoreInfo().GetCubeNum();
  std::vector<SuperKernelBaseNode *> notifyNodes;
  std::unordered_set<uint32_t> scopeStreams;
  // Collect notify nodes and stream information. Core counts come from the final SK core info.
  for (const auto *node : scope.GetNodes()) {
    if (node == nullptr) {
      continue;
    }
    if (node->GetNodeType() == SkNodeType::NODE_NOTIFY) {
      notifyNodes.push_back(const_cast<SuperKernelBaseNode *>(node));
    }
    scopeStreams.insert(node->GetStreamIdxInGraph());
  }

  // Set expand numbers for all notify nodes
  for (auto *notifyNode : notifyNodes) {
    notifyNode->SetNotifyExpandVecNum(maxExpandVecNum);
    notifyNode->SetNotifyExpandCubeNum(maxExpandCubeNum);
    SK_LOGI("[lock detector] Set Notify node %lu expandVecNum=%u, expandCubeNum=%u", notifyNode->GetNodeId(),
            maxExpandVecNum, maxExpandCubeNum);
  }
  // Set expand stream for all node
  for (auto *node : scope.GetNodes()) {
    node->SetScopeStreamIds(scopeStreams);
  }
}

void LockDetector::ResetNotifyExpandNumForScope(SuperKernelScopeInfo &scope) {
  for (auto *node : scope.GetNodes()) {
    if (node == nullptr) {
      continue;
    }
    if (node->GetNodeType() == SkNodeType::NODE_NOTIFY) {
      node->SetNotifyExpandVecNum(0);
      node->SetNotifyExpandCubeNum(0);
      SK_LOGD("[lock detector] Reset Notify node %lu expandVecNum=0, expandCubeNum=0", node->GetNodeId());
    }
    node->SetScopeStreamIds({});
  }
}
