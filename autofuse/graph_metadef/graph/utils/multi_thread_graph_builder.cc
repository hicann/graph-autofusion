/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/multi_thread_graph_builder.h"
#include "graph/normal_graph/operator_impl.h"
#include "graph_metadef/graph/debug/ge_util.h"

namespace af {
MultiThreadGraphBuilder::MultiThreadGraphBuilder(int32_t thread_num) : thread_num_(thread_num < 1 ? 1 : thread_num) {}

graphStatus MultiThreadGraphBuilder::GetGraphRelatedOperators(const std::vector<Operator> &inputs,
                                                              std::vector<OperatorImplPtr> &related_ops) {
  std::vector<OperatorImplPtr> vec_inputs;
  for (auto &it : inputs) {
    GE_CHECK_NOTNULL(it.operator_impl_);
    vec_inputs.push_back(it.operator_impl_);
  }
  GE_CHK_GRAPH_STATUS_RET(WalkForwardOperators(vec_inputs, related_ops), "Fail to walk all forward operators.");
  return GRAPH_SUCCESS;
}

void MultiThreadGraphBuilder::GetOutputLinkOps(const OperatorImplPtr &op_impl,
                                               std::vector<OperatorImplPtr> &output_op_impls) {
  for (const auto &out_link : op_impl->output_links_) {
    for (const auto &op_forward : out_link.second) {
      output_op_impls.push_back(op_forward.GetOwner());
    }
  }
  auto &out_control_links = op_impl->control_output_link_;
  for (const auto &out_control_link : out_control_links) {
    output_op_impls.push_back(out_control_link.lock());
  }
}

graphStatus MultiThreadGraphBuilder::WalkForwardOperators(const std::vector<OperatorImplPtr> &vec_ops,
                                                          std::vector<OperatorImplPtr> &related_ops) {
  std::set<OperatorImplPtr> all_impls;
  std::queue<std::vector<OperatorImplPtr>> que;
  que.push(vec_ops);
  while (!que.empty()) {
    const auto vec_tem = que.front();
    que.pop();
    for (const auto &op_impl : vec_tem) {
      GE_CHECK_NOTNULL(op_impl);
      if (all_impls.find(op_impl) == all_impls.cend()) {
        all_impls.emplace(op_impl);
        std::vector<OperatorImplPtr> vec_op_forward{};
        GetOutputLinkOps(op_impl, vec_op_forward);
        que.push(vec_op_forward);
      }
    }
  }

  for (auto impl : all_impls) {
    related_ops.emplace_back(impl);
  }
  return GRAPH_SUCCESS;
}

void MultiThreadGraphBuilder::ResetOpSubgraphBuilder(const OpDescPtr &op_desc, OperatorImplPtr &op_impl) {
  const auto &subgraph_names_to_index = op_desc->GetSubgraphNameIndexes();
  for (const auto &name_idx : subgraph_names_to_index) {
    const SubgraphBuilder &builder = op_impl->GetSubgraphBuilder(name_idx.first.c_str());
    if (builder == nullptr) {
      continue;
    }
    std::shared_future<Graph> future_graph = pool_->commit([builder]() -> Graph { return builder(); });
    auto future_graph_ptr = std::make_shared<std::shared_future<Graph>>(future_graph);
    auto graph_builder = [future_graph_ptr, builder]() mutable {
      Graph graph;
      if (future_graph_ptr->valid()) {
        graph = future_graph_ptr->get();
        // reset shared_future to release graph owner, can not be invoked twice
        *future_graph_ptr = std::shared_future<Graph>();
      } else {
        // use default builder
        graph = builder();
      }
      return graph;
    };
    op_impl->SetSubgraphBuilder(name_idx.first.c_str(), name_idx.second, graph_builder);
  }
}

Graph &MultiThreadGraphBuilder::SetInputs(const std::vector<Operator> &inputs, Graph &graph) {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (thread_num_ > 1 && pool_ == nullptr) {
      pool_ = ComGraphMakeUnique<GraphThreadPool>(thread_num_);
    }
  }

  if (pool_ != nullptr) {
    GELOGI("Build subgraph async, thread num = %d.", thread_num_);
    std::vector<OperatorImplPtr> all_related_ops;
    (void)GetGraphRelatedOperators(inputs, all_related_ops);
    for (auto &op_impl : all_related_ops) {
      if (op_impl->op_desc_ != nullptr) {
        ResetOpSubgraphBuilder(op_impl->op_desc_, op_impl);
      }
    }
  }
  return graph.SetInputs(inputs);
}
}  // namespace af
