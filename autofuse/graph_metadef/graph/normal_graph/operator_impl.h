/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_OPERATOR_IMPL_H
#define METADEF_CXX_OPERATOR_IMPL_H
#include <memory>
#include <string>
#include "graph/op_desc.h"
#include "graph/node.h"
#include "graph/operator_af.h"
#include "graph/inference_context_af.h"
#include "graph/runtime_inference_context.h"
#include "graph/normal_graph/op_io.h"
namespace af {
class OperatorImpl : public std::enable_shared_from_this<OperatorImpl> {
 public:
  using GetConstInputOnRuntimeFun =
      std::function<graphStatus(const ConstNodePtr &node, const size_t index, GeTensorPtr &tensor)>;
  explicit OperatorImpl(const std::string &name, const std::string &type);
  explicit OperatorImpl(const OpDescPtr &op_desc);
  explicit OperatorImpl(const ConstNodePtr node);
  ~OperatorImpl();

  void SetInputImpl(const std::string &dst_name, const Operator &src_oprt);
  void SetInputImpl(const std::string &dst_name, const OutHandler &out_handler);
  void AddControlInputImp(const Operator &src_oprt);
  graphStatus GetInputImpl(const std::string &dst_name, OpIO &out_handler) const;
  graphStatus GetInputImpl(const uint32_t idx, OpIO &out_handler) const;
  graphStatus GetInputConstData(const std::string &dst_name, Tensor &data);
  graphStatus GetInputConstData(const uint32_t idx, ConstGeTensorPtr &ge_tensor) const;
  graphStatus GetInputConstDataOut(const std::string &dst_name, Tensor &data) const;
  graphStatus GetInputConstDataOut(const uint32_t idx, ConstGeTensorPtr &ge_tensor) const;
  bool InputIsSet(const std::string &name);
  std::string GetName() const;
  GeTensorDesc GetInputDesc(const std::string &name) const;
  GeTensorDesc GetInputDesc(const uint32_t index) const;
  GeTensorDescPtr MutableInputDesc(const std::string &name);
  GeTensorDescPtr MutableInputDesc(const uint32_t index);
  graphStatus UpdateInputDesc(const std::string &name, const GeTensorDesc &tensor_desc);
  OutHandler GetOutput(const std::string &name);
  OutHandler GetOutput(uint32_t index);
  GeTensorDesc GetOutputDesc(const std::string &name) const;
  GeTensorDesc GetOutputDesc(const uint32_t index) const;
  GeTensorDescPtr MutableOutputDesc(const std::string &name);
  GeTensorDescPtr MutableOutputDesc(const uint32_t index);
  graphStatus UpdateOutputDesc(const std::string &name, const GeTensorDesc &tensor_desc);
  size_t GetInputsSize() const;
  size_t GetOutputsSize() const;
  graphStatus SetAttr(const std::string &name, AnyValue &&attr_value);
  graphStatus SetAttr(const std::string &name, const AnyValue &attr_value);
  graphStatus GetAttr(const std::string &name, AnyValue &attr_value) const;
  OpDescPtr GetOpDescImpl() const;
  void UpdateLinkMapImpl(const std::string &src_name, const OpIO &op_dst);
  Operator ToOperator();
  void ClearOutputLinks() noexcept;
  void ClearInputLinks() noexcept;
  ConstNodePtr GetNode() const;
  graphStatus SetNode(const ConstNodePtr &node);
  void SetInferenceContext(const InferenceContextPtr &inference_context);
  InferenceContextPtr GetInferenceContext() const;
  void SubgraphRegister(const std::string &ir_name, const bool dynamic);
  void SubgraphCountRegister(const std::string &ir_name, const uint32_t count);
  void SetSubgraphBuilder(const std::string &ir_name, const uint32_t index, const SubgraphBuilder &builder);
  SubgraphBuilder GetSubgraphBuilder(const std::string &ir_name, const uint32_t index) const;
  SubgraphBuilder GetSubgraphBuilder(const std::string &name) const;
  std::vector<std::string> GetSubgraphNames() const;
  size_t GetSubgraphNamesCount() const;

  static OpDescPtr GetOpDesc(const Operator &oprt);
  graphStatus UpdateInputDesc(const uint32_t index, const GeTensorDesc &tensor_desc);
  graphStatus UpdateOutputDesc(const uint32_t index, const GeTensorDesc &tensor_desc);

 private:
  graphStatus GetFromPeerNode(NodePtr &peer_node, const OutDataAnchorPtr &out_data_anchor,
                              ConstGeTensorPtr &ge_tensor) const;

 private:
  OpDescPtr op_desc_ = nullptr;
  ConstNodePtr node_{nullptr};
  InferenceContextPtr inference_context_;
  std::map<std::string, std::vector<OpIO>> output_links_{};
  std::map<std::string, OpIO> input_link_{};
  std::vector<std::weak_ptr<OperatorImpl>> control_input_link_{};
  std::vector<std::weak_ptr<OperatorImpl>> control_output_link_{};
  std::map<std::string, SubgraphBuilder> subgraph_names_to_builders_;
  RuntimeInferenceContext *runtime_context_{nullptr};  // deprecated, will delete when air support
  GetConstInputOnRuntimeFun get_const_input_runtime_ = nullptr;

 private:
  friend class GraphBuilderImpl;
  friend class MultiThreadGraphBuilder;
  friend class OpDescUtils;
};
// Used to manage OperatorImpl instances created by ge api.
class OperatorKeeper {
 public:
  static OperatorKeeper &GetInstance();
  void CheckInOperator(const OperatorImplPtr &op_impl) {
    if (op_impl) {
      const std::lock_guard<std::mutex> lock(mutex_);
      (void)(operators_.insert(op_impl));
    }
  }
  void CheckOutOperator(const OperatorImplPtr &op_impl) {
    if (op_impl) {
      const std::lock_guard<std::mutex> lock(mutex_);
      (void)(operators_.erase(op_impl));
    }
  }

  void ClearInvalidOp() {
    const std::lock_guard<std::mutex> lock(mutex_);
    for (auto iter = operators_.begin(); iter != operators_.end();) {
      auto op = iter->lock();
      if (op == nullptr) {
        iter = operators_.erase(iter);
      } else {
        ++iter;
      }
    }
  }

 private:
  OperatorKeeper() = default;
  ~OperatorKeeper() {
    for (const auto &iter : operators_) {
      if (!iter.expired()) {
        iter.lock()->ClearInputLinks();
      }
      if (!iter.expired()) {
        iter.lock()->ClearOutputLinks();
      }
    }
    // Manually clean up for `Operator` destructor may access `operators_`
    auto operators = std::move(operators_);
    operators.clear();
  }
  std::set<std::weak_ptr<OperatorImpl>, std::owner_less<std::weak_ptr<OperatorImpl>>> operators_;
  std::mutex mutex_;
};
}  // namespace af

#endif  // METADEF_CXX_OPERATOR_IMPL_H
