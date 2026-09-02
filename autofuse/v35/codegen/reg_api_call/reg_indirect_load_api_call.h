/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AUTOFUSE_REG_INDIRECT_LOAD_API_CALL_H__
#define __AUTOFUSE_REG_INDIRECT_LOAD_API_CALL_H__

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "codegen_kernel.h"
#include "indirect_load_utils.h"

namespace codegen {
struct SimtGmTensor {
  ascir::TensorId value_tensor_id;
  ascir::TensorId gm_tensor_id;
  af::DataType dtype;
  bool is_scalar{false};
};

struct SimtOutputChain {
  std::vector<af::AscNodePtr> nodes;
  ascir::TensorId result_tensor_id{af::kIdNone};
  ascir::TensorId target_tensor_id{af::kIdNone};
  af::DataType dtype{af::DT_UNDEFINED};
  bool local_target{false};
};

struct LogicalTensorInfo {
  LogicalTensorInfo() = default;
  explicit LogicalTensorInfo(const ascgen_utils::indirect_load::LogicalTensorView &view)
      : sizes(view.sizes), strides(view.strides) {}
  LogicalTensorInfo(const std::vector<ascir::SizeExpr> &sizes_in, const std::vector<ascir::SizeExpr> &strides_in)
      : sizes(sizes_in), strides(strides_in) {}

  std::vector<ascir::SizeExpr> sizes;
  std::vector<ascir::SizeExpr> strides;
};

enum class SimtAddressPolicy {
  kStaticPowerOfTwo,
  kStaticInner,
  kStructuredMagic,
  kEmbedding,
  kRecursive,
  kStrided,
};

struct SimtCodegenPlan {
  SimtAddressPolicy policy = SimtAddressPolicy::kRecursive;
  std::string offset_type = "uint64_t";
  af::Expression inner_span = af::sym::kSymbolOne;
  af::Expression output_axis_span = af::sym::kSymbolOne;
  af::Expression input_axis_stride = af::sym::kSymbolOne;
  af::Expression input_axis_span = af::sym::kSymbolOne;
  uint64_t inner_span_value = 0U;
  uint64_t output_axis_span_value = 0U;
  uint64_t input_axis_stride_value = 0U;
  uint64_t input_axis_span_value = 0U;
  uint64_t input_stride_mask = 0U;
  uint64_t index_stride_mask = 0U;
  bool use_per_load_index_offsets = false;
};

enum class SimtLoadAddressSource {
  kZeroOffset,
  kOutputOffset,
  kIndexOffset,
};

using SimtLoadAddressSources = std::unordered_map<const af::AscNode *, SimtLoadAddressSource>;
using SimtLoadIndexOffsetExpressions = std::unordered_map<const af::AscNode *, std::string>;

class IndirectLoadRegApiCall final : public ApiCall {
 public:
  explicit IndirectLoadRegApiCall(const std::string &api_name) : ApiCall(api_name) {}
  ~IndirectLoadRegApiCall() final = default;
  Status GenerateFuncDefinition(const TPipe &tpipe, const Tiler &tiler, std::stringstream &ss) const override;
  Status Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                  const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                  const std::vector<std::reference_wrapper<const Tensor>> &outputs, std::string &result) const override;
  Status Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                  std::string &result) const override;

 protected:
  Status ParseAttr(const ascir::NodeView &node) override;

 private:
  Status ParseSimtAttr(const ascir::NodeView &node);
  Status GenerateSk(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                    const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                    const std::vector<std::reference_wrapper<const Tensor>> &outputs, std::string &result) const;
  Status GenerateSimd(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                      const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                      const std::vector<std::reference_wrapper<const Tensor>> &outputs, std::string &result) const;
  Status GenerateSimtInvocation(const TPipe &tpipe, const std::string &input_dtype, const std::string &outer_tb_var,
                                std::stringstream &ss) const;
  Status GenerateSimt(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis, const Tensor &input,
                      std::string &result) const;

  int64_t axis_ = 0;
  ascir::AxisId outer_axis_ = af::kIdNone;
  ascir::TemplateId template_id_{ascir::TemplateId::kDefault};
  ascgen_utils::indirect_load::TemplateLogicalView logical_view_;
  ascgen_utils::indirect_load::IndirectLoadAccessInfo access_info_;
  bool has_post_reduce_{false};
  std::vector<af::AscNodePtr> index_nodes_;
  std::vector<af::AscNodePtr> output_nodes_;
  std::vector<SimtOutputChain> output_chains_;
  // SIMT layout metadata and policy are built once during ParseSimtAttr and reused by both emitters.
  LogicalTensorInfo simt_input_info_;
  LogicalTensorInfo simt_index_info_;
  SimtCodegenPlan simt_plan_;
  SimtLoadAddressSources simt_load_address_sources_;
  std::vector<SimtGmTensor> simt_gm_tensors_;
  std::unordered_set<ascir::TensorId> simt_gm_tensor_ids_;
  ascir::TensorId index_result_tensor_id_{af::kIdNone};
  ascir::TensorId simt_value_tensor_id_{af::kIdNone};
  ascir::TensorId output_result_tensor_id_{af::kIdNone};
  af::DataType index_dtype_{af::DT_UNDEFINED};
  af::DataType output_dtype_{af::DT_UNDEFINED};
  ascgen_utils::indirect_load::Implementation implementation_{ascgen_utils::indirect_load::Implementation::kDefault};
};
}  // namespace codegen

#endif  // __AUTOFUSE_REG_INDIRECT_LOAD_API_CALL_H__
