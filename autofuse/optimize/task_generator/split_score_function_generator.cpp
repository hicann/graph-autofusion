/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "split_score_function_generator.h"

#include "ascgraph_info_complete.h"
#include "ascir_utils.h"

namespace optimize {
SplitScoreFunctionGenerator::SplitScoreFunctionGenerator(const ascir::HintGraph &graph, af::AscNodePtr split_node,
                                                         uint32_t split_dim)
    : graph_(&graph), split_node_(std::move(split_node)), split_dim_(split_dim) {}

Status SplitScoreFunctionGenerator::Generate(std::string &score_func) {
  ss_ << "int32_t CalcScore(const AutofuseTilingData &tiling_data) {" << std::endl;
  GE_CHK_STATUS_RET(ParseStride());
  if (const_part_stride_ % kAlignment_ == 0U) {
    GenerateReturnValue(1);
    score_func = ss_.str();
    return ASCCOMMON_SUC;
  }

  int32_t score = 0;
  GE_CHK_STATUS_RET(TryGetScoreByConstExpr(score));
  if (score != 0) {
    GenerateReturnValue(score);
    score_func = ss_.str();
    return ASCCOMMON_SUC;
  }
  GE_CHK_STATUS_RET(GenerateForUnaligned());
  score_func = ss_.str();
  return ASCCOMMON_SUC;
}

af::Status SplitScoreFunctionGenerator::ParseStride() {
  num_outputs_ = split_node_->outputs().size();
  GE_ASSERT_TRUE(num_outputs_ > 0U);
  const auto &output_attr = split_node_->outputs[0].attr;
  auto dtype_size = af::GetSizeByDataType(output_attr.dtype);
  const auto num_dims = static_cast<int32_t>(output_attr.repeats.size());
  stride_ = af::Symbol(dtype_size);
  af::Expression const_part_stride = stride_;
  for (int32_t i = num_dims - 1; i > static_cast<int32_t>(split_dim_); --i) {
    const auto dim_expr = output_attr.repeats[i];
    stride_ = stride_ * dim_expr;
    if (dim_expr.IsConstExpr()) {
      const_part_stride = const_part_stride * dim_expr;
    }
  }
  GE_ASSERT_TRUE(const_part_stride.GetConstValue(const_part_stride_), "Failed to get int value, expr = %s",
                 const_part_stride.Str().get());
  return af::SUCCESS;
}

void SplitScoreFunctionGenerator::GenerateReturnValue(int32_t score) {
  ss_ << "  return " << score << ";" << std::endl;
  ss_ << "}" << std::endl;
}

Status SplitScoreFunctionGenerator::TryGetScoreByConstExpr(int32_t &score) {
  auto &output_split_dim = split_node_->outputs[0].attr.repeats[split_dim_];
  if (!output_split_dim.IsConstExpr()) {
    GELOGI("output split dim is non-const, cannot resolve score at compile time");
    return af::SUCCESS;
  }
  int64_t split_dim_size = -1;
  GE_ASSERT_TRUE(output_split_dim.GetConstValue(split_dim_size), "Failed to get int value, expr = %s",
                 output_split_dim.Str().get());
  int64_t num_aligned = 0U;
  auto align_threshold =
      static_cast<int64_t>(std::ceil(static_cast<double>(split_dim_size) * (1.0 - kMaxUnalignedRate)));
  for (size_t i = 0U; i < num_outputs_; ++i) {
    const auto &output = split_node_->outputs[i];
    if (!output.attr.repeats[split_dim_].IsConstExpr()) {
      GELOGI("output[%zu] split dim is non-const, cannot resolve score at compile time", i);
      score = 0;
      return ASCCOMMON_SUC;
    }
    int64_t dim = -1;
    GE_ASSERT_TRUE(output.attr.repeats[split_dim_].GetConstValue(dim), "Failed to get int value, expr = %s",
                   output.attr.repeats[split_dim_].Str().get());
    if ((dim * const_part_stride_) % kAlignment_ == 0) {
      GELOGD("output[%zu] is aligned, dim_size = %ld", i, dim);
      num_aligned += dim;
    } else {
      break;
    }
  }
  if (num_aligned >= align_threshold) {
    score = 1;
    GELOGI("aligned = %u/%u, reaches aligned threshold", num_aligned, split_dim_size);
    return ASCCOMMON_SUC;
  }
  // split dim之后的dim全为const, 不满足对齐率则为不对齐
  if (stride_.IsConstExpr()) {
    score = -1;
    GELOGI("aligned = %u/%u, did not reach aligned threshold", num_aligned, split_dim_size);
    return ASCCOMMON_SUC;
  }

  // split dim之后的dim不全为const, 无法确定是否对齐
  score = 0;
  GELOGI("Cannot resolve score at compile time");
  return ASCCOMMON_SUC;
}

Status SplitScoreFunctionGenerator::GenerateForUnaligned() {
  // 该模板必然为第0个result的第0个group
  const auto &tiling_data = "tiling_data.graph0_result0_g0_tiling_data";
  std::vector<std::pair<af::Expression, af::Expression>> replacements;
  SizeVarSet original_var_set;
  AscGraphInfoComplete::AppendOriginalSizeVar(*graph_, original_var_set);
  for (const auto &size_var : original_var_set) {
    if (!size_var.IsConstExpr()) {
      replacements.emplace_back(size_var, af::Symbol((std::string("t.") + size_var.Str().get()).c_str()));
    }
  }
  auto &output_split_dim = split_node_->inputs[0].attr.repeats[split_dim_];
  ss_ << "  const auto &t = " << tiling_data << ";" << std::endl;
  ss_ << "  auto stride = static_cast<int64_t>(" << stride_.Replace(replacements).Str().get() << ");" << std::endl;
  ss_ << "  if (stride % 32 == 0) { return 1; }" << std::endl;
  ss_ << "  std::vector<int64_t> split_dims;" << std::endl;
  ss_ << "  split_dims.reserve(" << num_outputs_ << ");" << std::endl;
  for (size_t i = 0U; i < num_outputs_; ++i) {
    ss_ << "  split_dims.emplace_back(static_cast<int64_t>("
        << split_node_->outputs[i].attr.repeats[split_dim_].Replace(replacements).Str().get() << "));" << std::endl;
  }
  ss_ << "  size_t num_unaligned = 0U;" << std::endl;
  ss_ << "  size_t max_unaligned = static_cast<int64_t>(std::ceil(static_cast<double>(";
  ss_ << output_split_dim.Replace(replacements).Str().get() << ") * " << kMaxUnalignedRate << "));" << std::endl;
  ss_ << "  bool already_unaligned = false;" << std::endl;
  ss_ << "  for (int32_t i = 0; i < " << num_outputs_ << "; ++i) {" << std::endl;
  ss_ << "    if (already_unaligned || (stride * split_dims[i] % 32) != 0) {" << std::endl;
  ss_ << "      num_unaligned += split_dims[i];" << std::endl;
  ss_ << "      already_unaligned = true;" << std::endl;
  ss_ << "      if (num_unaligned >= max_unaligned) { return -1; }" << std::endl;
  ss_ << "    }" << std::endl;
  ss_ << "  }" << std::endl;
  ss_ << "  return 1;" << std::endl;
  ss_ << "}" << std::endl;
  GELOGI("code generated: %s", ss_.str().c_str());
  return ASCCOMMON_SUC;
}
}  // namespace optimize
