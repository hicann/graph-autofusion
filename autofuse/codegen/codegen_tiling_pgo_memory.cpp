/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen_tiling.h"

#include <algorithm>
#include <unordered_map>

#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"

namespace codegen {
using namespace af::ascir_op;
using namespace af::ops;
using namespace ascgen_utils;
using namespace ascir;

std::string TilingLib::ExternFunctionDeclare(const ascir::FusedScheduledResult &fused_schedule_result,
                                             const std::string tiling) const {
  (void)tiling;
  std::stringstream ss;

  // 生成判断是否为静态shape的接口
  bool is_static = IsStaticSchedResult(fused_schedule_result);
  ss << GenCheckStaticShapeFunc(is_static);
  return ss.str();
}

std::string TilingLib::PGOTensorArgsDef() const {
  std::stringstream ss;
  ss << "#ifndef AUTOFUSE_PGO_TENSOR_ARGS_DEFINED" << std::endl;
  ss << "#define AUTOFUSE_PGO_TENSOR_ARGS_DEFINED" << std::endl;
  ss << "struct PgoTensorArgs {" << std::endl;
  ss << "  void **inputs = nullptr;" << std::endl;
  ss << "  uint32_t input_num = 0;" << std::endl;
  ss << "  void **outputs = nullptr;" << std::endl;
  ss << "  uint32_t output_num = 0;" << std::endl;
  ss << "};" << std::endl;
  ss << "#endif" << std::endl;
  return ss.str();
}

void TilingLib::AppendPgoConfigDef(std::stringstream &ss) const {
  ss << "class PgoConfig {" << std::endl;
  ss << "public:" << std::endl;
  ss << "  static PgoConfig& Instance() {" << std::endl;
  ss << "    static PgoConfig instance;" << std::endl;
  ss << "    return instance;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  void ResetRuntimeOverrides() {" << std::endl;
  ss << "    need_change_solver_run = false;" << std::endl;
  ss << "    pgo_threshold_index = 0;" << std::endl;
  ss << "    pgo_ub_threshold_list = {0.2, 0.1, 0, 0.05, 0.1};" << std::endl;
  ss << "    pgo_corenum_threshold_list = {0.4, 0.4, 1, 1, 0.8};" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ProfilingCallback single_callback;" << std::endl;
  ss << "  ProfilingBatchCallback batch_callback;" << std::endl;
  ss << "  PgoTensorArgs *tensor_args = nullptr;" << std::endl;
  ss << "  int32_t pgo_algorithm = 1; // 0 for pruning, 1 for core num" << std::endl;
  ss << "  bool need_change_solver_run = false;" << std::endl;
  ss << "  size_t pgo_threshold_index = 0;" << std::endl;
  ss << "  constexpr static size_t pgo_threshold_list_size = 5;" << std::endl;
  ss << "  std::array<double, pgo_threshold_list_size> pgo_ub_threshold_list{0.2, 0.1, 0, 0.05, 0.1};" << std::endl;
  ss << "  std::array<double, pgo_threshold_list_size> pgo_corenum_threshold_list{0.4, 0.4, 1, 1, 0.8};" << std::endl;
  ss << "private:" << std::endl;
  ss << "  PgoConfig() = default;" << std::endl;
  ss << "  ~PgoConfig() = default;" << std::endl;
  ss << "  PgoConfig(const PgoConfig &) = delete;" << std::endl;
  ss << "  PgoConfig &operator=(const PgoConfig &) = delete;" << std::endl;
  ss << "};" << std::endl;
  ss << "class PgoConfigRuntimeGuard {" << std::endl;
  ss << "public:" << std::endl;
  ss << "  PgoConfigRuntimeGuard() { PgoConfig::Instance().ResetRuntimeOverrides(); }" << std::endl;
  ss << "  ~PgoConfigRuntimeGuard() { PgoConfig::Instance().ResetRuntimeOverrides(); }" << std::endl;
  ss << "};" << std::endl;
}

std::string TilingLib::PGOProfilingCallbackDef(const ascir::FusedScheduledResult &fused_schedule_result,
                                               const std::string tiling, bool include_headers) const {
  std::stringstream ss;

  if (include_headers) {
    ss << "#include <cfloat>" << std::endl;
    ss << "#include <cstdint>" << std::endl;
    ss << "#include <vector>" << std::endl;
    ss << "#include <unordered_set>" << std::endl;
    ss << "#include <array>" << std::endl;
    ss << std::endl;
  }
  ss << PGOTensorArgsDef();
  ss << "typedef long int (*ProfilingCallback)(";
  ss << PGOSearchFuncInputOutputCallBackDef(fused_schedule_result);
  ss << "void *stream, uint32_t workspaceSize, " << tiling << " *tiling_data, double *cost_time);" << std::endl;
  ss << "typedef long int (*ProfilingBatchCallback)(";
  ss << PGOSearchFuncInputOutputCallBackDef(fused_schedule_result);
  ss << "void *stream, uint32_t workspaceSize, std::vector<AutofuseTilingDataPerf> *profiles);" << std::endl;
  AppendPgoConfigDef(ss);
  ss << std::endl;

  return ss.str();
}

std::string TilingLib::PGOSearchFuncInputOutputCallBackDef(
    const ascir::FusedScheduledResult &fused_schedule_result) const {
  (void)fused_schedule_result;
  return "PgoTensorArgs *tensor_args, ";
}

std::string TilingLib::PGOSearchFuncInputOutputDef(const ascir::FusedScheduledResult &fused_schedule_result) const {
  (void)fused_schedule_result;
  return "PgoTensorArgs *tensor_args = nullptr, ";
}

std::string TilingLib::PGOSearchFuncInputOutputCall(const ascir::FusedScheduledResult &fused_schedule_result) const {
  (void)fused_schedule_result;
  return "tensor_args, ";
}

std::string TilingLib::PGOSearchStructInputOutputDef(const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  int index = 0;
  for ([[maybe_unused]] auto &input : fused_schedule_result.input_nodes) {
    ss << "  uint64_t input" << index++ << ";" << std::endl;
  }
  index = 0;
  for (auto &node : fused_schedule_result.output_nodes) {
    if (af::ops::IsOps<af::ascir_op::Output>(node)) {
      ss << "  uint64_t output" << index++ << ";" << std::endl;
    }
  }

  return ss.str();
}

std::string TilingLib::PGOSearchTensorInputOutputDef(const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  int index = 0;
  for ([[maybe_unused]] auto &input : fused_schedule_result.input_nodes) {
    ss << "void* input" << index++ << ";" << std::endl;
  }
  index = 0;
  for (auto &node : fused_schedule_result.output_nodes) {
    if (af::ops::IsOps<af::ascir_op::Output>(node)) {
      ss << "void* output" << index++ << ";" << std::endl;
    }
  }
  ss << "uint64_t ffts;" << std::endl;
  const size_t input_array_size = std::max<size_t>(fused_schedule_result.input_nodes.size(), 1UL);
  const uint32_t output_count = PGOSearchFuncGetOutputCount(fused_schedule_result);
  const uint32_t output_array_size = std::max<uint32_t>(output_count, 1U);
  ss << "void *g_pgo_inputs[" << input_array_size << "] = {nullptr};" << std::endl;
  ss << "void *g_pgo_outputs[" << output_array_size << "] = {nullptr};" << std::endl;
  ss << "PgoTensorArgs g_pgo_tensor_args = {g_pgo_inputs, " << fused_schedule_result.input_nodes.size()
     << "U, g_pgo_outputs, " << output_count << "U};" << std::endl;

  return ss.str();
}

std::string TilingLib::PGOSearchTensorArgsUpdateDef(const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  int index = 0;
  for ([[maybe_unused]] auto &input : fused_schedule_result.input_nodes) {
    ss << "  g_pgo_inputs[" << index << "] = input" << index << ";" << std::endl;
    index++;
  }
  index = 0;
  for (auto &node : fused_schedule_result.output_nodes) {
    if (af::ops::IsOps<af::ascir_op::Output>(node)) {
      ss << "  g_pgo_outputs[" << index << "] = output" << index << ";" << std::endl;
      index++;
    }
  }
  ss << "  g_pgo_tensor_args.inputs = g_pgo_inputs;" << std::endl;
  ss << "  g_pgo_tensor_args.input_num = " << fused_schedule_result.input_nodes.size() << "U;" << std::endl;
  ss << "  g_pgo_tensor_args.outputs = g_pgo_outputs;" << std::endl;
  ss << "  g_pgo_tensor_args.output_num = " << PGOSearchFuncGetOutputCount(fused_schedule_result) << "U;" << std::endl;
  return ss.str();
}

std::string TilingLib::PGOSearchFuncInputOutputStructAssignDef(const ascir::FusedScheduledResult &fused_schedule_result,
                                                               const std::string &struct_var_name) const {
  std::stringstream ss;
  int index = 0;
  for ([[maybe_unused]] auto &input : fused_schedule_result.input_nodes) {
    ss << struct_var_name << ".input" << index << " = reinterpret_cast<uint64_t>(tensor_args->inputs[" << index << "]);"
       << std::endl;
    index++;
  }
  index = 0;
  for (auto &node : fused_schedule_result.output_nodes) {
    if (af::ops::IsOps<af::ascir_op::Output>(node)) {
      ss << struct_var_name << ".output" << index << " = reinterpret_cast<uint64_t>(tensor_args->outputs[" << index
         << "]);" << std::endl;
      index++;
    }
  }
  return ss.str();
}

uint32_t TilingLib::PGOSearchFuncGetInputOutputCount(const ascir::FusedScheduledResult &fused_schedule_result) const {
  return fused_schedule_result.input_nodes.size() + PGOSearchFuncGetOutputCount(fused_schedule_result);
}

uint32_t TilingLib::PGOSearchFuncGetOutputCount(const ascir::FusedScheduledResult &fused_schedule_result) const {
  uint32_t count = 0;
  for (auto &node : fused_schedule_result.output_nodes) {
    if (af::ops::IsOps<af::ascir_op::Output>(node)) {
      count++;
    }
  }
  return count;
}

std::string TilingLib::CalculateTensorMemorySizeStr(const ascir::TensorAttr &tensor) const {
  static const std::unordered_map<ge::DataType, af::Expression> type_size_map = {
      {ge::DT_FLOAT, af::Expression::Parse("4")},    // sizeof(float)
      {ge::DT_FLOAT16, af::Expression::Parse("2")},  // fp16 is 2 bytes
      {ge::DT_INT8, af::Expression::Parse("1")},     // sizeof(int8_t)
      {ge::DT_INT16, af::Expression::Parse("2")},    // sizeof(int16_t)
      {ge::DT_INT32, af::Expression::Parse("4")},    // sizeof(int32_t)
      {ge::DT_INT64, af::Expression::Parse("8")},    // sizeof(int64_t)
      {ge::DT_UINT8, af::Expression::Parse("1")},    // sizeof(uint8_t)
      {ge::DT_UINT16, af::Expression::Parse("2")},   // sizeof(uint16_t)
      {ge::DT_UINT32, af::Expression::Parse("4")},   // sizeof(uint32_t)
      {ge::DT_UINT64, af::Expression::Parse("8")},   // sizeof(uint64_t)
      {ge::DT_DOUBLE, af::Expression::Parse("8")},   // sizeof(double)
      {ge::DT_BF16, af::Expression::Parse("2")},     // bf16 is 2 bytes
      {ge::DT_BOOL, af::Expression::Parse("1")}      // sizeof(bool)
  };
  const auto dtype = tensor.attr.dtype.operator ge::DataType();
  auto it = type_size_map.find(dtype);
  if (it == type_size_map.end()) {
    GELOGE(ge::GRAPH_FAILED, "Unsupported data type: %d", static_cast<int32_t>(dtype));
    return "0";
  }
  af::Expression type_size = it->second;
  if (tensor.attr.repeats.empty() || tensor.attr.strides.empty()) {
    GELOGE(ge::GRAPH_FAILED, "Empty repeats or strides for tensor when calculating memory size");
    return "0";
  }

  // 跳过brc场景下的0 strides
  size_t stride_index = 0UL;
  for (; stride_index < tensor.attr.strides.size(); ++stride_index) {
    if (tensor.attr.strides[stride_index] != af::ops::Zero) {
      break;
    }
    GELOGD("Tensor stride %zu is zero, try to skip to next non-zero stride.", stride_index);
  }

  // 全为brc轴时，元素个数为1，其他情况下为repeats[index] * strides[index]
  af::Expression element_size = af::ops::One;
  if (stride_index < tensor.attr.repeats.size() && stride_index < tensor.attr.strides.size()) {
    element_size = af::sym::Mul(tensor.attr.repeats[stride_index], tensor.attr.strides[stride_index]).Simplify();
  }
  af::Expression need_malloc_size = af::sym::Mul(element_size, type_size).Simplify();
  GELOGD("Tensor element size: %s, need malloc size: %s", element_size.Str().get(), need_malloc_size.Str().get());
  return std::string(need_malloc_size.Str().get());
}

std::string TilingLib::PGOSearchTensorMallocDef(const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  int index = 0;
  for (auto &input : fused_schedule_result.input_nodes) {
    if (input->GetOutNodesPtr().empty()) {
      continue;
    }
    af::Node *out_node = input->GetOutNodesPtr()[0];
    af::AscNode *asc_out_node = static_cast<af::AscNode *>(out_node);
    ss << "  size_t input" << index << "_size = " << CalculateTensorMemorySizeStr(asc_out_node->outputs[0]) << ";"
       << std::endl;
    ss << "  ret = aclrtMalloc(&input" << index << ", input" << index << "_size, ACL_MEM_MALLOC_HUGE_FIRST);"
       << std::endl;
    ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
    ss << "    DLOGE(\"aclrtMalloc input" << index << " failed. ERROR: %d\", ret);" << std::endl;
    ss << "    return FAILED;" << std::endl;
    ss << "  }" << std::endl;
    index++;
  }
  index = 0;
  for (auto &output : fused_schedule_result.output_nodes) {
    if (af::ops::IsOps<af::ascir_op::Output>(output)) {
      ss << "  size_t output" << index << "_size = " << CalculateTensorMemorySizeStr(output->inputs[0]) << ";"
         << std::endl;
      ss << "  ret = aclrtMalloc(&output" << index << ", output" << index << "_size, ACL_MEM_MALLOC_HUGE_FIRST);"
         << std::endl;
      ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
      ss << "    DLOGE(\"aclrtMalloc output" << index << " failed. ERROR: %d\", ret);" << std::endl;
      ss << "    return FAILED;" << std::endl;
      ss << "  }" << std::endl;
      index++;
    }
  }
  return ss.str();
}

std::string TilingLib::PGOSearchTensorFreeDef(const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  int index = 0;
  for ([[maybe_unused]] auto &input : fused_schedule_result.input_nodes) {
    ss << "  if (input" << index << " != nullptr) {" << std::endl;
    ss << "    ret = aclrtFree(input" << index << ");" << std::endl;
    ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
    ss << "      DLOGW(\"aclrtFree input" << index << " failed. ERROR: %d\", ret);" << std::endl;
    ss << "    }" << std::endl;
    ss << "    input" << index << " = nullptr;" << std::endl;
    ss << "  }" << std::endl;
    index++;
  }
  index = 0;
  for (auto &output : fused_schedule_result.output_nodes) {
    if (af::ops::IsOps<af::ascir_op::Output>(output)) {
      ss << "  if (output" << index << " != nullptr) {" << std::endl;
      ss << "    ret = aclrtFree(output" << index << ");" << std::endl;
      ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
      ss << "      DLOGW(\"aclrtFree output" << index << " failed. ERROR: %d\", ret);" << std::endl;
      ss << "    }" << std::endl;
      ss << "    output" << index << " = nullptr;" << std::endl;
      ss << "  }" << std::endl;
      index++;
    }
  }
  return ss.str();
}

}  // namespace codegen
