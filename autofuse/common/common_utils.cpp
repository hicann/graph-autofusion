/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common_utils.h"

#include <sstream>
#include <vector>
#include <regex>
#include <climits>
#include "ascir_ops.h"
#include "ascir.h"
#include "ascir_utils.h"
#include "ascir_register.h"
#include "ascir_ops_utils.h"
#include "common/ge_common/debug/log.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "common/platform_context.h"
#include "autofuse_config/auto_fuse_config.h"

using namespace af::ascir_op;
using namespace af::ops;
namespace ascgen_utils {
constexpr int WORKSPACE_ALIGN_SIZE = 512;
constexpr uint32_t BRC_INLINE_INPUTS_SIZE = 2U;
const af::Expression ONE = af::Symbol(1);
constexpr int64_t kMaxGroupPerCompileUnit = 5;

std::string CamelToLowerSneak(const std::string &str) {
  std::string s1 = std::regex_replace(str, std::regex("(.)([A-Z][a-z]+)"), "$1_$2");
  std::string s2 = std::regex_replace(s1, std::regex("([a-z0-9])([A-Z])"), "$1_$2");

  std::transform(s2.begin(), s2.end(), s2.begin(), ::tolower);
  return s2;
}

std::string SubStringReplace(std::string &ori, const std::string &from, const std::string &to) {
  std::size_t pos = 0U;
  std::string result;

  if (from.empty()) {
    return ori;
  }
  while ((pos = ori.find(from, pos)) != std::string::npos) {
    result.append(ori, 0, pos);
    result.append(to);
    ori.erase(0, pos + from.length());  // 删除已处理部分
    pos = 0U;
  }
  result.append(ori);  // 追加剩余字符

  return result;
}

std::string GenValidName(const std::string &t_name) {
  string result;
  bool lastWasUnderscore = false;

  for (char c : t_name) {
    if (isalnum(c)) {
      result += c;
      lastWasUnderscore = false;
    } else {
      if (!lastWasUnderscore) {
        result += '_';
        lastWasUnderscore = true;
      }
    }
  }
  // 删除开头的下划线
  if (!result.empty() && result[0] == '_') {
    result = result.substr(1);
  }

  if (!result.empty() && std::isdigit(result[0]) != 0) {
    result = "t_" + result;
  }
  return result;
}

bool GetRealPath(const std::string &file_path, std::string &real_file_path) {
  char buf[PATH_MAX] = " ";
  if (realpath(file_path.c_str(), buf) == nullptr) {
    return false;
  }
  real_file_path = buf;
  return true;
}

af::Status GetApiTilingTypeName(const ascir::NodeView &node, std::string &type_name) {
  auto impl = ascgen_utils::GetAscIrCodegenImpl(node->GetType());
  GE_ASSERT_NOTNULL(impl, "GetAscIrCodegenImpl of node %s[%s] is null", node->GetTypePtr(), node->GetNamePtr());
  type_name = impl->GetApiTilingTypeName();
  if (type_name == "") {
    return af::FAILED;
  }
  return af::SUCCESS;
}

af::Status GetApiTilingFieldName(const ascir::NodeView &node, std::string &field_name) {
  auto impl = ascgen_utils::GetAscIrCodegenImpl(node->GetType());
  GE_ASSERT_NOTNULL(impl, "GetAscIrCodegenImpl of node %s[%s] is null", node->GetTypePtr(), node->GetNamePtr());
  auto type_name = impl->GetApiTilingTypeName();
  if (type_name == "") {
    return af::FAILED;
  }
  field_name = GenValidName(node->GetName() + "_tilingData");
  return af::SUCCESS;
}

std::string GenUpdateCurPerfAndBlockByGroupHelper(bool with_log, bool use_std_max) {
  std::stringstream ss;
  const std::string max_func = use_std_max ? "std::max" : "Max";
  ss << R"(
inline bool UpdateCurPerfAndBlockByGroup(std::pair<uint32_t, double> group_block_and_perf,
                                         const uint32_t limited_block,
                                         uint32_t &cur_block,
                                         double &cur_perf,
                                         double &cur_tmp_perf) {
  const auto &group_block = group_block_and_perf.first;
  const auto &group_perf = group_block_and_perf.second;
  if ((cur_block + group_block) > limited_block) {
)";
  if (with_log) {
    ss << R"(    OP_LOGD(OP_NAME, "Cur block %u + group block %u > limited block %u, need to update cur perf %lf.",
             cur_block, group_block, limited_block, cur_tmp_perf);
)";
  }
  ss << R"(    cur_block = group_block;
    cur_perf += cur_tmp_perf;
    cur_tmp_perf = group_perf;
    return true;
  } else {
    cur_block += group_block;
    cur_tmp_perf = )"
     << max_func << R"((cur_tmp_perf, group_perf);
    return false;
  }
}
)";
  return ss.str();
}

af::Expression GetTensorSize(const af::AscTensor &tensor) {
  if (tensor.attr.repeats.size() == 0U) {
    return af::Symbol(0);
  }
  af::Expression tensor_size = af::Symbol(1);  // 当stride全为0时，返回tensorSize为1
  GE_ASSERT_TRUE(tensor.attr.repeats.size() == tensor.attr.strides.size(),
                 "Check size failed, repeats size is %u, strides size is %u.", tensor.attr.repeats.size(),
                 tensor.attr.strides.size());
  for (size_t i = 0; i < tensor.attr.repeats.size(); i++) {
    if (tensor.attr.strides[i] != 0) {
      tensor_size = af::sym::Max(tensor_size, tensor.attr.repeats[i] * tensor.attr.strides[i]);
    }
  }
  auto dtype_size = af::GetSizeByDataType(tensor.attr.dtype);
  tensor_size = af::sym::Mul(tensor_size, af::Symbol(dtype_size));
  return tensor_size;
}

af::Expression CalculateOneWorkspaceSize(const af::AscNodePtr &workspace_nodes) {
  af::Expression ws_size = af::Symbol(0);
  if (workspace_nodes->inputs().size() > 0) {
    auto in = workspace_nodes->inputs[0U];
    auto in_size = GetTensorSize(in);
    ws_size = af::sym::Max(ws_size, in_size);
  }
  auto out = workspace_nodes->outputs[0U];
  for (auto &peer_input : out.anchor.GetPeerInDataAnchors()) {
    auto load_node = std::dynamic_pointer_cast<af::AscNode>(peer_input->GetOwnerNode());
    auto out_size = GetTensorSize(load_node->outputs[0U]);
    ws_size = af::sym::Max(ws_size, out_size);
  }
  return ws_size;
}

af::Expression CalculateWorkspaceSize(const std::vector<af::AscNodePtr> &workspace_nodes) {
  af::Expression total_workspace_size = af::Symbol(0);
  if (workspace_nodes.empty()) {
    return total_workspace_size;
  }
  std::unordered_map<int64_t, af::Expression> max_sizes;
  for (const auto &node : workspace_nodes) {
    if (node == nullptr) {
      GELOGW("[AscgenCommon][CalculateWorkspaceSize] node is nullptr");
      return total_workspace_size;
    }
    if (node->GetType() != "Workspace") {
      GELOGW("[AscgenCommon][CalculateWorkspaceSize] node[%s] is not workspace", node->GetName().c_str());
      return total_workspace_size;
    }
    auto ws_size = CalculateOneWorkspaceSize(node);
    int64_t tensor_id = node->outputs[0U].attr.mem.tensor_id;
    if (max_sizes.find(tensor_id) == max_sizes.end()) {
      max_sizes[tensor_id] = ws_size;
    } else {
      max_sizes[tensor_id] = af::sym::Max(max_sizes[tensor_id], ws_size);
    }
    GELOGD("[AscgenCommon][CalculateWorkspaceSize] node[%s] tensor id[%ld] tensor size[%s]", node->GetName().c_str(),
           tensor_id, af::SymbolicUtils::ToString(max_sizes[tensor_id]).c_str());
  }
  for (const auto &pair : max_sizes) {
    total_workspace_size = af::sym::Add(total_workspace_size, pair.second);
  }
  GELOGD("[AscgenCommon][CalculateWorkspaceSize] workspace total size[%s]",
         af::SymbolicUtils::ToString(total_workspace_size).c_str());
  return total_workspace_size;
}

bool IsStaticSchedResult(const ascir::FusedScheduledResult &fused_schedule_result) {
  for (auto &var : fused_schedule_result.origin_vars) {
    GELOGD("var:%s, is_const:%d", var.Str().get(), static_cast<int32_t>(var.IsConstExpr()));
    if (!var.IsConstExpr()) {
      return false;
    }
  }

  return true;
}

af::Status ScalarValuePreProcess(const std::string &ori_value, const std::string &dtype,
                                 std::string &after_pre_pro_value) {
  if (ori_value == "inf" || ori_value == "-inf") {
    if ((dtype != "float") && (dtype != "half")) {
      return af::FAILED;
    }
    after_pre_pro_value = ori_value == "inf" ? "AfInfinity<" + dtype + ">()" : "-AfInfinity<" + dtype + ">()";
  } else if (ori_value == "nan" || ori_value == "NaN" || ori_value == "NAN") {
    if ((dtype != "float") && (dtype != "half")) {
      return af::FAILED;
    }
    after_pre_pro_value = "AscendC::NumericLimits<" + dtype + ">::QuietNaN()";
  } else {
    after_pre_pro_value = ori_value;
  }
  return af::SUCCESS;
}

bool IsEmptyTensorSence(const ascir::FusedScheduledResult &fused_schedule_result) {
  if (fused_schedule_result.node_idx_to_scheduled_results.empty() ||
      fused_schedule_result.node_idx_to_scheduled_results[0].empty() ||
      fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups.empty() ||
      fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.empty()) {
    return false;
  }
  for (const auto &node :
       fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetType() == "Store") {
      GE_CHECK_NOTNULL_EXEC(node->GetOwnerComputeGraph(), return false;);
      auto attr = node->GetOwnerComputeGraph()->GetAttrsGroup<af::AscGraphAttr>();
      GE_CHECK_NOTNULL_EXEC(attr, return false;);
      for (const auto axis_id : node->outputs[0].attr.axis) {
        if (attr->axis[axis_id]->size == 0) {
          GELOGD("node[%s] axis sizes include 0.", node->GetName().c_str());
          return true;
        }
      }
    }
  }
  return false;
}

bool IsSupportBlkTensorInput(const af::AscNodePtr &next_node) {
  static const std::set<std::string> supported_ops = {"Where", "Select", "Eq", "Ne", "Gt", "Lt", "Ge", "Le", "Cast"};
  return (supported_ops.count(next_node->GetType()) > 0U);
}

void MergeBrcAxisRepeats(const std::vector<af::Expression> &input0_repeats,
                         const std::vector<af::Expression> &input1_repeats,
                         const std::vector<ascir::SizeExpr> &input1_strides,
                         std::vector<af::Expression> &i0_merge_repeats, std::vector<af::Expression> &i1_merge_repeats) {
  MergeBrcAxisParams in0(input0_repeats, input1_strides);
  MergeBrcAxisParams in1(input1_repeats, input1_strides);
  MergeBrcAxisRepeats(in0, in1);
  i0_merge_repeats = in0.merge_repeats;
  i1_merge_repeats = in1.merge_repeats;
}

/**
 * 合并连续的广播轴和非广播轴，默认各个入参合法（即数组长度一致），合轴逻辑：
 * 合轴后的最内轴是非广播轴，要求对齐；合轴后的其他轴不要求对齐；忽略全为1的轴。
 */
void MergeBrcAxisRepeats(MergeBrcAxisParams &in0, MergeBrcAxisParams &in1) {
  // 剔除输入对应位置全是1的轴
  for (size_t i = 0UL; i < in0.repeats.size(); i++) {
    if (ExpressEq(in0.repeats[i], ONE) && ExpressEq(in1.repeats[i], ONE)) {
      continue;
    }
    in0.repeats_no_one.push_back(in0.repeats[i]);
    in1.repeats_no_one.push_back(in1.repeats[i]);
    in0.strides_no_one.push_back(ExpressEq(in0.strides[i], Zero) ? ONE : in0.strides[i]);
    in1.strides_no_one.push_back(ExpressEq(in1.strides[i], Zero) ? ONE : in1.strides[i]);
    in0.is_axis_brc.push_back(!ExpressEq(in0.repeats[i], in1.repeats[i]) && ExpressEq(in0.repeats[i], ONE));
    in1.is_axis_brc.push_back(!ExpressEq(in0.repeats[i], in1.repeats[i]) && ExpressEq(in1.repeats[i], ONE));
  }
  // 两个输入全是1
  if (in0.repeats_no_one.empty()) {
    in0.merge_repeats.push_back(ONE);
    in1.merge_repeats.push_back(ONE);
    return;
  }
  bool pre_is_same = ExpressEq(in0.repeats_no_one[0], in1.repeats_no_one[0]);
  std::vector<size_t> partition = {0};
  for (size_t i = 1UL; i < in0.repeats_no_one.size(); i++) {
    const bool cur_is_same = ExpressEq(in0.repeats_no_one[i], in1.repeats_no_one[i]);
    const bool i0_can_merge = (in0.is_axis_brc[i - 1UL] == in0.is_axis_brc[i]);
    const bool i1_can_merge = (in1.is_axis_brc[i - 1UL] == in1.is_axis_brc[i]);
    if (pre_is_same != cur_is_same || !i0_can_merge || !i1_can_merge) {
      partition.push_back(i);
      pre_is_same = cur_is_same;
    }
  }
  in0.merge_repeats.resize(partition.size());
  in1.merge_repeats.resize(partition.size());

  // 合并后的前N-1个轴，直接repeats累乘
  size_t start = 0UL;
  for (size_t i = start + 1UL; i < partition.size(); i++) {
    const size_t end = partition[i];
    in0.merge_repeats[i - 1UL] = ONE;
    in1.merge_repeats[i - 1UL] = ONE;
    for (size_t j = start; j < end; j++) {
      in0.merge_repeats[i - 1UL] = in0.merge_repeats[i - 1UL] * in0.repeats_no_one[j];
      in1.merge_repeats[i - 1UL] = in1.merge_repeats[i - 1UL] * in1.repeats_no_one[j];
    }
    start = end;
  }
  // 最后1个轴，repeats * strides
  if (start < in1.strides_no_one.size()) {
    in0.merge_repeats[in0.merge_repeats.size() - 1UL] = in0.repeats_no_one[start] * in0.strides_no_one[start];
    in1.merge_repeats[in1.merge_repeats.size() - 1UL] = in1.repeats_no_one[start] * in1.strides_no_one[start];
  }
}

bool IsGeneralizeBrcInlineScene(const af::AscNodePtr &node, const af::AscTensor &input0, const af::AscTensor &input1,
                                std::vector<uint8_t> &input_idx_2_brc_inline) {
  GE_CHK_BOOL_RET_STATUS_NOLOG((input0.attr.repeats.size() == input1.attr.repeats.size()), false);

  // 构造input0的向量化轴id对应的索引
  vector<uint32_t> i0_vectorized_axis_pos;
  for (auto vec_axis : input0.attr.vectorized_axis) {
    auto pos = std::find(input0.attr.axis.begin(), input0.attr.axis.end(), vec_axis);
    GE_CHK_BOOL_RET_STATUS_NOLOG((pos != input0.attr.axis.end()), false);
    i0_vectorized_axis_pos.push_back(pos - input0.attr.axis.begin());
  }

  // 构造input1的向量化轴id对应的索引
  vector<uint32_t> i1_vectorized_axis_pos;
  for (auto vec_axis : input1.attr.vectorized_axis) {
    auto pos = std::find(input1.attr.axis.begin(), input1.attr.axis.end(), vec_axis);
    GE_CHK_BOOL_RET_STATUS_NOLOG((pos != input1.attr.axis.end()), false);
    i1_vectorized_axis_pos.push_back(pos - input1.attr.axis.begin());
  }

  GE_CHK_BOOL_RET_STATUS_NOLOG(i0_vectorized_axis_pos.size() == i1_vectorized_axis_pos.size(), false);

  GELOGD("node_name:%s, input0 axis_id:%s, repeats:%s, vectorized_axis:%s, vectorized_axis_pos:%s", node->GetNamePtr(),
         VectorToStr(input0.attr.axis).c_str(), VectorToStr(input0.attr.repeats).c_str(),
         VectorToStr(input0.attr.vectorized_axis).c_str(), VectorToStr(i0_vectorized_axis_pos).c_str());

  GELOGD("node_name:%s, input0 axis_id:%s, repeats:%s, vectorized_axis:%s, vectorized_axis_pos:%s", node->GetNamePtr(),
         VectorToStr(input1.attr.axis).c_str(), VectorToStr(input1.attr.repeats).c_str(),
         VectorToStr(input0.attr.vectorized_axis).c_str(), VectorToStr(i1_vectorized_axis_pos).c_str());

  /* 对input1, input2进行分组, check分组之后是否为(1, A)广播成(B, A)的场景
   * 分组原则：
   * 连续的广播轴合并，连续的非广播轴合并
   * input1/2都有广播轴, 合并终止, 返回不支持
   * 合并结果check：
   * 如果 input0是(1, A), input1是(B, A), 或者input1是(1, A), input0是(B, A)，则认为是支持场景, 否则不支持的场景
   */

  bool i0_has_brc_axis = false;
  bool i1_has_brc_axis = false;
  std::vector<af::Expression> i0_v_repeates;
  std::vector<af::Expression> i1_v_repeates;
  for (size_t i = 0; i < i0_vectorized_axis_pos.size(); i++) {
    const uint32_t i0_axis_idx = i0_vectorized_axis_pos[i];
    const uint32_t i1_axis_idx = i1_vectorized_axis_pos[i];
    af::Expression i0_cur_axis_repeate = input0.attr.repeats[i0_axis_idx];
    af::Expression i1_cur_axis_repeate = input1.attr.repeats[i1_axis_idx];
    i0_has_brc_axis = (i0_cur_axis_repeate == ONE && i1_cur_axis_repeate != ONE) ? true : i0_has_brc_axis;
    i1_has_brc_axis = (i1_cur_axis_repeate == ONE && i0_cur_axis_repeate != ONE) ? true : i1_has_brc_axis;
    i0_v_repeates.push_back(i0_cur_axis_repeate);
    i1_v_repeates.push_back(i1_cur_axis_repeate);
  }
  if (((i0_has_brc_axis == false) && (i1_has_brc_axis == false)) ||
      ((i0_has_brc_axis == true) && (i1_has_brc_axis == true))) {
    return false;
  }
  input_idx_2_brc_inline.resize(BRC_INLINE_INPUTS_SIZE);
  input_idx_2_brc_inline[0] = static_cast<uint8_t>(i0_has_brc_axis);
  input_idx_2_brc_inline[1] = static_cast<uint8_t>(i1_has_brc_axis);

  std::vector<af::Expression> i0_meger_repeates;
  std::vector<af::Expression> i1_meger_repeates;
  if (i0_has_brc_axis) {
    MergeBrcAxisRepeats(i0_v_repeates, i1_v_repeates, input1.attr.vectorized_strides, i0_meger_repeates,
                        i1_meger_repeates);
  } else {
    MergeBrcAxisRepeats(i1_v_repeates, i0_v_repeates, input0.attr.vectorized_strides, i1_meger_repeates,
                        i0_meger_repeates);
  }

  GELOGD("node_name:%s, i0_meger_repeates:%s, i1_meger_repeates:%s", node->GetNamePtr(),
         VectorToStr(i0_meger_repeates).c_str(), VectorToStr(i1_meger_repeates).c_str());

  if (i0_meger_repeates.size() == 2U && i1_meger_repeates.size() == 2U) {
    if ((i0_meger_repeates[0] == af::Symbol(1)) || (i1_meger_repeates[0] == af::Symbol(1))) {
      return true;
    }
  }

  return false;
}

bool IsGeneralizeBrcInlineScene(const af::AscNodePtr &node, std::vector<uint8_t> &input_idx_2_brc_inline) {
  // 目前只支持两个输入的场景
  if (node->inputs.Size() != BRC_INLINE_INPUTS_SIZE) {
    return false;
  }
  const af::AscTensor input0 = node->inputs[0];
  const af::AscTensor input1 = node->inputs[1];
  return IsGeneralizeBrcInlineScene(node, input0, input1, input_idx_2_brc_inline);
}

bool IsGeneralizeBrcInlineScene(const af::AscNodePtr &node) {
  std::vector<uint8_t> input_idx_2_brc_inline;
  bool is_brc_inline = IsGeneralizeBrcInlineScene(node, input_idx_2_brc_inline);
  input_idx_2_brc_inline.clear();
  return is_brc_inline;
}

int32_t GetBrcInlineIndex(const af::AscNodePtr &node) {
  std::vector<uint8_t> input_idx_2_brc_inline;
  const bool is_brc_inline = IsGeneralizeBrcInlineScene(node, input_idx_2_brc_inline);
  if (!is_brc_inline) {
    return NOT_SUPPORT_BRC_INLINE;
  }
  for (size_t i = 0UL; i < input_idx_2_brc_inline.size(); i++) {
    if (input_idx_2_brc_inline[i] == 1) {
      return static_cast<int32_t>(i);
    }
  }
  return NOT_SUPPORT_BRC_INLINE;
}

bool IsConstExpression(const std::string &expression) {
  for (char c : expression) {
    if (c < '0' || c > '9') {
      return false;
    }
  }
  return !expression.empty();
}

std::string FormatExpression(const std::string &expression) {
  GE_ASSERT_TRUE(!expression.empty(), "Check expression failed, expression is empty!");
  std::string formatted_expression = expression;
  if (IsConstExpression(expression)) {
    return formatted_expression;
  } else if (expression.front() != '(') {
    formatted_expression = "tiling_data.get_" + expression + "()";
  } else {
    // 表达式里的符号值由字母+数字组成，按该规则匹配替换成tiling_data.get_<符号>()
    const std::regex symbol_regex(R"(\b(?=\w*\d)(?=\w*[a-zA-Z])\w+\b)");
    formatted_expression =
        "static_cast<int64_t>" + std::regex_replace(expression, symbol_regex, "tiling_data.get_$&()");
  }
  return formatted_expression;
}

int32_t CalcReservedTmpBufSizeForAscGraph(const ascir::ImplGraph &graph) {
  constexpr int32_t one_blk_size = 1024;
  uint32_t total_reserve_blk_num = 0U;
  GetApiReservedBlockNum(graph, total_reserve_blk_num);
  return total_reserve_blk_num * one_blk_size;
}

void GetApiReservedBlockNum(const ascir::ImplGraph &graph, uint32_t &total_blk_num) {
  const std::unordered_set<std::string> type2api = {
      {Select::Type}, {Where::Type}, {Ge::Type}, {Eq::Type},     {Ne::Type},
      {Gt::Type},     {Le::Type},    {Lt::Type}, {Gather::Type},
  };
  for (const auto &node : graph.GetAllNodes()) {
    auto iter = type2api.find(node->GetType());
    if (iter != type2api.end()) {
      total_blk_num = 8U;
      return;
    }
  }
}

af::Expression CalcExtraTmpBufForAscGraph(const ascir::ImplGraph &graph) {
  constexpr int32_t one_blk_size = 32;
  uint32_t total_blk_num = 0U;
  std::set<std::pair<std::string, std::string>> pre_api_extract_dup;
  GetApiExtractDupSet(graph, pre_api_extract_dup, total_blk_num);
  uint32_t api_extract_blk_num = pre_api_extract_dup.size();
  total_blk_num += api_extract_blk_num;
  return af::Symbol(total_blk_num * one_blk_size);
}

void GetApiExtractDupSet(const ascir::ImplGraph &graph,
                         std::set<std::pair<std::string, std::string>> &pre_api_extract_dup, uint32_t &total_blk_num) {
  const std::unordered_map<std::string, std::string> type2api = {
      {LogicalNot::Type, "AscendcLogical_notStr"},
      {Rsqrt::Type, "AscendcRsqrtStr"},
  };
  const std::unordered_map<string, std::vector<std::pair<std::string, std::string>>> api_extract_dup_map = {
      {"AscendcRsqrtStr", {{"1", "float"}}}, {"AscendcLogical_notStr", {{"1", "half"}}}};

  for (const auto &node : graph.GetAllNodes()) {
    if (af::ops::IsOps<af::ascir_op::Scalar>(node) && IsScalarNextNodeSupportBlkTensor(node)) {
      total_blk_num++;
    }
    if (IsUbScalarLoad(node) && IsScalarNextNodeSupportBlkTensor(node)) {
      total_blk_num++;
    }
    auto iter = type2api.find(node->GetType());
    if (iter == type2api.end()) {
      continue;
    }
    auto it = api_extract_dup_map.find(iter->second);
    if (it == api_extract_dup_map.end()) {
      continue;
    }
    for (const auto &p : it->second) {
      pre_api_extract_dup.insert(p);  // 集合中插入pair {value, dtype}
    }
  }
}

std::unique_ptr<af::ascir::AscIrAtt> GetAscIrAttImpl(const string &ascir_type) {
  std::string platform_name;
  GE_ASSERT_SUCCESS(ge::PlatformContext::GetInstance().GetCurrentPlatformString(platform_name),
                    "Failed to get platform info.");
  return af::ascir::AscirRegistry::GetInstance().GetIrAttImpl(platform_name, ascir_type);
}

std::unique_ptr<af::ascir::AscIrCodegen> GetAscIrCodegenImpl(const string &ascir_type) {
  std::string platform_name;
  GE_ASSERT_SUCCESS(ge::PlatformContext::GetInstance().GetCurrentPlatformString(platform_name),
                    "Failed to get platform info.");
  return af::ascir::AscirRegistry::GetInstance().GetIrCodegenImpl(platform_name, ascir_type);
}

bool IsScalarInput(const std::vector<af::Expression> &repeats) {
  return std::all_of(repeats.begin(), repeats.end(),
                     [](const af::Expression &repeat) { return ExpressEq(repeat, One); });
}

bool IsNodeSupportsVectorFunction(const af::AscNodePtr &node) {
  const auto &codegen_impl = GetAscIrCodegenImpl(node->GetType());
  GE_ASSERT_NOTNULL(codegen_impl, "Failed to get AscIrCodegen implementation.");
  return codegen_impl->IsVectorFunctionSupported(*node);
}

bool IsNodeSupportsScalarInput(const af::AscNodePtr &node, const std::vector<bool> &is_scalar_list) {
  const auto &codegen_impl = GetAscIrCodegenImpl(node->GetType());
  GE_ASSERT_NOTNULL(codegen_impl, "Failed to get AscIrCodegen implementation.");
  return codegen_impl->IsScalarInputSupported(is_scalar_list);
}

bool IsNodeSupportsInplace(const af::AscNodePtr &node) {
  const auto &codegen_impl = GetAscIrCodegenImpl(node->GetType());
  GE_ASSERT_NOTNULL(codegen_impl, "Failed to get AscIrCodegen implementation.");
  return codegen_impl->IsInplaceSupported(*node);
}

bool IsNodeSupportsAllScalar(const af::AscNodePtr &node) {
  std::vector<bool> is_scalar_list(node->GetInDataNodesSize(), true);
  return IsNodeSupportsScalarInput(node, is_scalar_list);
}

bool IsNodeSupportsScalarIfExchangeInputs(const af::AscNodePtr &node, const std::vector<bool> &is_scalar_list) {
  const auto &codegen_impl = GetAscIrCodegenImpl(node->GetType());
  GE_ASSERT_NOTNULL(codegen_impl, "Failed to get AscIrCodegen implementation.");
  return codegen_impl->IsScalarInputSupportedIfExchangeInputs(is_scalar_list);
}

/**
 * 节点是否支持隐士广播(brc inline)特性
 * 【注意】与下方的IsNodeContainsBrcInline是否包含隐士广播不同
 */
bool IsNodeSupportsBrcInline(const af::AscNodePtr &node) {
  const auto &codegen_impl = GetAscIrCodegenImpl(node->GetType());
  GE_ASSERT_NOTNULL(codegen_impl, "Failed to get AscIrCodegen implementation.");
  return codegen_impl->IsBrcInlineSupported(*node);
}

/**
 * 判断节点是否包含隐士广播(brc inline)，判断逻辑是节点只有 2 个输入，并且两个输入的 repeats 不同。
 * 【注意】包含隐士广播的节点一定支持隐士广播特性
 */
bool IsNodeContainsBrcInline(const af::AscNodePtr &node) {
  if (node->inputs().size() != 2UL) {
    return false;
  }
  const std::vector<af::Expression> &vec_strides1 = node->inputs[0].attr.vectorized_strides;
  const std::vector<af::Expression> &vec_strides2 = node->inputs[1].attr.vectorized_strides;

  const auto is_scalar = [](const std::vector<af::Expression> &strides) {
    return std::all_of(strides.begin(), strides.end(),
                       [](const af::Expression &stride) { return ExpressEq(stride, One); });
  };

  // 节点输入是 Scalar 时，认为不属于隐士广播
  if (is_scalar(vec_strides1) || is_scalar(vec_strides2)) {
    return false;
  }

  // 若节点两个输入的 vector repeats 不同，则认为包含隐士广播
  for (size_t i = 0UL; i < vec_strides1.size(); ++i) {
    if (ExpressEq(vec_strides1[i], vec_strides2[i])) {
      continue;
    }
    if (ExpressEq(vec_strides1[i], Zero) || ExpressEq(vec_strides2[i], Zero)) {
      return true;
    }
  }
  return false;
}

std::vector<ascir::TensorId> GetWorkspaceTensorIdListInOneScheduleResult(
    const ascir::FusedScheduledResult &fused_schedule_result) {
  std::vector<ascir::TensorId> tensorId;
  for (auto workspace : fused_schedule_result.workspace_nodes) {
    GE_ASSERT_NOTNULL(workspace, "fused schedule result workspace node is null");
    ascir::TensorId tId = workspace->outputs[0].attr.mem.tensor_id;
    GELOGI("Get workspace tensor id: %ld", tId);
    auto index = std::find(tensorId.begin(), tensorId.end(), tId);
    if (index == tensorId.end()) {
      tensorId.emplace_back(tId);
    }
  }
  return tensorId;
}

bool IsScalarNextNodeSupportBlkTensor(const af::AscNodePtr &node) {
  for (auto &out : node->outputs()) {
    for (auto &peer_input : out->anchor.GetPeerInDataAnchors()) {
      auto next_node = std::dynamic_pointer_cast<af::AscNode>(peer_input->GetOwnerNode());
      if (IsSupportBlkTensorInput(next_node)) {
        return true;
      }
    }
  }
  return false;
}
bool IsUbScalarLoad(const af::AscNodePtr &node) {
  return (af::ops::IsOps<af::ascir_op::Load>(node)) && (node->outputs().size() == 1);
}

bool IsLinkToBrdcst(const ascir::NodeView &node, const std::set<std::string> &brc_types) {
  if (brc_types.find(node->GetType()) != brc_types.end()) {
    return true;
  }
  if (IsOps<Store>(node) || IsOps<Output>(node) || node->GetOutDataNodesSize() == 0U) {
    return false;
  }
  for (auto &out : node->outputs()) {
    if (out == nullptr) {
      continue;
    }
    for (auto &peer_input : out->anchor.GetPeerInDataAnchors()) {
      auto next_node = std::dynamic_pointer_cast<af::AscNode>(peer_input->GetOwnerNode());
      auto next_node_is_brc_inline = GetBrcInlineIndex(next_node) == peer_input->GetIdx();
      if (IsOps<Broadcast>(next_node) || next_node_is_brc_inline || IsLinkToBrdcst(next_node, brc_types)) {
        return true;
      }
    }
  }
  return false;
}

af::ExecuteCondition GetNodeExecCondition(const af::NodePtr &node) {
  const auto &asc_node = std::dynamic_pointer_cast<af::AscNode>(node);
  return asc_node != nullptr ? asc_node->attr.sched.exec_condition : af::ExecuteCondition::kConditionInvalid;
}

bool IsNodeCacheable(const af::NodePtr &node) {
  const auto exec_condition = GetNodeExecCondition(node);
  return exec_condition != af::ExecuteCondition::kConditionInvalid && exec_condition != af::ExecuteCondition::kNoCache;
}

bool IsSingleGroup(const ascir::FusedScheduledResult &fused_schedule_result) {
  return fused_schedule_result.node_idx_to_scheduled_results.size() == 1 &&
         fused_schedule_result.node_idx_to_scheduled_results[0].size() == 1 &&
         fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups.size() == 1;
}

bool CanUseTilingKey(const ascir::FusedScheduledResult &fused_schedule_result) {
  for (const auto &schedule_result_list : fused_schedule_result.node_idx_to_scheduled_results) {
    for (const auto &schedule_result : schedule_result_list) {
      if (schedule_result.enable_group_parallel) {
        return false;
      }
    }
  }
  return true;
}

bool IsJustCubeFixpip(const ascir::FusedScheduledResult &fused_schedule_result) {
  if ((fused_schedule_result.node_idx_to_scheduled_results.size() == 1U) &&
      (fused_schedule_result.node_idx_to_scheduled_results[0].size() == 1U) &&
      (fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups.size() == 1U) &&
      fused_schedule_result.node_idx_to_scheduled_results[0][0].cube_type == ascir::CubeTemplateType::kFixpip) {
    return true;
  }
  return false;
}

bool IsCubeFusedScheduled(const ascir::FusedScheduledResult &fused_schedule_result) {
  for (auto scheduled_results : fused_schedule_result.node_idx_to_scheduled_results) {
    for (auto scheduled_result : scheduled_results) {
      if (scheduled_result.cube_type != ascir::CubeTemplateType::kDefault) {
        return true;
      }
    }
  }
  return false;
}

bool IsCubeUBFusedScheduled(const ascir::FusedScheduledResult &fused_schedule_result) {
  for (auto scheduled_results : fused_schedule_result.node_idx_to_scheduled_results) {
    for (auto scheduled_result : scheduled_results) {
      if (scheduled_result.cube_type == ascir::CubeTemplateType::kUBFuse) {
        return true;
      }
    }
  }
  return false;
}

bool HasCubeUBFusedScheduled(const ascir::FusedScheduledResult &fused_schedule_result) {
  return IsCubeUBFusedScheduled(fused_schedule_result);
}

bool IsCubeCommonFusedScheduled(const ascir::FusedScheduledResult &fused_schedule_result) {
  for (auto scheduled_results : fused_schedule_result.node_idx_to_scheduled_results) {
    for (auto scheduled_result : scheduled_results) {
      if (scheduled_result.cube_type == ascir::CubeTemplateType::kCommon) {
        return true;
      }
    }
  }
  return false;
}

bool HasCubeCommonFusedScheduled(const ascir::FusedScheduledResult &fused_schedule_result) {
  return IsCubeCommonFusedScheduled(fused_schedule_result);
}

bool IsCubeType(const ascir::ImplGraph &impl_graph) {
  for (const auto &node : impl_graph.GetAllNodes()) {
    if (node->attr.api.compute_type == af::ComputeType::kComputeCube) {
      return true;
    }
  }
  return false;
}

bool IsMatMulTypeWithBatch(const ascir::ImplGraph &impl_graph) {
  for (const auto &node : impl_graph.GetAllNodes()) {
    if ((node->GetType() == kBatchMatMul) || (node->GetType() == kBatchMatMulBias) ||
        (node->GetType() == kBatchMatMulOffset) || (node->GetType() == kBatchMatMulOffsetBias)) {
      return true;
    }
  }
  return false;
}

bool IsMatMulTypeWithBias(const ascir::ImplGraph &impl_graph) {
  for (const auto &node : impl_graph.GetAllNodes()) {
    if ((node->GetType() == kMatMulBias) || (node->GetType() == kBatchMatMulBias) ||
        (node->GetType() == kMatMulOffsetBias) || (node->GetType() == kBatchMatMulOffsetBias)) {
      return true;
    }
  }
  return false;
}

bool IsMatMulTypeWithOffsetW(const ascir::ImplGraph &impl_graph) {
  for (const auto &node : impl_graph.GetAllNodes()) {
    if ((node->GetType() == kMatMulOffset) || (node->GetType() == kBatchMatMulOffset) ||
        (node->GetType() == kMatMulOffsetBias) || (node->GetType() == kBatchMatMulOffsetBias)) {
      return true;
    }
  }
  return false;
}

bool IsConv2DTypeWithBias(const ascir::ImplGraph &impl_graph) {
  for (const auto &node : impl_graph.GetAllNodes()) {
    if ((node->GetType() == kConv2DBias) || (node->GetType() == kConv2DOffsetBias)) {
      return true;
    }
  }
  return false;
}

bool IsConv2DTypeWithOffsetW(const ascir::ImplGraph &impl_graph) {
  for (const auto &node : impl_graph.GetAllNodes()) {
    if ((node->GetType() == kConv2DOffset) || (node->GetType() == kConv2DOffsetBias)) {
      return true;
    }
  }
  return false;
}

bool IsConv2DGraphType(const ascir::ImplGraph &impl_graph) {
  for (const auto &node : impl_graph.GetAllNodes()) {
    if ((node->GetType() == kConv2DOffset) || (node->GetType() == kConv2DOffsetBias) ||
        (node->GetType() == kConv2DBias) || (node->GetType() == kConv2D)) {
      return true;
    }
  }
  return false;
}

bool IsConv2DFusedScheduled(const ascir::FusedScheduledResult &fused_schedule_result) {
  auto check_conv2d_in_impl_graphs = [](const auto &schedule_groups) {
    for (const auto &schedule_group : schedule_groups) {
      for (const auto &impl_graph : schedule_group.impl_graphs) {
        if (IsConv2DGraphType(impl_graph)) {
          return true;
        }
      }
    }
    return false;
  };

  for (const auto &scheduled_results : fused_schedule_result.node_idx_to_scheduled_results) {
    for (const auto &scheduled_result : scheduled_results) {
      if (check_conv2d_in_impl_graphs(scheduled_result.schedule_groups)) {
        return true;
      }
    }
  }
  return false;
}

bool IsCubeGroupType(const ascir::ScheduleGroup &sched_group) {
  for (const auto &impl_graph : sched_group.impl_graphs) {
    if (IsCubeType(impl_graph)) {
      return true;
    }
  }
  return false;
}

bool IsSatetyResultType(const ascir::ScheduledResult &sched_result) {
  return sched_result.cube_type == ascir::CubeTemplateType::kCommon;
}

af::Status GetCubeOutputTypeSize(const ascir::NodeView &node, uint32_t &length) {
  if (node->attr.api.compute_type == af::ComputeType::kComputeCube) {
    GE_ASSERT_TRUE(node->outputs().size() > 0U);
    for (const auto output : node->outputs()) {
      GE_ASSERT_TRUE(ge::TypeUtils::GetDataTypeLength(output->attr.dtype, length));
      return af::SUCCESS;
    }
  }
  return af::FAILED;
}

af::Status GetCubeInputNum(const ascir::NodeView &node, uint32_t &num) {
  if (node->attr.api.compute_type == af::ComputeType::kComputeCube) {
    num = node->inputs().size();
    GE_ASSERT_TRUE(num > 1U);
  }
  return af::SUCCESS;
}

af::Status ParseMatmulAttr(const ascir::NodeView &node, MatMulAttr &mm_attr_data) {
  if (node->GetType() == kMatMul) {
    GET_MATMUL_ATTRS(node, MatMul, mm_attr_data);
  } else if (node->GetType() == kMatMulBias) {
    GET_MATMUL_ATTRS(node, MatMulBias, mm_attr_data);
    mm_attr_data.is_bias = true;
  } else if (node->GetType() == kMatMulOffset) {
    GET_MATMUL_ATTRS(node, MatMulOffset, mm_attr_data);
    mm_attr_data.is_offset_w = true;
  } else if (node->GetType() == kMatMulOffsetBias) {
    GET_MATMUL_ATTRS(node, MatMulOffsetBias, mm_attr_data);
    mm_attr_data.is_bias = true;
    mm_attr_data.is_offset_w = true;
  } else if (node->GetType() == kBatchMatMul) {
    GET_BATCH_MATMUL_ATTRS(node, BatchMatMul, mm_attr_data);
    mm_attr_data.is_batch = true;
  } else if (node->GetType() == kBatchMatMulBias) {
    GET_BATCH_MATMUL_ATTRS(node, BatchMatMulBias, mm_attr_data);
    mm_attr_data.is_batch = true;
    mm_attr_data.is_bias = true;
  } else if (node->GetType() == kBatchMatMulOffset) {
    GET_BATCH_MATMUL_ATTRS(node, BatchMatMulOffset, mm_attr_data);
    mm_attr_data.is_batch = true;
    mm_attr_data.is_offset_w = true;
  } else if (node->GetType() == kBatchMatMulOffsetBias) {
    GET_BATCH_MATMUL_ATTRS(node, BatchMatMulOffsetBias, mm_attr_data);
    mm_attr_data.is_batch = true;
    mm_attr_data.is_bias = true;
    mm_attr_data.is_offset_w = true;
  } else {
    GELOGE(af::FAILED, "can't parse matmul node attr, type=%s", node->GetType().c_str());
  }
  return af::SUCCESS;
}

af::Status ParseConv2DAttr(const ascir::NodeView &node, Conv2DAttr &conv_attr_data) {
  if (node->GetType() == kConv2D) {
    GET_CONV2D_ATTRS(node, Conv2D, conv_attr_data);
  } else if (node->GetType() == kConv2DBias) {
    GET_CONV2D_ATTRS(node, Conv2DBias, conv_attr_data);
    conv_attr_data.is_bias = true;
  } else if (node->GetType() == kConv2DOffset) {
    GET_CONV2D_ATTRS(node, Conv2DOffset, conv_attr_data);
    conv_attr_data.is_offset_w = true;
  } else if (node->GetType() == kConv2DOffsetBias) {
    GET_CONV2D_ATTRS(node, Conv2DOffsetBias, conv_attr_data);
    conv_attr_data.is_bias = true;
    conv_attr_data.is_offset_w = true;
  } else {
    GELOGE(af::FAILED, "can't parse conv2d node attr, type=%s", node->GetType().c_str());
  }
  return af::SUCCESS;
}

af::Status UpdateAttGroup(ascir::ScheduledResult &scheduled_result,
                          std::function<void(af::AscGraph &)> update_graph_axis) {
  for (auto &group : scheduled_result.schedule_groups) {
    std::vector<af::AscGraph> impl_graphs_tmp;
    for (auto &impl_graph : group.impl_graphs) {
      auto new_graph_name = impl_graph.GetName() + "_for_cube";
      af::AscGraph att_graph(new_graph_name.c_str());
      att_graph.CopyFrom(impl_graph);
      update_graph_axis(att_graph);
      impl_graphs_tmp.push_back(std::move(att_graph));
    }
    group.impl_graphs.clear();
    group.impl_graphs = impl_graphs_tmp;
  }
  return ge::GRAPH_SUCCESS;
}

af::Status CreateAttResult(ascir::FusedScheduledResult &elemwise_schedule_result,
                           std::function<void(af::AscGraph &)> update_graph_axis) {
  for (auto &scheduled_results : elemwise_schedule_result.node_idx_to_scheduled_results) {
    for (auto &scheduled_result : scheduled_results) {
      GE_ASSERT_SUCCESS(UpdateAttGroup(scheduled_result, update_graph_axis));
    }
  }
  return ge::GRAPH_SUCCESS;
}

af::Status CreateCVFusionResult(ascir::FusedScheduledResult &elemwise_schedule_result) {
  if (!IsCubeFusedScheduled(elemwise_schedule_result)) {
    return af::SUCCESS;
  }
  for (auto &scheduled_results : elemwise_schedule_result.node_idx_to_scheduled_results) {
    scheduled_results.erase(
        std::remove_if(scheduled_results.begin(), scheduled_results.end(),
                       [](const ascir::ScheduledResult &result) { return ascgen_utils::IsSatetyResultType(result); }),
        scheduled_results.end());
  }
  for (auto &scheduled_results : elemwise_schedule_result.node_idx_to_scheduled_results) {
    for (auto &scheduled_result : scheduled_results) {
      scheduled_result.schedule_groups.erase(
          std::remove_if(scheduled_result.schedule_groups.begin(), scheduled_result.schedule_groups.end(),
                         [](const ascir::ScheduleGroup &group) { return ascgen_utils::IsCubeGroupType(group); }),
          scheduled_result.schedule_groups.end());
    }
  }

  auto update_graph_axis = [](af::AscGraph &graph) {
    for (auto &ax : graph.GetAllAxis()) {
      if (ax == nullptr) {
        continue;
      }
      if (ax->type == ascir::Axis::kAxisTypeTileInner) {
        GELOGI("Add tile inner axis(%s) symbol align for graph(%s)", ax->name.c_str(), graph.GetName().c_str());
        ax->align = af::Symbol("get_g_basen_basem_align()");
      }
    }
  };
  GE_ASSERT_SUCCESS(CreateAttResult(elemwise_schedule_result, update_graph_axis));
  return ge::GRAPH_SUCCESS;
}

af::Status CreateCVFusionCommonResult(ascir::FusedScheduledResult &elemwise_schedule_result) {
  if (!IsCubeFusedScheduled(elemwise_schedule_result)) {
    return af::SUCCESS;
  }
  // 删除UBFuse的ScheduledResult
  for (auto &scheduled_results : elemwise_schedule_result.node_idx_to_scheduled_results) {
    scheduled_results.erase(std::remove_if(scheduled_results.begin(), scheduled_results.end(),
                                           [](const ascir::ScheduledResult &result) {
                                             return result.cube_type == ascir::CubeTemplateType::kUBFuse;
                                           }),
                            scheduled_results.end());
  }
  // 删除ScheduleGroup 中Cube AscGraph
  for (auto &scheduled_results : elemwise_schedule_result.node_idx_to_scheduled_results) {
    for (auto &scheduled_result : scheduled_results) {
      scheduled_result.schedule_groups.erase(
          std::remove_if(scheduled_result.schedule_groups.begin(), scheduled_result.schedule_groups.end(),
                         [](const ascir::ScheduleGroup &group) { return ascgen_utils::IsCubeGroupType(group); }),
          scheduled_result.schedule_groups.end());
    }
  }
  return ge::GRAPH_SUCCESS;
}

af::Status ProcessCubeFusionResultDynamic(ascir::FusedScheduledResult &fused_result) {
  ascir::FusedScheduledResult common_schedule_result = fused_result;
  if (ascgen_utils::IsCubeUBFusedScheduled(fused_result)) {
    GE_ASSERT_SUCCESS(ascgen_utils::CreateCVFusionResult(fused_result));
  }
  if (ascgen_utils::IsCubeCommonFusedScheduled(common_schedule_result)) {
    GE_ASSERT_SUCCESS(ascgen_utils::CreateCVFusionCommonResult(common_schedule_result));
  }

  for (size_t i = 0; i < common_schedule_result.node_idx_to_scheduled_results.size(); i++) {
    auto &common_scheduled_results = common_schedule_result.node_idx_to_scheduled_results[i];
    auto &target_scheduled_results = fused_result.node_idx_to_scheduled_results[i];

    for (size_t j = 0; j < common_scheduled_results.size() && j < target_scheduled_results.size(); j++) {
      auto &common_scheduled_result = common_scheduled_results[j];
      auto &target_scheduled_result = target_scheduled_results[j];

      for (size_t k = 0; k < common_scheduled_result.schedule_groups.size(); k++) {
        for (auto &impl_graph : common_scheduled_result.schedule_groups[k].impl_graphs) {
          if (k < target_scheduled_result.schedule_groups.size()) {
            target_scheduled_result.schedule_groups[k].impl_graphs.push_back(impl_graph);
          }
        }
      }
    }
  }

  return af::SUCCESS;
}

bool IsCVFusionUBGraph(const ascir::ImplGraph &impl_graph, ascir::CubeTemplateType cv_fusion_type) {
  if ((!ascgen_utils::IsCubeType(impl_graph)) && (cv_fusion_type == ascir::CubeTemplateType::kUBFuse)) {
    return true;
  }
  return false;
}

af::Status FilterCVFusionUBResult(ascir::FusedScheduledResult &ub_schedule_result) {
  if (!IsCubeFusedScheduled(ub_schedule_result)) {
    return af::SUCCESS;
  }
  for (auto &scheduled_results : ub_schedule_result.node_idx_to_scheduled_results) {
    scheduled_results.erase(std::remove_if(scheduled_results.begin(), scheduled_results.end(),
                                           [](const ascir::ScheduledResult &result) {
                                             return result.cube_type != ascir::CubeTemplateType::kUBFuse;
                                           }),
                            scheduled_results.end());
  }
  return af::SUCCESS;
}

af::Status GetCubeInfo(const ascir::FusedScheduledResult &fused_schedule_result, bool &is_batch, bool &is_conv,
                       std::string &input_type, std::string &output_type) {
  is_batch = false;
  is_conv = false;
  auto extract_from_impl_graphs = [&is_batch, &is_conv, &input_type, &output_type](const auto &schedule_groups) {
    for (const auto &schedule_group : schedule_groups) {
      for (const auto &impl_graph : schedule_group.impl_graphs) {
        for (const auto &node : impl_graph.GetAllNodes()) {
          if (node->attr.api.compute_type != af::ComputeType::kComputeCube) {
            continue;
          }
          if (IsConv2DGraphType(impl_graph)) {
            is_batch = false;
            is_conv = true;
          } else {
            ascgen_utils::MatMulAttr mm_attr_data;
            if (ascgen_utils::ParseMatmulAttr(node, mm_attr_data) != af::SUCCESS) {
              return false;
            }
            is_batch = mm_attr_data.is_batch;
            input_type = mm_attr_data.input_dtype;
            output_type = mm_attr_data.output_dtype;

            // 如果 ParseMatmulAttr 返回的 dtype 为空，从 node 的输入输出获取
            if (input_type.empty() && !node->inputs().empty()) {
              ge::DataType dtype = node->inputs()[0]->attr.dtype;
              if (DtypeName(dtype, input_type) != af::SUCCESS) {
                GELOGE(af::FAILED, "Failed to get input dtype from node");
                return false;
              }
            }
            if (output_type.empty() && !node->outputs().empty()) {
              ge::DataType dtype = node->outputs()[0]->attr.dtype;
              if (DtypeName(dtype, output_type) != af::SUCCESS) {
                GELOGE(af::FAILED, "Failed to get output dtype from node");
                return false;
              }
            }
          }

          return true;
        }
      }
    }
    return false;
  };

  auto process_scheduled_results = [&extract_from_impl_graphs](const auto &scheduled_results) {
    for (const auto &scheduled_result : scheduled_results) {
      if (scheduled_result.cube_type != ascir::CubeTemplateType::kDefault) {
        if (extract_from_impl_graphs(scheduled_result.schedule_groups)) {
          return true;
        }
      }
    }
    return false;
  };

  for (const auto &scheduled_results : fused_schedule_result.node_idx_to_scheduled_results) {
    if (process_scheduled_results(scheduled_results)) {
      return af::SUCCESS;
    }
  }
  return af::FAILED;
}

af::Status FilterCVFusionCommonResult(ascir::FusedScheduledResult &common_schedule_result) {
  if (!IsCubeFusedScheduled(common_schedule_result)) {
    return af::SUCCESS;
  }
  for (auto &scheduled_results : common_schedule_result.node_idx_to_scheduled_results) {
    scheduled_results.erase(
        std::remove_if(scheduled_results.begin(), scheduled_results.end(),
                       [](const ascir::ScheduledResult &result) { return !ascgen_utils::IsSatetyResultType(result); }),
        scheduled_results.end());
  }
  return af::SUCCESS;
}

af::Status DtypeName(ge::DataType dtype, std::string &dtype_name) {
  static const std::string kTypeNames[] = {
      [ge::DT_FLOAT] = "float",     [ge::DT_FLOAT16] = "half",    [ge::DT_INT8] = "int8_t",
      [ge::DT_INT32] = "int32_t",   [ge::DT_UINT8] = "uint8_t",   "",
      [ge::DT_INT16] = "int16_t",   [ge::DT_UINT16] = "uint16_t", [ge::DT_UINT32] = "uint32_t",
      [ge::DT_INT64] = "int64_t",   [ge::DT_UINT64] = "uint64_t", [ge::DT_DOUBLE] = "",
      [ge::DT_BOOL] = "bool",       [ge::DT_STRING] = "",         [ge::DT_DUAL_SUB_INT8] = "",
      [ge::DT_DUAL_SUB_UINT8] = "", [ge::DT_COMPLEX64] = "",      [ge::DT_COMPLEX128] = "",
      [ge::DT_QINT8] = "",          [ge::DT_QINT16] = "",         [ge::DT_QINT32] = "",
      [ge::DT_QUINT8] = "",         [ge::DT_QUINT16] = "",        [ge::DT_RESOURCE] = "",
      [ge::DT_STRING_REF] = "",     [ge::DT_DUAL] = "",           [ge::DT_VARIANT] = "",
      [ge::DT_BF16] = "bfloat16_t", [ge::DT_UNDEFINED] = "",      [ge::DT_INT4] = "int4_t",
      [ge::DT_UINT1] = "",          [ge::DT_INT2] = "",           [ge::DT_UINT2] = "",
      [ge::DT_COMPLEX32] = "",
  };
  GE_CHK_BOOL_RET_STATUS((dtype < (sizeof(kTypeNames) / sizeof(kTypeNames[0])) && kTypeNames[dtype] != ""), af::FAILED,
                         "Unsupported data type:%d", static_cast<int32_t>(dtype));
  dtype_name = kTypeNames[dtype];
  return af::SUCCESS;
}
}  // namespace ascgen_utils
