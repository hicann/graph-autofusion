/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascendc_graph_txt_dumper.h"

#include <sstream>
#include <functional>
#include <algorithm>
#include "graph/utils/type_utils.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace {
constexpr int32_t kAxisPriorityBlockOuter = 1;
constexpr int32_t kAxisPriorityBlockInner = 2;
constexpr int32_t kAxisPriorityTileOuter = 3;
constexpr int32_t kAxisPriorityTileInner = 4;
constexpr int32_t kAxisPriorityOriginal = 5;
constexpr int32_t kAxisPriorityMerged = 6;
constexpr int32_t kAxisPriorityDefault = 999;
}  // namespace
namespace ascir {
namespace dumper {
// =============================================================================
// 工具函数实现
// =============================================================================

/**
 * @brief 统一的 Dtype 映射表
 */
static const std::map<ge::DataType, DtypeInfo> kDtypeInfoMap = {
    {ge::DT_FLOAT, {"float32", "f32", "32f"}},   {ge::DT_FLOAT16, {"float16", "f16", "16f"}},
    {ge::DT_BF16, {"bfloat16", "bf16", "16f"}},  {ge::DT_INT8, {"int8_t", "i8", "8i"}},
    {ge::DT_INT16, {"int16_t", "i16", "16i"}},   {ge::DT_INT32, {"int32_t", "i32", "32i"}},
    {ge::DT_INT64, {"int64_t", "i64", "64i"}},   {ge::DT_UINT8, {"uint8_t", "u8", "8u"}},
    {ge::DT_UINT16, {"uint16_t", "u16", "16u"}}, {ge::DT_UINT32, {"uint32_t", "u32", "32u"}},
    {ge::DT_UINT64, {"uint64_t", "u64", "64u"}}, {ge::DT_BOOL, {"bool", "i1", "1i"}},
    {ge::DT_DOUBLE, {"float64", "f64", "64f"}},
};

const DtypeInfo *GetDtypeInfo(ge::DataType dtype) {
  const auto it = kDtypeInfoMap.find(dtype);
  if (it != kDtypeInfoMap.end()) {
    return &it->second;
  }
  return nullptr;
}

int32_t GetAxisTypePriority(af::Axis::Type type) {
  switch (type) {
    case af::Axis::Type::kAxisTypeBlockOuter:
      return kAxisPriorityBlockOuter;
    case af::Axis::Type::kAxisTypeBlockInner:
      return kAxisPriorityBlockInner;
    case af::Axis::Type::kAxisTypeTileOuter:
      return kAxisPriorityTileOuter;
    case af::Axis::Type::kAxisTypeTileInner:
      return kAxisPriorityTileInner;
    case af::Axis::Type::kAxisTypeOriginal:
      return kAxisPriorityOriginal;
    case af::Axis::Type::kAxisTypeMerged:
      return kAxisPriorityMerged;
    default:
      return kAxisPriorityDefault;
  }
}

std::string GetAxisTypeSuffix(af::Axis::Type type) {
  switch (type) {
    case af::Axis::Type::kAxisTypeOriginal:
      return "ORIGINAL";
    case af::Axis::Type::kAxisTypeTileOuter:
      return "TILE_OUT";
    case af::Axis::Type::kAxisTypeBlockOuter:
      return "BLOCK_OUT";
    case af::Axis::Type::kAxisTypeBlockInner:
      return "BLOCK_IN";
    case af::Axis::Type::kAxisTypeTileInner:
      return "TILE_IN";
    case af::Axis::Type::kAxisTypeMerged:
      return "MERGED";
    default:
      return "UNKNOWN";
  }
}

std::map<af::AxisId, std::string> BuildAxisIdToNameMap(const std::vector<af::AxisPtr> &axes) {
  std::map<af::AxisId, std::string> id_to_name;
  for (const auto &axis : axes) {
    id_to_name[axis->id] = axis->name;
  }
  return id_to_name;
}

std::map<int64_t, af::Axis::Type> BuildAxisIdToTypeMap(const std::vector<af::AxisPtr> &axes) {
  std::map<int64_t, af::Axis::Type> id_to_type;
  for (const auto &axis : axes) {
    id_to_type[axis->id] = axis->type;
  }
  return id_to_type;
}

/**
 * @brief 获取 DataType 的字符串表示（短名字）
 */
static std::string GetDtypeString(ge::DataType dtype) {
  const DtypeInfo *info = GetDtypeInfo(dtype);
  if (info != nullptr) {
    return info->short_name;
  }
  return ge::TypeUtils::DataTypeToSerialString(dtype);
}

/**
 * @brief 获取 tensor 类型字符串（用于函数签名）
 */
static std::string GetTensorTypeStr(const af::AscGraph &graph, const af::AscTensorAttr &attr,
                                    const std::map<af::AxisId, std::string> &axis_id_to_name) {
  (void)graph;
  std::stringstream ss;

  // 数据类型 - 使用简写类型名
  const auto dtype = attr.dtype;
  std::string dtype_str;
  const DtypeInfo *info = GetDtypeInfo(dtype);
  if (info != nullptr) {
    dtype_str = info->short_name;
  } else {
    // 使用完整类型名
    dtype_str = GetDtypeString(dtype);
  }

  ss << dtype_str << "[";

  // 形状
  for (size_t i = 0; i < attr.axis.size(); ++i) {
    if (i > 0) ss << ",";
    const auto axis_id = attr.axis[i];

    // 如果是 repeats，输出大小
    if (i < attr.repeats.size()) {
      const auto repeat = attr.repeats[i];
      if (repeat.GetExprType() == af::ExprType::kExprConstantRation) {
        int64_t val = 0;
        if (repeat.GetConstValue(val)) {
          ss << val;
        } else {
          const auto it = axis_id_to_name.find(axis_id);
          ss << (it != axis_id_to_name.end() ? it->second : "axis") << "_size";
        }
      } else {
        ss << af::SymbolicUtils::ToString(repeat);
      }
    } else {
      const auto it = axis_id_to_name.find(axis_id);
      ss << (it != axis_id_to_name.end() ? it->second : "axis") << "_size";
    }
  }

  ss << "]";
  return ss.str();
}

DumpContext BuildDumpContext(const ascir::Graph &graph) {
  DumpContext ctx;
  ctx.all_axis = graph.GetAllAxis();
  ctx.all_size_vars = graph.GetAllSizeVar();
  ctx.axis_id_to_name = BuildAxisIdToNameMap(ctx.all_axis);
  ctx.axis_id_to_type = BuildAxisIdToTypeMap(ctx.all_axis);
  ctx.ssa_mapping = BuildSSAMapping(graph.GetAllNodes());

  // 收集函数参数（data, workspace, output）
  for (auto node : graph.GetAllNodes()) {
    const auto node_type = node->GetType();
    if (node_type == NodeType::kData) {
      if (!node->outputs().empty()) {
        auto &output_attr = node->outputs()[0]->attr;
        ctx.func_params.data_params.push_back(
            {node->GetName(), GetTensorTypeStr(graph, output_attr, ctx.axis_id_to_name)});
      }
    } else if (node_type == NodeType::kWorkspace) {
      if (!node->outputs().empty()) {
        auto &output_attr = node->outputs()[0]->attr;
        ctx.func_params.workspace_params.push_back(
            {node->GetName(), GetTensorTypeStr(graph, output_attr, ctx.axis_id_to_name)});
      }
    } else if (node_type == NodeType::kOutput) {
      if (!node->outputs().empty()) {
        auto &output_attr = node->outputs()[0]->attr;
        ctx.func_params.output_params.push_back(
            {node->GetName(), GetTensorTypeStr(graph, output_attr, ctx.axis_id_to_name)});
      }
    }
  }

  return ctx;
}

std::string ExtractDtypeFromTensorType(const std::string &tensor_type) {
  // tensor_type 格式: f32[...] 或 float32[...]
  size_t pos = tensor_type.find('[');
  if (pos != std::string::npos) {
    std::string dtype = tensor_type.substr(0, pos);
    // 转换完整名称为简写
    if (dtype == "float32") return "f32";
    if (dtype == "float16") return "f16";
    if (dtype == "int32") return "i32";
    if (dtype == "int8") return "i8";
    if (dtype == "int16") return "i16";
    if (dtype == "int64") return "i64";
    if (dtype == "uint8") return "u8";
    if (dtype == "uint16") return "u16";
    if (dtype == "uint32") return "u32";
    if (dtype == "uint64") return "u64";
    if (dtype == "bfloat16") return "bf16";
    return dtype;
  }
  return tensor_type;
}

std::string ExtractAxisListFromTensorType(const std::string &tensor_type) {
  size_t pos = tensor_type.find('[');
  if (pos != std::string::npos) {
    return tensor_type.substr(pos);
  }
  return "[]";
}

std::vector<std::string> CollectInputNames(const ascir::Graph &graph, const af::AscNodePtr &node) {
  (void)graph;
  std::vector<std::string> input_names;

  for (uint32_t index = 0U; index < node->GetAllInDataAnchorsSize(); index++) {
    const auto in_anchor = node->GetInDataAnchor(static_cast<int32_t>(index));
    if (in_anchor == nullptr) {
      input_names.push_back("nil");
      continue;
    }
    const auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      input_names.push_back("nil");
    } else {
      const auto peer_name = peer_out_anchor->GetOwnerNode()->GetName();
      int32_t out_idx = peer_out_anchor->GetIdx();
      // 检查源节点是否有多个输出，如果有则显示索引
      auto peer_node = peer_out_anchor->GetOwnerNodeBarePtr();
      if (peer_node && peer_node->GetAllOutDataAnchorsSize() > 1) {
        input_names.push_back(peer_name + ".y[" + std::to_string(out_idx) + "]");
      } else {
        input_names.push_back(peer_name + ".y");
      }
    }
  }
  return input_names;
}

SSAMappingInfo BuildSSAMapping(af::AscNodeVisitor all_nodes) {
  SSAMappingInfo info;
  size_t topo_id = 0;

  for (auto node : all_nodes) {
    auto node_type = node->GetType();
    if (node_type == NodeType::kData) {
      info.data_node_names.insert(node->GetName());
    } else if (node_type != NodeType::kOutput && node_type != NodeType::kWorkspace) {
      info.node_name_to_ssa_id[node->GetName()] = topo_id + 1;
      info.node_name_to_topo_id[node->GetName()] = topo_id;
      topo_id++;
    }
  }

  return info;
}

// =============================================================================
// VIEW 1: Loop Execution 内部辅助函数
// =============================================================================

namespace {
// =============================================================================
// 向量化相关辅助函数实现
// =============================================================================

/**
 * @brief 检查向量化维度是否为广播维度
 */
bool IsBroadcastDimension(const af::Expression &stride) {
  if (stride.GetExprType() == af::ExprType::kExprConstantRation) {
    int64_t val = 0;
    if (stride.GetConstValue(val) && val == 0) {
      return true;
    }
  }
  return false;
}

/**
 * @brief 获取向量化维度的大小字符串
 */
std::string GetVectorizedDimSize(const af::AscTensorAttr &attr, af::AxisId axis_id,
                                 const std::map<af::AxisId, std::string> &axis_id_to_name) {
  // 找到 axis_id 在 attr.axis 中的位置
  size_t found_axis_idx = 0;
  for (; found_axis_idx < attr.axis.size(); ++found_axis_idx) {
    if (attr.axis[found_axis_idx] == axis_id) {
      break;
    }
  }

  // 获取对应的 repeat 值
  if (found_axis_idx < attr.repeats.size()) {
    auto repeat = attr.repeats[found_axis_idx];
    if (repeat.GetExprType() == af::ExprType::kExprConstantRation) {
      int64_t val = 0;
      if (repeat.GetConstValue(val)) {
        return std::to_string(val);
      }
      return "1";
    }
    return af::SymbolicUtils::ToString(repeat);
  }

  // 找不到对应的 repeat，使用轴名
  auto it = axis_id_to_name.find(axis_id);
  if (it != axis_id_to_name.end()) {
    return it->second + "_size";
  }
  return "1";
}

/**
 * @brief 获取 dtype 的简写后缀
 */
std::string GetDtypeSuffix(ge::DataType dtype) {
  const DtypeInfo *info = GetDtypeInfo(dtype);
  if (info != nullptr) {
    return info->short_name;
  }

  // 未知类型，生成位宽表示
  int32_t size_bytes = ge::GetSizeByDataType(dtype);
  std::string suffix = (size_bytes > 0) ? std::to_string(size_bytes * 8) : "32";

  std::string type_name = ge::TypeUtils::DataTypeToSerialString(dtype);
  if (type_name.find("UINT") != std::string::npos || type_name.find("uint") != std::string::npos) {
    suffix += "u";
  } else if (type_name.find("INT") != std::string::npos || type_name.find("int") != std::string::npos ||
             type_name.find("BOOL") != std::string::npos) {
    suffix += "i";
  } else {
    suffix += "f";
  }
  return suffix;
}

/**
 * @brief 获取向量化轴的字符串表示
 */
static std::string GetVectorizedAxesStr(const ascir::Graph &graph, const af::AscTensorAttr &attr,
                                        const std::map<af::AxisId, std::string> &axis_id_to_name) {
  (void)graph;
  if (attr.vectorized_axis.empty()) {
    return "";
  }

  std::stringstream ss;
  ss << "vector<";

  for (size_t i = 0; i < attr.vectorized_axis.size(); ++i) {
    if (i > 0) ss << "x";

    auto axis_id = attr.vectorized_axis[i];

    // 检查是否为广播维度
    bool is_broadcast = false;
    if (i < attr.vectorized_strides.size()) {
      is_broadcast = IsBroadcastDimension(attr.vectorized_strides[i]);
    }

    if (is_broadcast) {
      ss << "1";
    } else {
      ss << GetVectorizedDimSize(attr, axis_id, axis_id_to_name);
    }
  }

  ss << "x" << GetDtypeSuffix(attr.dtype) << ">";

  return ss.str();
}

/**
 * @brief 格式化输入参数列表
 */
static std::string FormatInputParams(const std::vector<std::string> &input_names, const SSAMappingInfo &ssa_info) {
  std::stringstream ss;
  for (size_t i = 0; i < input_names.size(); ++i) {
    if (i > 0) ss << ", ";
    if (input_names[i] != "nil") {
      std::string input_name = input_names[i];
      // 去掉 .y 或 .y[index] 后缀
      size_t pos = input_name.find(".y");
      if (pos != std::string::npos) {
        input_name = input_name.substr(0, pos);
      }

      // 判断是 Data 节点还是中间节点
      if (ssa_info.IsDataNode(input_name)) {
        // Data 节点，使用节点名称
        ss << "%" << input_name;
      } else {
        // 中间节点，使用 SSA 编号
        size_t ssa_id = ssa_info.GetSsaId(input_name);
        if (ssa_id > 0) {
          ss << "%" << ssa_id;
        } else {
          ss << "%" << input_name;
        }
      }
    }
  }
  return ss.str();
}

/**
 * @brief 收集并排序子图的 loop_axis
 */
static std::vector<int64_t> CollectSubgraphLoopAxes(const ascir::Graph &graph,
                                                    const std::map<int64_t, af::Axis::Type> &axis_id_to_type) {
  auto all_nodes = graph.GetAllNodes();

  // 收集所有节点的 loop_axis（去重）
  std::vector<int64_t> loop_axes_in_order;
  std::set<int64_t> seen_loop_axes;

  for (auto node : all_nodes) {
    auto node_type = node->GetType();
    if (node_type == NodeType::kData || node_type == NodeType::kOutput || node_type == NodeType::kWorkspace) {
      continue;
    }
    auto loop_axis = node->attr.sched.loop_axis;
    if (loop_axis != kInvalidLoopAxis && seen_loop_axes.find(loop_axis) == seen_loop_axes.end()) {
      seen_loop_axes.insert(loop_axis);
      loop_axes_in_order.push_back(loop_axis);
    }
  }

  // 按轴类型排序
  std::sort(loop_axes_in_order.begin(), loop_axes_in_order.end(), [&axis_id_to_type](int64_t a, int64_t b) {
    int32_t priority_a = GetAxisTypePriority(axis_id_to_type.at(a));
    int32_t priority_b = GetAxisTypePriority(axis_id_to_type.at(b));
    if (priority_a != priority_b) {
      return priority_a < priority_b;
    }
    return a < b;
  });

  return loop_axes_in_order;
}

/**
 * @brief 按 loop_axis 分组节点
 */
static std::map<int64_t, std::vector<af::AscNodePtr> > GroupNodesByLoopAxis(const ascir::Graph &graph) {
  std::map<int64_t, std::vector<af::AscNodePtr> > nodes_by_loop_axis;
  auto all_nodes = graph.GetAllNodes();

  for (auto node : all_nodes) {
    auto node_type = node->GetType();
    if (node_type == NodeType::kData || node_type == NodeType::kOutput || node_type == NodeType::kWorkspace) {
      continue;
    }
    auto loop_axis = node->attr.sched.loop_axis;
    nodes_by_loop_axis[loop_axis].push_back(node);
  }

  return nodes_by_loop_axis;
}

/**
 * @brief 输出子图中的单个节点
 */
static void DumpSubgraphNode(std::stringstream &ss, const ascir::Graph &graph, const af::AscNodePtr &node,
                             const SSAMappingInfo &ssa_info, size_t indent) {
  std::string node_name = node->GetName();
  size_t topo_id = ssa_info.GetTopoId(node_name);
  auto node_type = node->GetType();

  ss << std::string(indent, ' ') << "%" << (topo_id + 1) << " = ascir.ops." << node_type << "(";

  // 输入参数
  auto input_names = CollectInputNames(graph, node);
  ss << FormatInputParams(input_names, ssa_info);

  ss << ")";

  // 子图显示标量类型
  if (!node->outputs().empty()) {
    auto &output_attr = node->outputs()[0]->attr;
    ss << " → " << GetDtypeString(output_attr.dtype);
  }

  ss << "  # @" << node_name << " (topo_id=" << topo_id << ")" << std::endl;
}

/**
 * @brief 输出嵌套循环中的节点
 */
static void DumpNodesInLoops(std::stringstream &ss, const ascir::Graph &graph,
                             const std::vector<int64_t> &loop_axes_in_order,
                             const std::map<int64_t, std::vector<af::AscNodePtr> > &nodes_by_loop_axis,
                             const std::map<af::AxisId, std::string> &axis_id_to_name, const SSAMappingInfo &ssa_info) {
  std::set<int64_t> opened_loops;
  size_t current_depth = 0;

  for (auto axis_id : loop_axes_in_order) {
    // 打开循环
    ss << std::string(current_depth * kIndentSpaces, ' ') << "for %" << axis_id_to_name.at(axis_id) << " in "
       << axis_id_to_name.at(axis_id) << "_size {" << std::endl;
    opened_loops.insert(axis_id);
    current_depth++;

    // 输出 loop_axis = 当前轴的节点
    if (nodes_by_loop_axis.count(axis_id) > 0) {
      for (auto node : nodes_by_loop_axis.at(axis_id)) {
        DumpSubgraphNode(ss, graph, node, ssa_info, current_depth * kIndentSpaces);
      }
    }
  }

  // 闭合所有循环
  for (size_t i = 0; i < loop_axes_in_order.size(); ++i) {
    current_depth--;
    ss << std::string(current_depth * kIndentSpaces, ' ') << "}" << std::endl;
  }
}

/**
 * @brief 输出外层节点（loop_axis = kInvalidLoopAxis）
 */
static void DumpOuterNodes(std::stringstream &ss, const ascir::Graph &graph,
                           const std::map<int64_t, std::vector<af::AscNodePtr> > &nodes_by_loop_axis,
                           const SSAMappingInfo &ssa_info) {
  if (nodes_by_loop_axis.count(kInvalidLoopAxis) == 0) {
    return;
  }

  for (auto node : nodes_by_loop_axis.at(kInvalidLoopAxis)) {
    DumpSubgraphNode(ss, graph, node, ssa_info, 2);  // 2空格缩进
  }
}

/**
 * @brief 生成子图模式的循环执行视图
 */
static std::string DumpSubgraphLoopExecution(const ascir::Graph &graph,
                                             const std::map<af::AxisId, std::string> &axis_id_to_name,
                                             const std::map<int64_t, af::Axis::Type> &axis_id_to_type) {
  std::stringstream ss;

  auto all_nodes = graph.GetAllNodes();
  SSAMappingInfo ssa_info = BuildSSAMapping(all_nodes);

  // 收集并排序 loop_axis
  auto loop_axes_in_order = CollectSubgraphLoopAxes(graph, axis_id_to_type);

  // 按 loop_axis 分组节点
  auto nodes_by_loop_axis = GroupNodesByLoopAxis(graph);

  // 生成嵌套循环并输出节点
  DumpNodesInLoops(ss, graph, loop_axes_in_order, nodes_by_loop_axis, axis_id_to_name, ssa_info);

  // 输出外层节点
  DumpOuterNodes(ss, graph, nodes_by_loop_axis, ssa_info);

  return ss.str();
}
}  // namespace

// =============================================================================
// VIEW 1: Loop Execution 辅助函数实现
// =============================================================================

namespace {
/**
 * @brief 检测图是否为子图
 */
bool IsSubgraph(const std::string &graph_name) {
  return (graph_name.find("_VfSubgraph_") != std::string::npos || graph_name.find("_Subgraph_") != std::string::npos);
}

/**
 * @brief 收集所有被向量化的轴
 */
std::set<int64_t> CollectVectorizedAxes(const ascir::Graph &graph) {
  std::set<int64_t> vectorized_axes;
  auto all_nodes = graph.GetAllNodes();

  for (auto node : all_nodes) {
    auto node_type = node->GetType();
    if (node_type == NodeType::kData || node_type == NodeType::kOutput) {
      continue;
    }
    if (!node->outputs().empty()) {
      auto &output_attr = node->outputs()[0]->attr;
      for (auto axis_id : output_attr.vectorized_axis) {
        vectorized_axes.insert(axis_id);
      }
    }
  }

  return vectorized_axes;
}

/**
 * @brief 生成原始 tensor 形状的注释
 */
std::string GenerateOriginalShapesComment(const FunctionParams &params) {
  std::stringstream ss;

  ss << "# Original tensor shapes:" << std::endl;

  // Data 参数
  if (!params.data_params.empty()) {
    // 按类型分组输入
    std::map<std::string, std::vector<std::string> > inputs_by_type;
    for (const auto &input : params.data_params) {
      auto dtype = ExtractDtypeFromTensorType(input.type);
      auto axes = ExtractAxisListFromTensorType(input.type);
      std::string full_type = dtype + axes;
      inputs_by_type[full_type].push_back(input.name);
    }

    for (const auto &entry : inputs_by_type) {
      ss << "#   ";
      for (size_t i = 0; i < entry.second.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << entry.second[i];
      }
      ss << ": " << ExtractDtypeFromTensorType(entry.first) << ExtractAxisListFromTensorType(entry.first) << std::endl;
    }
  }

  // Workspace 参数
  for (const auto &param : params.workspace_params) {
    ss << "#   workspace: " << param.name << ": " << ExtractDtypeFromTensorType(param.type) << "[]" << std::endl;
  }

  // Output 参数
  for (const auto &param : params.output_params) {
    ss << "#   output: " << param.name << ": " << ExtractDtypeFromTensorType(param.type) << "[]" << std::endl;
  }

  ss << "#" << std::endl;

  return ss.str();
}

/**
 * @brief 构建 Tile/Block 分解树
 */
AxisTreeNode BuildAxisDecompositionTree(const af::AxisPtr &axis, const std::vector<af::AxisPtr> &all_axis,
                                        const std::set<int64_t> &merged_axes) {
  AxisTreeNode node;
  node.axis = axis;
  node.is_merge = (merged_axes.count(axis->id) > 0);

  // 找到所有直接从 axis 分解或合并出来的轴
  std::vector<af::AxisPtr> direct_derived;
  for (auto &target_axis : all_axis) {
    if (target_axis->id == axis->id) {
      continue;
    }
    if (!target_axis->from.empty()) {
      bool is_child = false;
      for (auto from_id : target_axis->from) {
        if (from_id == axis->id) {
          is_child = true;
          break;
        }
      }
      if (is_child) {
        direct_derived.push_back(target_axis);
      }
    }
  }

  // 按类型排序
  std::sort(direct_derived.begin(), direct_derived.end(), [](const af::AxisPtr &a, const af::AxisPtr &b) {
    return GetAxisTypePriority(a->type) < GetAxisTypePriority(b->type);
  });

  // 递归构建子树
  for (auto &derived : direct_derived) {
    node.children.push_back(BuildAxisDecompositionTree(derived, all_axis, merged_axes));
  }

  return node;
}

/**
 * @brief 输出 Tile/Block 分解树（递归）
 */
void PrintAxisDecompositionTree(std::stringstream &ss, const AxisTreeNode &node, const std::string &prefix,
                                const std::string &child_prefix) {
  ss << "#   " << prefix << node.axis->name;

  if (node.children.empty()) {
    if (node.is_merge) {
      ss << " ⋈";
    }
    ss << std::endl;
    return;
  }

  // 判断是否是合并操作
  if (node.is_merge) {
    ss << "-⋈" << std::endl;
  } else {
    ss << "-" << std::endl;
  }

  for (size_t i = 0; i < node.children.size(); ++i) {
    bool is_last = (i == node.children.size() - 1);
    std::string connector = is_last ? "└->" : "┬->";
    std::string next_prefix = child_prefix + "   " + connector;
    std::string next_child_prefix = child_prefix + (is_last ? "    " : "│   ");
    PrintAxisDecompositionTree(ss, node.children[i], next_prefix, next_child_prefix);
  }
}

/**
 * @brief 生成 Tile/Block 分解的注释
 */
std::string GenerateTileBlockDecompositionComment(const std::vector<af::AxisPtr> &all_axis) {
  std::stringstream ss;
  ss << "# Tile/Block decomposition:" << std::endl;

  // 收集所有涉及合并的轴
  std::set<int64_t> merged_axes;
  for (auto &axis : all_axis) {
    if (!axis->from.empty() && axis->from.size() > 1) {
      merged_axes.insert(axis->id);
    }
  }

  // 构建分解树并按树状图输出
  for (auto &axis : all_axis) {
    if (axis->type == af::Axis::Type::kAxisTypeOriginal) {
      AxisTreeNode root = BuildAxisDecompositionTree(axis, all_axis, merged_axes);
      if (root.children.empty()) {
        ss << "#   " << axis->name << ": original (no tiling)" << std::endl;
        continue;
      }

      PrintAxisDecompositionTree(ss, root, "", "");
    }
  }

  ss << "#" << std::endl;

  return ss.str();
}

/**
 * @brief 生成函数签名
 */
std::string GenerateFunctionSignature(const std::string &graph_name, const FunctionParams &params) {
  std::stringstream ss;

  ss << "func @" << graph_name << "(";

  // 按 data, workspace, output 顺序排布参数
  bool first = true;

  // Data 参数
  for (const auto &param : params.data_params) {
    if (!first) ss << ", ";
    ss << "%" << param.name << ": " << param.type;
    first = false;
  }

  // Workspace 参数
  for (const auto &param : params.workspace_params) {
    if (!first) ss << ", ";
    ss << "%" << param.name << ": " << param.type;
    first = false;
  }

  // Output 参数
  for (const auto &param : params.output_params) {
    if (!first) ss << ", ";
    ss << "%" << param.name << ": " << param.type;
    first = false;
  }

  // 核函数无返回值
  ss << ") {" << std::endl;

  return ss.str();
}

/**
 * @brief 检查是否有节点设置了 loop_axis
 */
bool HasLoopAxis(const ascir::Graph &graph) {
  auto all_nodes = graph.GetAllNodes();

  for (auto node : all_nodes) {
    auto node_type = node->GetType();
    if (node_type == NodeType::kData || node_type == NodeType::kOutput) {
      continue;
    }
    auto loop_axis = node->attr.sched.loop_axis;
    if (loop_axis != kInvalidLoopAxis) {
      return true;
    }
  }

  return false;
}

/**
 * @brief 收集所有需要循环的轴
 */
/**
 * @brief 收集有 loop_axis 时的循环轴
 */
static void CollectLoopAxesWithLoopAxis(const ascir::Graph &graph, const std::set<int64_t> &vectorized_axes,
                                        std::set<int64_t> &all_loop_axes) {
  auto all_nodes = graph.GetAllNodes();

  for (auto node : all_nodes) {
    auto node_type = node->GetType();
    if (node_type == NodeType::kData || node_type == NodeType::kOutput) {
      continue;
    }

    auto loop_axis = node->attr.sched.loop_axis;
    if (loop_axis != kInvalidLoopAxis) {
      auto &axis_list = node->attr.sched.axis;
      for (auto axis_id : axis_list) {
        if (axis_id == loop_axis) {
          break;  // 到 loop_axis 为止
        }
        if (vectorized_axes.count(axis_id) == 0) {
          all_loop_axes.insert(axis_id);
        }
      }
      if (vectorized_axes.count(loop_axis) == 0) {
        all_loop_axes.insert(loop_axis);
      }
    }
  }
}

/**
 * @brief 收集无 loop_axis 时的循环轴
 */
static void CollectLoopAxesWithoutLoopAxis(const ascir::Graph &graph, const std::set<int64_t> &vectorized_axes,
                                           std::set<int64_t> &all_loop_axes) {
  auto all_nodes = graph.GetAllNodes();

  for (auto node : all_nodes) {
    auto node_type = node->GetType();
    if (node_type == NodeType::kData || node_type == NodeType::kOutput) {
      continue;
    }

    auto &axis_list = node->attr.sched.axis;
    for (auto axis_id : axis_list) {
      if (vectorized_axes.count(axis_id) == 0) {
        all_loop_axes.insert(axis_id);
      }
    }
  }
}

/**
 * @brief 按轴类型优先级排序循环轴
 */
static std::vector<int64_t> SortLoopAxesByPriority(const std::set<int64_t> &all_loop_axes,
                                                   const std::map<int64_t, af::Axis::Type> &axis_id_to_type) {
  std::vector<int64_t> sorted_loop_axes(all_loop_axes.begin(), all_loop_axes.end());
  std::sort(sorted_loop_axes.begin(), sorted_loop_axes.end(), [&axis_id_to_type](int64_t a, int64_t b) {
    int32_t priority_a = GetAxisTypePriority(axis_id_to_type.at(a));
    int32_t priority_b = GetAxisTypePriority(axis_id_to_type.at(b));
    if (priority_a != priority_b) {
      return priority_a < priority_b;
    }
    return a < b;
  });
  return sorted_loop_axes;
}

/**
 * @brief 收集所有需要循环的轴
 */
std::vector<int64_t> CollectLoopAxes(const ascir::Graph &graph, const std::set<int64_t> &vectorized_axes,
                                     const std::map<int64_t, af::Axis::Type> &axis_id_to_type) {
  std::set<int64_t> all_loop_axes;
  bool has_loop_axis = HasLoopAxis(graph);
  if (has_loop_axis) {
    CollectLoopAxesWithLoopAxis(graph, vectorized_axes, all_loop_axes);
  } else {
    CollectLoopAxesWithoutLoopAxis(graph, vectorized_axes, all_loop_axes);
  }

  return SortLoopAxesByPriority(all_loop_axes, axis_id_to_type);
}

/**
 * @brief 输出 Scalar 节点
 */
void DumpScalarNode(std::stringstream &ss, const af::AscNodePtr &node, size_t indent_spaces, size_t topo_id) {
  ss << std::string(indent_spaces, ' ') << "%" << (topo_id + 1) << " = ";

  if (node->attr.ir_attr != nullptr) {
    std::string scalar_value;
    if (node->attr.ir_attr->GetAttrValue("value", scalar_value) == ge::GRAPH_SUCCESS) {
      ss << scalar_value << "f";
    } else {
      ss << "0.0f";
    }
  } else {
    ss << "0.0f";
  }

  ss << "  # @" << node->GetName() << " (topo_id=" << topo_id << ")" << std::endl;
}

/**
 * @brief 获取 ExecuteCondition 的字符串表示
 */
static std::string ExecuteConditionToString(af::ExecuteCondition condition) {
  switch (condition) {
    case af::ExecuteCondition::kNoCache:
      return "no_cache";
    case af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis:
      return "cache_block_split_fused_brc_axis";
    case af::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis:
      return "cache_block_split_origin_brc_axis";
    case af::ExecuteCondition::kConditionInvalid:
      return "invalid";
    default:
      return "unknown";
  }
}

/**
 * @brief 获取 Store 节点写入目标的注释字符串
 */
std::string GetStoreDestinationComment(const af::AscNodePtr &node) {
  if (node->GetType() != "Store" || node->GetAllOutDataAnchorsSize() == 0) {
    return "";
  }
  auto out_anchor = node->GetOutDataAnchor(0);
  if (out_anchor == nullptr) {
    return "";
  }
  for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchorsPtr()) {
    auto peer_node = peer_in_anchor->GetOwnerNodeBarePtr();
    if (peer_node == nullptr) continue;
    auto dest_type = peer_node->GetType();
    if (dest_type == NodeType::kOutput) {
      return " → output: %" + peer_node->GetName();
    }
    if (dest_type == NodeType::kWorkspace) {
      return " → workspace: %" + peer_node->GetName();
    }
  }
  return "";
}

/**
 * @brief 输出单个节点的执行语句
 */
void DumpNodeExecution(std::stringstream &ss, const ascir::Graph &graph, const af::AscNodePtr &node,
                       const SSAMappingInfo &ssa_info, const std::map<af::AxisId, std::string> &axis_id_to_name,
                       size_t indent_spaces) {
  std::string node_name = node->GetName();
  auto node_type = node->GetType();
  size_t topo_id = ssa_info.GetTopoId(node_name);

  // Scalar 节点特殊处理
  if (node_type == NodeType::kScalar) {
    DumpScalarNode(ss, node, indent_spaces, topo_id);
    return;
  }

  // ExecuteCondition 条件判断
  auto exec_condition = node->attr.sched.exec_condition;
  if (exec_condition != af::ExecuteCondition::kNoCache) {
    ss << std::string(indent_spaces, ' ') << "if (" << ExecuteConditionToString(exec_condition) << ") {" << std::endl;
    indent_spaces += kIndentSpaces;
  }

  // 非Scalar节点的通用处理
  ss << std::string(indent_spaces, ' ') << "%" << (topo_id + 1) << " = ascir.ops." << node_type << "(";

  // 输入参数
  auto input_names = CollectInputNames(graph, node);
  ss << FormatInputParams(input_names, ssa_info);
  ss << ")";

  // 类型转换（Store 节点不需要显示）
  if (!node->outputs().empty() && node_type != NodeType::kStore) {
    auto &output_attr = node->outputs()[0]->attr;
    auto vectorized_str = GetVectorizedAxesStr(graph, output_attr, axis_id_to_name);
    ss << " → " << (vectorized_str.empty() ? GetDtypeString(output_attr.dtype) : vectorized_str);
  }

  // 注释：节点名称 + topo_id + Store目标
  ss << "  # @" << node_name << " (topo_id=" << topo_id << ")";
  ss << GetStoreDestinationComment(node);
  ss << std::endl;

  // 关闭 ExecuteCondition 条件判断
  if (exec_condition != af::ExecuteCondition::kNoCache) {
    indent_spaces -= kIndentSpaces;
    ss << std::string(indent_spaces, ' ') << "}" << std::endl;
  }
}

/**
 * @brief 确定节点应该放置的循环深度
 * @param node 节点对象
 * @param has_loop_axis 图是否有 loop_axis
 * @param loop_axis_to_depth loop_axis 到深度的映射
 * @param sorted_loop_axes_size 排序后的 loop_axes 数量
 * @param current_depth 当前深度
 * @return 目标深度
 */
static size_t DetermineNodeTargetDepth(const af::AscNodePtr &node, bool has_loop_axis,
                                       const std::map<int64_t, size_t> &loop_axis_to_depth,
                                       size_t sorted_loop_axes_size, size_t current_depth) {
  auto node_type = node->GetType();
  auto loop_axis = node->attr.sched.loop_axis;
  bool is_scalar = (node_type == NodeType::kScalar);

  if (is_scalar) {
    return current_depth;
  } else if (has_loop_axis && loop_axis != kInvalidLoopAxis && loop_axis_to_depth.count(loop_axis) > 0) {
    return loop_axis_to_depth.at(loop_axis);
  } else if (!has_loop_axis) {
    return sorted_loop_axes_size;
  }
  return 0;
}

/**
 * @brief 关闭不需要的循环
 * @param ss 输出流
 * @param current_depth 当前深度（会被修改）
 * @param target_depth 目标深度
 * @param opened_loops 已打开的循环集合（会被修改）
 */
static void CloseUnneededLoops(std::stringstream &ss, size_t &current_depth, size_t target_depth,
                               std::set<int64_t> &opened_loops) {
  while (current_depth > target_depth) {
    current_depth--;
    ss << std::string(current_depth * kIndentSpaces, ' ') << "}" << std::endl;
    if (!opened_loops.empty()) {
      auto it = opened_loops.end();
      it--;
      opened_loops.erase(it);
    }
  }
}

/**
 * @brief 打开需要的循环
 * @param ss 输出流
 * @param sorted_loop_axes 排序后的 loop_axes
 * @param target_depth 目标深度
 * @param axis_id_to_name axis_id 到 name 的映射
 * @param current_depth 当前深度（会被修改）
 * @param opened_loops 已打开的循环集合（会被修改）
 */
static void OpenNeededLoops(std::stringstream &ss, const std::vector<int64_t> &sorted_loop_axes, size_t target_depth,
                            const std::map<af::AxisId, std::string> &axis_id_to_name, size_t &current_depth,
                            std::set<int64_t> &opened_loops) {
  for (auto axis_id : sorted_loop_axes) {
    if (opened_loops.count(axis_id) == 0) {
      auto depth_it = std::find(sorted_loop_axes.begin(), sorted_loop_axes.end(), axis_id);
      if (depth_it != sorted_loop_axes.end()) {
        size_t axis_depth = std::distance(sorted_loop_axes.begin(), depth_it) + 1;
        if (axis_depth <= target_depth) {
          ss << std::string(current_depth * kIndentSpaces, ' ') << "for %" << axis_id_to_name.at(axis_id) << " in "
             << axis_id_to_name.at(axis_id) << "_size {" << std::endl;
          opened_loops.insert(axis_id);
          current_depth++;
        }
      }
    }
    if (current_depth >= target_depth) {
      break;
    }
  }
}

/**
 * @brief 输出常规图（非子图）的循环执行内容
 * @param graph 图对象
 * @param axis_id_to_name axis_id 到 name 的映射
 * @param axis_id_to_type axis_id 到 type 的映射
 * @param vectorized_axes 向量化轴集合
 * @param ssa_info SSA 映射信息
 * @return 循环执行内容的字符串
 */
static std::string DumpRegularGraphLoopExecution(const ascir::Graph &graph,
                                                 const std::map<af::AxisId, std::string> &axis_id_to_name,
                                                 const std::map<int64_t, af::Axis::Type> &axis_id_to_type,
                                                 const std::set<int64_t> &vectorized_axes,
                                                 const SSAMappingInfo &ssa_info) {
  std::stringstream ss;

  auto all_nodes = graph.GetAllNodes();
  auto sorted_loop_axes = CollectLoopAxes(graph, vectorized_axes, axis_id_to_type);

  // 建立 loop_axis 到深度的映射
  std::map<int64_t, size_t> loop_axis_to_depth;
  for (size_t i = 0; i < sorted_loop_axes.size(); ++i) {
    loop_axis_to_depth[sorted_loop_axes[i]] = i + 1;
  }

  bool has_loop_axis = HasLoopAxis(graph);

  // 按拓扑序遍历节点，动态打开/关闭循环
  std::set<int64_t> opened_loops;
  size_t current_depth = 0;

  for (auto node : all_nodes) {
    auto node_type = node->GetType();
    if (node_type == NodeType::kData || node_type == NodeType::kOutput || node_type == NodeType::kWorkspace) {
      continue;
    }

    // 确定节点应该在哪个深度
    size_t target_depth =
        DetermineNodeTargetDepth(node, has_loop_axis, loop_axis_to_depth, sorted_loop_axes.size(), current_depth);

    // 关闭不需要的循环
    CloseUnneededLoops(ss, current_depth, target_depth, opened_loops);

    // 打开需要的循环
    OpenNeededLoops(ss, sorted_loop_axes, target_depth, axis_id_to_name, current_depth, opened_loops);

    // 输出节点
    DumpNodeExecution(ss, graph, node, ssa_info, axis_id_to_name, current_depth * kIndentSpaces);
  }

  // 闭合所有剩余循环
  while (current_depth > 0) {
    current_depth--;
    ss << std::string(current_depth * kIndentSpaces, ' ') << "}" << std::endl;
  }

  return ss.str();
}
}  // namespace

// =============================================================================
// VIEW 1: Loop Execution
// =============================================================================

std::string DumpLoopExecutionView(const ascir::Graph &graph, const DumpContext &ctx) {
  std::stringstream ss;

  // 获取基本信息
  std::string graph_name = graph.GetName();
  bool is_subgraph = IsSubgraph(graph_name);

  // 收集向量化轴
  std::set<int64_t> vectorized_axes = CollectVectorizedAxes(graph);

  // 生成说明性注释
  ss << GenerateOriginalShapesComment(ctx.func_params);

  // 生成 Tile/Block 分解注释（仅非子图）
  if (!is_subgraph) {
    ss << GenerateTileBlockDecompositionComment(ctx.all_axis);
  }

  // 生成函数签名（无返回值，按 data/workspace/output 顺序）
  ss << GenerateFunctionSignature(graph_name, ctx.func_params);

  // 生成函数体
  if (is_subgraph) {
    // 子图模式
    ss << DumpSubgraphLoopExecution(graph, ctx.axis_id_to_name, ctx.axis_id_to_type);
  } else {
    // 非子图模式：按照 loop_axis 分层输出节点
    ss << DumpRegularGraphLoopExecution(graph, ctx.axis_id_to_name, ctx.axis_id_to_type, vectorized_axes,
                                        ctx.ssa_mapping);
  }

  ss << "}" << std::endl;

  return ss.str();
}

// =============================================================================
// VIEW 2: Graph Structure 辅助函数
// =============================================================================

/**
 * @brief 获取 Position 的字符串表示
 */
std::string PositionToString(af::Position position) {
  switch (position) {
    case af::Position::kPositionVecIn:
      return "VECIN";
    case af::Position::kPositionVecCalc:
      return "VECCALC";
    case af::Position::kPositionVecOut:
      return "VECOUT";
    case af::Position::kPositionGM:
      return "GM";
    default:
      return "UNKNOWN";
  }
}

/**
 * @brief 获取 MemHardware 的字符串表示
 */
std::string MemHardwareToString(af::MemHardware hardware) {
  switch (hardware) {
    case af::MemHardware::kMemHardwareGM:
      return "GM";
    case af::MemHardware::kMemHardwareUB:
      return "UB";
    default:
      return "UNKNOWN";
  }
}

// =============================================================================
// VIEW 2: Graph Structure
// =============================================================================

namespace {
/**
 * @brief 输出形状信息 (axis, repeats, strides)
 */
static std::stringstream &OutputShapeStr(std::stringstream &ss, const ascir::Graph &graph,
                                         const af::AscTensorAttr &output_attr,
                                         const std::map<af::AxisId, std::string> &axis_id_to_name) {
  (void)graph;
  // 输出 axis 列表
  if (!output_attr.axis.empty()) {
    ss << std::string(kTensorPropertyIndent, ' ') << ".axis = {";
    for (size_t i = 0; i < output_attr.axis.size(); ++i) {
      if (i > 0) ss << ", ";
      auto it = axis_id_to_name.find(output_attr.axis[i]);
      ss << (it != axis_id_to_name.end() ? it->second : "unknown");
    }
    ss << "}" << std::endl;
  }

  // 输出 repeats
  if (!output_attr.repeats.empty()) {
    ss << std::string(kTensorPropertyIndent, ' ') << ".repeats = (";
    for (size_t i = 0; i < output_attr.repeats.size(); ++i) {
      if (i > 0) ss << ", ";
      ss << af::SymbolicUtils::ToString(output_attr.repeats[i]);
    }
    ss << ")" << std::endl;
  }

  // 输出 strides
  if (!output_attr.strides.empty()) {
    ss << std::string(kTensorPropertyIndent, ' ') << ".strides = (";
    for (size_t i = 0; i < output_attr.strides.size(); ++i) {
      if (i > 0) ss << ", ";
      ss << af::SymbolicUtils::ToString(output_attr.strides[i]);
    }
    ss << ")" << std::endl;
  }

  return ss;
}

/**
 * @brief 输出 vectorized 信息
 */
static std::stringstream &OutputVectorizedStr(std::stringstream &ss, const ascir::Graph &graph,
                                              const af::AscTensorAttr &output_attr,
                                              const std::map<af::AxisId, std::string> &axis_id_to_name) {
  (void)graph;
  if (!output_attr.vectorized_axis.empty()) {
    ss << std::string(kTensorPropertyIndent, ' ') << ".vectorized = {";
    for (size_t i = 0; i < output_attr.vectorized_axis.size(); ++i) {
      if (i > 0) ss << ", ";
      auto it = axis_id_to_name.find(output_attr.vectorized_axis[i]);
      std::string axis_name = (it != axis_id_to_name.end()) ? it->second : "unknown";
      ss << axis_name << ":";
      if (i < output_attr.vectorized_strides.size()) {
        ss << af::SymbolicUtils::ToString(output_attr.vectorized_strides[i]);
      }
    }
    ss << "}" << std::endl;
  }
  return ss;
}

/**
 * @brief 输出内存信息
 */
static std::stringstream &OutputMemStr(std::stringstream &ss, const af::AscTensorAttr &output_attr, bool verbose) {
  if (!verbose && (output_attr.mem.alloc_type != af::AllocType::kAllocTypeQueue) &&
      (output_attr.mem.alloc_type != af::AllocType::kAllocTypeBuffer)) {
    return ss;
  }

  std::string pos_str = PositionToString(output_attr.mem.position);
  std::string hardware_str = MemHardwareToString(output_attr.mem.hardware);

  ss << std::string(kTensorPropertyIndent, ' ') << ".mem = " << hardware_str << "[";

  // 输出 tensor_id（如果存在）
  if (output_attr.mem.tensor_id != af::kIdNone) {
    ss << "tensor_id=" << output_attr.mem.tensor_id << ", ";
  }

  if (output_attr.mem.alloc_type == af::AllocType::kAllocTypeQueue) {
    const auto &que = output_attr.que;
    ss << "que_id=" << que.id;
    if (que.buf_num > 0) {
      ss << ", buf_num=" << que.buf_num;
    }
    if (output_attr.mem.reuse_id >= 0) {
      ss << ", reuse_id=" << output_attr.mem.reuse_id;
    }
    ss << ", depth=" << que.depth << ", pos=" << pos_str;
  } else if (output_attr.mem.alloc_type == af::AllocType::kAllocTypeBuffer) {
    ss << "buf_id=" << output_attr.buf.id;
    if (output_attr.mem.reuse_id >= 0) {
      ss << ", reuse_id=" << output_attr.mem.reuse_id;
    }
    ss << ", pos=" << pos_str;
  } else {
    ss << "pos=" << pos_str;
  }

  ss << "]" << std::endl;

  return ss;
}
}  // namespace

// =============================================================================
// VIEW 2: Graph Structure 辅助函数实现
// =============================================================================

/**
 * @brief 输出 Size 变量列表
 */
void DumpSizeVars(std::stringstream &ss, const ascir::Graph &graph) {
  ss << "Sizes:" << std::endl;
  auto all_size_var = graph.GetAllSizeVar();

  for (const auto &size_var : all_size_var) {
    if (size_var->expr.GetExprType() == af::ExprType::kExprVariable) {
      ss << "  " << size_var->expr.Str().get() << ": VAR" << std::endl;
    } else {
      ss << "  " << size_var->name << ": " << af::SymbolicUtils::ToString(size_var->expr) << std::endl;
    }
  }
}

/**
 * @brief 输出 Axis 列表
 */
void DumpAxisList(std::stringstream &ss, const ascir::Graph &graph,
                  const std::map<af::AxisId, std::string> &axis_id_to_name) {
  ss << std::endl << "Axis:" << std::endl;
  auto all_axis = graph.GetAllAxis();

  for (auto &axis : all_axis) {
    ss << "  " << axis->name << "(" << axis->id << ") : ";
    ss << GetAxisTypeSuffix(axis->type);
    ss << ", size:" << af::SymbolicUtils::ToString(axis->size);

    if (!axis->from.empty()) {
      ss << ", from: {";
      for (size_t i = 0; i < axis->from.size(); ++i) {
        if (i > 0) ss << ", ";
        auto it = axis_id_to_name.find(axis->from[i]);
        ss << (it != axis_id_to_name.end() ? it->second : "unknown");
      }
      ss << "}";
    }

    ss << std::endl;
  }
}

/**
 * @brief 输出节点调度属性（axis, loop_axis）
 */
static void DumpNodeSchedProps(std::stringstream &ss, const af::AscNodePtr &node,
                               const std::map<af::AxisId, std::string> &axis_id_to_name) {
  // 输出 axis 列表
  if (!node->attr.sched.axis.empty()) {
    ss << std::string(kPropertyIndent, ' ') << ".axis = {";
    for (size_t i = 0; i < node->attr.sched.axis.size(); ++i) {
      if (i > 0) ss << ", ";
      auto it = axis_id_to_name.find(node->attr.sched.axis[i]);
      ss << (it != axis_id_to_name.end() ? it->second : "unknown");
    }
    ss << "}" << std::endl;
  }

  // 输出 loop_axis
  if (node->attr.sched.loop_axis >= 0) {
    auto it = axis_id_to_name.find(node->attr.sched.loop_axis);
    ss << std::string(kPropertyIndent, ' ') << ".loop_axis = " << (it != axis_id_to_name.end() ? it->second : "unknown")
       << std::endl;
  }

  // 输出 exec_condition（只显示非默认值）
  if (node->attr.sched.exec_condition != af::ExecuteCondition::kNoCache) {
    ss << std::string(kPropertyIndent, ' ')
       << ".exec_condition = " << ExecuteConditionToString(node->attr.sched.exec_condition) << std::endl;
  }

  const auto &tmp_buffers = node->attr.tmp_buffers;
  if (!tmp_buffers.empty()) {
    ss << std::string(kPropertyIndent, ' ') << ".tmp_buf = {";
    for (size_t i = 0; i < tmp_buffers.size(); ++i) {
      if (i > 0) ss << ", ";
      ss << "{id=" << tmp_buffers[i].id << ", size=" << af::SymbolicUtils::ToString(tmp_buffers[i].buf_desc.size)
         << ", life_cycle=" << tmp_buffers[i].buf_desc.life_time_axis_id << "}";
    }
    ss << "}" << std::endl;
  }
}

/**
 * @brief 输出节点的 ir_attr 属性
 */
static void DumpNodeIrAttr(std::stringstream &ss, const af::AscNodePtr &node) {
  auto &ir_attr = node->GetOpDesc()->GetAttrsGroup<af::AscNodeAttr>()->ir_attr;
  if (ir_attr != nullptr) {
    ascendc_ir::proto::AscIrAttrDef asc_ir_attr_def;
    (void)ir_attr->Serialize(asc_ir_attr_def);
    if (!asc_ir_attr_def.attr().empty()) {
      for (const auto &pair : asc_ir_attr_def.attr()) {
        ss << std::string(kPropertyIndent, ' ') << ".ir_attr." << pair.first << " = " << pair.second.ShortDebugString()
           << std::endl;
      }
    }
  }
}

/**
 * @brief 输出节点输入
 */
static void DumpNodeInputs(std::stringstream &ss, const ascir::Graph &graph, const af::AscNodePtr &node) {
  auto input_names = CollectInputNames(graph, node);
  if (input_names.empty()) {
    return;
  }

  // 检查是否全部为 nil
  bool all_nil = true;
  for (const auto &name : input_names) {
    if (name != "nil") {
      all_nil = false;
      break;
    }
  }
  if (all_nil) {
    return;
  }

  ss << std::string(kPropertyIndent, ' ') << ".x = {";
  for (size_t i = 0; i < input_names.size(); ++i) {
    if (i > 0) ss << ", ";
    if (input_names[i] != "nil") {
      ss << input_names[i];
    }
  }
  ss << "}" << std::endl;
}

/**
 * @brief 输出节点输出
 */
static void DumpNodeOutputs(std::stringstream &ss, const ascir::Graph &graph, const af::AscNodePtr &node,
                            const std::map<af::AxisId, std::string> &axis_id_to_name, bool verbose, bool is_subgraph) {
  size_t output_count = node->outputs().size();
  for (size_t i = 0; i < output_count; ++i) {
    auto &output_attr = node->outputs()[i]->attr;

    // 构建 tensor 名称：多输出用 y[0]，单输出用 y
    std::string tensor_name = (output_count > 1) ? ("y[" + std::to_string(i) + "]") : "y";
    ss << std::string(kPropertyIndent, ' ') << "." << tensor_name << ": " << GetDtypeString(output_attr.dtype)
       << std::endl;

    // 输出形状、向量化信息
    OutputShapeStr(ss, graph, output_attr, axis_id_to_name);
    OutputVectorizedStr(ss, graph, output_attr, axis_id_to_name);

    // mem 信息 - 仅非子图显示
    if (!is_subgraph) {
      OutputMemStr(ss, output_attr, verbose);
    }
  }
}

/**
 * @brief 输出单个节点的详细信息
 */
void DumpNodeDetails(std::stringstream &ss, const ascir::Graph &graph, const af::AscNodePtr &node, size_t idx,
                     const std::map<af::AxisId, std::string> &axis_id_to_name, bool verbose, bool is_subgraph) {
  // 节点名和类型
  ss << "  [" << idx << "] " << node->GetName() << " : ascir.ops." << node->GetType() << std::endl;

  // 输出调度属性
  DumpNodeSchedProps(ss, node, axis_id_to_name);

  // 输出 ir_attr 属性
  DumpNodeIrAttr(ss, node);

  // 输出输入
  DumpNodeInputs(ss, graph, node);

  // 输出输出
  DumpNodeOutputs(ss, graph, node, axis_id_to_name, verbose, is_subgraph);
}

std::string DumpGraphStructureView(const ascir::Graph &graph, const DumpContext &ctx, bool verbose, bool is_subgraph) {
  std::stringstream ss;

  // Header
  ss << "Graph: " << graph.GetName() << std::endl;

  // Sizes
  DumpSizeVars(ss, graph);

  // Axis
  DumpAxisList(ss, graph, ctx.axis_id_to_name);

  // Nodes
  ss << std::endl << "Nodes:" << std::endl;
  size_t idx = 0UL;

  for (auto node : graph.GetAllNodes()) {
    DumpNodeDetails(ss, graph, node, idx++, ctx.axis_id_to_name, verbose, is_subgraph);
  }

  return ss.str();
}

namespace {
void CollectQueueInfo(const af::AscNodePtr &node, size_t topo_id, std::map<int32_t, dumper::QueueInfo> &queues) {
  if (node->outputs().empty()) {
    return;
  }
  auto &output_attr = node->outputs()[0]->attr;
  auto &mem = output_attr.mem;
  if (mem.alloc_type != af::AllocType::kAllocTypeQueue) {
    return;
  }

  int32_t que_id = output_attr.que.id;
  if (queues.find(que_id) == queues.end()) {
    dumper::QueueInfo info;
    info.que_id = que_id;
    info.depth = output_attr.que.depth;
    info.buf_num = static_cast<int32_t>(output_attr.que.buf_num);
    info.position = "TPosition::" + PositionToString(mem.position);
    queues[que_id] = info;
  }
  queues[que_id].nodes.push_back({topo_id, node->GetName(), static_cast<int32_t>(mem.reuse_id), ""});
}

void CollectBufferInfo(const af::AscNodePtr &node, size_t topo_id, std::map<int32_t, dumper::BufferInfo> &buffers) {
  if (node->outputs().empty()) {
    return;
  }
  auto &output_attr = node->outputs()[0]->attr;
  auto &mem = output_attr.mem;
  if (mem.alloc_type != af::AllocType::kAllocTypeBuffer) {
    return;
  }

  int32_t buf_id = output_attr.buf.id;
  if (buffers.find(buf_id) == buffers.end()) {
    dumper::BufferInfo info;
    info.buf_id = buf_id;
    buffers[buf_id] = info;
  }
  buffers[buf_id].nodes.push_back({topo_id, node->GetName(), "", false, 0});
}

std::string GetTmpBufSizeStr(const af::TmpBufDesc &buf_desc) {
  if (buf_desc.size.GetExprType() == af::ExprType::kExprConstantRation) {
    int64_t val = 0;
    if (buf_desc.size.GetConstValue(val)) {
      return std::to_string(val);
    }
  }
  return af::SymbolicUtils::ToString(buf_desc.size);
}

void CollectTmpBufferInfo(const af::AscNodePtr &node, size_t topo_id, std::map<int32_t, dumper::BufferInfo> &buffers) {
  const auto &tmp_buffers = node->attr.tmp_buffers;
  for (size_t i = 0; i < tmp_buffers.size(); ++i) {
    const auto &tmp_buf = tmp_buffers[i];
    int32_t buf_id = static_cast<int32_t>(tmp_buf.id);
    if (buf_id < 0) {
      continue;
    }

    if (buffers.find(buf_id) == buffers.end()) {
      dumper::BufferInfo info;
      info.buf_id = buf_id;
      buffers[buf_id] = info;
    }
    std::string size_str = GetTmpBufSizeStr(tmp_buf.buf_desc);
    buffers[buf_id].nodes.push_back({topo_id, node->GetName(), size_str, true, static_cast<int32_t>(i)});
  }
}

/**
 * @brief 收集 Queue 和 Buffer 信息
 */
void CollectMemoryInfo(const ascir::Graph &graph, std::map<int32_t, dumper::QueueInfo> &queues,
                       std::map<int32_t, dumper::BufferInfo> &buffers) {
  size_t topo_id = 0;
  for (auto node : graph.GetAllNodes()) {
    auto node_type = node->GetType();
    if (node_type != NodeType::kData && node_type != NodeType::kOutput && node_type != NodeType::kWorkspace) {
      CollectQueueInfo(node, topo_id, queues);
      CollectBufferInfo(node, topo_id, buffers);
      CollectTmpBufferInfo(node, topo_id, buffers);
    }
    topo_id++;
  }
}

/**
 * @brief 输出 Queues 部分
 */
void DumpQueues(std::stringstream &ss, const std::map<int32_t, dumper::QueueInfo> &queues) {
  ss << "# Queues (" << queues.size() << " queues)" << std::endl;
  ss << std::endl;

  for (auto &entry : queues) {
    auto &info = entry.second;
    ss << "Queue " << info.que_id << " [" << info.position;
    if (info.buf_num > 0) {
      ss << ", buf_num=" << info.buf_num;
    }
    ss << ", depth=" << info.depth << "]:" << std::endl;

    // 按 reuse_id 分组
    std::map<int32_t, std::vector<dumper::QueueNodeInfo> > reuse_groups;
    for (auto &node_info : info.nodes) {
      reuse_groups[node_info.reuse_id].push_back(node_info);
    }

    // 输出每个 reuse 组
    for (auto &reuse_entry : reuse_groups) {
      auto &nodes = reuse_entry.second;
      for (size_t i = 0; i < nodes.size(); ++i) {
        ss << "  [" << nodes[i].topo_id << "] " << nodes[i].node_name << ".y" << std::endl;
      }
    }
    ss << std::endl;
  }
}

/**
 * @brief 输出 Buffers 部分
 */
void DumpBuffers(std::stringstream &ss, const std::map<int32_t, dumper::BufferInfo> &buffers) {
  ss << "# Buffers (" << buffers.size() << " buffers)" << std::endl;
  ss << std::endl;

  for (auto &entry : buffers) {
    auto &info = entry.second;
    ss << "Buffer " << info.buf_id << ":" << std::endl;

    // 按 topo_id 排序，tmpbuf 排在普通节点之后
    auto sorted_nodes = info.nodes;
    std::sort(sorted_nodes.begin(), sorted_nodes.end(),
              [](const dumper::BufferNodeInfo &a, const dumper::BufferNodeInfo &b) {
                if (a.is_tmpbuf != b.is_tmpbuf) {
                  return !a.is_tmpbuf;  // 普通 buffer 在前
                }
                if (a.topo_id != b.topo_id) {
                  return a.topo_id < b.topo_id;
                }
                return a.tmpbuf_idx < b.tmpbuf_idx;
              });

    for (auto &node_info : sorted_nodes) {
      ss << "  [" << node_info.topo_id << "] " << node_info.node_name;
      if (node_info.is_tmpbuf) {
        ss << ".tmpbuf[" << node_info.tmpbuf_idx << "]";
        // tmpbuf 显示 size
        if (!node_info.size_str.empty()) {
          ss << "  # size:" << node_info.size_str;
        }
      } else {
        ss << ".y";
      }
      ss << std::endl;
    }
    ss << std::endl;
  }
}
}  // namespace

std::string DumpMemoryLayoutView(const ascir::Graph &graph, bool verbose) {
  if (!verbose) {
    return "";
  }

  std::stringstream ss;

  // 收集内存信息
  std::map<int32_t, QueueInfo> queues;
  std::map<int32_t, BufferInfo> buffers;
  CollectMemoryInfo(graph, queues, buffers);

  // 输出 Queues 和 Buffers
  DumpQueues(ss, queues);
  DumpBuffers(ss, buffers);

  return ss.str();
}

std::string DumpGraphText(const ascir::Graph &graph, bool verbose, bool is_subgraph) {
  std::stringstream ss;

  // 构建 Dump 上下文（只获取一次数据，避免重复计算）
  DumpContext ctx = BuildDumpContext(graph);
  // Header
  ss << "================================================================================" << std::endl;
  ss << "Graph: " << graph.GetName() << std::endl;
  ss << "================================================================================" << std::endl;
  ss << std::endl;

  // VIEW 1: Loop Execution
  ss << "--------------------------------------------------------------------------------" << std::endl;
  ss << "VIEW 1: Loop Execution" << std::endl;
  ss << "--------------------------------------------------------------------------------" << std::endl;
  ss << DumpLoopExecutionView(graph, ctx);
  ss << std::endl;

  // VIEW 2: Graph Structure
  ss << "--------------------------------------------------------------------------------" << std::endl;
  ss << "VIEW 2: Graph Structure" << std::endl;
  ss << "--------------------------------------------------------------------------------" << std::endl;
  ss << DumpGraphStructureView(graph, ctx, verbose, is_subgraph);
  ss << std::endl;

  // VIEW 3: Memory Layout (子图不显示，非子图仅在 verbose=true 时显示)
  if (!is_subgraph) {
    auto memory_layout = DumpMemoryLayoutView(graph, verbose);
    if (!memory_layout.empty()) {
      ss << "--------------------------------------------------------------------------------" << std::endl;
      ss << "VIEW 3: Memory Layout" << std::endl;
      ss << "--------------------------------------------------------------------------------" << std::endl;
      ss << memory_layout;
      ss << std::endl;
    }
  }

  ss << "================================================================================" << std::endl;
  ss << "End of Dump" << std::endl;
  ss << "================================================================================" << std::endl;

  return ss.str();
}
}  // namespace dumper
}  // namespace ascir
