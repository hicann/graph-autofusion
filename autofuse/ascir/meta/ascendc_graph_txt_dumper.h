/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_GRAPH_TXT_DUMPER_H
#define ASCENDC_GRAPH_TXT_DUMPER_H

#include <string>
#include <map>
#include <set>
#include <vector>
#include "ascir.h"

namespace ascir {
namespace dumper {
namespace NodeType {
constexpr const char *kData = "Data";
constexpr const char *kOutput = "Output";
constexpr const char *kWorkspace = "Workspace";
constexpr const char *kScalar = "Scalar";
constexpr const char *kStore = "Store";
}  // namespace NodeType

// 魔法数字常量
constexpr int64_t kInvalidLoopAxis = -1;
constexpr int64_t kInvalidAxisId = -1;
constexpr size_t kIndentSpaces = 2UL;           // 循环缩进空格数
constexpr size_t kPropertyIndent = 8UL;         // 属性缩进空格数 (VIEW 2)
constexpr size_t kTensorPropertyIndent = 12UL;  // tensor 属性缩进空格数 (VIEW 2)
constexpr size_t kNodeIndent = 4UL;             // 节点缩进空格数

/**
 * @brief Dtype 信息结构体
 */
struct DtypeInfo {
  const char *full_name;   // 完整类型名 (如 "float32")
  const char *short_name;  // 简写类型名 (如 "f32")
  const char *suffix;      // 后缀 (如 "32f")
};

/**
 * @brief 获取 Dtype 信息
 * @param dtype 数据类型
 * @return Dtype 信息指针，如果找不到返回 nullptr
 */
const DtypeInfo *GetDtypeInfo(ge::DataType dtype);

/**
 * @brief 获取轴类型的优先级（用于排序）
 * @param type 轴类型
 * @return 优先级值，越小越外层
 */
int32_t GetAxisTypePriority(af::Axis::Type type);

/**
 * @brief 获取轴类型对应的字符串后缀
 * @param type 轴类型
 * @return 类型字符串 (如 "TILE_OUT", "BLOCK_IN")
 */
std::string GetAxisTypeSuffix(af::Axis::Type type);

/**
 * @brief 构建 axis_id 到 axis_name 的映射
 * @param axes 所有轴列表
 * @return axis_id -> axis_name 的映射表
 */
std::map<af::AxisId, std::string> BuildAxisIdToNameMap(const std::vector<af::AxisPtr> &axes);

/**
 * @brief 构建 axis_id 到 axis_type 的映射
 * @param axes 所有轴列表
 * @return axis_id -> axis_type 的映射表
 */
std::map<int64_t, af::Axis::Type> BuildAxisIdToTypeMap(const std::vector<af::AxisPtr> &axes);

/**
 * @brief 从 tensor 类型字符串中提取 dtype
 * @param tensor_type tensor 类型字符串 (如 "f32[...]")
 * @return 完整的 dtype 名称 (如 "float32")
 */
std::string ExtractDtypeFromTensorType(const std::string &tensor_type);

/**
 * @brief 从 tensor 类型字符串中提取 axis 列表
 * @param tensor_type tensor 类型字符串 (如 "f32[...]")
 * @return axis 列表部分 (如 "[...]")
 */
std::string ExtractAxisListFromTensorType(const std::string &tensor_type);

/**
 * @brief 收集节点的输入名称列表
 * @param graph 图对象
 * @param node 节点对象
 * @return 输入名称列表
 */
std::vector<std::string> CollectInputNames(const ascir::Graph &graph, const af::AscNodePtr &node);

/**
 * @brief SSA 编号映射信息
 */
struct SSAMappingInfo {
  std::map<std::string, size_t> node_name_to_ssa_id;   // 节点名 -> SSA 编号
  std::set<std::string> data_node_names;               // Data 节点名称集合
  std::map<std::string, size_t> node_name_to_topo_id;  // 节点名 -> topo_id

  /**
   * @brief 获取节点的 SSA 编号
   * @param node_name 节点名称
   * @return SSA 编号，如果找不到返回 0
   */
  size_t GetSsaId(const std::string &node_name) const {
    auto it = node_name_to_ssa_id.find(node_name);
    return (it != node_name_to_ssa_id.end()) ? it->second : 0;
  }

  /**
   * @brief 获取节点的 topo_id
   * @param node_name 节点名称
   * @return topo_id，如果找不到返回 0
   */
  size_t GetTopoId(const std::string &node_name) const {
    auto it = node_name_to_topo_id.find(node_name);
    return (it != node_name_to_topo_id.end()) ? it->second : 0;
  }

  /**
   * @brief 判断是否为 Data 节点
   * @param node_name 节点名称
   * @return 如果是 Data 节点返回 true
   */
  bool IsDataNode(const std::string &node_name) const {
    return data_node_names.count(node_name) > 0;
  }
};

/**
 * @brief 构建 SSA 映射信息
 * @param all_nodes 所有节点列表
 * @return SSA 映射信息
 */
SSAMappingInfo BuildSSAMapping(af::AscNodeVisitor all_nodes);

/**
 * @brief 参数信息（用于输入/输出）
 */
struct ParamInfo {
  std::string name;
  std::string type;
};

/**
 * @brief 函数签名参数（按 data, workspace, output 顺序）
 */
struct FunctionParams {
  std::vector<ParamInfo> data_params;       // Data 节点
  std::vector<ParamInfo> workspace_params;  // Workspace 节点
  std::vector<ParamInfo> output_params;     // Output 节点
};

/**
 * @brief Dump 上下文，缓存重复计算的数据
 */
struct DumpContext {
  std::vector<af::AxisPtr> all_axis;
  std::vector<af::SizeVarPtr> all_size_vars;
  std::map<af::AxisId, std::string> axis_id_to_name;
  std::map<int64_t, af::Axis::Type> axis_id_to_type;
  SSAMappingInfo ssa_mapping;
  FunctionParams func_params;  // 函数参数（data, workspace, output）
};

/**
 * @brief 构建 Dump 上下文
 * @param graph 图对象
 * @return Dump 上下文
 */
DumpContext BuildDumpContext(const ascir::Graph &graph);

/**
 * @brief Tile/Block 分解树的节点
 */
struct AxisTreeNode {
  af::AxisPtr axis;
  bool is_merge;
  std::vector<AxisTreeNode> children;
};

/**
 * @brief Queue 节点信息
 */
struct QueueNodeInfo {
  size_t topo_id = 0;
  std::string node_name;
  int32_t reuse_id = -1;
  std::string size_str;       // tensor 的 vector<> 格式大小
  std::string tensor_suffix;  // 多输出节点为 "[i]"，单输出为 ""
};

/**
 * @brief Queue 信息
 */
struct QueueInfo {
  int32_t que_id = 0;
  int32_t depth = 0;
  int32_t buf_num = 0;  // queue 的 buf_num
  std::string position;
  std::vector<QueueNodeInfo> nodes;
};

/**
 * @brief Buffer 节点信息
 */
struct BufferNodeInfo {
  size_t topo_id = 0;
  std::string node_name;
  std::string size_str;       // tensor 的 vector<> 格式大小
  bool is_tmpbuf = false;     // 是否为节点的 tmpbuf
  int32_t tmpbuf_idx = 0;     // tmpbuf 的索引（仅当 is_tmpbuf=true 时有效）
  std::string tensor_suffix;  // 多输出节点为 "[i]"，单输出为 ""
};

/**
 * @brief Buffer 信息
 */
struct BufferInfo {
  int32_t buf_id = 0;
  std::vector<BufferNodeInfo> nodes;
};

/**
 * @brief 生成 VIEW 1: Loop Execution 的文本
 * @param graph 图对象
 * @param ctx Dump 上下文（缓存重复计算的数据）
 * @return VIEW 1 的文本内容
 */
std::string DumpLoopExecutionView(const ascir::Graph &graph, const DumpContext &ctx);

/**
 * @brief 生成 VIEW 2: Graph Structure 的文本
 * @param graph 图对象
 * @param ctx Dump 上下文（缓存重复计算的数据）
 * @param verbose 是否显示详细信息
 * @param is_subgraph 是否为子图
 * @return VIEW 2 的文本内容
 */
std::string DumpGraphStructureView(const ascir::Graph &graph, const DumpContext &ctx, bool verbose, bool is_subgraph);

/**
 * @brief 生成 VIEW 3: Memory Layout 的文本
 * @param graph 图对象
 * @param verbose 是否显示详细信息
 * @return VIEW 3 的文本内容
 */
std::string DumpMemoryLayoutView(const ascir::Graph &graph, bool verbose);

/**
 * @brief 生成完整的图转储文本（包含三个 VIEW）
 * @param graph 图对象
 * @param verbose 是否显示详细信息
 * @param is_subgraph 是否为子图
 * @return 完整的转储文本
 */
std::string DumpGraphText(const ascir::Graph &graph, bool verbose = false, bool is_subgraph = false);
}  // namespace dumper
}  // namespace ascir

#endif  // ASCENDC_GRAPH_TXT_DUMPER_H
