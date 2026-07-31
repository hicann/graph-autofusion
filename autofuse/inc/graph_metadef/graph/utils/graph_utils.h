/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_UTILS_GRAPH_UTILS_H_
#define INC_GRAPH_UTILS_GRAPH_UTILS_H_

#include <fstream>
#include <sstream>
#include <list>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "graph/anchor.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/model.h"
#include "graph/node.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/anchor_utils.h"
#include "graph/utils/cycle_detector.h"

/**
 * 图dump接口，用于把`compute_graph`对象序列化到文件，默认落盘到当前路径;
 * 如果`compute_graph`挂载了子图对象，子图对象也尝试进行落盘
 * 图的落盘行为受`DUMP_GE_GRAPH`和`DUMP_GRAPH_LEVEL`和`DUMP_GRAPH_PATH`环境变量的控制
 * DUMP_GE_GRAPH含义说明：
 * 1-全量dump
 * 2-不含有权重等数据的基础版dump
 * 3-只显示节点关系的精简版dump
 * DUMP_GRAPH_LEVEL含义说明：
 * 1-dump根图在所有阶段的图
 * 2-dump白名单阶段的图
 * 3-dump最后阶段的生成图
 * 4-dump入口阶段的生成图
 * a|b-会dump匹配到a或者b的字串的图，支持自定义dump图
 * DUMP_GRAPH_PATH含义说明：
 * 控制图的落盘的路径
 * @param compute_graph
 * @param name 用于拼接文件的名称
 */
#define GE_DUMP(compute_graph, name)                                \
  do {                                                              \
    af::GraphUtils::DumpGEGraph((compute_graph), (name));           \
    af::GraphUtils::DumpGEGraphToOnnx(*(compute_graph), (name));    \
    af::GraphUtils::DumpGEGraphToReadable((compute_graph), (name)); \
  } while (false)

namespace af {
using NodeCloneMap = std::unordered_map<const Node *, NodePtr>;

enum class DumpLevel { NO_DUMP = 0, DUMP_ALL = 1, DUMP_WITH_OUT_DATA = 2, DUMP_WITH_OUT_DESC = 3, DUMP_LEVEL_END = 4 };
enum class DumpFormat { GE_PROTO = 0, ONNX = 1, READABLE = 2 };
enum class MemType { OUTPUT_MEM, WORKSPACE_MEM };

struct MemReuseInfo {
  NodePtr node;
  MemType mem_type;
  uint32_t index;
};

enum IOType { kIn, kOut };
class NodeIndexIO {
 public:
  NodeIndexIO(const NodePtr &node, const uint32_t index, const IOType io_type)
      : node_(node), index_(index), io_type_(io_type), node_ptr_(node.get()) {
    ToValue();
  }
  NodeIndexIO(const NodePtr &node, const int32_t index, const IOType io_type)
      : node_(node), index_(static_cast<uint32_t>(index)), io_type_(io_type), node_ptr_(node.get()) {
    ToValue();
  }
  NodeIndexIO(const NodePtr &node, const int64_t index, const IOType io_type)
      : node_(node), index_(static_cast<uint32_t>(index)), io_type_(io_type), node_ptr_(node.get()) {
    ToValue();
  }
  NodeIndexIO(const Node *node, const uint32_t index, const IOType io_type)
      : node_(nullptr), index_(index), io_type_(io_type), node_ptr_(node) {
    ToValue();
  }
  ~NodeIndexIO() {}

  const std::string &ToString() const {
    return value_;
  }

  void ToValue() {
    if (node_ptr_ != nullptr) {
      value_ = node_ptr_->GetName() + ((io_type_ == kOut) ? "_out_" : "_in_") + std::to_string(index_);
    }
  }

  bool operator==(const NodeIndexIO &other) const {
    return value_ == other.value_;
  }

  NodePtr node_ = nullptr;
  uint32_t index_ = 0U;
  IOType io_type_ = kOut;
  std::string value_;
  const Node *node_ptr_ = nullptr;
};

// Symbol: nodeName, out or in and index, for example: Sqrt0trans_Cast_4_in_0. @see NodeIndexIO::ToValue()
using SymbolToAnchors = std::unordered_map<std::string, std::list<NodeIndexIO>>;
using AnchorToSymbol = std::unordered_map<std::string, std::string>;  // symbol and its peer out anchor's symbol

class GraphUtils {
 public:
  /**
   * pipline拆分场景获取`compute_graph`的`PARTITIONEDCALL`子图
   * @param compute_graph
   * @param independent_compile_subgraphs:出参，pipline拆分场景返回子图对象，非拆分场景返回`compute_graph`本身
   * @return 成功返回GRAPH_SUCCESS, 失败返回GRAPH_FAILED
   */
  static graphStatus GetIndependentCompileGraphs(const ComputeGraphPtr &compute_graph,
                                                 std::vector<ComputeGraphPtr> &independent_compile_subgraphs);

  /**
   * `src`和`dst`进行连边，`dst`作为InDataAnchorPtr, 最多允许一个对端OutDataAnchorPtr
   * @param src
   * @param dst
   * @return 如果`dst`已经有数据输入，则返回GRAPH_FAILED
   */
  static graphStatus AddEdge(const OutDataAnchorPtr &src, const InDataAnchorPtr &dst);

  static graphStatus AddEdge(const AnchorPtr &src, const AnchorPtr &dst);

  static graphStatus AddEdge(const OutControlAnchorPtr &src, const InControlAnchorPtr &dst);

  static graphStatus AddEdge(const OutDataAnchorPtr &src, const InControlAnchorPtr &dst);

  /**
   * `src`和`dst`进行断边
   * @param src
   * @param dst
   * @return 如果`src`和`dst`没有连边关系，则返回GRAPH_FAILED
   */
  static graphStatus RemoveEdge(const OutDataAnchorPtr &src, const InDataAnchorPtr &dst);

  static graphStatus RemoveEdge(const AnchorPtr &src, const AnchorPtr &dst);

  static graphStatus RemoveEdge(const OutControlAnchorPtr &src, const InControlAnchorPtr &dst);

  static graphStatus RemoveEdge(const OutDataAnchorPtr &src, const InControlAnchorPtr &dst);

  /**
   * 替换`dst`的对端`src`为`new_src`
   * @param src
   * @param dst
   * @param new_src
   * @return 替换成功返回GRAPH_SUCCESS, 替换失败返回GRAPH_FAILED
   */
  static graphStatus ReplaceEdgeSrc(const OutDataAnchorPtr &src, const InDataAnchorPtr &dst,
                                    const OutDataAnchorPtr &new_src);
  /**
   * 替换`dst`的对端`src`为`new_src`
   * @param src
   * @param dst
   * @param new_src
   * @return 替换成功返回GRAPH_SUCCESS, 替换失败返回GRAPH_FAILED
   */
  static graphStatus ReplaceEdgeSrc(const OutControlAnchorPtr &src, const InControlAnchorPtr &dst,
                                    const OutControlAnchorPtr &new_src);
  /**
   * 替换`src`的对端`dst`为`new_dst`
   * @param src
   * @param dst
   * @param new_dst
   * @return 替换成功返回GRAPH_SUCCESS, 替换失败返回GRAPH_FAILED
   */
  static graphStatus ReplaceEdgeDst(const OutDataAnchorPtr &src, const InDataAnchorPtr &dst,
                                    const InDataAnchorPtr &new_dst);
  /**
   * 替换`src`的对端`dst`为`new_dst`
   * @param src
   * @param dst
   * @param new_dst
   * @return 替换成功返回GRAPH_SUCCESS, 替换失败返回GRAPH_FAILED
   */
  static graphStatus ReplaceEdgeDst(const OutControlAnchorPtr &src, const InControlAnchorPtr &dst,
                                    const InControlAnchorPtr &new_dst);
  /**
   * 在`src`所属的node对象和`dst`所属的node对象之间插入`new_node`, 行为等价于替换`src`的对端`dst`为`new_node`的第`0`个
   * InDataAnchor, 同时替换`dst`的对端`src`为`new_node`的第`0`个OutDataAnchor
   * @param src
   * @param dst
   * @param new_node
   * @return 替换成功返回GRAPH_SUCCESS, 替换失败返回GRAPH_FAILED
   */
  static graphStatus InsertNodeBetweenDataAnchors(const OutDataAnchorPtr &src, const InDataAnchorPtr &dst,
                                                  const NodePtr &new_node);
  /**
   * 从`compute_graph`上删除`直接`或者`间接`父节点为remove_node的所有子图对象
   * @param compute_graph
   * @param remove_node
   * @return 成功返回GRAPH_SUCCESS, 失败返回GRAPH_FAILED
   */
  static graphStatus RemoveSubgraphRecursively(const ComputeGraphPtr &compute_graph, const NodePtr &remove_node);

  /**
   * 从`compute_graph`中删除算子类型为`node_type`的所有关系，包括子图关系，从属关系，作为`compute_graph`的输入，输出的关系；
   * 仅删除，不进行断边连边，不保证删除后节点前后的控制关系传递
   * @param compute_graph
   * @param node_type
   * @return 成功返回GRAPH_SUCCESS, 失败返回GRAPH_FAILED
   */
  static graphStatus RemoveNodesByTypeWithoutRelink(const ComputeGraphPtr &compute_graph, const std::string &node_type);

  /**
   * 从`compute_graph`中删除`node`对象的所有关系，包括子图关系，从属关系，作为`compute_graph`的输入，输出的关系；
   * 仅删除，不进行断边连边，不保证删除后节点前后的控制关系传递
   * @param compute_graph
   * @param node
   * @return 成功返回GRAPH_SUCCESS, 失败返回GRAPH_FAILED
   */
  static graphStatus RemoveNodeWithoutRelink(const ComputeGraphPtr &compute_graph, const NodePtr &node);

  /**
   * 从`compute_graph`中删除`nodes`对象们的所有关系，包括子图关系，从属关系，作为`compute_graph`的输入，输出的关系；
   * 仅删除，不进行断边连边，不保证删除后节点前后的控制关系传递
   * 此接口在图规模比较大的时候，比遍历`nodes`节点依次调用`RemoveNodeWithoutRelink`更高效一些
   * @param compute_graph
   * @param nodes
   * @return
   */
  static graphStatus RemoveNodesWithoutRelink(const ComputeGraphPtr &compute_graph,
                                              const std::unordered_set<NodePtr> &nodes);
  /**
   * ComputeGraph图对象的全量深拷贝接口
   * @param src_compute_graph 需要是根图对象
   * @param dst_compute_graph
   * @return
   */
  static graphStatus CopyComputeGraph(const ComputeGraphPtr &src_compute_graph, ComputeGraphPtr &dst_compute_graph);

  /**
   * ComputeGraph图对象的深拷贝接口
   * @param src_compute_graph 需要是根图对象
   * @param node_filter 节点拷贝白名单过滤器，可以通过传递此参数实现满足条件的节点的复制，不传递时代表全量拷贝
   * @param graph_filter 子图拷贝白名单过滤器，可以通过传递此参数实现满足条件的子图的复制，不传递时代表全量拷贝
   * @param attr_filter 节点上属性拷贝白名单过滤器，可以通过传递此参数实现满足条件的属性复制，不传递时代表全量拷贝
   * @param dst_compute_graph
   * @return
   */
  static graphStatus CopyComputeGraph(const ComputeGraphPtr &src_compute_graph, const NodeFilter &node_filter,
                                      const GraphFilter &graph_filter, const AttrFilter &attr_filter,
                                      ComputeGraphPtr &dst_compute_graph);

  /**
   * ComputeGraph图对象的深拷贝接口
   * @param src_compute_graph
   * @param node_filter 节点拷贝白名单过滤器，可以通过传递此参数实现满足条件的节点的复制，不传递时代表全量拷贝
   * @param graph_filter 子图拷贝白名单过滤器，可以通过传递此参数实现满足条件的子图的复制，不传递时代表全量拷贝
   * @param attr_filter 节点上属性拷贝白名单过滤器，可以通过传递此参数实现满足条件的属性复制，不传递时代表全量拷贝
   * @param dst_compute_graph
   * @param node_old_2_new 新旧节点映射关系
   * @param op_desc_old_2_new 新旧节点描述信息的映射关系
   * @param depth 子图拷贝深度, 最大支持为10
   * @return
   */
  static graphStatus CopyComputeGraph(const ComputeGraphPtr &src_compute_graph, const NodeFilter &node_filter,
                                      const GraphFilter &graph_filter, const AttrFilter &attr_filter,
                                      ComputeGraphPtr &dst_compute_graph,
                                      std::map<ConstNodePtr, NodePtr> &node_old_2_new,
                                      std::map<ConstOpDescPtr, OpDescPtr> &op_desc_old_2_new, const int32_t depth);

  /**
   * ComputeGraph图对象的深拷贝接口
   * @param src_compute_graph
   * @param dst_compute_graph
   * @param node_old_2_new 新旧节点映射关系
   * @param op_desc_old_2_new 新旧节点描述信息的映射关系
   * @param depth 子图拷贝深度, 最大支持为10
   * @return
   */
  static graphStatus CopyComputeGraph(const ComputeGraphPtr &src_compute_graph, ComputeGraphPtr &dst_compute_graph,
                                      std::map<ConstNodePtr, NodePtr> &node_old_2_new,
                                      std::map<ConstOpDescPtr, OpDescPtr> &op_desc_old_2_new, const int32_t depth);
  /**
   * 拷贝OpDesc对象，跟`CopyOpDesc`方法的区别是`CloneOpDesc`的拷贝内容精简一些
   * 注意：此接口性能较差，且拷贝内容存在丢失，为了考虑兼容目前保留此接口，推荐直接使用OpDesc
   * 的拷贝构造函数来实现拷贝，拷贝构造函数性能好且内容不存在丢失
   * @param org_op_desc
   * @return
   */
  static OpDescPtr CloneOpDesc(const ConstOpDescPtr &org_op_desc);
  /**
   * 拷贝OpDesc对象，跟`CloneOpDesc`方法的区别是`CopyOpDesc`的拷贝内容多一些，
   * 包括函数指针成员等
   * 注意：此接口性能较差，且拷贝内容存在丢失，为了考虑兼容目前保留此接口，推荐直接使用OpDesc
   * 的拷贝构造函数来实现拷贝，拷贝构造函数性能好且内容不存在丢失
   * @param org_op_desc
   * @return
   */
  static OpDescPtr CopyOpDesc(const ConstOpDescPtr &org_op_desc);
  /**
   * `CopyOpDesc`的重载接口
   * @param org_op_desc
   * @param attr_filter 节点上属性拷贝白名单过滤器，可以通过传递此参数实现满足条件的属性复制，不传递时代表全量拷贝
   * @return
   */
  static OpDescPtr CopyOpDesc(const ConstOpDescPtr &org_op_desc, const AttrFilter &attr_filter);

  /**
   * 接口行为是在数据`src`锚点所属的`src_node`节点和数据`dsts`锚点所属的`dst_node`节点们之间插入一个`insert_node`节点,
   * 默认是`insert_node`的`0`号数据输入锚点和`0`号输出数据锚点参与连边，`insert_node`插入之后, `src_node`和`insert_node`
   * 作为一个整体与原来的`src_node`具备等价的控制和数据关系
   * `insert_node`继承`src_node`的用户属性
   * @param src 源数据输出锚点
   * @param dsts 源数据输出锚点连接的目的数据输入锚点，使用vector的原因是存在一个源锚点给到多个目的锚点的情况
   * @param insert_node 表示要插入的节点
   * @param input_index 表示插入节点的哪个数据输入锚点要跟src相连，如果不传递，默认取0
   * @param output_index 表示插入节点的哪个数据输出锚点要跟dsts依次相连，如果不传递，默认取0
   * @return 如果插入成功返回GRAPH_SUCCESS，失败返回GRAPH_FAILED
   */
  static graphStatus InsertNodeAfter(const OutDataAnchorPtr &src, const std::vector<InDataAnchorPtr> &dsts,
                                     const NodePtr &insert_node, const uint32_t input_index = 0U,
                                     const uint32_t output_index = 0U);

  /**
   * 接口行为是通过insert_op在图中生成一个insert_node节点，
   * 在数据`src`锚点所属的`src_node`节点和数据`dsts`锚点所属的`dst_node`节点们之间插入一个`insert_node`节点,
   * 默认是`insert_node`的`0`号数据输入锚点和`0`号输出数据锚点参与连边
   * `insert_node`插入之后, `src_node`和`insert_node`
   * 作为一个整体与原来的`src_node`具备等价的控制和数据关系
   * `insert_node`继承`src_node`的用户属性
   * @param src 源数据输出锚点
   * @param dsts 源数据输出锚点连接的目的数据输入锚点，使用vector的原因是存在一个源锚点给到多个目的锚点的情况
   * @param insert_op 表示要插入的opDesc，需要用其在src_node的图上生成一个node
   * @param input_index 表示插入节点的哪个数据输入锚点要跟src相连，如果不传递，默认取0
   * @param output_index 表示插入节点的哪个数据输出锚点要跟dsts依次相连，如果不传递，默认取0
   * @return 如果插入成功返回insert_node，失败返回nullptr
   */
  static NodePtr InsertNodeAfter(const OutDataAnchorPtr &src, const std::vector<InDataAnchorPtr> &dsts,
                                 const OpDescPtr &insert_op, const uint32_t input_index = 0U,
                                 const uint32_t output_index = 0U);

  /**
   * 接口行为是在数据`dst`锚点所属的`dst_node`节点和其对端`src_node`节点之间插入一个`insert_node`节点,
   * 默认是`insert_node`的`0`号数据输入锚点和`0`号数据输出数据锚点参与连边，`insert_node`插入之后,
   * `dst_node`和`insert_node`作为一个整体与原来的`dst_node`具备等价的控制和数据关系
   * `insert_node`继承`dst_node`的用户属性
   * @param dst 目的数据输入锚点
   * @param insert_node 表示要插入的节点
   * @param input_index 表示插入节点的哪个数据输入锚点要跟dst的对端src锚点相连，如果不传递，默认取0
   * @param output_index 表示插入节点的哪个数据输出锚点要跟dst相连，如果不传递，默认取0
   * @return 如果插入成功返回GRAPH_SUCCESS，失败返回GRAPH_FAILED
   */
  static graphStatus InsertNodeBefore(const InDataAnchorPtr &dst, const NodePtr &insert_node,
                                      const uint32_t input_index = 0U, const uint32_t output_index = 0U);

  /**
   * 接口行为是通过insert_op在图中生成一个insert_node节点，
   * 在数据`dst`锚点所属的`dst_node`节点和其对端`src_node`节点之间插入一个`insert_node`节点,
   * 默认是`insert_node`的`0`号数据输入锚点和`0`号数据输出数据锚点参与连边，`insert_node`插入之后,
   * `dst_node`和`insert_node`作为一个整体与原来的`dst_node`具备等价的控制和数据关系
   * `insert_node`继承`dst_node`的用户属性
   * @param dst 目的数据输入锚点
   * @param insert_op 表示要插入的opDesc，需要用其在src_node的图上生成一个node
   * @param input_index 表示插入节点的哪个数据输入锚点要跟dst的对端src锚点相连，如果不传递，默认取0
   * @param output_index 表示插入节点的哪个数据输出锚点要跟dst相连，如果不传递，默认取0
   * @return 如果插入成功返回insert_node，失败返回nullptr
   */
  static NodePtr InsertNodeBefore(const InDataAnchorPtr &dst, const OpDescPtr &insert_op,
                                  const uint32_t input_index = 0U, const uint32_t output_index = 0U);

  /**
   * 从`compute_graph`智能指针管理的图对象的包含的nodes列表中删除`node`节点，仅仅是删除节点，
   * 不包含对node的断边和重新连边等操作
   * @param compute_graph
   * @param node
   * @return 如果删除成功返回GRAPH_SUCCESS，失败返回GRAPH_FAILED
   */
  static graphStatus RemoveJustNode(const ComputeGraphPtr compute_graph, const NodePtr &node);

  /**
   * `RemoveJustNode`的批量删除接口，节点个数较多的时候，调用批量删除接口的性能比遍历`nodes`节点依次调用`RemoveJustNode`更高效一些
   * @param compute_graph
   * @param nodes
   * @return
   */
  static graphStatus RemoveJustNodes(const ComputeGraphPtr &compute_graph, const std::unordered_set<NodePtr> &nodes);

  /**
   * 从`compute_graph`图对象的包含的nodes列表中删除`node`节点，仅仅是删除节点，不包含对`node`的断边和重新连边等操作
   * @param compute_graph
   * @param node
   * @return 如果删除成功返回GRAPH_SUCCESS，失败返回GRAPH_FAILED
   */
  static graphStatus RemoveJustNode(ComputeGraph &compute_graph, const NodePtr &node);

  /**
   * 记录`original_nodes`的原始name到node上
   * @param original_nodes
   * @param node
   */
  static void RecordOriginalNames(const std::vector<NodePtr> original_nodes, const NodePtr &node);
  /**
   * 记录`names_tmp`中的字段到node上
   * @param original_nodes
   * @param node
   */
  static void RecordOriginalNames(std::vector<std::string> names_tmp, const NodePtr &node);
  /**
   * 图dump接口，用于把`graph`对象序列化到文件，默认落盘到当前路径
   * 接口内部受DUMP_GE_GRAPH/DUMP_GRAPH_LEVEL环境变量控制，除非is_always_dump为true
   * @param graph
   * @param suffix 用于拼接文件的名称
   * @param is_always_dump 如果值为true，则接口行为不受环境变量约束
   * @param user_graph_name 用于指定落盘的文件名和文件路径
   */
  static void DumpGEGraph(const ComputeGraphPtr &graph, const std::string &suffix, const bool is_always_dump = false,
                          const std::string &user_graph_name = "");
  /**
   * 图dump接口，用于把`graph`对象序列化到文件，落盘到`path`指定的路径, 文件名将在函数内生成
   * @param graph
   * @param path 指定落盘的路径
   * @param suffix 用于拼接文件的名称
   */
  static void DumpGEGrph(const ComputeGraphPtr &graph, const std::string &path, const std::string &suffix);
  /**
   * 图dump接口，用于把`graph`对象序列化到文件，落盘到`file_path`，文件名由file_path指定
   * @param graph
   * @param file_path 路径+文件名
   * @param dump_level DUMP_GE_GRAPH环境变量以函数入参的表达
   * @return
   */
  static graphStatus DumpGEGraphByPath(const ComputeGraphPtr &graph, const std::string &file_path,
                                       const DumpLevel dump_level);
  static graphStatus DumpGEGraphByPath(const ComputeGraphPtr &graph, const std::string &file_path,
                                       const int64_t dump_level);
  /**
   * 从`file`文件反序列化得到`compute_graph`的图对象
   * @param file
   * @param compute_graph
   * @return
   */
  static bool LoadGEGraph(const char_t *const file, af::ComputeGraph &compute_graph);
  /**
   * 从`file`文件反序列化得到`compute_graph`的智能指针对象
   * @param file
   * @param compute_graph
   * @return
   */
  static bool LoadGEGraph(const char_t *const file, af::ComputeGraphPtr &compute_graph);

  static graphStatus GenDumpOnnxFileName(const ComputeGraphPtr &compute_graph, const std::string &suffix,
                                         std::string &real_path_name);

  static graphStatus GenDumpTxtFileName(const ComputeGraphPtr &compute_graph, const std::string &suffix,
                                        const std::string &user_graph_name, std::string &real_path_name);

  /**
   * @brief 生成文件绝对路径
   * @param compute_graph Dump图
   * @param suffix 用于拼接文件名
   * @param user_graph_name 用于指定落盘的文件名和文件路径
   * @param [out] real_path_name 生成的文件绝对路径
   * @return af::SUCCESS：成功，其他：失败
   */
  static graphStatus GenDumpReadableTxtFileName(const ComputeGraphPtr &compute_graph, const std::string &suffix,
                                                const std::string &user_graph_name, std::string &real_path_name);

  /**
   * 图dump接口，用于把`graph`对象按照onnx的格式序列化到文件，默认落盘到当前路径
   * 该接口受DUMP_GRAPH_LEVEL/DUMP_GE_GRAPH环境变量控制
   * @param compute_graph
   * @param suffix 用于拼接文件名
   */
  static void DumpGEGraphToOnnx(const ComputeGraph &compute_graph, const std::string &suffix);

  /**
   * 图dump接口，用于把`graph`对象按照onnx的格式序列化到文件，默认落盘到当前路径
   * 该接口受DUMP_GRAPH_LEVEL/DUMP_GE_GRAPH环境变量控制, 除非is_always_dump为true
   * @param compute_graph
   * @param suffix 用于拼接文件名
   * @param is_always_dump 如果值为true，则接口行为不受环境变量约束
   */
  static void DumpGEGraphToOnnx(const ComputeGraph &compute_graph, const std::string &suffix, bool is_always_dump);

  /**
   * 图dump接口，用于把`graph`对象按照onnx的格式序列化到文件，默认落盘到当前路径
   * 该接口不受DUMP_GRAPH_LEVEL环境变量控制，调用就会dump
   * @param compute_graph
   * @param suffix 用于拼接文件名
   * @param content_level DUMP_GE_GRAPH环境变量表示的级别，表示dump file内容的级别。取值由低到高，内容逐渐简化
   *                      1：包含连边关系和数据信息的全量dump。
   *                      2：不含有权重等数据的基本版dump。
   *                      3：只显示节点关系的精简版dump。
   */
  static void DumpGEGraphToOnnxByContentLevel(const ComputeGraph &compute_graph, const std::string &suffix,
                                              DumpLevel content_level);
  /**
   * 图dump接口，用于把`graph`对象按照onnx的格式序列化到文件，默认落盘到`path`路径
   * @param compute_graph
   * @param path 路径名
   * @param suffix 拼接的文件名
   */
  static void DumpGrphToOnnx(const ComputeGraph &compute_graph, const std::string &path, const std::string &suffix);

  static bool ReadProtoFromTextFile(const char_t *const file, google::protobuf::Message *const proto);

  static graphStatus WriteProtoToOStream(const google::protobuf::Message &proto, std::ostream &o_stream);

  /**
   * @brief 将Readable Dump写入输出流
   * @param readable_ss 字符串流
   * @param o_stream 输出流
   * @return af::SUCCESS：成功，其他：失败
   */
  static graphStatus WriteReadableDumpToOStream(const std::stringstream &readable_ss, std::ostream &o_stream);

  static void WriteProtoToTextFile(const google::protobuf::Message &proto, const char_t *const real_path);

  /**
   * @brief 将Readable Dump以txt格式输出到制定路径
   * @param readable_ss 字符串流
   * @param real_path 输出路径
   */
  static void WriteReadableDumpToTextFile(const std::stringstream &readable_ss, const char_t *const real_path);

  /**
   * 图dump接口，用于把`graph`对象进行可读性转换后存储到文件，默认落盘到当前路径
   * 接口内部受DUMP_GE_GRAPH/DUMP_GRAPH_FORMAT环境变量控制，除非is_always_dump为true
   * @param graph Dump图
   * @param suffix 用于拼接文件的名称
   * @param is_always_dump 如果值为true，则接口行为不受环境变量约束
   * @param user_graph_name 用于指定落盘的文件名和文件路径
   */
  static void DumpGEGraphToReadable(const ComputeGraphPtr &graph, const std::string &suffix,
                                    const bool is_always_dump = false, const std::string &user_graph_name = "");

  static graphStatus AppendInputNode(const ComputeGraphPtr &graph, const NodePtr &node);

  /**
   * 孤立`node`, 根据`io_map`完成node的输入输出数据边的重连；同时会添加必要的控制边保证`node`的所有输入节点
   * 均在`node`的输出节点之前执行
   * @param node
   * @param io_map 把第`io_map[i]`个输入的对端输出，连接到第`i`个输出的对端输入。因此`io_map`的元素个数应该与
   * `node`的输出锚点的个数相等，如果`io_map[i]`小于0，则仅断开第`i`个输出锚点到对端的所有连边
   * @return
   */
  static graphStatus IsolateNode(const NodePtr &node, const std::initializer_list<int32_t> &io_map);
  static graphStatus IsolateNode(const NodePtr &node, const std::vector<int32_t> &io_map);
  /**
   * `node`应该是单输入单输出的节点，接口行为等价于`IsolateNode(node, {0})`
   * @param node
   * @return
   */
  static graphStatus IsolateNodeOneIO(const NodePtr &node);

  /**
   * 此接口对数据关系的处理与`ReplaceNodeDataAnchors`的处理行为一致， 在此基础上，
   * 复制了`old_node`的所有控制关系到`new_node`上，这也是要注意的一点：
   * `数据`关系是`移动`操作，`控制`关系是`复制`操作
   * @param new_node
   * @param old_node
   * @param inputs_map 用于指导输入数据锚点的替换，注意元素个数不应该超过`new_node`的输入锚点总个数
   * @param outputs_map 用于指导输出锚点的替换，注意元素个数不应该超过`new_node`的输出锚点总个数
   * @return 如果替换成功返回GRAPH_SUCCESS，失败返回GRAPH_FAILED
   */
  static graphStatus ReplaceNodeAnchors(const NodePtr &new_node, const NodePtr &old_node,
                                        const std::initializer_list<int32_t> inputs_map,
                                        const std::initializer_list<int32_t> outputs_map);

  /**
   * `ReplaceNodeAnchors`的重载接口
   * @param new_node
   * @param old_node
   * @param inputs_map
   * @param outputs_map
   * @return
   */
  static graphStatus ReplaceNodeAnchors(const NodePtr &new_node, const NodePtr &old_node,
                                        const std::vector<int32_t> &inputs_map,
                                        const std::vector<int32_t> &outputs_map);

  /**
   * 接口行为是根据`inputs_map`和`outputs_map`把`old_node`上的数据关系`移动`到`new_node`上；具体操作是
   * 把`old_node`的第`inputs_map[i]`/`outputs_map[i]`个数据锚点的数据关系替换到`new_node`的第`i`个
   * 数据锚点上, `i`的取值范围是[0, `inputs_map`/`outputs_map`的元素个数）; 如果`inputs_map[i]`/`outputs_map[i]`
   * 的值小于0或者不在`old_node`的锚点索引范围之内，那么`new_node`的第`i`个数据锚点的数据关系保持原样
   * @param new_node
   * @param old_node
   * @param inputs_map 用于指导输入数据锚点的替换，注意元素个数不应该超过`new_node`的输入锚点总个数
   * @param outputs_map 用于指导输出锚点的替换，注意元素个数不应该超过`new_node`的输出锚点总个数
   * @return
   */
  static graphStatus ReplaceNodeDataAnchors(const NodePtr &new_node, const NodePtr &old_node,
                                            const std::initializer_list<int32_t> inputs_map,
                                            const std::initializer_list<int32_t> outputs_map);

  static graphStatus ReplaceNodeDataAnchors(const NodePtr &new_node, const NodePtr &old_node,
                                            const std::vector<int32_t> &inputs_map,
                                            const std::vector<int32_t> &outputs_map);

  /**
   * 将`old_nodes`替换为`new_nodes`，并完成数据边的处理。
   * ！！！注意：本接口不做成环校验！！！
   * new_nodes需要跟old_nodes处于相同的图对象,接口内部并不会将old_nodes从图里面清除
   * 按照在`vector`中的顺序，依次将所有`nodes`的输入/输出`index`排列，生成`vector`中全量`nodes`的`indexes`，
   *
   * `inputs_map`、`outputs_map`基于上述`indexes`工作，描述如何将`old_nodes`的数据输入、数据输出映射到`new_nodes`上。
   * 映射规则为：
   * 把`old_node`的第`inputs_map[i]`/`outputs_map[i]`个数据锚点的数据关系替换到`new_nodes`的第`i`个数据锚点上。
   * `i`的取值范围是[0, `inputs_map`/`outputs_map`的元素个数）;
   * 如果`inputs_map[i]`/`outputs_map[i]`的值小于0，那么`new_nodes`的第`i`个数据锚点的数据关系保持原样；
   * 如果`inputs_map[i]`/`outputs_map[i]`的值超出`old_nodes`的索引范围，函数报错。
   */
  static graphStatus ReplaceNodesDataAnchors(const std::vector<NodePtr> &new_nodes,
                                             const std::vector<NodePtr> &old_nodes,
                                             const std::vector<int32_t> &inputs_map,
                                             const std::vector<int32_t> &outputs_map);

  /**
   * 将`old_nodes`替换为`new_nodes`，并完成输入数据边的处理。
   * ！！！注意：本接口不做成环校验！！！
   * new_nodes需要跟old_nodes处于相同的图对象,接口内部并不会将old_nodes从图里面清除
   * 按照在`vector`中的顺序，依次将所有`nodes`的输入`index`排列，生成`vector`中全量`nodes`的`indexes`，
   *
   * `inputs_map`基于上述`indexes`工作，描述如何将`old_nodes`的数据输入映射到`new_nodes`上。
   * 映射规则为：
   * 把`old_node`的第`inputs_map[i]`个数据锚点的数据关系替换到`new_nodes`的第`i`个数据锚点上。
   * `i`的取值范围是[0, `inputs_map`的元素个数）;
   * 如果`inputs_map[i]`的值小于0，那么`new_nodes`的第`i`个数据锚点的数据关系保持原样；
   * 如果`inputs_map[i]`的值超出`old_nodes`的索引范围，函数报错。
   */
  static graphStatus ReplaceNodesInDataAnchors(const std::vector<NodePtr> &new_nodes,
                                               const std::vector<NodePtr> &old_nodes,
                                               const std::vector<int32_t> &inputs_map);

  /**
   * 将`old_nodes`替换为`new_nodes`，并完成输出数据边的处理。
   * ！！！注意：本接口不做成环校验！！！
   * new_nodes需要跟old_nodes处于相同的图对象,接口内部并不会将old_nodes从图里面清除
   * 按照在`vector`中的顺序，依次将所有`nodes`的输出`index`排列，生成`vector`中全量`nodes`的`indexes`，
   *
   * `outputs_map`基于上述`indexes`工作，描述如何将`old_nodes`的数据输出映射到`new_nodes`上。
   * 映射规则为：
   * 把`old_node`的第`outputs_map[i]`个数据锚点的数据关系替换到`new_nodes`的第`i`个数据锚点上。
   * `i`的取值范围是[0, `outputs_map`的元素个数）;
   * 如果`outputs_map[i]`的值小于0，那么`new_nodes`的第`i`个数据锚点的数据关系保持原样；
   * 如果`outputs_map[i]`的值超出`old_nodes`的索引范围，函数报错。
   */
  static graphStatus ReplaceNodesOutDataAnchors(const std::vector<NodePtr> &new_nodes,
                                                const std::vector<NodePtr> &old_nodes,
                                                const std::vector<int32_t> &outputs_map);

  /**
   * 将`old_nodes`的数据输入拷贝到`new_nodes`上
   * ！！！注意：本接口不做成环校验！！！
   * new_nodes需要跟old_nodes处于相同的图对象,接口内部并不会将old_nodes从图里面清除
   * 按照在`vector`中的顺序，依次将所有`nodes`的输入`index`排列，生成`vector`中全量`nodes`的`indexes`，
   *
   * `inputs_map`基于上述`indexes`工作，描述如何将`old_nodes`的数据输入映射到`new_nodes`上。
   * 映射规则为：
   * 把`old_node`的第`inputs_map[i]`个数据锚点的数据关系拷贝到`new_nodes`的第`i`个数据锚点上。
   * `i`的取值范围是[0, `inputs_map`的元素个数）;
   * 如果`inputs_map[i]`的值小于0，那么`new_nodes`的第`i`个数据锚点的数据关系保持原样；
   * 如果`inputs_map[i]`的值超出`old_nodes`的索引范围，函数报错。
   */
  static graphStatus CopyNodesInDataAnchors(const std::vector<NodePtr> &new_nodes,
                                            const std::vector<NodePtr> &old_nodes,
                                            const std::vector<int32_t> &inputs_map);

  /*
   * `old_nodes`、`new_nodes`均被认为是一个整体，`old_nodes`内部节点之间的控制边被丢弃。
   * 全量`old_nodes`作为整体，输入、输出控制关系被保留，生效到全量`new_nodes`上。
   * 具体来说，`new_nodes`前后创建`noop`，输入/输出控制节点全连接到前/后面的`noop`节点上，
   * 同时前后`noop`向所有`new_nodes`做控制边全连接
   * need_convert_data_edges_2_ctrl_edges = false
   * cp所有old node的in ctrl到noop_in的in ctrl上, 添加noop_in的ctrl out到所有new nodes的ctrl in
   * cp所有old node的out ctrl到noop_out的out ctrl上, 添加所有new nodes的ctrl out到noop_out的ctrl in
   *
   * 更严格的行为：need_convert_data_edges_2_ctrl_edges = true
   * 在之前行为的基础上，
   * 把old node的in data也转换为noop_in的in ctrl;
   * 把old node的out data也转换为noop_in的out ctrl;
   */
  static graphStatus InheritExecutionOrder(const std::vector<NodePtr> &new_nodes, const std::vector<NodePtr> &old_nodes,
                                           const ComputeGraphPtr &graph,
                                           bool need_convert_data_edges_2_ctrl_edges = false);
  /**
   * 拷贝`src_node`的控制输入边到`dst_node`上
   * @param src_node
   * @param dst_node
   * @return
   */
  static graphStatus CopyInCtrlEdges(const NodePtr &src_node, const NodePtr &dst_node);

  /**
   * 拷贝`src_node`的控制输入边到`dst_node`上
   * @param src_node
   * @param dst_node
   * @param node_filter 控制边拷贝白名单过滤器，可以通过传递此参数实现满足条件的边的复制，不传递时代表全量拷贝
   * @return
   */
  static graphStatus CopyInCtrlEdges(const NodePtr &src_node, const NodePtr &dst_node, const NodeFilter &node_filter);

  /**
   * 把`src_node`的数据输入边转换成`dst_node`上的控制输入边
   * @param src_node
   * @param dst_node
   * @param node_filter 数据边转换白名单过滤器，可以通过传递此参数实现满足条件的边的转换，不传递时代表全量数据边转换
   * @return
   */
  static graphStatus ConvertInDataEdgesToInCtrlEdges(const NodePtr &src_node, const NodePtr &dst_node,
                                                     const NodeFilter &node_filter);

  /**
   * 移动`src_node`的控制输入边到`dst_node`上
   * @param src_node
   * @param dst_node
   * @return
   */
  static graphStatus MoveInCtrlEdges(const NodePtr &src_node, const NodePtr &dst_node);
  /**
   * 拷贝`src_node`的控制输出边到`dst_node`上
   * @param src_node
   * @param dst_node
   * @return
   */
  static graphStatus CopyOutCtrlEdges(const NodePtr &src_node, const NodePtr &dst_node);
  /**
   * 选择性拷贝`src_node`的控制输出边到`dst_node`上
   * @param src_node
   * @param dst_node
   * @param node_filter 控制边拷贝白名单过滤器，可以通过传递此参数实现满足条件的边的复制，不传递时代表全量拷贝
   * @return
   */
  static graphStatus CopyOutCtrlEdges(const NodePtr &src_node, const NodePtr &dst_node, const NodeFilter &node_filter);

  /**
   * 把`src_node`的数据输出边转换成`dst_node`上的控制输出边
   * @param src_node
   * @param dst_node
   * @param node_filter 数据边转换白名单过滤器，可以通过传递此参数实现满足条件的边的转换，不传递时代表全量数据边转换
   * @return
   */
  static graphStatus ConvertOutDataEdgesToOutCtrlEdges(const NodePtr &src_node, const NodePtr &dst_node,
                                                       const NodeFilter &node_filter);

  /**
   * 移动`src_node`的控制输出边到`dst_node`上
   * @param src_node
   * @param dst_node
   * @return
   */
  static graphStatus MoveOutCtrlEdges(NodePtr &src_node, NodePtr &dst_node);

  /**
   * 查找`graph`的根图，如果当前图就是根图或者当前图没有父图，则返回当前图
   * @param graph
   * @return
   */
  static ComputeGraphPtr FindRootGraph(ComputeGraphPtr graph);
  /**
   * 浅拷贝`graph`，并不会同步拷贝`graph`的子图
   * @param graph
   * @param suffix 用于拼接克隆图中的节点名称
   * @param input_nodes 返回克隆图中的输入节点
   * @param output_nodes 返回克隆图中的输出节点
   * @return
   */
  static ComputeGraphPtr CloneGraph(const ComputeGraphPtr &graph, const std::string &suffix,
                                    std::vector<NodePtr> &input_nodes, std::vector<NodePtr> &output_nodes);
  /**
   * 拷贝`src_compute_graph`图上的attr属性到`dst_compute_graph`上，
   * 需要注意的是 如果`dst_compute_graph`图上已经存在了某些同名属性，则会跳过这些属性的值的拷贝
   * @param src_compute_graph
   * @param dst_compute_graph
   */
  static void InheritOriginalAttr(const ComputeGraphPtr &src_compute_graph, ComputeGraphPtr &dst_compute_graph);
  /**
   * 拷贝`src_node`节点及其所有有效的输入输出tensor上的attr属性到`dst_desc`上
   * @param dst_desc 目的OpDesc对象
   * @param src_node 源Node对象
   * @return 拷贝成功返回GRAPH_SUCCESS， 拷贝失败返回GRAPH_FAILED
   */
  static graphStatus CopyTensorAttrs(const OpDescPtr &dst_desc, const NodePtr &src_node);

  /**
   * 获取当前图里面的所有的节点的输入输出tensor的复用关系
   * @param graph
   * @param symbol_to_anchors
   * @param anchor_to_symbol
   * @return
   */
  static graphStatus GetRefMapping(const ComputeGraphPtr &graph, SymbolToAnchors &symbol_to_anchors,
                                   AnchorToSymbol &anchor_to_symbol);

  /// Determine if the graph is a UNKNOWN_SHAPE graph based on whether the graph and all subgraphs
  /// of the graph have UNKNOWN_SHAPE operators or not.
  /// Note: This function will only look 'down' from the graph, not 'up'. For example, the following
  /// scenario (K for known shape, U for unknown shape), ROOT graph is UNKNOWN_SHAPE while SUB graph is KNOWN_SHAPE
  /// ROOT graph:      A -----> B -----> C
  ///                  K    subgraph     U
  ///                           |
  ///                           V
  /// SUB graph:          D --> E --> F
  ///                     K     K     K
  /// @param [in] graph
  /// @return bool
  static bool IsUnknownShapeGraph(const ComputeGraphPtr &graph);

  static NodePtr FindNodeFromAllNodes(ComputeGraphPtr &graph, const std::string &name);
  static std::vector<NodePtr> FindNodesByTypeFromAllNodes(ComputeGraphPtr &graph, const std::string &type);
  static std::vector<Node *> FindBareNodesByTypeFromAllNodes(ComputeGraphPtr &graph, const char_t *const type);
  /**
   * 判断当前`out_data_anchor`是否复用了输入anchor的内存
   * @param out_data_anchor
   * @param reuse_in_index 复用的输入anchor的index
   * @return 如果存在复用关系，返回true, 否则返回false
   */
  static bool IsRefFromInput(const OutDataAnchorPtr &out_data_anchor, int32_t &reuse_in_index);
  /**
   * 判断当前`out_data_anchor`是否引用RefData的输出
   * @param out_data_anchor
   * @param ref_data 复用的RefData节点
   * @param 如果存在复用关系，返回true, 否则返回false
   * @return status
   */
  static graphStatus CheckIsRefFromOther(const OutDataAnchorPtr &out_data_anchor, NodePtr &refed_node,
                                         bool &is_ref_from_other);
  /**
   * 针对含有`ATTR_NAME_NOPADDING_CONTINUOUS_INPUT`和`ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT`类型的节点
   * 单独封装的复用接口
   * @param out_data_anchor
   * @param reuse_in_index 出参，如果存在复用，值为0
   * @return 如果存在复用，返回true,负责返回false
   */
  static bool IsNoPaddingRefFromInput(const OutDataAnchorPtr &out_data_anchor, int32_t &reuse_in_index);
  /**
   * 用于判断`node`是否直接或者间接从属于`graph`, `间接`的一种含义是`node`的父图是`graph`
   * @param graph
   * @param node
   * @return
   */
  static bool IsNodeInGraphRecursively(const ComputeGraphPtr &graph, const Node &node);

  /**
   * 获取所有`直接`父图为`graph`和`间接`父图为`graph`的子图对象合集
   * @param graph
   * @param subgraphs 子图对象的合集
   * @return 成功返回GRAPH_SUCCESS，失败返回GRAPH_FAILED
   */
  static graphStatus GetSubgraphsRecursively(const ComputeGraphPtr &graph, std::vector<ComputeGraphPtr> &subgraphs);

  /**
   * 创建以`subgraph_name`拼接命名的子图对象`subgraph`，把`nodes`中的节点从`graph`中抽取出来放在`subgraph`中，
   * 完成图归属和节点连边关系的重建,`nodes`作为一个整体与`subgraph`父节点等价
   * @param graph
   * @param nodes
   * @param subgraph_name
   * @return 成功返回GRAPH_SUCCESS，失败返回GRAPH_FAILED
   */
  static ComputeGraphPtr BuildSubgraphWithNodes(const ComputeGraphPtr &graph, const std::set<NodePtr> &nodes,
                                                const std::string &subgraph_name);
  /**
   * `BuildSubgraphWithNodes`的重载接口
   * @param graph
   * @param nodes
   * @param subgraph_name
   * @return 成功返回GRAPH_SUCCESS，失败返回GRAPH_FAILED
   */
  static ComputeGraphPtr BuildSubgraphWithNodes(ComputeGraph &graph, const std::set<NodePtr> &nodes,
                                                const std::string &subgraph_name);
  /**
   * 作为`BuildSubgraphWithNodes`函数的逆向操作，会把`graph`展开其父图上，
   * 此接口支持子图的递归展开操作
   * @param graph 要展开的子图
   * @param filter 子图过滤器，用于过滤子图的子图是否要展开，不传递时不进行递归操作
   * @return 成功返回GRAPH_SUCCESS，失败返回GRAPH_FAILED
   */
  static graphStatus UnfoldSubgraph(const ComputeGraphPtr &graph,
                                    const std::function<bool(const ComputeGraphPtr &)> &filter);
  /**
   * 作为`UnfoldSubgraph`的高阶版本，支持没有父子关系的图的展开操作，展开`graph`到`target_graph`上，
   * `graph`作为一个整体，等价替换掉`target_graph`中的`target_node`；
   * 此接口支持子图的递归展开操作
   * @param graph 要展开的子图
   * @param target_graph 展开到的目标图
   * @param target_node 子图要替换的目标节点
   * @param filter 图过滤器，用于过滤子图的子图是否要展开，不传递时不进行递归操作
   * @param depth
   * @return 成功返回GRAPH_SUCCESS，失败返回GRAPH_FAILED
   */
  static graphStatus UnfoldGraph(const ComputeGraphPtr &graph, const ComputeGraphPtr &target_graph,
                                 const NodePtr &target_node, const function<bool(const ComputeGraphPtr &)> &filter,
                                 int32_t depth = 0);

  /**
   * 创建以`name`命名的图对象`ComputeGraphPtr`，根据`nodes`中的节点
   * 完成输入输出和节点连边关系的重建,`nodes`作为一个整体与`ComputeGraphPtr`等价
   * @param nodes
   * @param name
   * @return
   */
  static ComputeGraphPtr BuildGraphFromNodes(const std::unordered_set<NodePtr> &nodes, const std::string &name);

  static bool IsSingleOpScene(const ComputeGraphPtr &graph);

  static CycleDetectorPtr CreateCycleDetector(const ComputeGraphPtr &graph);

  static CycleDetectorSharedPtr CreateSharedCycleDetector(const ComputeGraphPtr &graph);

  /**
   * 将node所有的输入、输出边断开，并移动到dst_graph
   * @param dst_graph 目的Graph，
   * @param node 需要移动的Node
   * @return 成功时，返回ge::GRAPH_SUCCESS
   */
  static graphStatus MoveNodeToGraph(const NodePtr &node, ComputeGraph &dst_graph);

  /**
   * 判断当前节点的某个输出是否可以复用输入
   * @param node 目标节点，
   * @param out_index_to_refable_in_indexes 返回节点的输出-输入(单输出多输入)复用关系
   * @return 成功时，返回SUCCESS
   */
  static graphStatus GetSupportInplaceOutput(const NodePtr &node,
                                             std::map<size_t, std::vector<size_t>> &out_index_to_refable_in_indexes);

  /**
   * 用expand_graph图替换target_node节点，先将expand_graph中的算子插入到图中，再建立连边关系，最后删掉target_node
   * @param target_node 目标节点
   * @param expand_graph 需要被展开的图
   * @return 成功时，返回SUCCESS 失败返回FAILED
   */
  static graphStatus ExpandNodeWithGraph(const NodePtr &target_node, const ComputeGraphPtr &expand_graph);

  /// Relink all edges for cloned ComputeGraph.
  /// @param [in] node: original node.
  /// @param [in] node_clone_map: source-node to cloned-node mapping.
  /// @return success: GRAPH_SUCESS
  static graphStatus RelinkGraphEdges(const NodePtr &node, const NodeCloneMap &node_clone_map);

  /// Relink all edges for cloned ComputeGraph.
  /// @param [in] node: original node.
  /// @param [in] suffix: node name suffix of new node.
  /// @param [in] all_nodes: all nodes in new graph.
  /// @return success: GRAPH_SUCESS
  static graphStatus RelinkGraphEdges(const NodePtr &node, const std::string &suffix,
                                      const std::unordered_map<std::string, NodePtr> &all_nodes);

  /**
   * 将src_graph中所有节点插入到target_graph中，插入节点在列表中的位置在target_node之后
   * @param target_graph 需要被插入算子的图
   * @param target_node 新插入的算子需要被插在target_node的后面，且target_node需要在target_graph上
   * @param insert_nodes 表示要插入的节点
   * @return 如果插入成功返回GRAPH_SUCCESS，失败返回GRAPH_FAILED
   */
  static graphStatus MoveNodesToGraphAfterTargetNode(const ComputeGraphPtr &target_graph, const NodePtr &target_node,
                                                     const ComputeGraphPtr &src_graph);

 private:
  class GraphInfo {
   public:
    GraphInfo() = default;
    ~GraphInfo() = default;

   private:
    std::set<NodePtr> nodes_;
    std::map<uint32_t, std::pair<OutDataAnchorPtr, std::list<InDataAnchorPtr>>> data_inputs_;
    std::map<uint32_t, std::pair<OutDataAnchorPtr, std::list<InDataAnchorPtr>>> data_outputs_;
    std::list<std::pair<OutControlAnchorPtr, InControlAnchorPtr>> ctrl_inputs_;
    std::list<std::pair<OutControlAnchorPtr, InControlAnchorPtr>> ctrl_outputs_;
    std::list<std::pair<OutDataAnchorPtr, InDataAnchorPtr>> inner_data_edges_;
    std::list<std::pair<OutControlAnchorPtr, InControlAnchorPtr>> inner_ctrl_edges_;
    friend class GraphUtils;
  };

  /**
   * 创建节点输入tensor的内存符号，正常情况下，应该跟其对端的输出tensor的内存符号相同，因为二者是一块地址
   * @param graph
   * @param node
   * @param symbol_to_anchors
   * @param anchor_to_symbol
   * @return
   */
  static graphStatus HandleInAnchorMapping(const ComputeGraphPtr &graph, const NodePtr &node,
                                           SymbolToAnchors &symbol_to_anchors, AnchorToSymbol &anchor_to_symbol);

  /**
   * 创建节点输出tensor的内存符号
   * @param node
   * @param symbol_to_anchors
   * @param anchor_to_symbol
   * @return
   */
  static graphStatus HandleOutAnchorMapping(const NodePtr &node, SymbolToAnchors &symbol_to_anchors,
                                            AnchorToSymbol &anchor_to_symbol);

  /**
   * 创建子图内Data节点的输出tensor的内存符号，正常应该跟父节点对应的输入tensor内存符号相同
   * @param node
   * @param symbol_to_anchors
   * @param anchor_to_symbol
   * @return
   */
  static graphStatus HandleSubgraphInput(const NodePtr &node, SymbolToAnchors &symbol_to_anchors,
                                         AnchorToSymbol &anchor_to_symbol);

  /**
   * 创建merge节点的内存符号，merge是特殊的v1控制算子，其输出和多个输入存在复用关系，因此单独封装函数处理
   * @param node
   * @param symbol_to_anchors
   * @param anchor_to_symbol
   * @return
   */
  static graphStatus HandleMergeInput(const NodePtr &node, SymbolToAnchors &symbol_to_anchors,
                                      AnchorToSymbol &anchor_to_symbol);

  /**
   * 创建子图内Netoutput节点的输出tensor的内存符号，正常应该跟父节点对应的输出tensor内存符号相同
   * @param node
   * @param symbol_to_anchors
   * @param anchor_to_symbol
   * @return
   */
  static graphStatus HandleSubgraphOutput(const NodePtr &node, SymbolToAnchors &symbol_to_anchors,
                                          AnchorToSymbol &anchor_to_symbol);

  /**
   * 合并代表同一块地址的不同符号
   * @param exist_node_info1
   * @param exist_node_info2
   * @param symbol_to_anchors
   * @param anchor_to_symbol
   * @param symbol
   * @return
   */
  static graphStatus UnionSymbolMapping(const NodeIndexIO &exist_node_info1, const NodeIndexIO &exist_node_info2,
                                        SymbolToAnchors &symbol_to_anchors, AnchorToSymbol &anchor_to_symbol,
                                        std::string &symbol);

  /**
   * 对于同一块地址，使用已有tensor的符号设置当前tensor的符号
   * @param cur_node_info
   * @param exist_node_info
   * @param symbol_to_anchors
   * @param anchor_to_symbol
   * @return
   */
  static graphStatus UpdateRefMapping(const NodeIndexIO &cur_node_info, const NodeIndexIO &exist_node_info,
                                      SymbolToAnchors &symbol_to_anchors, AnchorToSymbol &anchor_to_symbol);
  template <typename Container>
  static void BuildGraphInfoFromNodes(const Container &nodes, GraphInfo &graph_info);

  template <typename Container>
  static void BuildInDataEdgesFromNode(const NodePtr &node, const Container &nodes,
                                       std::map<OutDataAnchorPtr, size_t> &data_input_index_map, GraphInfo &graph_info);

  static NodePtr BuildSubgraphNode(ComputeGraph &graph, const std::string &graph_name, const GraphInfo &graph_info);

  static ComputeGraphPtr BuildSubgraph(const NodePtr &subgraph_node, const GraphInfo &graph_info,
                                       const std::string &subgraph_name);
  static ComputeGraphPtr BuildGraph(const GraphInfo &graph_info, const std::string &name);

  static ComputeGraphPtr BuildGraphInternal(const GraphInfo &graph_info, const std::string &name,
                                            const NodePtr &parent_node);
  static graphStatus RelinkDataEdges(const NodePtr &subgraph_node, const GraphInfo &graph_info);

  static graphStatus RelinkCtrlEdges(const NodePtr &subgraph_node, const GraphInfo &graph_info);

  static graphStatus MergeInputNodes(const ComputeGraphPtr &graph, const NodePtr &target_node);

  static graphStatus MergeNetOutputNode(const ComputeGraphPtr &graph, const NodePtr &target_node);

  static bool NoNeedDumpGraphBySuffix(const std::string &suffix);

  static graphStatus CopyOpAndSubgraph(const ComputeGraphPtr &src_compute_graph, const NodeFilter &node_filter,
                                       const GraphFilter &graph_filter, const AttrFilter &attr_filter,
                                       ComputeGraphPtr &dst_compute_graph,
                                       std::map<ConstNodePtr, NodePtr> &node_old_2_new,
                                       std::map<ConstOpDescPtr, OpDescPtr> &op_desc_old_2_new,
                                       std::unordered_map<std::string, NodePtr> &all_new_nodes, const int32_t depth);

  static graphStatus CopyMembers(const ComputeGraphPtr &src_compute_graph, ComputeGraphPtr &dst_compute_graph,
                                 const std::unordered_map<std::string, NodePtr> &all_new_nodes);

  static graphStatus CopyGraphImpl(const Graph &src_graph, Graph &dst_graph,
                                   const std::map<ConstNodePtr, NodePtr> &node_old_2_new,
                                   const std::map<ConstOpDescPtr, OpDescPtr> &op_desc_old_2_new);

  /**
   * 改图接口，将图中的“FileConstant”节点用属性中对应的权重文件恢复成“Const”节点
   * @param compute_graph
   * @return
   */
  static graphStatus ConvertFileConstToConst(const ComputeGraphPtr &graph);
  static graphStatus RecoverConstByWeightFile(const OpDescPtr &op_desc, const GeTensorPtr &weight);
};

/**
 * 黑匣子图dump接口，用于把`compute_graph`对象序列化到文件，默认落盘到当前路径;
 * 如果`compute_graph`挂载了子图对象，子图对象也尝试进行落盘
 * 黑匣子图的落盘行为不受`DUMP_GE_GRAPH`和`DUMP_GRAPH_LEVEL`环境变量的控制，用于异常场景保留现场
 * @param compute_graph
 * @param name 用于拼接文件的名称
 */
inline void GE_DUMP_BLACK_BOX(const ComputeGraphPtr &compute_graph, const std::string &name) {
  GraphUtils::DumpGEGraph((compute_graph), (name), true);
  GraphUtils::DumpGEGraphToOnnx(*(compute_graph), (name), true);
  GraphUtils::DumpGEGraphToReadable(compute_graph, name, true);
  uint64_t i = 0U;
  for (const auto &sub_graph_func : (compute_graph)->GetAllSubgraphs()) {
    const auto sub_graph_func_name = std::string(name) + std::string("_sub_graph_") + std::to_string(i++);
    GraphUtils::DumpGEGraph(sub_graph_func, sub_graph_func_name, true);
    GraphUtils::DumpGEGraphToOnnx(*sub_graph_func, sub_graph_func_name, true);
    GraphUtils::DumpGEGraphToReadable(sub_graph_func, sub_graph_func_name, true);
  }
}

class ComputeGraphBuilder {
 public:
  ComputeGraphBuilder() : owner_graph_(nullptr) {}
  virtual ~ComputeGraphBuilder() = default;

  /// @brief Add node to graph
  /// @param [in] op_desc
  /// @return ComputeGraphBuilder
  virtual ComputeGraphBuilder &AddNode(const OpDescPtr &op_desc);

  /// @brief Add data-link among nodes in graph
  /// @param [in] src_name
  /// @param [in] out_anchor_ind
  /// @param [in] dst_name
  /// @param [in] in_anchor_ind
  /// @return ComputeGraphBuilder
  virtual ComputeGraphBuilder &AddDataLink(const std::string &src_name, const uint32_t out_anchor_ind,
                                           const std::string &dst_name, const uint32_t in_anchor_ind);

  /// @brief Add ctrl-link among nodes in graph
  /// @param [in] src_name
  /// @param [in] dst_name
  /// @return ComputeGraphBuilder
  virtual ComputeGraphBuilder &AddControlLink(const std::string &src_name, const std::string &dst_name);

  /// @brief Build graph
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return ComputeGraphPtr
  virtual ComputeGraphPtr Build(graphStatus &error_code, std::string &error_msg) = 0;

  /// @brief Get node with name
  /// @param [in] name
  /// @return NodePtr
  NodePtr GetNode(const std::string &name);

  /// @brief Get all nodes
  /// @return std::vector<NodePtr>
  std::vector<NodePtr> GetAllNodes();

 protected:
  /// @brief Build nodes
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  void BuildNodes(graphStatus &error_code, std::string &error_msg);

  /// @brief Build data-links
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  void BuildDataLinks(graphStatus &error_code, std::string &error_msg);

  /// @brief Build ctrl-links
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  void BuildCtrlLinks(graphStatus &error_code, std::string &error_msg);

 private:
  ComputeGraphBuilder(const ComputeGraphBuilder &) = delete;
  ComputeGraphBuilder &operator=(const ComputeGraphBuilder &) = delete;
  ComputeGraphBuilder(const ComputeGraphBuilder &&) = delete;
  ComputeGraphBuilder &operator=(const ComputeGraphBuilder &&) = delete;

  ComputeGraphPtr owner_graph_;
  // node_name -> node
  std::map<std::string, NodePtr> node_names_;
  std::vector<OpDescPtr> nodes_;
  // <src_node_name, out_anchor_ind> -> <dst_node_name, in_anchor_ind>
  std::vector<std::pair<std::pair<std::string, uint32_t>, std::pair<std::string, uint32_t>>> data_links_;
  // src_node_name -> dst_node_name
  std::vector<std::pair<std::string, std::string>> ctrl_links_;

  friend class CompleteGraphBuilder;
  friend class PartialGraphBuilder;
};

class CompleteGraphBuilder : public ComputeGraphBuilder {
 public:
  explicit CompleteGraphBuilder(const std::string name, const bool retval_flag = true)
      : ComputeGraphBuilder(), name_(name), parent_node_(nullptr), retval_flag_(retval_flag) {}
  CompleteGraphBuilder(const CompleteGraphBuilder &) = delete;
  CompleteGraphBuilder &operator=(const CompleteGraphBuilder &) = delete;
  CompleteGraphBuilder(const CompleteGraphBuilder &&) = delete;
  CompleteGraphBuilder &operator=(const CompleteGraphBuilder &&) = delete;
  ~CompleteGraphBuilder() = default;

  /// @brief Add node to graph
  /// @param [in] op_desc
  /// @return CompleteGraphBuilder
  CompleteGraphBuilder &AddNode(const OpDescPtr &op_desc) override;

  /// @brief Add data-link among nodes in graph
  /// @param [in] src_name
  /// @param [in] out_anchor_ind
  /// @param [in] dst_name
  /// @param [in] in_anchor_ind
  /// @return CompleteGraphBuilder
  CompleteGraphBuilder &AddDataLink(const std::string &src_name, const uint32_t out_anchor_ind,
                                    const std::string &dst_name, const uint32_t in_anchor_ind) override;

  /// @brief Add ctrl-link among nodes in graph
  /// @param [in] src_name
  /// @param [in] dst_name
  /// @return CompleteGraphBuilder
  CompleteGraphBuilder &AddControlLink(const std::string &src_name, const std::string &dst_name) override;

  /// @brief Set index_th input anchor for graph
  /// @param [in] index
  /// @param [in] node_names
  /// @param [in] anchor_inds
  /// @return CompleteGraphBuilder
  CompleteGraphBuilder &SetInput(const uint32_t index, const std::vector<std::string> &node_names,
                                 const std::vector<uint32_t> &anchor_inds);

  /// @brief Set index_th input of graph as useless
  /// @param [in] index
  /// @return CompleteGraphBuilder
  CompleteGraphBuilder &SetUselessInput(const uint32_t index);

  /// @brief Add output anchor for graph
  /// @param [in] owner_node_name
  /// @param [in] anchor_ind
  /// @return CompleteGraphBuilder
  CompleteGraphBuilder &AddOutput(const std::string &owner_node_name, uint32_t anchor_ind);

  /// @brief Add target for graph
  /// @param [in] target_name
  /// @return CompleteGraphBuilder
  CompleteGraphBuilder &AddTarget(const std::string &target_name);

  /// @brief Set parent-node of graph
  /// @param [in] parent_node
  /// @return CompleteGraphBuilder
  CompleteGraphBuilder &SetParentNode(const NodePtr &parent_node);

  /// @brief Set mapping-relation of parent-node in_anchor_ind & Data-node
  /// @param [in] input_mapping: index_of_graph_input -> in_anchor_index_of_parent_node
  /// @return CompleteGraphBuilder
  CompleteGraphBuilder &SetInputMapping(const std::map<uint32_t, uint32_t> &input_mapping);

  /// @brief Set mapping-relation of parent-node out_anchor_ind & NetOutput-node out_anchor_ind
  /// @param [in] output_mapping: index_of_graph_output -> out_anchor_index_of_parent_node
  /// @return CompleteGraphBuilder
  CompleteGraphBuilder &SetOutputMapping(const std::map<uint32_t, uint32_t> &output_mapping);

  /// @brief Build graph
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return ComputeGraphPtr
  ComputeGraphPtr Build(graphStatus &error_code, std::string &error_msg) override;

 private:
  /// @brief Add data nodes
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  void AddDataNodes(graphStatus &error_code, std::string &error_msg);

  /// @brief Add data node
  /// @param [in] index
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  NodePtr AddDataNode(const uint32_t index, graphStatus &error_code, std::string &error_msg);

  /// @brief Add RetVal nodes
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  void AddRetValNodes(graphStatus &error_code, std::string &error_msg);

  /// @brief Build target-nodes for graph
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  void BuildGraphTargets(graphStatus &error_code, std::string &error_msg);

  /// @brief Add NetOutput node
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  void AddNetOutputNode(graphStatus &error_code, std::string &error_msg);

  /// @brief Build NetOutput nodes with data & ctrl edges
  /// @param [in] net_output_desc
  /// @param [in] peer_out_anchors
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  void BuildNetOutputNodeWithLink(const OpDescPtr &net_output_desc,
                                  const std::vector<OutDataAnchorPtr> &peer_out_anchors, graphStatus &error_code,
                                  std::string &error_msg);

  /// @brief process after build
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  void PostProcess(graphStatus &error_code, std::string &error_msg);

  std::string name_;
  NodePtr parent_node_;
  bool retval_flag_;
  std::map<uint32_t, std::pair<std::vector<std::string>, std::vector<uint32_t>>> graph_inputs_;
  std::vector<std::pair<std::string, uint32_t>> graph_outputs_;
  std::vector<std::string> graph_targets_;

  // index_of_graph_input -> in_anchor_index_of_parent_node
  std::map<uint32_t, uint32_t> input_mapping_;
  // index_of_graph_output -> out_anchor_index_of_parent_node
  std::map<uint32_t, uint32_t> output_mapping_;
};

class PartialGraphBuilder : public ComputeGraphBuilder {
 public:
  PartialGraphBuilder() = default;
  PartialGraphBuilder(const PartialGraphBuilder &) = delete;
  PartialGraphBuilder &operator=(const PartialGraphBuilder &) = delete;
  PartialGraphBuilder(const PartialGraphBuilder &&) = delete;
  PartialGraphBuilder &operator=(const PartialGraphBuilder &&) = delete;
  ~PartialGraphBuilder() = default;

  /// @brief Add node to graph
  /// @param [in] op_desc
  /// @return PartialGraphBuilder
  PartialGraphBuilder &AddNode(const OpDescPtr &op_desc) override;

  /// @brief Add data-link among nodes in graph
  /// @param [in] src_name
  /// @param [in] out_anchor_ind
  /// @param [in] dst_name
  /// @param [in] in_anchor_ind
  /// @return PartialGraphBuilder
  PartialGraphBuilder &AddDataLink(const std::string &src_name, const uint32_t out_anchor_ind,
                                   const std::string &dst_name, const uint32_t in_anchor_ind) override;

  /// @brief Add ctrl-link among nodes in graph
  /// @param [in] src_name
  /// @param [in] dst_name
  /// @return PartialGraphBuilder
  PartialGraphBuilder &AddControlLink(const std::string &src_name, const std::string &dst_name) override;

  /// @brief Set owner graph
  /// @param [in] graph
  /// @return PartialGraphBuilder
  PartialGraphBuilder &SetOwnerGraph(const ComputeGraphPtr &graph);

  /// @brief Add exist node
  /// @param [in] node
  /// @return PartialGraphBuilder
  PartialGraphBuilder &AddExistNode(const NodePtr &exist_node);

  /// @brief Build multi nodes with links
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return ComputeGraphPtr
  ComputeGraphPtr Build(graphStatus &error_code, std::string &error_msg) override;

 private:
  /// @brief Build exist nodes
  /// @param [out] error_code
  /// @param [out] error_msg
  /// @return void
  void BuildExistNodes(graphStatus &error_code, std::string &error_msg);

  std::vector<NodePtr> exist_nodes_;
};
}  // namespace af

#endif  // INC_GRAPH_UTILS_GRAPH_UTILS_H_
