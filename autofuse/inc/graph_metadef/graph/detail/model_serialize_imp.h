/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_DETAIL_MODEL_SERIALIZE_IMP_H_
#define INC_GRAPH_DETAIL_MODEL_SERIALIZE_IMP_H_

#include <google/protobuf/repeated_field.h>
#include "ge_common_af/ge_api_types_af.h"
#include "graph/model.h"

namespace af {
using ComputeGraphPtr = std::shared_ptr<ComputeGraph>;
using AnchorWithIndex = std::pair<AnchorPtr, int64_t>;
struct MyCmp {
  bool operator()(const AnchorWithIndex &anchor1, const AnchorWithIndex &anchor2) const {
    return anchor1.second < anchor2.second;
  }
};
using DstAnchors = std::set<AnchorWithIndex, MyCmp>;

struct NodeNameGraphReq {
 public:
  NodeNameGraphReq(const std::string &name, const int32_t index, const ComputeGraphPtr &graph)
      : node_name(name), index(index), graph(graph) {}
  friend class ModelSerializeImp;

 private:
  std::string node_name;
  int32_t index;
  ComputeGraphPtr graph;
};

struct NodeNameNodeReq {
 public:
  NodeNameNodeReq(const std::string &src_name, const int32_t src_index, const int32_t src_out_peer_index,
                  const NodePtr dst_node, const int32_t dst_index, const std::string &dst_name)
      : src_node_name(src_name),
        src_out_index(src_index),
        src_out_peer_index(src_out_peer_index),
        dst_node(dst_node),
        dst_in_index(dst_index),
        dst_node_name(dst_name) {}

  friend class ModelSerializeImp;

 private:
  std::string src_node_name;
  int32_t src_out_index;
  int32_t src_out_peer_index;
  NodePtr dst_node;
  int32_t dst_in_index;
  std::string dst_node_name;
};

class ModelSerializeImp {
 public:
  bool SerializeModel(const Model &model, proto::ModelDef *const model_proto, const bool not_dump_all = false) const;
  // if is_dump_graph is true, ensure peer anchors of node in the same order during serialization and deserialization
  // if is_dump_graph is false, cannot guarantee peer anchors in the same order during serialization and deserialization
  bool SerializeModel(const Model &model, const bool is_dump_graph, proto::ModelDef *const model_proto,
                      const bool not_dump_all = false) const;

  bool SerializeGraph(const ConstComputeGraphPtr &graph, proto::GraphDef *const graph_proto,
                      const bool not_dump_all = false) const;
  bool SerializeGraph(const ConstComputeGraphPtr &graph, const bool is_dump_graph, proto::GraphDef *const graph_proto,
                      const bool not_dump_all = false) const;

  bool SerializeEdge(const NodePtr &node, proto::OpDef *const op_def_proto, const bool is_dump_graph = false) const;

  bool SerializeOpDesc(const ConstOpDescPtr &op_desc, proto::OpDef *const op_def_proto,
                       const bool not_dump_all = false) const;

  bool SerializeNode(const NodePtr &node, proto::OpDef *const op_def_proto, const bool not_dump_all = false) const;
  bool SerializeNode(const NodePtr &node, const bool is_dump_graph, proto::OpDef *const op_def_proto,
                     const bool not_dump_all = false) const;

  bool SeparateModelDef(Buffer &buffer, const std::string &path, proto::ModelDef &model_def) const;

  bool SerializeToBuffer(const proto::ModelDef &model_def, Buffer &buffer) const;

  bool UnserializeModel(Model &model, proto::ModelDef &model_proto);
  bool SetWeightForModel(proto::OpDef &op_def) const;

  bool UnserializeGraphWithoutEdge(ComputeGraphPtr &graph, proto::GraphDef &graph_proto);

  bool UnserializeGraph(ComputeGraphPtr &graph, proto::GraphDef &graph_proto);

  bool HandleNodeNameRef();

  void AttrDefToOpDescIrDef(OpDescPtr &op_desc, proto::OpDef &op_def_proto) const;
  bool UnserializeOpDesc(OpDescPtr &op_desc, proto::OpDef &op_def_proto) const;
  void AttrDefToOpDescIn(OpDescPtr &op_desc, std::vector<std::string> &key_in, std::vector<uint32_t> &value_in) const;
  void AttrDefToOpDesc(OpDescPtr &op_desc, std::vector<std::string> &key_out, std::vector<uint32_t> &value_out,
                       const std::vector<std::string> &opt_input) const;
  void OpDescToAttrDef(const ConstOpDescPtr &op_desc, proto::OpDef *const op_def_proto,
                       const bool not_dump_all = false) const;
  void OpDescIrDefToAttrDef(const ConstOpDescPtr &op_desc,
                            google::protobuf::Map<std::string, af::proto::AttrDef> *op_desc_attr) const;
  bool UnserializeNode(ComputeGraphPtr &graph, proto::OpDef &op_def_proto);

  bool ParseNodeIndex(const std::string &node_index, std::string &node_name, int32_t &index) const;

  void SetProtobufOwner(const ProtoMsgOwner &buffer_proto_buf_onwer) {
    protobuf_owner_ = buffer_proto_buf_onwer;
  }

  bool LoadWeightFromFile(const std::string &file_path, const int64_t &length, std::string &weight) const;

  void SetAirModelPath(const std::string &path) {
    air_path_ = path;
  }
  static bool SerializeAllAttrsFromAnyMap(const std::map<std::string, AnyValue> &attr_map,
                                          google::protobuf::Map<std::string, ::af::proto::AttrDef> *const mutable_attr);
  static bool DeserializeAllAttrsToAttrHolder(
      const google::protobuf::Map<std::string, ::af::proto::AttrDef> &proto_attr_map, AttrHolder *const attr_holder);

 private:
  bool RebuildOwnership(ComputeGraphPtr &compute_graph, std::map<std::string, ComputeGraphPtr> &subgraphs) const;
  Status UnserializeGraph(std::map<std::string, ComputeGraphPtr> &graphs,
                          ::google::protobuf::RepeatedPtrField<proto::GraphDef> &graphs_proto);
  void FixOpDefSubgraphInstanceName(const ConstOpDescPtr &op_desc) const;

  void ExtractMetaDataAttrIn(proto::OpDef &op_def_proto, std::vector<std::string> &opt_input,
                             std::vector<std::string> &key_in, std::vector<uint32_t> &value_in) const;
  void ExtractMetaDataAttr(proto::OpDef &op_def_proto, std::vector<std::string> &key_out,
                           std::vector<uint32_t> &value_out) const;

  int64_t GenDataInputInfo(const OutDataAnchorPtr &src_anchor, const InDataAnchorPtr &dst_anchor) const;
  int64_t GenCtrlInputInfo(const OutControlAnchorPtr &src_anchor, const InControlAnchorPtr &dst_anchor) const;
  void SaveEdgeInfo(const AnchorPtr &src_anchor, const AnchorPtr &dst_anchor, const int64_t src_out_peer_index,
                    const int64_t cur_index, std::unordered_map<AnchorPtr, DstAnchors> &edges) const;
  bool LinkEdges(const std::unordered_map<AnchorPtr, DstAnchors> &edges) const;

  std::vector<NodeNameGraphReq> graph_input_node_names_;
  std::vector<NodeNameGraphReq> graph_output_node_names_;
  std::vector<NodeNameNodeReq> node_input_node_names_;
  std::map<std::string, NodePtr> node_map_;
  ProtoMsgOwner protobuf_owner_;
  std::string air_path_;  // path store air model path
};
}  // namespace af

#endif  // INC_GRAPH_DETAIL_MODEL_SERIALIZE_IMP_H_
