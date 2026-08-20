/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_ASCEND_GRAPH_CODE_DUMPER_H
#define AUTOFUSE_ASCEND_GRAPH_CODE_DUMPER_H

#include <string>
#include <fstream>
#include <iostream>

#include "common/checker.h"
#include "graph/compute_graph.h"
#include "graph/node.h"
#include "graph/anchor.h"
#include "graph/ascendc_ir/ascir_register.h"
namespace af::ascir {
// c++中的node name可能有/等python中无法作为对象名的非法字符，所以用这个类来根据type生成全局唯一的名字
class NameGenerator {
 public:
  std::string GenerateUniqueName(const af::Node &node) {
    const std::string &type = node.GetType();
    const std::string &original_name = node.GetName();
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string &unique_name = type + "_" + std::to_string(type_counter_[type]++);
    name_mapping_[original_name] = unique_name;
    return unique_name;
  }

  const std::unordered_map<std::string, std::string> &GetNameMapping() const {
    return name_mapping_;
  }

 private:
  std::unordered_map<std::string, int64_t> type_counter_;
  // 记录原始名字和python中的名字的映射关系
  std::unordered_map<std::string, std::string> name_mapping_;
  std::mutex mutex_;
};

class PythonCodeDumper {
 public:
  void GenerateInputCode(const std::string &op_name, const std::string &input_name, const af::NodePtr &src_node,
                         uint32_t out_idx, std::ostream &output_file);
  Status GenerateDynamicInputCode(const af::Node::Vistor<std::pair<af::NodePtr, af::OutDataAnchorPtr>> &src_nodes,
                                  size_t start_index, size_t count, const std::string &op_name,
                                  const std::string &input_name, std::ostream &output_file);
  static void GenerateHeader(std::ofstream &output_file);
  void GenerateGraphInstance(const af::AscGraph &asc_graph, std::ostream &output_file);
  static void GenerateFooter(std::ofstream &output_file);
  Status DumpAscGraphNode(const af::AscGraph &graph, std::ostream &output_file);
  Status Dump(const af::AscGraph &graph, const std::string &out_file_path);
  Status GenerateNodeCode(const af::NodePtr &node, std::ostream &output_file);
  Status GenerateDataEdgeCode(const af::Node::Vistor<std::pair<af::NodePtr, af::OutDataAnchorPtr>> &src_nodes,
                              const af::NodePtr &dst_node, std::ostream &output_file);
  Status GenerateTensorCode(const af::NodePtr &node, std::ostream &output_file);
  Status GenerateIrAttrCode(const af::NodePtr &node, std::ostream &output_file);
  explicit PythonCodeDumper(const std::shared_ptr<NameGenerator> &name_generator = nullptr)
#ifdef AUTOFUSE_USE_GE_METADEF
      : name_generator_(name_generator ? name_generator : ge::ComGraphMakeShared<NameGenerator>()),
#else
      : name_generator_(name_generator ? name_generator : af::ComGraphMakeShared<NameGenerator>()),
#endif
        types_to_ascir_(af::ascir::AscirRegistry::GetInstance().GetAll()) {
  }
  std::string node_name_of_python_;

 private:
  std::vector<af::AxisPtr> asis_ptrs;
  std::shared_ptr<NameGenerator> name_generator_;
  const std::unordered_map<std::string, af::ascir::AscIrDef> &types_to_ascir_;
};

class PythonCodeDumperFused {
 public:
  static void GenerateHeader(std::ofstream &output_file);
  void GenerateGraphInstance(const af::ComputeGraph &compute_graph, std::ofstream &output_file) const;
  Status DumpAscGraphNode(const af::NodePtr &node, std::ofstream &output_file);
  Status Dump(const af::ComputeGraph &graph, const std::string &out_file_path);
  void GenerateFooter(std::ofstream &output_file) const;
  Status GenerateDataEdgeCodeWithOutIr(const af::Node::Vistor<std::pair<af::NodePtr, af::OutDataAnchorPtr>> &src_nodes,
                                       const af::NodePtr &dst_node, std::ofstream &output_file);
  Status GenerateDataEdgeCode(const af::Node::Vistor<std::pair<af::NodePtr, af::OutDataAnchorPtr>> &src_nodes,
                              const af::NodePtr &dst_node, std::ofstream &output_file);
  explicit PythonCodeDumperFused()
#ifdef AUTOFUSE_USE_GE_METADEF
      : name_generator_(ge::ComGraphMakeShared<NameGenerator>()),
#else
      : name_generator_(af::ComGraphMakeShared<NameGenerator>()),
#endif
        code_dumper_asc_graph_(PythonCodeDumper(name_generator_)) {
  }

 private:
  std::string node_name_of_python_;
  std::shared_ptr<NameGenerator> name_generator_;
  PythonCodeDumper code_dumper_asc_graph_;
};
}  // namespace af::ascir
#endif  // AUTOFUSE_ASCEND_GRAPH_CODE_DUMPER_H
