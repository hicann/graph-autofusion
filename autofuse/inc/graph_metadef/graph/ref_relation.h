/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_GRAPH_REF_RELATION_H_
#define COMMON_GRAPH_REF_RELATION_H_

#include <deque>
#include <string>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "graph/compute_graph.h"
#include "graph/ge_error_codes_af.h"
#include "node.h"

namespace af {
enum InOutFlag {
  NODE_IN = 0,   // input flag
  NODE_OUT = 1,  // output flag
};

// RefCell的对象一经创建，便不允许去修改其数据成员。
struct RefCell {
  const std::string node_name;
  const NodePtr node;
  const InOutFlag in_out;
  const int32_t in_out_idx;
  const std::string hash_key;

  explicit RefCell(const std::string &name, const NodePtr &node_ptr, const InOutFlag in_out_flag, const int32_t idx)
      : node_name(name),
        node(node_ptr),
        in_out(in_out_flag),
        in_out_idx(idx),
        hash_key(std::string("")
                     .append(node_name)
                     .append(std::to_string(in_out))
                     .append(std::to_string(in_out_idx))
                     .append(std::to_string(ge::PtrToValue(node.get())))) {}
  RefCell(const RefCell &ref_cell)
      : node_name(ref_cell.node_name),
        node(ref_cell.node),
        in_out(ref_cell.in_out),
        in_out_idx(ref_cell.in_out_idx),
        hash_key(ref_cell.hash_key) {}
  RefCell &operator=(const RefCell &ref_cell) = delete;
  bool operator==(const RefCell &c) const {
    return node_name == c.node_name && node == c.node && in_out == c.in_out && in_out_idx == c.in_out_idx;
  }
  ~RefCell() = default;
};

struct RefCellHash {
  size_t operator()(const RefCell &c) const {
    return std::hash<std::string>()(c.hash_key);
  }
};

class RefRelations {
 public:
  graphStatus LookUpRefRelations(const RefCell &key, std::unordered_set<RefCell, RefCellHash> &result);
  graphStatus BuildRefRelations(ComputeGraph &graph);
  graphStatus Clear();

  RefRelations();
  ~RefRelations() = default;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_ = nullptr;
};

}  // namespace af
#endif  // COMMON_GRAPH_REF_RELATION_H_
