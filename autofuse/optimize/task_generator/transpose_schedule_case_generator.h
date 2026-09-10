/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_TRANSPOSE_SCHEDULE_CASE_GENERATOR_H
#define AUTOFUSE_TRANSPOSE_SCHEDULE_CASE_GENERATOR_H
#include "ascir.h"
#include "ascgen_log.h"
#include "task_generator/schedule_case_generator.h"

namespace optimize {
class TransposeFusionCaseGenerator : public FusionCaseGenerator {
 public:
  Status Generate(ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &graphs,
                  std::vector<std::string> &score_functions) override;
  bool HasLoadStoreConversion() const override {
    return true;
  }

 protected:
  virtual Status GenerateScoreFuncForUbReorder(const ascir::HintGraph &graph, const af::AscNodePtr &transpose_node,
                                               std::string &score_func);

 private:
  static std::vector<af::AscNodePtr> FindTransposeNodes(const ascir::HintGraph &owner_graph);
  Status TransposeNodeInputsAndOutputsCheck(const af::AscNodePtr &transpose_node) const;
  void UpdateAxisByPath(::ascir::ImplGraph &owner_graph, const af::NodePtr &input_node,
                        std::set<af::Node *> &visited_nodes, const std::vector<int64_t> &reordered_axis,
                        const std::vector<int64_t> &reordered_sched_axis) const;
  void UpdateAxis(ascir::HintGraph &graph, const af::AscNodePtr &transpose_node) const;
  Status TransposeConvertProcess(ascir::HintGraph &graph, const af::AscNodePtr &transpose_node) const;
};

class TransposeScoreFunctionGenerator {
 public:
  TransposeScoreFunctionGenerator(const ascir::HintGraph &graph, af::AscNodePtr transpose_node);
  Status Generate(std::string &score_func);

 private:
  Status ParseRepeat();
  Status GetScoreByExpr(int32_t &score) const;
  void GenerateReturnValue(const int32_t score);

  const ascir::HintGraph *graph_;
  af::AscNodePtr transpose_node_;
  af::Expression repeat_;
  std::stringstream ss_;
};
}  // namespace optimize
#endif  // AUTOFUSE_TRANSPOSE_SCHEDULE_CASE_GENERATOR_H
