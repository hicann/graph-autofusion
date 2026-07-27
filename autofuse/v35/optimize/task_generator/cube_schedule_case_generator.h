/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCGEN_DEV_OPTIMIZE_TASK_GENERATOR_CUBE_SCHEDULE_CASE_GENERATOR_H_
#define ASCGEN_DEV_OPTIMIZE_TASK_GENERATOR_CUBE_SCHEDULE_CASE_GENERATOR_H_

#include <utility>

#include "ascir_ops.h"
#include "ascir/meta/ascir.h"
#include "common/ascgen_log.h"
#include "optimize/task_generator/schedule_case_generator.h"

namespace optimize {
class CubeFusionCaseGenerator : public FusionCaseGenerator {
 public:
  Status GeneratorTask(ascir::HintGraph &optimize_graph, std::vector<ScheduleTask> &tasks,
                       const OptimizerOptions &options) override;
  Status Generate(ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &graphs,
                  std::vector<std::string> &score_functions) override;

 private:
  Status GenerateGeneralCase(ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &graphs);
  static Status GeneratorUbTask(const std::vector<::ascir::ImplGraph> &grouped_graphs, ScheduleTask &ub_task,
                                std::vector<ScheduleTask> &tasks);
  static Status GenNddmaNode(const af::AscNodePtr &node_load, const af::AscNodePtr &node_brc, af::AscGraph &new_case);
  static Status SwapCastBrcAndGenNddma(const af::AscNodePtr &node_cast, const af::AscNodePtr &node_load,
                                       af::AscGraph &new_case);
  std::vector<af::AscNodePtr> node_order_{};
  std::vector<std::pair<std::string, std::string>> split_data_names_{};
  bool partition_ = false;
};
}  // namespace optimize

#endif  // ASCGEN_DEV_OPTIMIZE_TASK_GENERATOR_CUBE_SCHEDULE_CASE_GENERATOR_H_
