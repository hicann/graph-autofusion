/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_EXTRA_INFO_GENERATOR_H_
#define ATT_EXTRA_INFO_GENERATOR_H_
#include <set>
#include <vector>
#include <string>
#include "common/checker.h"
#include "base/model_info.h"
#include "preprocess/args_manager.h"
#include "extra_info_config.h"
#include "tiling_data_gen/tiling_data_generator.h"
namespace att {
class ExtraInfoGenerator {
 public:
  ExtraInfoGenerator(const ExtraInfoConfig &config, const std::vector<ModelInfo> &model_info_list,
                     const TilingDataGenerator &tiling_data_manager)
      : config_(config), model_info_list_(model_info_list), tiling_data_generator_(tiling_data_manager) {}
  ~ExtraInfoGenerator() = default;
  /**
   * @brief 获取所有的modelinfo tilingdata字段的定义
   * @param model_info_list
   * @param type_name_to_definition_map tilingdata字段的类型名--tilingdata定义 例如 LoopNumData -- "struct LoopNumData
   * {... }"
   */
  af::Status GetExtraTilingDataDef(std::map<std::string, std::string> &type_name_to_definition);
  /**
   * @brief 获取tilingdata字段
   * @param const uint32_t tiling_key 一个modelinfo
   * @param tiling_vars 变量名
   */
  af::Status GetExtraTilingVars(const uint32_t tiling_key, std::set<std::string> &tiling_vars);

 private:
  std::string WriteCoreParamData(const ModelInfo &model_info, const TilingDataGenType tiling_data_gen_type,
                                 std::set<std::string> &tiling_data_vars);
  const ModelInfo *GetModelInfo(const uint32_t tiling_key) const;
  const ExtraInfoConfig &config_;
  const std::vector<ModelInfo> &model_info_list_;
  const TilingDataGenerator &tiling_data_generator_;
};
}  // namespace att
#endif  // ATT_EXTRA_INFO_GENERATOR_H_
