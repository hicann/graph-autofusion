/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_REGISTER_CORE_NUM_UTILS_H
#define INC_REGISTER_CORE_NUM_UTILS_H

#include "graph/ge_error_codes_af.h"
#include "platform/platform_info_def.h"
#include "platform/platform_infos_def.h"
#include "graph/op_desc.h"

namespace af {
const std::string kAiCoreCntIni = "ai_core_cnt";
const std::string kCubeCoreCntIni = "cube_core_cnt";
const std::string kVectorCoreCntIni = "vector_core_cnt";
const std::string kVectorCoreNum = "ge.vectorcoreNum";
const std::string kAiCoreNumOp = "_op_aicore_num";
const std::string kVectorCoreNumOp = "_op_vectorcore_num";
const std::string kSocInfo = "SoCInfo";

class CoreNumUtils {
 public:
  static graphStatus ParseAicoreNumFromOption(std::map<std::string, std::string> &options);

  static graphStatus ParseAndValidateCoreNum(const std::string &param_name, const std::string &param_value_str,
                                             int32_t min_value, int32_t max_value, int32_t &parsed_value);

  static graphStatus GetGeDefaultPlatformInfo(const std::string &soc_version, fe::PlatformInfo &platform_info);

  static graphStatus UpdateCoreCountWithOpDesc(const std::string &param_name, const std::string &op_core_num_str,
                                               int32_t soc_core_num, const std::string &res_key,
                                               std::map<std::string, std::string> &res);

  static graphStatus UpdatePlatformInfosWithOpDesc(const fe::PlatformInfo &platform_info, const ge::OpDescPtr &op_desc,
                                                   fe::PlatFormInfos &platform_infos, bool &is_op_core_num_set);
};
}  // namespace af

#endif  // INC_REGISTER_CORE_NUM_UTILS_H
