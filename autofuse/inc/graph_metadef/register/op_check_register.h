/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_REGISTER_OP_CHECK_REGISTER_H_
#define INC_REGISTER_OP_CHECK_REGISTER_H_

#include <map>
#include <vector>
#include <memory>
#include <string>
#include <functional>

#include "graph/ascend_string_af.h"
#include "graph/operator_af.h"
#include "graph/ge_error_codes_af.h"
#include "register/op_def_af.h"

namespace af {
class Operator;
}

namespace optiling {
struct ReplayFuncParam {
  int32_t block_dim = 0;
  const char *tiling_data = nullptr;
  const char *kernel_name = nullptr;
  const char *entry_file = nullptr;
  int32_t gentype = 0;
  const char *output_kernel_file = nullptr;
  char **objptr = nullptr;
  int32_t task_ration = 0;
  int32_t tiling_key = 0;
};

using REPLAY_FUNC = int32_t (*)(ReplayFuncParam &param, const int32_t core_type);
using GEN_SIMPLIFIEDKEY_FUNC = bool (*)(const ge::Operator &op, ge::AscendString &result);

class OpCheckFuncRegistryImpl;
class OpCheckFuncRegistry {
 public:
  static void RegisterOpCapability(const ge::AscendString &check_type, const ge::AscendString &op_type,
                                   OP_CHECK_FUNC func);

  static OP_CHECK_FUNC GetOpCapability(const ge::AscendString &check_type, const ge::AscendString &op_type);

  static void RegisterGenSimplifiedKeyFunc(const ge::AscendString &op_type, GEN_SIMPLIFIEDKEY_FUNC func);

  static GEN_SIMPLIFIEDKEY_FUNC GetGenSimplifiedKeyFun(const ge::AscendString &op_type);

  static PARAM_GENERALIZE_FUNC GetParamGeneralize(const ge::AscendString &op_type);

  static void RegisterParamGeneralize(const ge::AscendString &op_type, PARAM_GENERALIZE_FUNC func);

  static void RegisterReplay(const ge::AscendString &op_type, const ge::AscendString &soc_version, REPLAY_FUNC func);
  static REPLAY_FUNC GetReplay(const ge::AscendString &op_type, const ge::AscendString &soc_version);

 private:
  static std::map<ge::AscendString, std::map<ge::AscendString, OP_CHECK_FUNC>> check_op_capability_instance_;
  static std::map<ge::AscendString, GEN_SIMPLIFIEDKEY_FUNC> gen_simplifiedkey_instance_;
  static std::map<ge::AscendString, PARAM_GENERALIZE_FUNC> param_generalize_instance_;
  static std::map<ge::AscendString, std::map<ge::AscendString, REPLAY_FUNC>> replay_instance_;
};

class ReplayFuncHelper {
 public:
  ReplayFuncHelper(const ge::AscendString &op_type, const ge::AscendString &soc_version, REPLAY_FUNC func);
};
}  // end of namespace optiling
#endif  // INC_REGISTER_OP_CHECK_REGISTER_H_
