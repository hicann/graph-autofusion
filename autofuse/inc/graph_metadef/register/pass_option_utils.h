/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_REGISTER_PASS_OPTION_UTILS_H
#define INC_REGISTER_PASS_OPTION_UTILS_H

#include "optimization_option_registry.h"
#include "graph/ge_error_codes_af.h"
namespace af {
class PassOptionUtils {
 public:
  static graphStatus CheckIsPassEnabled(const std::string &pass_name, bool &is_enabled);

  static graphStatus CheckIsPassEnabledByOption(const std::string &pass_name, bool &is_enabled);
};
}  // namespace af

#endif  // INC_REGISTER_PASS_OPTION_UTILS_H
