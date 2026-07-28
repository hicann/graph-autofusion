/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTS_AUTOFUSE_ST_ATT_UTILS_COMMON_GEN_UTILS_H_
#define TESTS_AUTOFUSE_ST_ATT_UTILS_COMMON_GEN_UTILS_H_

#include <string>
#include <map>
#include "base/base_types.h"

namespace att {
namespace test {

// 拼接tiling函数（参考test_concat.cpp和test_add_layer_norm.cpp）
void CombineTilings(const std::map<std::string, std::string> &tilings, std::string &result);

// 移除AutoFuse tiling head的宏保护
std::string RemoveAutoFuseTilingHeadGuards(const std::string &input);

// 为文件添加头文件宏保护
void AddHeaderGuardToFile(const std::string &file_name, const std::string &macro_name);

}  // namespace test
}  // namespace att

#endif  // TESTS_AUTOFUSE_ST_ATT_UTILS_COMMON_GEN_UTILS_H_
