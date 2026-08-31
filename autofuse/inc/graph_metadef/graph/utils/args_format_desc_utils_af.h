/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_ARGS_FORMAT_DESC_UTILS_H
#define METADEF_CXX_ARGS_FORMAT_DESC_UTILS_H

#include <string>
#include <vector>

#include "graph/ge_error_codes_af.h"
#include "graph/op_desc.h"
#include "register/hidden_inputs_func_registry_af.h"

namespace af {
enum class AddrType {
  INPUT = 0,
  OUTPUT,
  WORKSPACE,
  TILING,
  INPUT_DESC,
  OUTPUT_DESC,
  FFTS_ADDR,
  OVERFLOW_ADDR,
  TILING_FFTS,
  HIDDEN_INPUT,
  TILING_CONTEXT,
  OP_TYPE,
  PLACEHOLDER,
  CUSTOM_VALUE,
  INPUT_INSTANCE,
  OUTPUT_INSTANCE,
  SUPER_KERNEL_SUB_NODE,
  EVENT_ADDR,
  MAX  // the end, add new value before MAX.
};

enum class TilingContextSubType {
  TILING_CONTEXT = -1,
  TILING_DATA,
  TILING_KEY,
  BLOCK_DIM,
  MAX  // the end, add new value before MAX.
};

// i* -> ir_idx = -1，folded=false
// 对于输入输出，idx表示ir定义的idx，-1表示所有输入、所有输出，此时非动态输入、输出默认展开，动态输出要i1*这样才表示展开
// 对于workspace -1表示个数未知，folded暂时无意义
// 对ffts尾块非尾块地址，idx=0表示非尾块，idx=1表示尾块
// 对于hidden input，支持多个，idx表示索引，从0开始，reserved字段表示类型（uint32）
// 对于custom value，reserved字段表示需要透传的值（uint64），其他字段无意义
// 对于其他类型, idx和fold暂时没有意义
struct ArgDesc {
  AddrType addr_type;
  int32_t ir_idx;
  bool folded;
  uint8_t reserved[8];
};
static_assert(std::is_standard_layout<ArgDesc>::value, "The class ArgDesc must be a POD");

class ArgsFormatDescUtils {
 public:
  static void Append(std::vector<ArgDesc> &arg_descs, AddrType type, int32_t ir_idx = -1, bool folded = false);

  static void AppendTilingContext(std::vector<ArgDesc> &arg_descs,
                                  TilingContextSubType sub_type = TilingContextSubType::TILING_CONTEXT);

  // insert_pos为插入位置，-1表示添加到最后，0表示添加到最前面，以此类推，注意不能超过arg_descs的个数，否则会报错
  // input_cnt为插入hidden input的个数
  static graphStatus InsertHiddenInputs(std::vector<ArgDesc> &arg_descs, int32_t insert_pos,
                                        HiddenInputsType hidden_type, size_t input_cnt = 1U);

  // insert_pos为插入位置，-1表示添加到最后，0表示添加到最前面，以此类推，注意不能超过arg_descs的个数，否则会报错
  static graphStatus InsertCustomValue(std::vector<ArgDesc> &arg_descs, int32_t insert_pos, uint64_t custom_value);

  static std::string ToString(const std::vector<ArgDesc> &arg_descs);

  // 字符串用i*这样的通配符时，返回的argDesc不会按照实际个数展开
  static graphStatus Parse(const std::string &str, std::vector<ArgDesc> &arg_descs);

  static std::string Serialize(const std::vector<ArgDesc> &arg_descs);
};
}  // namespace af

#endif  // METADEF_CXX_ARGS_FORMAT_DESC_UTILS_H
