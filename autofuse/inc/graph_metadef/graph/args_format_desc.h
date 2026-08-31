/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_ARGS_FORMAT_H
#define METADEF_CXX_ARGS_FORMAT_H

#include <string>
#include <vector>

#include "framework/common/debug/ge_log.h"
#include "graph/ge_error_codes_af.h"
#include "graph/op_desc.h"
#include "graph/node.h"
#include "register/hidden_inputs_func_registry_af.h"
#include "graph/utils/args_format_desc_utils_af.h"

namespace af {

// Meaningful only for PLACEHOLDER and CUSTOM_VALUE.
enum class ArgsFormatWidth : int32_t {
  BIT64 = -1,
  BIT32 = -2,
};

struct SkArgDesc {
  AddrType addr_type;
  int32_t ir_idx;
  bool folded;
  AddrType sub_addr_type;
  int32_t sub_idx;
};
static_assert(std::is_standard_layout<SkArgDesc>::value, "The class SkArgDesc must be a POD");

struct SkArgDescV2 {
  AddrType addr_type;
  int32_t ir_idx;
  uint32_t reserved;
  AddrType sub_addr_type;
  int32_t sub_idx;
};
static_assert(std::is_standard_layout<SkArgDescV2>::value, "The class SkArgDescV2 must be a POD");

class ArgsFormatDesc {
 public:
  // i* -> ir_idx = -1, folded=false
  // 对于输入输出，idx表示ir定义的idx，-1表示所有输入、所有输出，此时非动态输入、输出默认展开,动态输出要i1*这样才表示展开
  // 对于workspace -1表示个数未知，folded暂时无意义
  // 对ffts尾块非尾块地址，idx=0表示非尾块，idx=1表示尾块
  // 对于其他类型, idx和fold 暂时没有意义
  void Append(AddrType type, int32_t ir_idx = -1, bool folded = false);

  void Clear();

  void AppendTilingContext(TilingContextSubType sub_type = TilingContextSubType::TILING_CONTEXT);
  void AppendCustomValue(uint64_t value, ArgsFormatWidth width = ArgsFormatWidth::BIT64);
  void AppendPlaceholder(ArgsFormatWidth width = ArgsFormatWidth::BIT64);

  std::string ToString() const;

  graphStatus GetArgsSize(const OpDescPtr &op_desc, size_t &args_size) const;

  static graphStatus GetArgSize(const OpDescPtr &op_desc, const ArgDesc arg_desc, size_t &arg_size);

  static graphStatus Parse(const OpDescPtr &op_desc, const std::string &str, std::vector<ArgDesc> &arg_descs);

  // 为了方便使用，字符串用i*这样的通配符时，返回的argDesc会按照实际个数展开
  // easy mode 不需要进行展开，只根据字面值做反序列化，允许不传op_desc
  static graphStatus Parse(const OpDescPtr &op_desc, const std::string &str, std::vector<ArgDesc> &arg_descs,
                           const bool easy_mode);

  // 抽取公共序列化/反序列化函数
  static std::string Serialize(const std::vector<ArgDesc> &arg_descs);

  using const_iterator = std::vector<ArgDesc>::const_iterator;
  const_iterator begin() const {
    return arg_descs_.begin();
  }
  const_iterator end() const {
    return arg_descs_.end();
  }

  static graphStatus FromString(ArgsFormatDesc &format, const OpDescPtr &op_desc, const std::string &str,
                                const bool easy_mode = false) {
    return Parse(op_desc, str, format.arg_descs_, easy_mode);
  }

  static graphStatus ConvertArgDescSkToNormal(const ArgDesc &sk_arg_desc, ArgDesc &arg_desc, int32_t &sub_op_id);

  static graphStatus ConvertToSuperKernelArgFormat(const NodePtr &sk_node, const NodePtr &sub_node,
                                                   const std::string &sub_node_arg_format, std::string &sk_arg_format);

 private:
  std::vector<ArgDesc> arg_descs_;
};
}  // namespace af

#endif  // METADEF_CXX_ARGS_FORMAT_H
