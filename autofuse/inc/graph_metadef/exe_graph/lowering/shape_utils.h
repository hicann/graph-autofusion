/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_SHAPE_UTILS_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_SHAPE_UTILS_H_
#include <string>
#include <sstream>
#include "exe_graph/runtime/shape.h"
#include "graph/ge_error_codes_af.h"
#include "graph/types_af.h"

namespace gert {
extern const Shape g_vec_1_shape;
/**
 * 确保返回的shape是非scalar的。
 * 当一个shape的dim num为0时，此shape被认为表达了一个scalar。
 * 本函数在接受一个非scalar的shape时，会返回原有shape；在接收到scalar shape时，会返回返回一个{1}的vector shape
 * @param in_shape 输入shape
 * @return 保证非scalar的shape
 */
inline const Shape &EnsureNotScalar(const Shape &in_shape) {
  if (in_shape.IsScalar()) {
    return g_vec_1_shape;
  }
  return in_shape;
}
/**
 * 返回shape的字符串，本函数性能较低，不可以在执行时的正常流程中使用
 * @param shape 需要转为字符串的shape实例
 * @param join_char 每个Dim的间隔，默认为`,`
 * @return 转好的字符串
 */
inline std::string ShapeToString(const Shape &shape, const char *join_char = ",") {
  if (join_char == nullptr) {
    join_char = ",";
  }
  std::stringstream ss;
  for (size_t i = 0U; i < shape.GetDimNum(); ++i) {
    if (i > 0U) {
      ss << join_char;
    }
    ss << shape[i];
  }
  return ss.str();
}

ge::graphStatus CalcAlignedSizeByShape(const Shape &shape, ge::DataType data_type, uint64_t &ret_tensor_size);
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_SHAPE_UTILS_H_
