/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <set>
#include <vector>
#include <sstream>
#include <algorithm>
#include "graph/node.h"
#include "common/checker.h"
#include "common_utils.h"
#include "base/base_types.h"
#include "api_tiling_gen/api_tiling_gen_register.h"

namespace att {
const std::string kPadType = "Pad";

af::Status GetPadHeightAndWidth(const af::AscTensor &node_input, af::Expression &height, af::Expression &width) {
  for (size_t i = 0; i < node_input.attr.vectorized_axis.size(); i++) {
    auto vectorized_axis_id = node_input.attr.vectorized_axis[i];
    auto pos = std::find(node_input.attr.axis.begin(), node_input.attr.axis.end(), vectorized_axis_id);
    GE_ASSERT_TRUE(pos != node_input.attr.axis.end(), "Incorrect axis ID in vectorized_axis");
    auto axis_id = static_cast<uint32_t>(pos - node_input.attr.axis.begin());
    if (i == 0) {
      height = node_input.attr.repeats[axis_id];
    } else if (i == node_input.attr.vectorized_axis.size() - 1) {
      width = node_input.attr.repeats[axis_id];
    } else {
      height = height * node_input.attr.repeats[axis_id];
    }
  }
  return af::SUCCESS;
}

af::Status GetPadTilingCall([[maybe_unused]] const std::string &tiling_data_type,
                            [[maybe_unused]] const af::AscGraph &graph, const af::AscNodePtr &node,
                            std::string &code_string, uint32_t tiling_case_id) {
  af::AscTensorAttr input_tensor_attr = node->inputs[0].attr;

  // 生成字段名
  const std::string field_name =
      ascgen_utils::GenValidName(node->GetName()) + "_tilingData_" + std::to_string(tiling_case_id);

  /* 增加optiling:: PadTiling函数调用 */
  std::ostringstream oss;
  oss << "optiling::PadTiling apiPadTiling;" << std::endl;

  // 添加函数调用
  af::Expression height;
  af::Expression width;
  GE_ASSERT_SUCCESS(GetPadHeightAndWidth(node->inputs[0], height, width), "GetPadHeightAndWidth Failed.");
  oss << "int64_t height = " << ascgen_utils::FormatExpression(Str(height)) << ";" << std::endl;
  oss << "int64_t width = " << ascgen_utils::FormatExpression(Str(width)) << ";" << std::endl;
  oss << "const ge::Shape srcShape{{height, width}};" << std::endl;
  oss << "uint32_t maxValue;" << std::endl;
  oss << "uint32_t minValue;" << std::endl;
  oss << "const uint32_t typeSize = " << GetSizeByDataType(input_tensor_attr.dtype) << ";" << std::endl;
  oss << "AscendC::GetPadMaxMinTmpSize(srcShape, typeSize, maxValue, minValue);" << std::endl;
  oss << "AscendC::PadTilingFunc(srcShape, srcShape, minValue, typeSize, apiPadTiling);\n";

  oss << "apiPadTiling.SaveToBuffer((void *)&tiling_data." << field_name << ", sizeof(PadTiling));" << std::endl;

  code_string = oss.str();
  return af::SUCCESS;
}

af::Status GetPadTilingDefine([[maybe_unused]] const std::string &tiling_data_type,
                              [[maybe_unused]] const af::AscGraph &graph, [[maybe_unused]] const af::AscNodePtr &node,
                              [[maybe_unused]] const std::string &code_string,
                              [[maybe_unused]] uint32_t tiling_case_id) {
  return af::SUCCESS;
}

af::Status GetPadTilingHeadFiles([[maybe_unused]] const std::string &tiling_data_type,
                                 [[maybe_unused]] const af::AscGraph &graph,
                                 [[maybe_unused]] const af::AscNodePtr &node, std::string &code_string,
                                 [[maybe_unused]] uint32_t tiling_case_id) {
  static constexpr char kHeaderContent[] = R"(
#include <vector>
#include <array>
#include "graph/tensor.h"
#include "lib/pad/pad_tilingdata.h"
#include "lib/pad/pad_tiling.h"
)";

  code_string = kHeaderContent;
  return af::SUCCESS;
}

REGISTER_API_TILING_FUNC(kPadType, GetPadTilingCall, GetPadTilingDefine, GetPadTilingHeadFiles);
}  // namespace att
