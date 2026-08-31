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
#include "graph/node.h"
#include "common/checker.h"
#include "common_utils.h"
#include "base/base_types.h"
#include "api_tiling_gen/api_tiling_gen_register.h"
#include "transpose_base_type.h"

namespace att {
const std::string kConfusionTransposeType = "Transpose";
std::string GetConfusionTransposeOnlyTilingFunc() {
  return R"(
// scene7：srcShape[H, W]
void GetConfusionTransposeOnlyTilingInfo(const ge::Shape &srcShape, const uint32_t stackBufferSize,
                                         const uint32_t typeSize, ConfusionTransposeTiling &tiling)
{
  (void)stackBufferSize;
  std::vector<int64_t> shapeDims = srcShape.GetDims();
  uint32_t blockSize = ONE_BLK_SIZE / typeSize;
  uint32_t height = shapeDims[0];
  uint32_t width = shapeDims[1];
  uint32_t highBlock = height / BLOCK_CUBE;
  uint32_t repeat = width / blockSize;

  uint32_t firstAxisAlign = ALIGN_UP(height, BLOCK_CUBE);
  uint32_t firstAxisRem = height % BLOCK_CUBE;
  uint32_t secondAxisAlign = ALIGN_UP(width, 16); // float32和float16的尾轴都对齐到16
  uint32_t secondAxisRem = width % blockSize;
  uint32_t stride = firstAxisAlign;

  tiling.param0 = height;
  tiling.param1 = width;
  tiling.param2 = highBlock;
  tiling.param3 = stride;
  tiling.param4 = blockSize;
  tiling.param5 = repeat;
  tiling.param6 = firstAxisAlign;
  tiling.param7 = firstAxisRem;
  tiling.param8 = secondAxisAlign;
  tiling.param9 = secondAxisRem;
}
)";
}

std::string AlignUpFunc() {
  return R"(
inline uint32_t ALIGN_UP(uint32_t origin_size, uint32_t align_num) {
  return (0 == (origin_size & (align_num - 1))) ? origin_size  :  (origin_size + align_num - (origin_size & (align_num - 1)));
}
)";
}

std::string GetConfusionTranspose102TilingFunc() {
  return R"(
// scene8：srcShape[s0,s1,s2] -> dstShape[s1,s0,s2]
void GetConfusionTranspose102TilingInfo(const ge::Shape &srcShape, const uint32_t stackBufferSize,
                                         const uint32_t typeSize, ConfusionTransposeTiling &tiling)
{
    (void)stackBufferSize;
    std::vector<int64_t> shapeDims = srcShape.GetDims();

    /* 尾轴要求block对齐 */
    uint32_t blockSize = ONE_BLK_SIZE / typeSize;

    uint32_t lastDim = shapeDims[2];
    uint32_t lastDimAlign = ALIGN_UP(lastDim, 16); // float32和float16的尾轴都对齐到16

    /* s2为单位搬移数据快大小，s1为blockCount，单位为block */
    uint32_t blockLen = lastDimAlign / blockSize;
    uint32_t blockCount = shapeDims[1];

    /* 输入连续，输出不连续，单位为block */
    uint32_t srcStride = 0;
    uint32_t dstStride = (shapeDims[0] * lastDimAlign - lastDimAlign) / blockSize;

    /* 输入连续，输出不连续，单位为数据个数 */
    uint32_t thirdDimSrcStride = shapeDims[1] * lastDimAlign;
    uint32_t thirdDimDstStride = lastDimAlign;
    uint32_t thirdDimCnt = shapeDims[0];

    tiling.param0 = blockLen;
    tiling.param1 = blockCount;
    tiling.param2 = srcStride;
    tiling.param3 = dstStride;
    tiling.param4 = thirdDimSrcStride;
    tiling.param5 = thirdDimDstStride;
    tiling.param6 = thirdDimCnt;
}
)";
}

std::string GetTransposeNLast4DCommonPrefix(const std::string &funcName) {
  return R"( void )" + funcName + R"((const ge::Shape &srcShape, const uint32_t stackBufferSize,
  const uint32_t typeSize, ConfusionTransposeTiling &tiling) {
    (void)stackBufferSize;

    std::vector<int64_t> shapeDims = srcShape.GetDims();

    /* 尾轴要求block对齐 */
    uint32_t blockSize = ONE_BLK_SIZE / typeSize;
    uint32_t lastDim = shapeDims[3];
    uint32_t lastDimAlign = ALIGN_UP(lastDim, 16); // float32和float16的尾轴都对齐到16

    /* s3为单位搬移数据快大小，s1为blockCount，单位为block */
    uint32_t blockLen = lastDimAlign / blockSize;
    uint32_t blockCount = shapeDims[2];

    /* 输入连续，输出不连续，单位为block */
    uint32_t srcStride = 0;
)";
}

std::string GetTransposeNLast4DCommonSuffix() {
  return R"(
    tiling.param0 = blockLen;
    tiling.param1 = blockCount;
    tiling.param2 = srcStride;
    tiling.param3 = dstStride;
    tiling.param4 = thirdDimSrcStride;
    tiling.param5 = thirdDimDstStride;
    tiling.param6 = thirdDimCnt;
    tiling.param7 = fourthDimSrcStride;
    tiling.param8 = fourthDimDstStride;
    tiling.param9 = fourthDimCnt;
}
)";
}

std::string GetConfusionTranspose0213TilingFunc() {
  return GetTransposeNLast4DCommonPrefix("GetConfusionTranspose0213TilingInfo") +
         R"(

    uint32_t dstStride = (shapeDims[1] * lastDimAlign - lastDimAlign) / blockSize;

    /* 输入连续，输出不连续，单位为数据个数,内层循环 */
    uint32_t thirdDimSrcStride = shapeDims[2] * lastDimAlign;
    uint32_t thirdDimDstStride = lastDimAlign;
    uint32_t thirdDimCnt = shapeDims[1];

    /* 输入连续，输出不连续，单位为数据个数,外层循环 */
    uint32_t fourthDimSrcStride = shapeDims[1] * shapeDims[2] * lastDimAlign;
    uint32_t fourthDimDstStride = shapeDims[1] * shapeDims[2] * lastDimAlign;
    uint32_t fourthDimCnt = shapeDims[0];
)" + GetTransposeNLast4DCommonSuffix();
}

std::string GetConfusionTranspose2103TilingFunc() {
  return GetTransposeNLast4DCommonPrefix("GetConfusionTranspose2103TilingInfo") +
         R"(
    uint32_t dstStride = (shapeDims[0] * shapeDims[1] * lastDimAlign - lastDimAlign) / blockSize;

    /* 输入连续，输出不连续，单位为数据个数,内层循环 */
    uint32_t thirdDimSrcStride = shapeDims[2] * lastDimAlign;
    uint32_t thirdDimDstStride = shapeDims[0] * lastDimAlign;
    uint32_t thirdDimCnt = shapeDims[1];

    /* 输入连续，输出不连续，单位为数据个数,外层循环 */
    uint32_t fourthDimSrcStride = shapeDims[1] * shapeDims[2] * lastDimAlign;
    uint32_t fourthDimDstStride = lastDimAlign;
    uint32_t fourthDimCnt = shapeDims[0];
)" + GetTransposeNLast4DCommonSuffix();
}

std::string GetTranspose3DCommonPrefix(const std::string &funcName) {
  return R"( void )" + funcName + R"((const ge::Shape &srcShape, const uint32_t stackBufferSize,
  const uint32_t typeSize, ConfusionTransposeTiling &tiling) {
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    uint32_t channel = shapeDims[0];
    uint32_t height = shapeDims[1];
    uint32_t width = shapeDims[2];

    uint32_t blockSize = ONE_BLK_SIZE / typeSize;
)";
}

std::string GetTranspose3DCommonSuffix() {
  return R"(
  uint32_t repeat = width / blockSize;

  tiling.param0 = height;
  tiling.param1 = width;
  tiling.param2 = channel;
  tiling.param3 = highBlock;
  tiling.param4 = stride;
  tiling.param5 = blockSize;
  tiling.param6 = repeat;
  tiling.param7 = firstAxisAlign;
  tiling.param8 = firstAxisRem;
  tiling.param9 = secondAxisAlign;
  tiling.param10 = secondAxisRem;
}
)";
}

// 021转置函数
std::string GetConfusionTranspose021TilingFunc() {
  return GetTranspose3DCommonPrefix("GetConfusionTranspose021TilingInfo") +
         R"(  uint32_t highBlock = height / BLOCK_CUBE;
  uint32_t firstAxisAlign = ALIGN_UP(height, BLOCK_CUBE);
  uint32_t firstAxisRem = height % BLOCK_CUBE;
  uint32_t secondAxisAlign = ALIGN_UP(width, 16); // float32和float16的尾轴都对齐到16
  uint32_t secondAxisRem = width % blockSize;

  uint32_t stride = firstAxisAlign;
)" + GetTranspose3DCommonSuffix();
}

// 210转置函数
std::string GetConfusionTranspose210TilingFunc() {
  return GetTranspose3DCommonPrefix("GetConfusionTranspose210TilingInfo") +
         R"(  uint32_t highBlock = channel / BLOCK_CUBE;
  uint32_t firstAxisAlign = ALIGN_UP(channel, BLOCK_CUBE);
  uint32_t firstAxisRem = channel % BLOCK_CUBE;
  uint32_t secondAxisAlign = ALIGN_UP(width, 16); // float32和float16的尾轴都对齐到16
  uint32_t secondAxisRem = width % blockSize;
  uint32_t stride = firstAxisAlign * height;
)" + GetTranspose3DCommonSuffix();
}

std::string GetConfusionTranspose0321TilingFunc() {
  return R"(
// scene13 shape[s0, s1, s2, s3]->shape[s0, s3, s2, s1]
void GetConfusionTranspose0321TilingInfo(const ge::Shape &srcShape, const uint32_t stackBufferSize,
                                         const uint32_t typeSize, ConfusionTransposeTiling &tiling) {
  std::vector<int64_t> shapeDims = srcShape.GetDims();
  uint32_t batch = shapeDims[0];
  uint32_t channel = shapeDims[1];
  uint32_t height = shapeDims[2];
  uint32_t width = shapeDims[3];

  uint32_t blockSize = ONE_BLK_SIZE / typeSize;
  uint32_t highBlock = channel / BLOCK_CUBE;
  uint32_t repeat = width / blockSize;

  uint32_t firstAxisAlign = ALIGN_UP(channel, BLOCK_CUBE);
  uint32_t firstAxisRem = channel % BLOCK_CUBE;
  uint32_t secondAxisAlign = ALIGN_UP(width, 16); // float32和float16的尾轴都对齐到16
  uint32_t secondAxisRem = width % blockSize;
  uint32_t stride = firstAxisAlign * height;

  tiling.param0 = height;
  tiling.param1 = width;
  tiling.param2 = channel;
  tiling.param3 = batch;
  tiling.param4 = highBlock;
  tiling.param5 = stride;
  tiling.param6 = blockSize;
  tiling.param7 = repeat;
  tiling.param8 = firstAxisAlign;
  tiling.param9 = firstAxisRem;
  tiling.param10 = secondAxisAlign;
  tiling.param11 = secondAxisRem;
}
)";
}

std::string GetConfusionTransposeTilingMainFunc() {
  return R"(
void GetConfusionTransposeTilingInfo(const ge::Shape &srcShape, const uint32_t stackBufferSize,
    const uint32_t typeSize, const uint32_t transposeTypeIn, ConfusionTransposeTiling &tiling)
{
    if (static_cast<AutoFuseTransposeType>(transposeTypeIn) == AutoFuseTransposeType::TRANSPOSE_ND2ND_ONLY) {
        GetConfusionTransposeOnlyTilingInfo(srcShape, stackBufferSize, typeSize, tiling);
    } else if (static_cast<AutoFuseTransposeType>(transposeTypeIn) == AutoFuseTransposeType::TRANSPOSE_ND2ND_102) {
        GetConfusionTranspose102TilingInfo(srcShape, stackBufferSize, typeSize, tiling);
    } else if (static_cast<AutoFuseTransposeType>(transposeTypeIn) == AutoFuseTransposeType::TRANSPOSE_ND2ND_0213) {
        GetConfusionTranspose0213TilingInfo(srcShape, stackBufferSize, typeSize, tiling);
    } else if (static_cast<AutoFuseTransposeType>(transposeTypeIn) == AutoFuseTransposeType::TRANSPOSE_ND2ND_2103) {
        GetConfusionTranspose2103TilingInfo(srcShape, stackBufferSize, typeSize, tiling);
    } else if (static_cast<AutoFuseTransposeType>(transposeTypeIn) == AutoFuseTransposeType::TRANSPOSE_ND2ND_021) {
        GetConfusionTranspose021TilingInfo(srcShape, stackBufferSize, typeSize, tiling);
    } else if (static_cast<AutoFuseTransposeType>(transposeTypeIn) == AutoFuseTransposeType::TRANSPOSE_ND2ND_210) {
        GetConfusionTranspose210TilingInfo(srcShape, stackBufferSize, typeSize, tiling);
    } else if (static_cast<AutoFuseTransposeType>(transposeTypeIn) == AutoFuseTransposeType::TRANSPOSE_ND2ND_0321) {
        GetConfusionTranspose0321TilingInfo(srcShape, stackBufferSize, typeSize, tiling);
    }
}
)";
}

af::Status GetConfusionTransposeTilingDefine([[maybe_unused]] const std::string &tiling_data_type,
                                             [[maybe_unused]] const af::AscGraph &graph,
                                             [[maybe_unused]] const af::AscNodePtr &node, std::string &code_string,
                                             [[maybe_unused]] uint32_t tiling_case_id) {
  static constexpr auto kTilingFuncs = {AlignUpFunc,
                                        GetConfusionTransposeOnlyTilingFunc,
                                        GetConfusionTranspose102TilingFunc,
                                        GetConfusionTranspose0213TilingFunc,
                                        GetConfusionTranspose2103TilingFunc,
                                        GetConfusionTranspose021TilingFunc,
                                        GetConfusionTranspose210TilingFunc,
                                        GetConfusionTranspose0321TilingFunc,
                                        GetConfusionTransposeTilingMainFunc};

  std::ostringstream oss;
  for (const auto &func : kTilingFuncs) {
    oss << func();
  }
  code_string = oss.str();

  return af::SUCCESS;
}

AutoFuseTransposeType ConvertPermuteToTransposeType(af::AscTensorAttr &input_tensor_attr,
                                                    af::AscTensorAttr &output_tensor_attr, Permutation &permutation) {
  std::vector<uint32_t> output_vectorized_axis;
  const auto &input_axes = input_tensor_attr.vectorized_axis;
  const auto &output_axes = output_tensor_attr.vectorized_axis;
  GELOGD("Got input vectorized_axis=[%s], output vectorized_axis=[%s]",
         DebugString(input_tensor_attr.vectorized_axis).c_str(),
         DebugString(output_tensor_attr.vectorized_axis).c_str());
  /* 建立输入到输出的轴映射关系 */
  for (auto output_axis : output_axes) {
    auto pos = std::find(input_axes.begin(), input_axes.end(), output_axis);
    if (pos != input_axes.end()) {
      output_vectorized_axis.emplace_back(static_cast<uint32_t>(pos - input_axes.begin()));
    }
  }
  auto it = kPermutationTable.find(output_vectorized_axis);
  if (it != kPermutationTable.end()) {
    permutation = it->first;
    return it->second.true_transpose_type;
  }
  std::ostringstream oss;
  for (size_t i = 0; i < output_vectorized_axis.size(); ++i) {
    if (i != 0) oss << ", ";
    oss << output_vectorized_axis[i];
  }
  GELOGE(af::FAILED, "Unsupported transpose pattern. Axes count: %zu, pattern: [%s]", output_vectorized_axis.size(),
         oss.str().c_str());
  return AutoFuseTransposeType::TRANSPOSE_INVALID;
}

std::vector<string> CreateShapeString(af::AscTensorAttr &input_tensor_attr, const af::AscGraph &graph,
                                      std::vector<size_t> indices) {
  std::map<af::AxisId, af::AxisPtr> axis_id_to_axis;
  std::vector<af::AxisPtr> asis_ptrs;
  std::vector<string> shapeString;
  asis_ptrs = graph.GetAllAxis();
  for (const auto &axis_ptr : asis_ptrs) {
    axis_id_to_axis[axis_ptr->id] = axis_ptr;
  }
  for (size_t index : indices) {
    auto &aixs_id = input_tensor_attr.axis[index];
    auto axis_ptr = axis_id_to_axis[aixs_id];
    std::string repeat = Str(input_tensor_attr.repeats[index]);
    if (axis_ptr->size.IsConstExpr()) {
      // 当repeat为常量表达式时，可以直接通过axis_ptr->size.Str().get()来获取值
      shapeString.emplace_back(std::string(axis_ptr->size.Str().get()));
    } else {
      shapeString.emplace_back(ascgen_utils::FormatExpression(repeat));
    }
  }
  return shapeString;
}

af::Status GenSrcShapeCode(af::AscTensorAttr &input_tensor_attr, const af::AscGraph &graph, std::string &shape_code,
                           Permutation &permutation) {
  std::vector<size_t> indices;
  std::vector<int32_t> merge_flags;
  size_t axis_num = input_tensor_attr.vectorized_axis.size();
  indices.reserve(axis_num);
  merge_flags.reserve(axis_num);
  for (size_t i = 0; i < axis_num; i++) {
    merge_flags.push_back(-1);  // 初始化merge_flags = -1, -1表示该axis未参与合轴
  }
  auto it = kPermutationTable.find(permutation);
  GE_ASSERT_TRUE(it != kPermutationTable.end(), "Permutation not found in Permutation Table");
  auto permute_param = it->second;
  auto merge_axis = permute_param.potential_axis_idx;
  for (size_t i = 0; i < axis_num; i++) {
    int64_t val = input_tensor_attr.vectorized_axis[i];
    auto it = std::find(input_tensor_attr.axis.begin(), input_tensor_attr.axis.end(), val);
    if (it == input_tensor_attr.axis.end()) {
      return af::FAILED;
    }
    indices.push_back(std::distance(input_tensor_attr.axis.begin(), it));
    for (size_t group_idx = 0; group_idx < merge_axis.size(); group_idx++) {
      auto axis_it = std::find(merge_axis[group_idx].begin(), merge_axis[group_idx].end(), permutation[i]);
      if (axis_it != merge_axis[group_idx].end()) {
        merge_flags[permutation[i]] = group_idx;
      }
    }
  }
  vector<string> shapeString = CreateShapeString(input_tensor_attr, graph, indices);
  std::string dims_str = "const std::vector<int64_t> dims{";
  for (size_t i = 0; i < merge_flags.size(); i++) {
    if (i == 0) {
      dims_str += shapeString[i];
      continue;
    }
    if (merge_flags[i] != -1 && merge_flags[i] == merge_flags[i - 1]) {
      dims_str += " * ";
    } else {
      dims_str += ", ";
    }
    dims_str += shapeString[i];
  }
  dims_str += "};\nge::Shape srcShape{dims};\n";
  shape_code = std::move(dims_str);
  return af::SUCCESS;
}

af::Status GetConfusionTransposeTilingCall([[maybe_unused]] const std::string &tiling_data_type,
                                           [[maybe_unused]] const af::AscGraph &graph, const af::AscNodePtr &node,
                                           std::string &code_string, uint32_t tiling_case_id) {
  /* 计算TransposeType */
  af::AscTensorAttr input_tensor_attr = node->inputs[0].attr;
  af::AscTensorAttr output_tensor_attr = node->outputs[0].attr;
  Permutation permutation;
  AutoFuseTransposeType transpose_type =
      ConvertPermuteToTransposeType(input_tensor_attr, output_tensor_attr, permutation);
  GE_ASSERT_TRUE(transpose_type != AutoFuseTransposeType::TRANSPOSE_INVALID, "node=%s, graph=%s",
                 node->GetName().c_str(), graph.GetName().c_str());

  const std::vector<std::string> kTransTypeValue = {
      "AutoFuseTransposeType::TRANSPOSE_ND2ND_ONLY", "AutoFuseTransposeType::TRANSPOSE_ND2ND_102",
      "AutoFuseTransposeType::TRANSPOSE_ND2ND_0213", "AutoFuseTransposeType::TRANSPOSE_ND2ND_2103",
      "AutoFuseTransposeType::TRANSPOSE_ND2ND_021",  "AutoFuseTransposeType::TRANSPOSE_ND2ND_210",
      "AutoFuseTransposeType::TRANSPOSE_ND2ND_0321", "AutoFuseTransposeType::TRANSPOSE_INVALID"};
  std::ostringstream oss;
  oss << "AutoFuseTransposeType transpose_type = " << kTransTypeValue[static_cast<uint8_t>(transpose_type)] << ";\n";

  std::string shape_code;
  GE_ASSERT_SUCCESS(GenSrcShapeCode(input_tensor_attr, graph, shape_code, permutation),
                    "GenSrcShapeCode failed, graph[%s], node[%s] tiling data type[%s]", graph.GetName().c_str(),
                    node->GetName().c_str(), tiling_data_type.c_str());
  oss << shape_code;

  // 生成字段名
  const std::string field_name =
      ascgen_utils::GenValidName(node->GetName()) + "_tilingData_" + std::to_string(tiling_case_id);

  /* 增加ConfusionTransposeTiling函数调用 */
  oss << "ConfusionTransposeTiling &apiConfusionTransposeTiling = tiling_data." << field_name << ";" << std::endl;

  // 添加函数调用
  oss << "uint32_t stackBufferSize = 0;" << std::endl;
  oss << "GetConfusionTransposeTilingInfo(srcShape, stackBufferSize, " << GetSizeByDataType(input_tensor_attr.dtype)
      << ", static_cast<uint32_t>(transpose_type), apiConfusionTransposeTiling);\n";

  code_string = oss.str();
  return af::SUCCESS;
}

af::Status GetConfusionTransposeTilingHeadFiles([[maybe_unused]] const std::string &tiling_data_type,
                                                [[maybe_unused]] const af::AscGraph &graph,
                                                [[maybe_unused]] const af::AscNodePtr &node, std::string &code_string,
                                                [[maybe_unused]] uint32_t tiling_case_id) {
  static constexpr char kHeaderContent[] = R"(
#include <vector>
#include <array>
#include "graph/tensor.h"

#ifndef AUTOFUSE_CONFUSION_TRANSPOSE_TILING_DEFS
#define AUTOFUSE_CONFUSION_TRANSPOSE_TILING_DEFS
using graphStatus = uint32_t;
const graphStatus GRAPH_FAILED = 0xFFFFFFFF;
const graphStatus GRAPH_SUCCESS = 0;
const uint32_t ONE_BLK_SIZE = 32;
const uint32_t BLOCK_CUBE = 16;

enum class AutoFuseTransposeType: uint8_t {
  TRANSPOSE_ND2ND_ONLY = 0,
  TRANSPOSE_ND2ND_102 = 1,
  TRANSPOSE_ND2ND_0213 = 2,
  TRANSPOSE_ND2ND_2103 = 3,
  TRANSPOSE_ND2ND_021 = 4,
  TRANSPOSE_ND2ND_210 = 5,
  TRANSPOSE_ND2ND_0321 = 6,
  TRANSPOSE_INVALID = 7
};
#endif
)";

  code_string = kHeaderContent;
  return af::SUCCESS;
}

REGISTER_API_TILING_FUNC(kConfusionTransposeType, GetConfusionTransposeTilingCall, GetConfusionTransposeTilingDefine,
                         GetConfusionTransposeTilingHeadFiles);
}  // namespace att
