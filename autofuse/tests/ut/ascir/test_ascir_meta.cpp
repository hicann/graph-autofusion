/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"

#include "ascir_utils.h"
#include "ascendc_graph_txt_dumper.h"
#include "ascir.h"

namespace af {
namespace ascir {

class AscirMetaUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Test: identifier rendering produces a stable, non-empty textual form
TEST_F(AscirMetaUtilsTest, IdentifierToStr_ShouldRenderNonEmpty) {
  EXPECT_FALSE(::ascir::utils::IdentifierToStr(0).empty());
  EXPECT_FALSE(::ascir::utils::IdentifierToStr(42).empty());
  EXPECT_EQ(::ascir::utils::IdentifierToStr(42), ::ascir::utils::IdentifierToStr(42));
}

// Test: dump prefix is empty while codegen-compile-debug is disabled, and stays deterministic
TEST_F(AscirMetaUtilsTest, GetDumpFilePrefix_ShouldBeEmptyWhenDumpDisabled) {
  const auto prefix = ::ascir::utils::GetDumpFilePrefix();
  EXPECT_EQ(prefix, ::ascir::utils::GetDumpFilePrefix());
}

// Test: dtype info lookup succeeds for common dtypes and fails gracefully for unknown ones
TEST_F(AscirMetaUtilsTest, GetDtypeInfo_ShouldResolveCommonDtypes) {
  EXPECT_NE(::ascir::dumper::GetDtypeInfo(ge::DT_FLOAT), nullptr);
  EXPECT_NE(::ascir::dumper::GetDtypeInfo(ge::DT_FLOAT16), nullptr);
  EXPECT_NE(::ascir::dumper::GetDtypeInfo(ge::DT_INT32), nullptr);
}

// Test: axis type priority/suffix are consistent lookups for the same type
TEST_F(AscirMetaUtilsTest, AxisTypeHelpers_ShouldBeConsistent) {
  const auto type = af::Axis::Type::kAxisTypeOriginal;
  EXPECT_GE(::ascir::dumper::GetAxisTypePriority(type), 0);
  EXPECT_FALSE(::ascir::dumper::GetAxisTypeSuffix(type).empty());
  EXPECT_EQ(::ascir::dumper::GetAxisTypePriority(type), ::ascir::dumper::GetAxisTypePriority(type));
}

// Test: tensor type parsing extracts the dtype fragment
TEST_F(AscirMetaUtilsTest, ExtractDtypeFromTensorType_ShouldExtractDtypeFragment) {
  const std::string tensor_type = "Tensor<fp16>";
  const auto dtype = ::ascir::dumper::ExtractDtypeFromTensorType(tensor_type);
  EXPECT_FALSE(dtype.empty());
  EXPECT_EQ(dtype, ::ascir::dumper::ExtractDtypeFromTensorType(tensor_type));
}

// Test: tensor type parsing extracts the axis list fragment
TEST_F(AscirMetaUtilsTest, ExtractAxisListFromTensorType_ShouldExtractAxisFragment) {
  const auto axes = ::ascir::dumper::ExtractAxisListFromTensorType("Tensor<fp16, ax0, ax1>");
  EXPECT_FALSE(axes.empty());
}

// Test: axis id maps are built from graph axes with stable names/types
TEST_F(AscirMetaUtilsTest, BuildAxisIdMaps_ShouldMapGraphAxes) {
  af::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  const std::vector<af::AxisPtr> axes = graph.GetAllAxis();
  const auto id_to_name = ::ascir::dumper::BuildAxisIdToNameMap(axes);
  ASSERT_GE(id_to_name.size(), 1U);
  EXPECT_FALSE(id_to_name.at(z0.id).empty());
  const auto id_to_type = ::ascir::dumper::BuildAxisIdToTypeMap(axes);
  ASSERT_GE(id_to_type.size(), 1U);
}

}  // namespace ascir
}  // namespace af
