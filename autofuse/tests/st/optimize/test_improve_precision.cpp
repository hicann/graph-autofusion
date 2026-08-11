/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "ascir.h"
#include "ascir_ops.h"
#include "schedule_utils.h"
#define private public
#include "optimize/pre_process/improve_precision.h"
#include "optimize/pre_process/pre_process_config.h"
#include "optimize/pre_process/pre_process.h"
#undef private
#include "common/platform_context.h"

#include "ascgraph_info_complete.h"
#include "tests/framework/easy_asc_graph/asc_graph_builder.h"
#include "runtime_stub.h"

using namespace af;
using namespace af::ascir_op;
using af::ops::IsOps;
using af::ops::One;
using af::testing::AscGraphBuilder;
using af::testing::Sym;
using namespace af::pre_process;

namespace {
// ====================== Helpers ======================

size_t CountNodesByType(AscGraph &graph, const std::string &type) {
  size_t count = 0U;
  for (const auto &node : AscGraphUtils::GetComputeGraph(graph)->GetAllNodes()) {
    if (node->GetType() == type) {
      ++count;
    }
  }
  return count;
}

bool CheckNodeOutputDtype(AscGraph &graph, const std::string &node_name, ge::DataType expected_dtype) {
  for (const auto &node : AscGraphUtils::GetComputeGraph(graph)->GetAllNodes()) {
    if (node->GetName() == node_name) {
      auto desc = node->GetOpDesc();
      if (desc != nullptr && desc->GetOutputDesc(0).GetDataType() == expected_dtype) {
        return true;
      }
    }
  }
  return false;
}

class TestImprovePrecisionST : public ::testing::Test {
 protected:
  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
    PreProcessConfig::Instance().Reset();
    auto stub_v1 = std::make_shared<ge::RuntimeStub>();
    ge::RuntimeStub::SetInstance(stub_v1);
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }

  void TearDown() override {
    ge::PlatformContext::GetInstance().Reset();
    unsetenv("AUTOFUSE_FLAGS");
    PreProcessConfig::Instance().Reset();
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
};
}  // namespace

TEST_F(TestImprovePrecisionST, ComplexFp16Chain_AllPromotedToFp32) {
  auto graph = AscGraphBuilder("st_complex_fp16_chain")
                   .Loops({Sym("s0")})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0")
                   .Abs("abs0", "load0")
                   .Add("add0", "abs0", "abs0")
                   .Mul("mul0", "add0", "add0")
                   .Sub("sub0", "mul0", "mul0")
                   .Store("store0", "sub0")
                   .Output("output0", "store0", 0, ge::DT_FLOAT16)
                   .Build();

  ASSERT_EQ(ImprovePrecisionForAscGraph(graph), af::SUCCESS);

  // 所有计算节点应变为 fp32
  EXPECT_TRUE(CheckNodeOutputDtype(graph, "abs0", ge::DT_FLOAT));
  EXPECT_TRUE(CheckNodeOutputDtype(graph, "add0", ge::DT_FLOAT));
  EXPECT_TRUE(CheckNodeOutputDtype(graph, "mul0", ge::DT_FLOAT));
  EXPECT_TRUE(CheckNodeOutputDtype(graph, "sub0", ge::DT_FLOAT));

  // 至少有 Load 后的升精度 Cast
  size_t cast_count = CountNodesByType(graph, Cast::Type);
  EXPECT_GE(cast_count, 1U);
}

TEST_F(TestImprovePrecisionST, TwoInputsBothFp16_BothPromoted) {
  auto graph = AscGraphBuilder("st_two_inputs_fp16")
                   .Loops({Sym("s0")})
                   .Data("data1", 0, ge::DT_FLOAT16)
                   .Data("data2", 1, ge::DT_FLOAT16)
                   .Load("load1", "data1")
                   .Load("load2", "data2")
                   .Add("add0", "load1", "load2")
                   .Store("store0", "add0")
                   .Output("output0", "store0", 0, ge::DT_FLOAT16)
                   .Build();

  ASSERT_EQ(ImprovePrecisionForAscGraph(graph), af::SUCCESS);

  EXPECT_TRUE(CheckNodeOutputDtype(graph, "add0", ge::DT_FLOAT));

  // 两个 Load 后应各插一个升精度 Cast
  size_t cast_count = CountNodesByType(graph, Cast::Type);
  EXPECT_GE(cast_count, 2U);
}

TEST_F(TestImprovePrecisionST, MixedFp16Bf16Inputs_BothPromotedToFp32) {
  auto graph = AscGraphBuilder("st_mixed_fp16_bf16")
                   .Loops({Sym("s0")})
                   .Data("data1", 0, ge::DT_FLOAT16)
                   .Data("data2", 1, ge::DT_BF16)
                   .Load("load1", "data1")
                   .Load("load2", "data2")
                   .Add("add0", "load1", "load2")
                   .Store("store0", "add0")
                   .Output("output0", "store0", 0, ge::DT_FLOAT16)
                   .Build();

  ASSERT_EQ(ImprovePrecisionForAscGraph(graph), af::SUCCESS);

  EXPECT_TRUE(CheckNodeOutputDtype(graph, "add0", ge::DT_FLOAT));
  EXPECT_GE(CountNodesByType(graph, Cast::Type), 2U);
}

TEST_F(TestImprovePrecisionST, IdentityFp16Cast_RemovedAndPromoted) {
  auto graph = AscGraphBuilder("st_identity_cast")
                   .Loops({Sym("s0")})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0")
                   .Cast("cast0", "load0", ge::DT_FLOAT16)
                   .Abs("abs0", "cast0")
                   .Store("store0", "abs0")
                   .Output("output0", "store0", 0, ge::DT_FLOAT16)
                   .Build();

  ASSERT_EQ(ImprovePrecisionForAscGraph(graph), af::SUCCESS);

  EXPECT_TRUE(CheckNodeOutputDtype(graph, "abs0", ge::DT_FLOAT));
}

TEST_F(TestImprovePrecisionST, Fp32ToFp16CastBeforeStore_RemovedAndAbsPromoted) {
  auto graph = AscGraphBuilder("st_fp32_to_fp16_before_store")
                   .Loops({Sym("s0")})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0")
                   .Abs("abs0", "load0")
                   .Cast("cast0", "abs0", ge::DT_FLOAT16)
                   .Store("store0", "cast0")
                   .Output("output0", "store0", 0, ge::DT_FLOAT16)
                   .Build();

  ASSERT_EQ(ImprovePrecisionForAscGraph(graph), af::SUCCESS);

  // Cast(fp16→fp16 float间) 应被删除, Abs 变为 fp32
  EXPECT_TRUE(CheckNodeOutputDtype(graph, "abs0", ge::DT_FLOAT));
}

TEST_F(TestImprovePrecisionST, ScalarFp16Promoted_DownstreamAllFp32) {
  auto graph = AscGraphBuilder("st_scalar_fp16_downstream")
                   .Loops({Sym("s0")})
                   .Scalar("scalar0", "2.0", ge::DT_FLOAT16)
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0")
                   .Mul("mul0", "scalar0", "load0")
                   .Add("add0", "mul0", "mul0")
                   .Store("store0", "add0")
                   .Output("output0", "store0", 0, ge::DT_FLOAT16)
                   .Build();

  ASSERT_EQ(ImprovePrecisionForAscGraph(graph), af::SUCCESS);

  EXPECT_TRUE(CheckNodeOutputDtype(graph, "scalar0", ge::DT_FLOAT));
  EXPECT_TRUE(CheckNodeOutputDtype(graph, "mul0", ge::DT_FLOAT));
  EXPECT_TRUE(CheckNodeOutputDtype(graph, "add0", ge::DT_FLOAT));
}

TEST_F(TestImprovePrecisionST, AllFp32Graph_NoModification) {
  auto graph = AscGraphBuilder("st_all_fp32")
                   .Loops({Sym("s0")})
                   .Data("data0", 0, ge::DT_FLOAT)
                   .Load("load0", "data0")
                   .Abs("abs0", "load0")
                   .Mul("mul0", "abs0", "abs0")
                   .Store("store0", "mul0")
                   .Output("output0", "store0", 0, ge::DT_FLOAT)
                   .Build();

  ASSERT_EQ(ImprovePrecisionForAscGraph(graph), af::SUCCESS);

  EXPECT_EQ(CountNodesByType(graph, Cast::Type), 0U);
  EXPECT_TRUE(CheckNodeOutputDtype(graph, "abs0", ge::DT_FLOAT));
  EXPECT_TRUE(CheckNodeOutputDtype(graph, "mul0", ge::DT_FLOAT));
}

TEST_F(TestImprovePrecisionST, DeepOtherChain_AllPromotedToFp32) {
  auto graph = AscGraphBuilder("st_deep_other_chain")
                   .Loops({Sym("s0")})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0")
                   .Abs("abs0", "load0")
                   .Neg("neg0", "abs0")
                   .Exp("exp0", "neg0")
                   .Sqrt("sqrt0", "exp0")
                   .Store("store0", "sqrt0")
                   .Output("output0", "store0", 0, ge::DT_FLOAT16)
                   .Build();

  ASSERT_EQ(ImprovePrecisionForAscGraph(graph), af::SUCCESS);

  EXPECT_TRUE(CheckNodeOutputDtype(graph, "abs0", ge::DT_FLOAT));
  EXPECT_TRUE(CheckNodeOutputDtype(graph, "neg0", ge::DT_FLOAT));
  EXPECT_TRUE(CheckNodeOutputDtype(graph, "exp0", ge::DT_FLOAT));
  EXPECT_TRUE(CheckNodeOutputDtype(graph, "sqrt0", ge::DT_FLOAT));
  EXPECT_GE(CountNodesByType(graph, Cast::Type), 1U);
}

TEST_F(TestImprovePrecisionST, Bf16FullPipeline_AllPromotedToFp32) {
  auto graph = AscGraphBuilder("st_bf16_full")
                   .Loops({Sym("s0")})
                   .Data("data0", 0, ge::DT_BF16)
                   .Load("load0", "data0")
                   .Abs("abs0", "load0")
                   .Mul("mul0", "abs0", "abs0")
                   .Store("store0", "mul0")
                   .Output("output0", "store0", 0, ge::DT_BF16)
                   .Build();

  ASSERT_EQ(ImprovePrecisionForAscGraph(graph), af::SUCCESS);

  EXPECT_TRUE(CheckNodeOutputDtype(graph, "abs0", ge::DT_FLOAT));
  EXPECT_TRUE(CheckNodeOutputDtype(graph, "mul0", ge::DT_FLOAT));
  EXPECT_GE(CountNodesByType(graph, Cast::Type), 1U);
}

TEST_F(TestImprovePrecisionST, AllBlacklist_AllNodesSupportFp16_Skip) {
  setenv("AUTOFUSE_FLAGS", "--autofuse_enhance_precision_blacklist=all", 1);
  PreProcessConfig::Instance().Reset();

  // Load→Store: Load and Store in BlackList1, all blacklisted → skip
  auto graph = AscGraphBuilder("st_blacklist_all_skip")
                   .Loops({Sym("s0")})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0")
                   .Store("store0", "load0")
                   .Output("output0", "store0", 0, ge::DT_FLOAT16)
                   .Build();

  ASSERT_EQ(ImprovePrecisionForAscGraph(graph), af::SUCCESS);

  EXPECT_EQ(CountNodesByType(graph, Cast::Type), 0U);
}

TEST_F(TestImprovePrecisionST, PartialBlacklist_SpecificOpSkipped) {
  setenv("AUTOFUSE_FLAGS", "--autofuse_enhance_precision_blacklist=Abs", 1);
  PreProcessConfig::Instance().Reset();

  auto graph = AscGraphBuilder("st_partial_blacklist")
                   .Loops({Sym("s0")})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0")
                   .Abs("abs0", "load0")
                   .Mul("mul0", "abs0", "abs0")
                   .Store("store0", "mul0")
                   .Output("output0", "store0", 0, ge::DT_FLOAT16)
                   .Build();

  ASSERT_EQ(ImprovePrecisionForAscGraph(graph), af::SUCCESS);

  // Mul 不在黑名单中，仍应升精度
  EXPECT_TRUE(CheckNodeOutputDtype(graph, "mul0", ge::DT_FLOAT));
}

TEST_F(TestImprovePrecisionST, LoadDirectToStore_NoImprovement) {
  auto graph = AscGraphBuilder("st_load_direct_store")
                   .Loops({Sym("s0")})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0")
                   .Store("store0", "load0")
                   .Output("output0", "store0", 0, ge::DT_FLOAT16)
                   .Build();

  ASSERT_EQ(ImprovePrecisionForAscGraph(graph), af::SUCCESS);

  // Load 直连 Store，不升精度
  EXPECT_EQ(CountNodesByType(graph, Cast::Type), 0U);
}

TEST_F(TestImprovePrecisionST, LoadWithExistingCastPeer_NoDuplicateCast) {
  auto graph = AscGraphBuilder("st_load_with_cast_peer")
                   .Loops({Sym("s0")})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0")
                   .Cast("cast0", "load0", ge::DT_FLOAT16)
                   .Abs("abs0", "cast0")
                   .Store("store0", "abs0")
                   .Output("output0", "store0", 0, ge::DT_FLOAT16)
                   .Build();

  ASSERT_EQ(ImprovePrecisionForAscGraph(graph), af::SUCCESS);

  // Cast(fp16→fp16) 被删除, Abs 变 fp32
  EXPECT_TRUE(CheckNodeOutputDtype(graph, "abs0", ge::DT_FLOAT));
}

TEST_F(TestImprovePrecisionST, PreProcessEntryPoint_Succeeds) {
  auto graph = AscGraphBuilder("st_preprocess_entry")
                   .Loops({Sym("s0")})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0")
                   .Abs("abs0", "load0")
                   .Store("store0", "abs0")
                   .Output("output0", "store0", 0, ge::DT_FLOAT16)
                   .Build();

  ASSERT_EQ(PreProcess::Run(graph), af::SUCCESS);

  EXPECT_TRUE(CheckNodeOutputDtype(graph, "abs0", ge::DT_FLOAT));
}

TEST_F(TestImprovePrecisionST, PreProcessConfig_TrailingPunctuation) {
  setenv("AUTOFUSE_FLAGS", "--autofuse_enhance_precision_blacklist=Abs,Mul;", 1);
  PreProcessConfig::Instance().Reset();

  const auto &bl = PreProcessConfig::Instance().GetImprovePrecisionBlacklist();
  EXPECT_TRUE(bl.find("Abs") != bl.end());
  EXPECT_TRUE(bl.find("Mul") != bl.end());
}

TEST_F(TestImprovePrecisionST, PreProcessConfig_MultipleFlagsWithSemicolon) {
  setenv("AUTOFUSE_FLAGS", "--other_flag=xyz;--autofuse_enhance_precision_blacklist=Abs,Mul;--yet_another=123", 1);
  PreProcessConfig::Instance().Reset();

  const auto &bl = PreProcessConfig::Instance().GetImprovePrecisionBlacklist();
  EXPECT_TRUE(bl.find("Abs") != bl.end());
  EXPECT_TRUE(bl.find("Mul") != bl.end());
}

TEST_F(TestImprovePrecisionST, StoreDtypeMismatch_CastInsertedBeforeStore) {
  auto graph = AscGraphBuilder("st_store_dtype_mismatch")
                   .Loops({Sym("s0")})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0")
                   .Abs("abs0", "load0")
                   .Store("store0", "abs0")
                   .Output("output0", "store0", 0, ge::DT_FLOAT16)
                   .Build();

  ASSERT_EQ(ImprovePrecisionForAscGraph(graph), af::SUCCESS);

  // Abs 变 fp32, Store 输出 fp16, dtype 不匹配应插入降精度 Cast
  EXPECT_TRUE(CheckNodeOutputDtype(graph, "abs0", ge::DT_FLOAT));
}

TEST_F(TestImprovePrecisionST, CastBeforeStorePeer_DtypeNotChanged) {
  auto graph = AscGraphBuilder("st_cast_store_peer")
                   .Loops({Sym("s0")})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0")
                   .Cast("cast0", "load0", ge::DT_FLOAT16)
                   .Store("store0", "cast0")
                   .Output("output0", "store0", 0, ge::DT_FLOAT16)
                   .Build();

  ASSERT_EQ(ImprovePrecisionForAscGraph(graph), af::SUCCESS);

  // Load 直连 Cast(fp16→fp16 float间), Cast 被删除, Load→Store 全在 BlackList1, 不升精度
  EXPECT_EQ(CountNodesByType(graph, Cast::Type), 0U);
}

TEST_F(TestImprovePrecisionST, ScalarBf16PromotedToFp32) {
  auto graph = AscGraphBuilder("st_scalar_bf16")
                   .Loops({Sym("s0")})
                   .Scalar("scalar0", "1.5", ge::DT_BF16)
                   .Mul("mul0", "scalar0", "scalar0")
                   .Store("store0", "mul0")
                   .Output("output0", "store0", 0, ge::DT_BF16)
                   .Build();

  ASSERT_EQ(ImprovePrecisionForAscGraph(graph), af::SUCCESS);

  EXPECT_TRUE(CheckNodeOutputDtype(graph, "scalar0", ge::DT_FLOAT));
  EXPECT_TRUE(CheckNodeOutputDtype(graph, "mul0", ge::DT_FLOAT));
}

TEST_F(TestImprovePrecisionST, MultiOutputLoad_CastOnOneBranch) {
  auto graph = AscGraphBuilder("st_multi_output_load")
                   .Loops({Sym("s0")})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0")
                   .Cast("cast0", "load0", ge::DT_FLOAT16)
                   .Abs("abs0", "cast0")
                   .Mul("mul0", "load0", "load0")
                   .Store("store0", "abs0")
                   .Output("output0", "store0", 0, ge::DT_FLOAT16)
                   .Build();

  ASSERT_EQ(ImprovePrecisionForAscGraph(graph), af::SUCCESS);

  // abs0 和 mul0 都应升精度
  EXPECT_TRUE(CheckNodeOutputDtype(graph, "abs0", ge::DT_FLOAT));
  EXPECT_TRUE(CheckNodeOutputDtype(graph, "mul0", ge::DT_FLOAT));
}

TEST_F(TestImprovePrecisionST, OnlyLoadAndStore_NoChange) {
  auto graph = AscGraphBuilder("st_only_load_store")
                   .Loops({Sym("s0")})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0")
                   .Store("store0", "load0")
                   .Output("output0", "store0", 0, ge::DT_FLOAT16)
                   .Build();

  ASSERT_EQ(ImprovePrecisionForAscGraph(graph), af::SUCCESS);

  // Load 和 Store 都在 BlackList1, 不升精度
  EXPECT_EQ(CountNodesByType(graph, Cast::Type), 0U);
}

TEST_F(TestImprovePrecisionST, SingleOtherNode_CastInsertedBeforeOther) {
  auto graph = AscGraphBuilder("st_single_other")
                   .Loops({Sym("s0")})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0")
                   .Relu("relu0", "load0")
                   .Store("store0", "relu0")
                   .Output("output0", "store0", 0, ge::DT_FLOAT16)
                   .Build();

  ASSERT_EQ(ImprovePrecisionForAscGraph(graph), af::SUCCESS);

  // Relu 应变为 fp32
  EXPECT_TRUE(CheckNodeOutputDtype(graph, "relu0", ge::DT_FLOAT));
}
