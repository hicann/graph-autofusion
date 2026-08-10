/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <exception>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include "codegen_kernel.h"
#include "codegen.h"
#include "optimize.h"
#include "share_graph.h"
#include "../backend_codegen_common.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "ascgraph_info_complete.h"
#include "ascgen_log.h"
#include "common/platform_context.h"
#include "runtime_stub.h"
#include "common_utils.h"
#include "../../../../../v35/codegen/reg_api_call/reg_api_call_utils.h"

namespace {
codegen::Tensor MakeCvFusionTensor(const ascir::TensorAttr &tensor_attr) {
  std::string dtype_name;
  EXPECT_EQ(codegen::Tensor::DtypeName(tensor_attr.attr.dtype, dtype_name), af::SUCCESS);
  codegen::Tensor tensor(tensor_attr, dtype_name);
  EXPECT_EQ(tensor.Init(), af::SUCCESS);
  return tensor;
}

struct CvFusionDataCopyTensors {
  codegen::Tensor gm;
  codegen::Tensor ub;
};

CvFusionDataCopyTensors MakeCvFusionDataCopyTensors() {
  af::AscGraph graph("cv_fusion_data_copy_params");
  af::ascir_op::Data data("data", graph);
  af::ascir_op::Load load("load");
  graph.AddNode(load);
  load.x = data.y;

  auto gm_node = graph.FindNode("data");
  gm_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  gm_node->outputs[0].attr.mem.tensor_id = 0;
  gm_node->outputs[0].attr.mem.reuse_id = 0;
  gm_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  gm_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  gm_node->outputs[0].attr.mem.position = af::Position::kPositionGM;
  gm_node->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto ub_node = graph.FindNode("load");
  ub_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  ub_node->outputs[0].attr.mem.tensor_id = 1;
  ub_node->outputs[0].attr.mem.reuse_id = 0;
  ub_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  ub_node->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  ub_node->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  ub_node->outputs[0].attr.opt.merge_scope = af::kIdNone;
  return {MakeCvFusionTensor(gm_node->outputs[0]), MakeCvFusionTensor(ub_node->outputs[0])};
}
}  // namespace

class TestBackendMatmulEleBrc : public testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v2 = std::make_shared<af::RuntimeStubV2>();
    ge::RuntimeStub::SetInstance(stub_v2);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::RuntimeStub::Reset();
  }
};

TEST_F(TestBackendMatmulEleBrc, MatmulEleBrcCodegen) {
  bool gen_success = true;
  const std::map<std::string, std::string> shape_info;
  auto graph = ascir::ShareGraph::LoadMatmulElewiseBrcFusedGraph();

  std::cout << "KERNEL_SRC_LIST=" << KERNEL_SRC_LIST << std::endl;
  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  std::string kernel_src_file_name =
      "matmul_elemwise_brc_test_kernel_ub.cpp";  // matmul_elemwise_brc_test_kernel_ub.cpp
  std::string tiling_src_file_name =
      "matmul_elemwise_brc_test_tiling_ub.cpp";      // matmul_elemwise_brc_test_tiling_ub.cpp
  std::string tiling_data_src_file_name = parts[2];  // autofuse_tiling_data.h

  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    codegen::Codegen codegen(codegen::CodegenOptions{});

    std::fstream kernel_file(kernel_src_file_name, std::ios::out);
    std::fstream tiling_file(tiling_src_file_name, std::ios::out);
    std::fstream tiling_data_file(tiling_data_src_file_name, std::ios::out);

    std::vector<::ascir::ScheduledResult> schedule_results;
    ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);
    codegen::CodegenResult result;
    ::ascir::FusedScheduledResult ub_schedule_result = fused_schedule_result;
    ::ascir::FusedScheduledResult common_schedule_result = fused_schedule_result;
    if (ascgen_utils::IsCubeFusedScheduled(fused_schedule_result)) {
      // 过滤CVFusion的UBResult ub模板结果
      ascgen_utils::FilterCVFusionUBResult(ub_schedule_result);
      // 过滤CVFusion的CommonResult 兜底模板结果
      ascgen_utils::FilterCVFusionCommonResult(common_schedule_result);
    }

    // 分别生成ub和common模板的kernel和tiling
    EXPECT_EQ(codegen.Generate(shape_info, ub_schedule_result, result), 0);
    const std::string ub_kernel = RemoveSubDirInclude(result.kernel);
    EXPECT_NE(ub_kernel.find("uint32_t curAivM, uint32_t curAivN, uint32_t curAlignN"), std::string::npos);
    EXPECT_NE(ub_kernel.find("uint16_t cv_m_loop_size = static_cast<uint16_t>(curAivM);"), std::string::npos);
    EXPECT_NE(ub_kernel.find("const int64_t output_dims_8[2] = {curAivM, curAivN};"), std::string::npos);
    EXPECT_NE(ub_kernel.find("const int64_t input_stride_8[2] = {1, 0};"), std::string::npos);
    EXPECT_NE(ub_kernel.find("const int64_t output_stride_8[2] = {KernelUtils::BlkAlign<float>(curAivN), 1};"),
              std::string::npos);
    EXPECT_NE(ub_kernel.find("DataCopyNddma(local_8, global_2[offset / shapeN], output_dims_8, "
                             "output_stride_8, input_stride_8);"),
              std::string::npos);
    EXPECT_NE(ub_kernel.find("const int64_t input_stride_10[2] = {0, 1};"), std::string::npos);
    EXPECT_NE(ub_kernel.find("DataCopyNddma(local_10, global_3[offset % shapeN + batch_num * shapeN], output_dims_10, "
                             "output_stride_10, input_stride_10);"),
              std::string::npos);
    EXPECT_NE(ub_kernel.find("const int64_t input_stride_11[2] = {0, 0};"), std::string::npos);
    EXPECT_NE(ub_kernel.find("AscendC::MicroAPI::LoadAlign(vreg_0, local_9 + 0 + cv_m * "
                             "KernelUtils::BlkAlign<float>(curAivN) + cv_n * ELEMENT_PER_VECTOR_LENGTH);"),
              std::string::npos);
    EXPECT_NE(ub_kernel.find("AscendC::MicroAPI::LoadAlign(vreg_1, local_7 + 0 + cv_m * curAlignN + cv_n * "
                             "ELEMENT_PER_VECTOR_LENGTH);"),
              std::string::npos);
    EXPECT_NE(ub_kernel.find("AscendC::MicroAPI::LoadAlign(vreg_2, local_11 + 0 + cv_m * "
                             "KernelUtils::BlkAlign<float>(curAivN) + cv_n * ELEMENT_PER_VECTOR_LENGTH);"),
              std::string::npos);
    EXPECT_NE(ub_kernel.find("AscendC::MicroAPI::StoreAlign(local_12 + 0 + cv_m * "
                             "KernelUtils::BlkAlign<float>(curAivN) + cv_n * ELEMENT_PER_VECTOR_LENGTH"),
              std::string::npos);
    EXPECT_NE(ub_kernel.find("DataCopyPadExtend<float, AscendC::PaddingMode::Normal>(global_4[offset], local_12[0], "
                             "curAivM, curAivN, (KernelUtils::BlkAlign<float>(curAivN) - curAivN), "
                             "(shapeN - curAivN));"),
              std::string::npos);
    kernel_file << ub_kernel;
    tiling_file << result.tiling;
    tiling_data_file << result.tiling_data;

    // 校验RemoveSubDirInclude(result.kernel)中是否包含IncludeMatmulHeadFiles方法返回的所有头文件内容
    std::vector<std::string> expected_headers = {"#include \"arch35/mat_mul_v3_tiling_key_public.h\"",
                                                 "#include \"arch35/mat_mul_tiling_data.h\"",
                                                 "#include \"mat_mul_v3_common.h\"",
                                                 "#include \"arch35/mat_mul_asw_block.h\"",
                                                 "#include \"arch35/mat_mul_asw_kernel.h\"",
                                                 "#include \"arch35/mat_mul_stream_k_block.h\"",
                                                 "#include \"arch35/mat_mul_stream_k_kernel.h\"",
                                                 "#include \"arch35/mat_mul_v3_full_load_kernel_helper.h\"",
                                                 "#include \"arch35/mat_mul_full_load.h\"",
                                                 "#include \"arch35/mm_extension_interface/mm_copy_cube_out.h\"",
                                                 "#include \"arch35/mm_extension_interface/mm_custom_mm_policy.h\"",
                                                 "#include \"arch35/mat_mul_fixpipe_opti.h\"",
                                                 "#include \"arch35/block_scheduler_aswt.h\"",
                                                 "#include \"arch35/block_scheduler_streamk.h\"",
                                                 "#include \"arch35/mat_mul_streamk_basic_cmct.h\"",
                                                 "#include \"arch35/mat_mul_fixpipe_opti_basic_cmct.h\"",
                                                 "#include \"arch35/mat_mul_input_k_eq_zero_clear_output.h\""};

    for (const auto &header : expected_headers) {
      EXPECT_NE(RemoveSubDirInclude(result.kernel).find(header), std::string::npos)
          << "Expected header not found in kernel: " << header;
    }

    kernel_src_file_name = "matmul_elemwise_brc_test_kernel_common.cpp";  // matmul_elemwise_brc_test_kernel_common.cpp
    tiling_src_file_name = "matmul_elemwise_brc_test_tiling_common.cpp";  // matmul_elemwise_brc_test_tiling_common.cpp
    tiling_data_src_file_name = parts[2];                                 // autofuse_tiling_data.h
    std::fstream kernel_file_common(kernel_src_file_name, std::ios::out);
    std::fstream tiling_file_common(tiling_src_file_name, std::ios::out);
    std::fstream tiling_data_file_common(tiling_data_src_file_name, std::ios::out);
    codegen::CodegenResult result_common;
    EXPECT_EQ(codegen.Generate(shape_info, common_schedule_result, result_common), 0);
    kernel_file_common << RemoveSubDirInclude(result_common.kernel);
    tiling_file_common << result_common.tiling;
    tiling_data_file_common << result_common.tiling_data;

    // 校验result_common.kernel中是否包含IncludeMatmulHeadFiles方法返回的所有头文件内容
    for (const auto &header : expected_headers) {
      EXPECT_NE(RemoveSubDirInclude(result_common.kernel).find(header), std::string::npos)
          << "Expected header not found in common kernel: " << header;
    }
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}

TEST_F(TestBackendMatmulEleBrc, MatmulToIntCastCvCodegen) {
  auto graph = ascir::ShareGraph::LoadMatmulToIntCastFusedGraph();
  GenerateCvBackendUbKernelWithCheck(
      graph, "matmul_to_int_cast_test_kernel_ub.cpp", "matmul_to_int_cast_test_tiling_ub.cpp", "autofuse_tiling_data.h",
      [](const std::string &kernel) {
        EXPECT_NE(kernel.find("AscendC::Cast(local_5[0], local_4[0], AscendC::RoundMode::CAST_RINT, "
                              "{ConvertToUint32(curAivM), ConvertToUint32(curAivN)}, "
                              "{ConvertToUint32(((curAivN + 8 - 1) / 8 * 8)), ConvertToUint32(1)}, "
                              "{ConvertToUint32(curAlignN), ConvertToUint32(1)});"),
                  std::string::npos);
        EXPECT_NE(kernel.find("CastExtend(local_6[0], local_5[0], "
                              "{ConvertToUint32(curAivM), ConvertToUint32(curAivN)}, "
                              "{ConvertToUint32(((curAivN + 8 - 1) / 8 * 8)), ConvertToUint32(1)}, "
                              "{ConvertToUint32(((curAivN + 8 - 1) / 8 * 8)), ConvertToUint32(1)});"),
                  std::string::npos);
        EXPECT_NE(kernel.find("AscendC::Cast(local_7[0], local_6[0], AscendC::RoundMode::CAST_TRUNC, "
                              "{ConvertToUint32(curAivM), ConvertToUint32(curAivN)}, "
                              "{ConvertToUint32(((curAivN + 8 - 1) / 8 * 8)), ConvertToUint32(1)}, "
                              "{ConvertToUint32(((curAivN + 8 - 1) / 8 * 8)), ConvertToUint32(1)});"),
                  std::string::npos);
        EXPECT_NE(kernel.find("AscendC::Cast(local_9[0], local_8[0], AscendC::RoundMode::CAST_FLOOR, "
                              "{ConvertToUint32(curAivM), ConvertToUint32(curAivN)}, "
                              "{ConvertToUint32(((curAivN + 8 - 1) / 8 * 8)), ConvertToUint32(1)}, "
                              "{ConvertToUint32(((curAivN + 8 - 1) / 8 * 8)), ConvertToUint32(1)});"),
                  std::string::npos);
        EXPECT_NE(kernel.find("KernelUtils::BlkAlign<int32_t>(curAivN)"), std::string::npos);
      });
}

TEST_F(TestBackendMatmulEleBrc, CvFusionDataCopyParamsUseDtypeAlignedFallback) {
  codegen::Tiler tiler;
  const CvFusionDataCopyTensors tensors = MakeCvFusionDataCopyTensors();
  std::string dtype_name = "half";

  codegen::ApiCallContext normal_context;
  EXPECT_EQ(codegen::GetCvInputAlignedSize(normal_context, tensors.gm, "curAivN"), "curAivN");
  codegen::ApiCallContext cv_context;
  cv_context.stage = codegen::ComputeStage::kCVFuseStage1;
  EXPECT_EQ(codegen::GetCvInputAlignedSize(cv_context, tensors.gm, "curAivN"), "((curAivN + 16 - 1) / 16 * 16)");

  codegen::TPipe ub_fuse_tpipe("tpipe", tiler);
  ub_fuse_tpipe.cv_fusion_type = ascir::CubeTemplateType::kUBFuse;
  codegen::CodegenApiParam ub_fuse_api_param;
  codegen::DmaSpecificParams ub_fuse_dma_params;
  codegen::BuildDataCopyApiParamInCVFusion(ub_fuse_tpipe, ub_fuse_api_param, ub_fuse_dma_params, tensors.gm, tensors.ub,
                                           dtype_name, true);
  EXPECT_EQ(ub_fuse_api_param.template_params[0], "AscendC::PaddingMode::Normal");
  EXPECT_EQ(ub_fuse_dma_params.data_copy_params.block_count.DebugStr(), "curAivM");
  EXPECT_EQ(ub_fuse_dma_params.data_copy_params.block_len.DebugStr(), "curAivN");
  EXPECT_EQ(ub_fuse_dma_params.data_copy_params.src_stride.DebugStr(), "(shapeN - curAivN)");
  EXPECT_EQ(ub_fuse_dma_params.data_copy_params.dst_stride.DebugStr(),
            "(KernelUtils::BlkAlign<half>(curAivN) - curAivN)");

  codegen::TPipe common_tpipe("tpipe", tiler);
  codegen::CodegenApiParam common_api_param;
  codegen::DmaSpecificParams common_dma_params;
  codegen::BuildDataCopyApiParamInCVFusion(common_tpipe, common_api_param, common_dma_params, tensors.gm, tensors.ub,
                                           dtype_name, true);
  EXPECT_EQ(common_dma_params.data_copy_params.block_len.DebugStr(), "load_block_len");
  EXPECT_EQ(common_dma_params.data_copy_params.src_stride.DebugStr(), "load_src_stride");
  EXPECT_EQ(common_dma_params.data_copy_params.dst_stride.DebugStr(), "load_dst_stride");
  ASSERT_EQ(common_api_param.api_post_process.size(), 1U);
  EXPECT_NE(common_api_param.api_post_process[0].find("AscendC::GatherMask"), std::string::npos);
  EXPECT_NE(common_api_param.api_post_process[0].find("KernelUtils::BlkAlign<half>(curAlignN)"), std::string::npos);

  codegen::CvApi2DParams cv_params;
  cv_params.first_dim = "curAivM";
  cv_params.last_dim = "curAivN";
  EXPECT_EQ(codegen::GenCvUint16Dims(cv_params), "{static_cast<uint16_t>(curAivM), static_cast<uint16_t>(curAivN)}");
  EXPECT_EQ(codegen::GenCvUint16Stride("curAlignN"), "{static_cast<uint16_t>(curAlignN), static_cast<uint16_t>(1)}");
}
