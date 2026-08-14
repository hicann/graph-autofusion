/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "perf_param_v2.h"
#include "v35/att/api_perf_register/ascir_reduce_api_perf_v2.h"
#include "v35/att/api_perf_register/ascendc_regbase_perf.h"
#include "api_perf_register/api_perf_factory.h"
#include "api_perf_register/ascendc_api_perf.h"
namespace att {
namespace {
constexpr int32_t kMaxDmaLen = 4;
constexpr int32_t kMaxNddmaLen = 5;
PerfParamTableV2 perf_param_table_v2;
TilingScheduleConfigTableV2 tiling_schedule_config_table_v2;
ApiPerfRegister<ApiPerf> ApiPerfRegisterV2(const std::string &api_name, Perf perf_func, MicroPerfFunc micro_perf_func,
                                           const PerfParamTable *perf_param,
                                           const TilingScheduleConfigTable *tiling_schedule_config_table) {
  return ApiPerfRegister<ApiPerf>(api_name + "V2", perf_func, micro_perf_func, perf_param,
                                  tiling_schedule_config_table);
}
namespace ascir_v2 {
/*
LoadApi(DataCopy from GM to UB)的性能公式：（其中a-b-c-d-e为待拟合参数）
  1. 单次MTE2 = S(数据量Byte)/T + h(指令头开销)，针对非连续搬运场景会增加stride建模值(0.043 * (stride % (256) *
block_count))
  2. 总MTE2 = 单次MTE2 * 调用次数 + H(pipe启动头开销)
  当Shape > 256B时：
  3. H = 1174.3
  4. h = 34
  5. T = 11.8292 + 6.6155 / blockdim
     (单核的峰值带宽，核数越多，带宽抢占越严重，直到收敛到稳定值)
  当前Shape <= 256B时：
  3. H = 775.0
  4. h = 15.01
  5. T = 13.1355 + 6.4088 / blockdim
  6. mte2 = S/T + h
  7. overall_mte2 = mte2 * mte2_count + H
  8. 外抛for循环：最外侧4个维度丢到循环次数里面去
*/
af::Status LoadApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                   [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                   [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  auto const &node_ptr = node.node_ptr;
  GE_ASSERT_TRUE(!input_shapes.empty());
  GE_ASSERT_TRUE(!output_shapes.empty());
  std::string node_name = node_ptr != nullptr ? node_ptr->GetName() : "LoadNode";
  auto merged_output_shapes = output_shapes[0];
  GE_ASSERT_SUCCESS(MergeTensorContinuousDims(node_ptr, GetNodeOutTensorName(node_ptr, 0), merged_output_shapes));
  NodeDetail dma_info;
  dma_info.name = node_name;
  dma_info.optype = node_ptr->GetType();
  dma_info.input_dtype = {merged_output_shapes.data_type};
  dma_info.output_dtype = {merged_output_shapes.data_type};
  GE_ASSERT_SUCCESS(SetDims(merged_output_shapes, dma_info));
  GE_ASSERT_SUCCESS(GetDmaPerf(merged_output_shapes, dma_info, perf_res, kMaxDmaLen));
  return af::SUCCESS;
}

/*
NddmaApi(MultiDataCopy from GM to UB)的性能公式：
  1. 单次Nddma = S(数据量Byte)/T + h(指令头开销)
  2. 总Nddma = 单次Nddma * 调用次数 + H(pipe启动头开销)
  3. 当Shape > 256B时：H = 1174.3，当Shape <= 256B时：H = 775.0
  4. h = 418.9789
  5. T = 7.61 + 6.39 / blockdim
     (单核的峰值带宽，核数越多，带宽抢占越严重，直到收敛到稳定值)
  6. nddma = S/T + h
  7. overall_nddma = nddma * nddma_count + H
  8. 外抛for循环：最外侧4个维度丢到循环次数里面去
*/
af::Status NddmaApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                    [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                    [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  auto const &node_ptr = node.node_ptr;
  GE_ASSERT_TRUE(!input_shapes.empty());
  GE_ASSERT_TRUE(!output_shapes.empty());
  std::string node_name = node_ptr != nullptr ? node_ptr->GetName() : "NddmaNode";
  auto merged_output_shapes = output_shapes[0];
  GE_ASSERT_SUCCESS(MergeTensorContinuousDims(node_ptr, GetNodeOutTensorName(node_ptr, 0), merged_output_shapes));
  NodeDetail dma_info;
  dma_info.name = node_name;
  dma_info.optype = node_ptr->GetType();
  dma_info.input_dtype = {merged_output_shapes.data_type};
  dma_info.output_dtype = {merged_output_shapes.data_type};
  GE_ASSERT_SUCCESS(SetDims(merged_output_shapes, dma_info));
  GE_ASSERT_SUCCESS(GetDmaPerf(merged_output_shapes, dma_info, perf_res, kMaxNddmaLen, false));
  return af::SUCCESS;
}

/*
StoreApiV2(DataCopy from UB to GM)的性能公式：（其中a-b-c-d为待拟合参数）
  1. 单次MTE3 = S(数据量Byte)/T + h(指令头开销)，针对非连续搬运场景会增加stride建模值(k*(stride%(256)*block_count))
  2. 总MTE3 = 单次MTE3 * 调用次数 + H(pipe启动头开销)
  3. H = 571
  4. h = 160
  5. T = 11.774 + 10.265 / blockdim(单核的峰值带宽，核数越多，带宽抢占越严重，直到收敛到稳定值)
  6. mte3 = S/T + h
  7. overall_mte3 = mte3 * mte3_count + H
  8. 外抛for循环：最外侧4个维度丢到循环次数里面去
*/
af::Status StoreApiV2([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                      [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                      [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  auto const &store_node_ptr = node.node_ptr;
  GE_ASSERT_TRUE(!input_shapes.empty() && !output_shapes.empty());
  std::string store_node_name = store_node_ptr != nullptr ? store_node_ptr->GetName() : "StoreNode";
  auto merged_output_shapes = output_shapes[0];
  GE_ASSERT_SUCCESS(
      MergeTensorContinuousDims(store_node_ptr, GetNodeOutTensorName(store_node_ptr, 0), merged_output_shapes));
  NodeDetail dma_info;
  dma_info.name = store_node_name;
  dma_info.optype = store_node_ptr->GetType();
  dma_info.input_dtype = {merged_output_shapes.data_type};
  dma_info.output_dtype = {merged_output_shapes.data_type};
  GE_ASSERT_SUCCESS(SetDims(merged_output_shapes, dma_info));
  GE_ASSERT_SUCCESS(GetDmaPerf(merged_output_shapes, dma_info, perf_res, kMaxDmaLen));
  return af::SUCCESS;
}

inline af::Status CompareSpecificPerf(const std::string &mode, const NodeDetail &node_info, PerfOutputInfo &perf) {
  if (mode == kGe) {
    ascendcperf_v2::CompareGEPerf(node_info, perf);
  } else if (mode == kEq) {
    ascendcperf_v2::CompareEQPerf(node_info, perf);
  } else if (mode == kNe) {
    ascendcperf_v2::CompareNEPerf(node_info, perf);
  } else if (mode == kGt) {
    ascendcperf_v2::CompareGTPerf(node_info, perf);
  } else if (mode == kLe) {
    ascendcperf_v2::CompareLEPerf(node_info, perf);
  } else if (mode == kLt) {
    ascendcperf_v2::CompareLTPerf(node_info, perf);
  } else {
    GELOGW("compare mode %s is not registered", mode.c_str());
  }
  return af::SUCCESS;
}

af::Status CompareApiV2([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, const std::string &mode, PerfOutputInfo &perf_res) {
  GE_ASSERT_TRUE(input_shapes.size() >= 2U && !output_shapes.empty());
  NodeDetail node_info;
  Expr outer_repeat;
  vector<Expr> used_dims;
  GE_ASSERT_SUCCESS(GetOuterParams(output_shapes[0].dims, outer_repeat, used_dims));
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(SetDims(used_dims, node_info));
  node_info.compare_node_params = node.compare_node_params;
  GE_ASSERT_SUCCESS(CompareSpecificPerf(mode, node_info, perf_res));
  return af::SUCCESS;
}

af::Status CompareGeApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  return CompareApiV2(input_shapes, output_shapes, node, kGe, perf_res);
}

af::Status CompareEqApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  return CompareApiV2(input_shapes, output_shapes, node, kEq, perf_res);
}

af::Status CompareNeApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  return CompareApiV2(input_shapes, output_shapes, node, kNe, perf_res);
}

af::Status CompareGtApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  return CompareApiV2(input_shapes, output_shapes, node, kGt, perf_res);
}

af::Status CompareLeApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  return CompareApiV2(input_shapes, output_shapes, node, kLe, perf_res);
}

af::Status CompareLtApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  return CompareApiV2(input_shapes, output_shapes, node, kLt, perf_res);
}

af::Status AbsApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                  [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                  [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::AbsPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status ExpApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                  [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                  [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::ExpPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status LnApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                 [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                 [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::LnPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status SqrtApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                   [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                   [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::SqrtPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status RsqrtApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                    [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                    [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::RsqrtPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status DivApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                  [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                  [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::DivPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status ReciprocalApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                         [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                         [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::ReciprocalPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status ReluApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                   [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                   [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::ReluPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status NegApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                  [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                  [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::NegPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status AddApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                  [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                  [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::AddPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status SubApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                  [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                  [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::SubPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status MulApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                  [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                  [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::MulPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status LeakyReluApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::LeakyReluPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status CastApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                   [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                   [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  node_info.cast_node_params = node.cast_node_params;
  GE_ASSERT_SUCCESS(ascendcperf_v2::CastPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status RemovePadApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::RemovePadPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status WhereApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                    [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                    [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  node_info.where_node_params = node.where_node_params;
  GE_ASSERT_SUCCESS(ascendcperf_v2::WherePerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status PowApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                  [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                  [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::PowPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status ErfApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                  [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                  [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::ErfPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status TanhApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                   [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                   [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::TanhPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status SigmoidApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                      [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                      [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::SigmoidPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status GeluApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                   [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                   [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::GeluPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status SignApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                   [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                   [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::SignPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status LogicalNotApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                         [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                         [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::LogicalNotPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status LogicalOrApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::LogicalOrPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status LogicalAndApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                         [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                         [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::LogicalAndPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status ClipByValueApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                          [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                          [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::ClipByValuePerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status BitwiseAndApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                         [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                         [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::BitwiseAndPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status FloorDivApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                       [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                       [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::FloorDivPerf(node_info, perf_res));
  return af::SUCCESS;
}
}  // namespace ascir_v2

REGISTER_EVAL_FUNC_TAG(kStore, V2, ascir_v2::StoreApiV2);
REGISTER_EVAL_FUNC_TAG(kLoad, V2, ascir_v2::LoadApi);
REGISTER_EVAL_FUNC_TAG(kNddma, V2, ascir_v2::NddmaApi);
REGISTER_EVAL_FUNC_TAG(kGe, V2, ascir_v2::CompareGeApi);
REGISTER_EVAL_FUNC_TAG(kEq, V2, ascir_v2::CompareEqApi);
REGISTER_EVAL_FUNC_TAG(kNe, V2, ascir_v2::CompareNeApi);
REGISTER_EVAL_FUNC_TAG(kGt, V2, ascir_v2::CompareGtApi);
REGISTER_EVAL_FUNC_TAG(kLe, V2, ascir_v2::CompareLeApi);
REGISTER_EVAL_FUNC_TAG(kLt, V2, ascir_v2::CompareLtApi);
REGISTER_EVAL_FUNC_TAG(kAbs, V2, ascir_v2::AbsApi);
REGISTER_EVAL_FUNC_TAG(kExp, V2, ascir_v2::ExpApi);
REGISTER_EVAL_FUNC_TAG(kLn, V2, ascir_v2::LnApi);
REGISTER_EVAL_FUNC_TAG(kSqrt, V2, ascir_v2::SqrtApi);
REGISTER_EVAL_FUNC_TAG(kRsqrt, V2, ascir_v2::RsqrtApi);
REGISTER_EVAL_FUNC_TAG(kDiv, V2, ascir_v2::DivApi);
REGISTER_EVAL_FUNC_TAG(kReciprocal, V2, ascir_v2::ReciprocalApi);
REGISTER_EVAL_FUNC_TAG(kRelu, V2, ascir_v2::ReluApi);
REGISTER_EVAL_FUNC_TAG(kMax, V2, ascir_reduce_v2::MaxApi);
REGISTER_EVAL_FUNC_TAG(kMin, V2, ascir_reduce_v2::MinApi);
REGISTER_EVAL_FUNC_TAG(kMaximum, V2, ascir_reduce_v2::ElementwiseMaxApi);
REGISTER_EVAL_FUNC_TAG(kMinimum, V2, ascir_reduce_v2::ElementwiseMinApi);
REGISTER_EVAL_FUNC_TAG(kAny, V2, ascir_reduce_v2::AnyApi);
REGISTER_EVAL_FUNC_TAG(kAll, V2, ascir_reduce_v2::AllApi);
REGISTER_EVAL_FUNC_TAG(kReduceMax, V2, ascir_reduce_v2::ReduceMaxApi);
REGISTER_EVAL_FUNC_TAG(kReduceMin, V2, ascir_reduce_v2::ReduceMinApi);
REGISTER_EVAL_FUNC_TAG(kReduceAny, V2, ascir_reduce_v2::ReduceAnyApi);
REGISTER_EVAL_FUNC_TAG(kReduceAll, V2, ascir_reduce_v2::ReduceAllApi);
REGISTER_EVAL_FUNC_TAG(kReduceSum, V2, ascir_reduce_v2::ReduceSumApi);
REGISTER_EVAL_FUNC_TAG(kReduceMean, V2, ascir_reduce_v2::ReduceMeanApi);
REGISTER_EVAL_FUNC_TAG(kReduceProd, V2, ascir_reduce_v2::ReduceProdApi);
REGISTER_EVAL_FUNC_TAG(kNeg, V2, ascir_v2::NegApi);
REGISTER_EVAL_FUNC_TAG(kMean, V2, ascir_reduce_v2::MeanApi);
REGISTER_EVAL_FUNC_TAG(kAdd, V2, ascir_v2::AddApi);
REGISTER_EVAL_FUNC_TAG(kSub, V2, ascir_v2::SubApi);
REGISTER_EVAL_FUNC_TAG(kMul, V2, ascir_v2::MulApi);
REGISTER_EVAL_FUNC_TAG(kProd, V2, ascir_reduce_v2::ProdApi);
REGISTER_EVAL_FUNC_TAG(kLeakyRelu, V2, ascir_v2::LeakyReluApi);
REGISTER_EVAL_FUNC_TAG(kCast, V2, ascir_v2::CastApi);
REGISTER_EVAL_FUNC_TAG(kSum, V2, ascir_reduce_v2::SumApi);
REGISTER_EVAL_FUNC_TAG(kRemovePad, V2, ascir_v2::RemovePadApi);
REGISTER_EVAL_FUNC_TAG(kWhere, V2, ascir_v2::WhereApi);
REGISTER_EVAL_FUNC_TAG(kPow, V2, ascir_v2::PowApi);
REGISTER_EVAL_FUNC_TAG(kErf, V2, ascir_v2::ErfApi);
REGISTER_EVAL_FUNC_TAG(kTanh, V2, ascir_v2::TanhApi);
REGISTER_EVAL_FUNC_TAG(kSigmoid, V2, ascir_v2::SigmoidApi);
REGISTER_EVAL_FUNC_TAG(kGelu, V2, ascir_v2::GeluApi);
REGISTER_EVAL_FUNC_TAG(kSign, V2, ascir_v2::SignApi);
REGISTER_EVAL_FUNC_TAG(kLogicalNot, V2, ascir_v2::LogicalNotApi);
REGISTER_EVAL_FUNC_TAG(kLogicalOr, V2, ascir_v2::LogicalOrApi);
REGISTER_EVAL_FUNC_TAG(kLogicalAnd, V2, ascir_v2::LogicalAndApi);
REGISTER_EVAL_FUNC_TAG(kClipByValue, V2, ascir_v2::ClipByValueApi);
REGISTER_EVAL_FUNC_TAG(kBitwiseAnd, V2, ascir_v2::BitwiseAndApi);
REGISTER_EVAL_FUNC_TAG(kFloorDiv, V2, ascir_v2::FloorDivApi);
ApiPerfRegister<ApiPerf> add_api_perf_v2(ApiPerfRegisterV2(kAdd, GetPerfFunc(kAdd + "V2"), nullptr,
                                                           &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> gather_api_perf_v2(ApiPerfRegisterV2(kGather, GetPerfFunc(kGather), nullptr,
                                                              &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> indirect_load_api_perf_v2(ApiPerfRegisterV2(kIndirectLoad, GetPerfFunc(kUnitVector), nullptr,
                                                                     &perf_param_table_v2,
                                                                     &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> abs_api_perf_v2(ApiPerfRegisterV2(kAbs, GetPerfFunc(kAbs + "V2"), nullptr,
                                                           &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> broadcast_api_perf_v2(ApiPerfRegisterV2(kBroadcast, GetPerfFunc(kBroadcast), nullptr,
                                                                 &perf_param_table_v2,
                                                                 &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> cast_api_perf_v2(ApiPerfRegisterV2(kCast, GetPerfFunc(kCast + "V2"), nullptr,
                                                            &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> div_api_perf_v2(ApiPerfRegisterV2(kDiv, GetPerfFunc(kDiv + "V2"), nullptr,
                                                           &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> erf_api_perf_v2(ApiPerfRegisterV2(kErf, GetPerfFunc(kErf + "V2"), nullptr,
                                                           &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> exp_api_perf_v2(ApiPerfRegisterV2(kExp, GetPerfFunc(kExp + "V2"), nullptr,
                                                           &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> exp2_api_perf_v2(ApiPerfRegisterV2(kExp2, GetPerfFunc(kExp2 + "V2"), nullptr,
                                                            &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> floor_api_perf_v2(ApiPerfRegisterV2(kFloor, GetPerfFunc(kFloor + "V2"), nullptr,
                                                             &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> fma_api_perf_v2(ApiPerfRegisterV2(kFma, GetPerfFunc(kFma + "V2"), nullptr,
                                                           &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> bitwise_not_api_perf_v2(ApiPerfRegisterV2(kBitwiseNot, GetPerfFunc(kUnitVector), nullptr,
                                                                   &perf_param_table_v2,
                                                                   &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> bitwise_or_api_perf_v2(ApiPerfRegisterV2(kBitwiseOr, GetPerfFunc(kUnitVector), nullptr,
                                                                  &perf_param_table_v2,
                                                                  &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> bitwise_xor_api_perf_v2(ApiPerfRegisterV2(kBitwiseXor, GetPerfFunc(kUnitVector), nullptr,
                                                                   &perf_param_table_v2,
                                                                   &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> ceil_api_perf_v2(ApiPerfRegisterV2(kCeil, GetPerfFunc(kUnitVector), nullptr,
                                                            &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> cos_api_perf_v2(ApiPerfRegisterV2(kCos, GetPerfFunc(kUnitVector), nullptr,
                                                           &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> acos_api_perf_v2(ApiPerfRegisterV2(kAcos, GetPerfFunc(kUnitVector), nullptr,
                                                            &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> cosh_api_perf_v2(ApiPerfRegisterV2(kCosh, GetPerfFunc(kUnitVector), nullptr,
                                                            &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> atan2_api_perf_v2(ApiPerfRegisterV2(kAtan2, GetPerfFunc(kUnitVector), nullptr,
                                                             &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> copysign_api_perf_v2(ApiPerfRegisterV2(kCopySign, GetPerfFunc(kUnitVector), nullptr,
                                                                &perf_param_table_v2,
                                                                &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> ceil2int_api_perf_v2(ApiPerfRegisterV2(kCeil2Int, GetPerfFunc(kUnitVector), nullptr,
                                                                &perf_param_table_v2,
                                                                &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> logical_and_api_perf_v2(ApiPerfRegisterV2(kLogicalAnd, GetPerfFunc(kLogicalAnd + "V2"),
                                                                   nullptr, &perf_param_table_v2,
                                                                   &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> logical_or_api_perf_v2(ApiPerfRegisterV2(kLogicalOr, GetPerfFunc(kLogicalOr + "V2"), nullptr,
                                                                  &perf_param_table_v2,
                                                                  &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> logical_not_api_perf_v2(ApiPerfRegisterV2(kLogicalNot, GetPerfFunc(kLogicalNot + "V2"),
                                                                   nullptr, &perf_param_table_v2,
                                                                   &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> maximum_api_perf_v2(ApiPerfRegisterV2(kMaximum, GetPerfFunc(kMaximum + "V2"), nullptr,
                                                               &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> minimum_api_perf_v2(ApiPerfRegisterV2(kMinimum, GetPerfFunc(kMinimum + "V2"), nullptr,
                                                               &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> max_api_perf_v2(ApiPerfRegisterV2(kMax, GetPerfFunc(kMax + "V2"), nullptr,
                                                           &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> reduce_max_api_perf_v2(ApiPerfRegisterV2(kReduceMax, GetPerfFunc(kReduceMax + "V2"), nullptr,
                                                                  &perf_param_table_v2,
                                                                  &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> min_api_perf_v2(ApiPerfRegisterV2(kMin, GetPerfFunc(kMin + "V2"), nullptr,
                                                           &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> reduce_min_api_perf_v2(ApiPerfRegisterV2(kReduceMin, GetPerfFunc(kReduceMin + "V2"), nullptr,
                                                                  &perf_param_table_v2,
                                                                  &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> mul_api_perf_v2(ApiPerfRegisterV2(kMul, GetPerfFunc(kMul + "V2"), nullptr,
                                                           &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> neg_api_perf_v2(ApiPerfRegisterV2(kNeg, GetPerfFunc(kNeg + "V2"), nullptr,
                                                           &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> reciprocal_api_perf_v2(ApiPerfRegisterV2(kReciprocal, GetPerfFunc(kReciprocal + "V2"), nullptr,
                                                                  &perf_param_table_v2,
                                                                  &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> relu_api_perf_v2(ApiPerfRegisterV2(kRelu, GetPerfFunc(kRelu + "V2"), nullptr,
                                                            &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> remove_pad_api_perf_v2(ApiPerfRegisterV2(kRemovePad, GetPerfFunc(kRemovePad + "V2"), nullptr,
                                                                  &perf_param_table_v2,
                                                                  &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> rsqrt_api_perf_v2(ApiPerfRegisterV2(kRsqrt, GetPerfFunc(kRsqrt + "V2"), nullptr,
                                                             &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> sign_api_perf_v2(ApiPerfRegisterV2(kSign, GetPerfFunc(kSign + "V2"), nullptr,
                                                            &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> sqrt_api_perf_v2(ApiPerfRegisterV2(kSqrt, GetPerfFunc(kSqrt + "V2"), nullptr,
                                                            &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> sub_api_perf_v2(ApiPerfRegisterV2(kSub, GetPerfFunc(kSub + "V2"), nullptr,
                                                           &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> tanh_api_perf_v2(ApiPerfRegisterV2(kTanh, GetPerfFunc(kTanh + "V2"), nullptr,
                                                            &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> sin_api_perf_v2(ApiPerfRegisterV2(kSin, GetPerfFunc(kSin + "V2"), nullptr,
                                                           &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> asin_api_perf_v2(ApiPerfRegisterV2(kAsin, GetPerfFunc(kUnitVector), nullptr,
                                                            &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> asinh_api_perf_v2(ApiPerfRegisterV2(kAsinh, GetPerfFunc(kUnitVector), nullptr,
                                                             &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> atan_api_perf_v2(ApiPerfRegisterV2(kAtan, GetPerfFunc(kUnitVector), nullptr,
                                                            &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> atanh_api_perf_v2(ApiPerfRegisterV2(kAtanh, GetPerfFunc(kUnitVector), nullptr,
                                                             &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> digamma_api_perf_v2(ApiPerfRegisterV2(kDigamma, GetPerfFunc(kUnitVector), nullptr,
                                                               &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> erfc_api_perf_v2(ApiPerfRegisterV2(kErfc, GetPerfFunc(kUnitVector), nullptr,
                                                            &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> erfcx_api_perf_v2(ApiPerfRegisterV2(kErfcx, GetPerfFunc(kUnitVector), nullptr,
                                                             &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> acosh_api_perf_v2(ApiPerfRegisterV2(kAcosh, GetPerfFunc(kUnitVector), nullptr,
                                                             &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> rshift_api_perf_v2(ApiPerfRegisterV2(kRShift, GetPerfFunc(kRShift + "V2"), nullptr,
                                                              &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> where_api_perf_v2(ApiPerfRegisterV2(kWhere, GetPerfFunc(kWhere + "V2"), nullptr,
                                                             &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> select_api_perf_v2(ApiPerfRegisterV2(kSelect, GetPerfFunc(kWhere + "V2"), nullptr,
                                                              &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> ge_api_perf_v2(ApiPerfRegisterV2(kGe, GetPerfFunc(kGe + "V2"), nullptr, &perf_param_table_v2,
                                                          &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> eq_api_perf_v2(ApiPerfRegisterV2(kEq, GetPerfFunc(kEq + "V2"), nullptr, &perf_param_table_v2,
                                                          &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> ne_api_perf_v2(ApiPerfRegisterV2(kNe, GetPerfFunc(kNe + "V2"), nullptr, &perf_param_table_v2,
                                                          &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> gt_api_perf_v2(ApiPerfRegisterV2(kGt, GetPerfFunc(kGt + "V2"), nullptr, &perf_param_table_v2,
                                                          &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> le_api_perf_v2(ApiPerfRegisterV2(kLe, GetPerfFunc(kLe + "V2"), nullptr, &perf_param_table_v2,
                                                          &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> lt_api_perf_v2(ApiPerfRegisterV2(kLt, GetPerfFunc(kLt + "V2"), nullptr, &perf_param_table_v2,
                                                          &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> ub2ub_api_perf_v2(ApiPerfRegisterV2(kUb2ub, GetPerfFunc(kUb2ub), nullptr, &perf_param_table_v2,
                                                             &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> load_api_perf_v2(ApiPerfRegisterV2(kLoad, GetPerfFunc(kLoad + "V2"), nullptr,
                                                            &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> store_api_perf_v2(ApiPerfRegisterV2(kStore, GetPerfFunc(kStore + "V2"), nullptr,
                                                             &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> nddma_api_perf_v2(ApiPerfRegisterV2(kNddma, GetPerfFunc(kNddma + "V2"), nullptr,
                                                             &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> all_api_perf_v2(ApiPerfRegisterV2(kAll, GetPerfFunc(kAll + "V2"), nullptr,
                                                           &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> reduce_all_api_perf_v2(ApiPerfRegisterV2(kReduceAll, GetPerfFunc(kReduceAll + "V2"), nullptr,
                                                                  &perf_param_table_v2,
                                                                  &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> any_api_perf_v2(ApiPerfRegisterV2(kAny, GetPerfFunc(kAny + "V2"), nullptr,
                                                           &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> reduce_any_api_perf_v2(ApiPerfRegisterV2(kReduceAny, GetPerfFunc(kReduceAny + "V2"), nullptr,
                                                                  &perf_param_table_v2,
                                                                  &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> reduce_mean_api_perf_v2(ApiPerfRegisterV2(kReduceMean, GetPerfFunc(kReduceMean + "V2"),
                                                                   nullptr, &perf_param_table_v2,
                                                                   &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> reduce_prod_api_perf_v2(ApiPerfRegisterV2(kReduceProd, GetPerfFunc(kReduceProd + "V2"),
                                                                   nullptr, &perf_param_table_v2,
                                                                   &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> reduce_sum_api_perf_v2(ApiPerfRegisterV2(kReduceSum, GetPerfFunc(kReduceSum + "V2"), nullptr,
                                                                  &perf_param_table_v2,
                                                                  &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> mean_api_perf_v2(ApiPerfRegisterV2(kMean, GetPerfFunc(kMean + "V2"), nullptr,
                                                            &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> prod_api_perf_v2(ApiPerfRegisterV2(kProd, GetPerfFunc(kProd + "V2"), nullptr,
                                                            &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> sum_api_perf_v2(ApiPerfRegisterV2(kSum, GetPerfFunc(kSum + "V2"), nullptr,
                                                           &perf_param_table_v2, &tiling_schedule_config_table_v2));
// 不需要建模的ASCIR
ApiPerfRegister<ApiPerf> data_api_perf_v2(ApiPerfRegisterV2(kData, DefaultGetPerf, nullptr, &perf_param_table_v2,
                                                            &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> scalar_api_perf_v2(ApiPerfRegisterV2(kScalar, DefaultGetPerf, nullptr, &perf_param_table_v2,
                                                              &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> scalar_data_api_perf_v2(ApiPerfRegisterV2(kScalarData, DefaultGetPerf, nullptr,
                                                                   &perf_param_table_v2,
                                                                   &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> index_expr_api_perf_v2(ApiPerfRegisterV2(kIndexExpr, DefaultGetPerf, nullptr,
                                                                  &perf_param_table_v2,
                                                                  &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> output_api_perf_v2(ApiPerfRegisterV2(kOutput, DefaultGetPerf, nullptr, &perf_param_table_v2,
                                                              &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> workspace_api_perf_v2(ApiPerfRegisterV2(kWorkspace, DefaultGetPerf, nullptr,
                                                                 &perf_param_table_v2,
                                                                 &tiling_schedule_config_table_v2));
// 目前无建模的ASCIR
ApiPerfRegister<ApiPerf> pad_api_perf_v2(ApiPerfRegisterV2(kPad, GetPerfFunc(kUnitVector), nullptr,
                                                           &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> round_api_perf_v2(ApiPerfRegisterV2(kRound, GetPerfFunc(kUnitVector), nullptr,
                                                             &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> nop_api_perf_v2(ApiPerfRegisterV2(kNop, GetPerfFunc(kUnitVector), nullptr,
                                                           &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> ln_api_perf_v2(ApiPerfRegisterV2(kLn, GetPerfFunc(kLn + "V2"), nullptr, &perf_param_table_v2,
                                                          &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> floor_to_int_api_perf_v2(ApiPerfRegisterV2(kFloorToInt, GetPerfFunc(kFloorToInt + "V2"),
                                                                    nullptr, &perf_param_table_v2,
                                                                    &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> fmod_api_perf_v2(ApiPerfRegisterV2(kFmod, GetPerfFunc(kFmod + "V2"), nullptr,
                                                            &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> hypot_api_perf_v2(ApiPerfRegisterV2(kHypot, GetPerfFunc(kHypot + "V2"), nullptr,
                                                             &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> lgamma_api_perf_v2(ApiPerfRegisterV2(kLgamma, GetPerfFunc(kLgamma + "V2"), nullptr,
                                                              &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> log10_api_perf_v2(ApiPerfRegisterV2(kLog10, GetPerfFunc(kLog10 + "V2"), nullptr,
                                                             &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> logical_xor_api_perf_v2(ApiPerfRegisterV2(kLogicalXor, GetPerfFunc(kLogicalXor + "V2"),
                                                                   nullptr, &perf_param_table_v2,
                                                                   &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> log1p_api_perf_v2(ApiPerfRegisterV2(kLog1p, GetPerfFunc(kLog1p + "V2"), nullptr,
                                                             &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> expm_api_perf_v2(ApiPerfRegisterV2(kExpm, GetPerfFunc(kExpm + "V2"), nullptr,
                                                            &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> log2_api_perf_v2(ApiPerfRegisterV2(kLog2, GetPerfFunc(kLog2 + "V2"), nullptr,
                                                            &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> lShift_api_perf_v2(ApiPerfRegisterV2(kLShift, GetPerfFunc(kLShift + "V2"), nullptr,
                                                              &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> mod_api_perf_v2(ApiPerfRegisterV2(kMod, GetPerfFunc(kMod + "V2"), nullptr,
                                                           &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> isnan_api_perf_v2(ApiPerfRegisterV2(kIsnan, GetPerfFunc(kUnitVector), nullptr,
                                                             &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> isfinite_api_perf_v2(ApiPerfRegisterV2(kIsFinite, GetPerfFunc(kUnitVector), nullptr,
                                                                &perf_param_table_v2,
                                                                &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> isinf_api_perf_v2(ApiPerfRegisterV2(kIsInf, GetPerfFunc(kUnitVector), nullptr,
                                                             &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> maskedfill_api_perf_v2(ApiPerfRegisterV2(kMaskedFill, GetPerfFunc(kUnitVector), nullptr,
                                                                  &perf_param_table_v2,
                                                                  &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> sigmoid_api_perf_v2(ApiPerfRegisterV2(kSigmoid, GetPerfFunc(kSigmoid + "V2"), nullptr,
                                                               &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> true_div_api_perf_v2(ApiPerfRegisterV2(kTrueDiv, GetPerfFunc(kDiv + "V2"), nullptr,
                                                                &perf_param_table_v2,
                                                                &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> pow_api_perf_v2(ApiPerfRegisterV2(kPow, GetPerfFunc(kPow + "V2"), nullptr,
                                                           &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> clip_by_value_api_perf_v2(ApiPerfRegisterV2(kClipByValue, GetPerfFunc(kClipByValue + "V2"),
                                                                     nullptr, &perf_param_table_v2,
                                                                     &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> concat_api_perf_v2(ApiPerfRegisterV2(kConcat, GetPerfFunc(kUnitVector), nullptr,
                                                              &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> leaky_relu_api_perf_v2(ApiPerfRegisterV2(kLeakyRelu, GetPerfFunc(kLeakyRelu + "V2"), nullptr,
                                                                  &perf_param_table_v2,
                                                                  &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> bitwise_and_api_perf_v2(ApiPerfRegisterV2(kBitwiseAnd, GetPerfFunc(kBitwiseAnd + "V2"),
                                                                   nullptr, &perf_param_table_v2,
                                                                   &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> transpose_api_perf_v2(ApiPerfRegisterV2(kTranspose, GetPerfFunc(kUnitVector), nullptr,
                                                                 &perf_param_table_v2,
                                                                 &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> floor_div_api_perf_v2(ApiPerfRegisterV2(kFloorDiv, GetPerfFunc(kFloorDiv + "V2"), nullptr,
                                                                 &perf_param_table_v2,
                                                                 &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> gelu_api_perf_v2(ApiPerfRegisterV2(kGelu, GetPerfFunc(kGelu + "V2"), nullptr,
                                                            &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> trunc_api_perf_v2(ApiPerfRegisterV2(kTrunc, GetPerfFunc(kTrunc + "V2"), nullptr,
                                                             &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> tan_api_perf_v2(ApiPerfRegisterV2(kTan, GetPerfFunc(kTan + "V2"), nullptr,
                                                           &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> sinh_api_perf_v2(ApiPerfRegisterV2(kSinh, GetPerfFunc(kSinh + "V2"), nullptr,
                                                            &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> trunc_div_api_perf_v2(ApiPerfRegisterV2(kTruncDiv, GetPerfFunc(kTruncDiv + "V2"), nullptr,
                                                                 &perf_param_table_v2,
                                                                 &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> trunc_to_int_api_perf_v2(ApiPerfRegisterV2(kTruncToInt, GetPerfFunc(kTruncToInt + "V2"),
                                                                    nullptr, &perf_param_table_v2,
                                                                    &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> round_to_int_api_perf_v2(ApiPerfRegisterV2(kRoundToInt, GetPerfFunc(kRoundToInt + "V2"),
                                                                    nullptr, &perf_param_table_v2,
                                                                    &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> xor_api_perf_v2(ApiPerfRegisterV2(kXor, GetPerfFunc(kXor + "V2"), nullptr,
                                                           &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> remainder_api_perf_v2(ApiPerfRegisterV2(kRemainder, GetPerfFunc(kRemainder + "V2"), nullptr,
                                                                 &perf_param_table_v2,
                                                                 &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> modified_bessel_i0_api_perf_v2(ApiPerfRegisterV2(kModifiedBesselI0, GetPerfFunc(kUnitVector),
                                                                          nullptr, &perf_param_table_v2,
                                                                          &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> modified_bessel_i1_api_perf_v2(ApiPerfRegisterV2(kModifiedBesselI1, GetPerfFunc(kUnitVector),
                                                                          nullptr, &perf_param_table_v2,
                                                                          &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> modified_bessel_k0_api_perf_v2(ApiPerfRegisterV2(kModifiedBesselK0, GetPerfFunc(kUnitVector),
                                                                          nullptr, &perf_param_table_v2,
                                                                          &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> modified_bessel_k1_api_perf_v2(ApiPerfRegisterV2(kModifiedBesselK1, GetPerfFunc(kUnitVector),
                                                                          nullptr, &perf_param_table_v2,
                                                                          &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> i0_api_perf_v2(ApiPerfRegisterV2(kI0, GetPerfFunc(kUnitVector), nullptr, &perf_param_table_v2,
                                                          &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> i0e_api_perf_v2(ApiPerfRegisterV2(kI0e, GetPerfFunc(kUnitVector), nullptr,
                                                           &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> i1e_api_perf_v2(ApiPerfRegisterV2(kI1e, GetPerfFunc(kUnitVector), nullptr,
                                                           &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> bessel_j0_api_perf_v2(ApiPerfRegisterV2(kBesselJ0, GetPerfFunc(kUnitVector), nullptr,
                                                                 &perf_param_table_v2,
                                                                 &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> bessel_j1_api_perf_v2(ApiPerfRegisterV2(kBesselJ1, GetPerfFunc(kUnitVector), nullptr,
                                                                 &perf_param_table_v2,
                                                                 &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> bessel_y0_api_perf_v2(ApiPerfRegisterV2(kBesselY0, GetPerfFunc(kUnitVector), nullptr,
                                                                 &perf_param_table_v2,
                                                                 &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> bessel_y1_api_perf_v2(ApiPerfRegisterV2(kBesselY1, GetPerfFunc(kUnitVector), nullptr,
                                                                 &perf_param_table_v2,
                                                                 &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> scaled_modified_bessel_k0_api_perf_v2(ApiPerfRegisterV2(kScaledModifiedBesselK0,
                                                                                 GetPerfFunc(kUnitVector), nullptr,
                                                                                 &perf_param_table_v2,
                                                                                 &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> scaled_modified_bessel_k1_api_perf_v2(ApiPerfRegisterV2(kScaledModifiedBesselK1,
                                                                                 GetPerfFunc(kUnitVector), nullptr,
                                                                                 &perf_param_table_v2,
                                                                                 &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> spherical_bessel_j0_api_perf_v2(ApiPerfRegisterV2(kSphericalBesselJ0, GetPerfFunc(kUnitVector),
                                                                           nullptr, &perf_param_table_v2,
                                                                           &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> igamma_api_perf_v2(ApiPerfRegisterV2(kIgamma, GetPerfFunc(kUnitVector), nullptr,
                                                              &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> igammac_api_perf_v2(ApiPerfRegisterV2(kIgammac, GetPerfFunc(kUnitVector), nullptr,
                                                               &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> zeta_api_perf_v2(ApiPerfRegisterV2(kZeta, GetPerfFunc(kUnitVector), nullptr,
                                                            &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> ndtr_api_perf_v2(ApiPerfRegisterV2(kNdtr, GetPerfFunc(kUnitVector), nullptr,
                                                            &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> ndtri_api_perf_v2(ApiPerfRegisterV2(kNdtri, GetPerfFunc(kUnitVector), nullptr,
                                                             &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> logndtr_api_perf_v2(ApiPerfRegisterV2(kLogNdtr, GetPerfFunc(kUnitVector), nullptr,
                                                               &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> nextafter_api_perf_v2(ApiPerfRegisterV2(kNextAfter, GetPerfFunc(kUnitVector), nullptr,
                                                                 &perf_param_table_v2,
                                                                 &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> polygamma_api_perf_v2(ApiPerfRegisterV2(kPolyGamma, GetPerfFunc(kUnitVector), nullptr,
                                                                 &perf_param_table_v2,
                                                                 &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> signbit_api_perf_v2(ApiPerfRegisterV2(kSignBit, GetPerfFunc(kUnitVector), nullptr,
                                                               &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> frexp_api_perf_v2(ApiPerfRegisterV2(kFrexp, GetPerfFunc(kUnitVector), nullptr,
                                                             &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> shifted_chebyshev_polynomial_t_api_perf_v2(
    ApiPerfRegisterV2(kShiftedChebyshevPolynomialT, GetPerfFunc(kUnitVector), nullptr, &perf_param_table_v2,
                      &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> shifted_chebyshev_polynomial_u_api_perf_v2(
    ApiPerfRegisterV2(kShiftedChebyshevPolynomialU, GetPerfFunc(kUnitVector), nullptr, &perf_param_table_v2,
                      &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> shifted_chebyshev_polynomial_v_api_perf_v2(
    ApiPerfRegisterV2(kShiftedChebyshevPolynomialV, GetPerfFunc(kUnitVector), nullptr, &perf_param_table_v2,
                      &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> shifted_chebyshev_polynomial_w_api_perf_v2(
    ApiPerfRegisterV2(kShiftedChebyshevPolynomialW, GetPerfFunc(kUnitVector), nullptr, &perf_param_table_v2,
                      &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> chebyshev_polynomial_t_api_perf_v2(ApiPerfRegisterV2(
    kChebyshevPolynomialT, GetPerfFunc(kUnitVector), nullptr, &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> chebyshev_polynomial_u_api_perf_v2(ApiPerfRegisterV2(
    kChebyshevPolynomialU, GetPerfFunc(kUnitVector), nullptr, &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> chebyshev_polynomial_v_api_perf_v2(ApiPerfRegisterV2(
    kChebyshevPolynomialV, GetPerfFunc(kUnitVector), nullptr, &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> chebyshev_polynomial_w_api_perf_v2(ApiPerfRegisterV2(
    kChebyshevPolynomialW, GetPerfFunc(kUnitVector), nullptr, &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> hermite_polynomial_h_api_perf_v2(ApiPerfRegisterV2(
    kHermitePolynomialH, GetPerfFunc(kUnitVector), nullptr, &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> hermite_polynomial_he_api_perf_v2(ApiPerfRegisterV2(
    kHermitePolynomialHe, GetPerfFunc(kUnitVector), nullptr, &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> laguerre_polynomial_l_api_perf_v2(ApiPerfRegisterV2(
    kLaguerrePolynomialL, GetPerfFunc(kUnitVector), nullptr, &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> legendre_polynomial_p_api_perf_v2(ApiPerfRegisterV2(
    kLegendrePolynomialP, GetPerfFunc(kUnitVector), nullptr, &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> airy_ai_api_perf_v2(ApiPerfRegisterV2(kAiryAi, GetPerfFunc(kUnitVector), nullptr,
                                                               &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> erfinv_api_perf_v2(ApiPerfRegisterV2(kErfinv, GetPerfFunc(kUnitVector), nullptr,
                                                              &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> square_api_perf_v2(ApiPerfRegisterV2(kSquare, GetPerfFunc(kSquare + "V2"), nullptr,
                                                              &perf_param_table_v2, &tiling_schedule_config_table_v2));

ApiPerfRegister<ApiPerf> vector_func_api_perf(kVectorFunc, DefaultGetPerf, nullptr, &perf_param_table_v2,
                                              &tiling_schedule_config_table_v2);
}  // namespace
}  // namespace att
