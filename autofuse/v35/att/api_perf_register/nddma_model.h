/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef V35_ATT_API_PERF_REGISTER_NDDMA_MODEL_H_
#define V35_ATT_API_PERF_REGISTER_NDDMA_MODEL_H_

#include "api_perf_register/api_perf.h"

namespace att {
/**
 * @brief NDDMA 精确性能模型的组成和计算原理。
 *
 * 当前范围：
 * - 只注册 NDDMA_1D_MULTICORE_V1，覆盖默认 Codegen 路径下 raw rank=1、effective rank=1 的搬运。
 * - raw rank=2～5 暂无正式模型，保留完整 descriptor 后回退 legacy NDDMA 模型；不会因连续轴合并而伪装成 1D。
 * - kUBFuse Codegen 使用 {curAivM, curAlignN} 和固定 2D stride，与 raw descriptor 不等价，因此通过
 *   NodeInfo::is_cv_ub_fusion 门禁回退 legacy 模型。
 *
 * 输入和 Codegen 映射：
 * - TensorShapeInfo::repeats       -> output_dims，单核单次搬运的各维元素个数；
 * - TensorShapeInfo::gm_strides    -> input_strides，GM 元素 stride；
 * - TensorShapeInfo::strides       -> output_strides，UB 元素 stride；
 * - AscTensorAttr::vectorized_axis -> descriptor 轴序；
 * - dtype                          -> dtype_size，并选择 B8/B16/B32/B64 参数；
 * - block_dim                      -> 参与搬运的核数。
 * 三组向量均为 legacy 连续轴合并前的原始值，顺序与 vectorized_axis 一致，stride 单位均为元素。
 * 默认 Codegen 使用同源的 DataCopyParams，经 SetNddmaParams 左侧补 1 后生成 5 元素 API 数组；补 1 不改变
 * effective rank 和搬运语义。
 *
 * 1D 特征：
 *   B = output_dims[0] * dtype_size                       // 每核每次搬运字节数
 *   is = input_strides[0], s = min(is, 128)              // GM 非连续程度，is 允许为 0
 *   os = output_strides[0]
 *   g = max(0, min(1, os - 1))                           // os=1 连续，os>=2 时修正饱和
 *
 * 参数组成（每种 dtype_size 各有一组）：
 * - low_os_one / low_os_ge_two：k<=2 且 os=1/os>=2 时的 L0、LB、Ls、LBs；
 * - high_os_one / high_os_ge_two：k>2 且 os=1/os>=2 时的 C0、C1、C2、E0、E1、E2。
 * 合并后的参数是原始 t/h/a/b/c 参数在合法 os 分支上的代数展开结果，参数表直接对应最终 cycles。
 *
 * 1D 公式：
 *   low(os=1)   = L0 + LB*B + Ls*s + LBs*B*s
 *   high(os=1)  = C0 + C1*s + C2*s² + B*(E0 + E1*s + E2*s²)
 *   low(os>=2)  = L0 + LB*B + Ls*s + LBs*B*s
 *   high(os>=2) = C0 + C1*s + C2*s² + B*(E0 + E1*s + E2*s²)
 *   cycles = low(block_dim <= 2) or high(block_dim > 2)
 * 合法 stride 为整数，因此动态 os 使用 g=max(0,min(1,os-1)) 在两个 os 多项式之间选择；g=0 对应 os=1，
 * g=1 对应 os>=2。对于非法分数 stride，不承诺与原始连续 g 公式等价。
 *
 * 静态和动态 shape：
 * - 两者使用同一组参数和同一公式；静态表达式直接折叠，动态 block_dim 生成一个 TernaryOp；
 * - 动态 is/os 保留符号 Min/Max，g 的非负保护避免动态 os 产生负修正；
 * - 静态非正 dim、负 input_stride、非正 output_stride、非法 dtype/schema 均记录原因并回退 legacy。
 *
 * 处理阶段：
 * 1. BuildNddmaDescriptor：从 TensorShapeInfo 构造原始 descriptor，不从 legacy 标量 stride 反推；
 * 2. NormalizeNddmaDescriptor：校验 rank、向量长度、轴序、静态值及默认 Codegen 一致性；
 * 3. EvaluateNddmaModel：选择 dtype 参数、构造静态/动态 cycles，并返回 ternary_ops；
 * 4. LogNddmaFallback：未选择新模型时记录一次稳定 reason，调用方继续执行原有 GetDmaPerf。
 * 模型只输出单次调用的 AIV_MTE2 cycles；全局 pipe head 仍由 PipePerfExpr 统一添加。
 *
 * 扩展约束：原始 NddmaDescriptorInfo 与物理 NddmaNormalizedDesc 分离。后续 2D～5D 可在归一化阶段构造
 * effective view，并独立选择 raw 或 normalized 统计特征，不改变现有 descriptor 字段及 legacy 数据结构。
 */
enum class NddmaFallbackReason : int32_t {
  kNone = 0,
  kNoDescriptor,
  kRankUnsupported,
  kSchemaMismatch,
  kDtypeUnsupported,
  kStrideInvalid,
  kCodegenMismatch,
  kNoRegisteredModel,
};

// schema 和 Codegen 一致性校验后的物理描述。物理描述与统计特征归一化分离，使后续 2D～5D 模型可选择
// raw 或 normalized 特征而不改变 descriptor 语义。
struct NddmaNormalizedDesc {
  std::vector<Expr> output_dims;
  std::vector<Expr> input_strides;
  std::vector<Expr> output_strides;
  std::vector<int64_t> vectorized_axis;
  size_t raw_rank{0U};
  size_t effective_rank{0U};
};

// 一次模型计算要么返回 cycles 和所需动态 ternary，要么返回一个稳定的 legacy fallback 原因。
struct NddmaModelResult {
  bool selected{false};
  Expr cycles;
  std::string model_name{"NDDMA_1D_MULTICORE_V1"};
  NddmaFallbackReason fallback_reason{NddmaFallbackReason::kNone};
  size_t raw_rank{0U};
  size_t effective_rank{0U};
  TernaryOpMap ternary_ops;
};

const char *NddmaFallbackReasonToString(NddmaFallbackReason reason);

NddmaFallbackReason BuildNddmaDescriptor(const TensorShapeInfo &shape_info, const std::vector<int64_t> &vectorized_axis,
                                         NddmaDescriptorInfo &descriptor);

NddmaFallbackReason NormalizeNddmaDescriptor(const NddmaDescriptorInfo &descriptor, NddmaNormalizedDesc &normalized);

af::Status EvaluateNddmaModel(const NddmaDescriptorInfo &descriptor, const std::string &dtype, const Expr &block_dim,
                              NddmaModelResult &result);

void LogNddmaFallback(const std::string &node_name, const std::string &dtype, const NddmaDescriptorInfo *descriptor,
                      const NddmaModelResult &result);
}  // namespace att

#endif  // V35_ATT_API_PERF_REGISTER_NDDMA_MODEL_H_
