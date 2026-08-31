/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_OPERATOR_FACTORY_IMPL_H_
#define INC_GRAPH_OPERATOR_FACTORY_IMPL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "graph/operator_factory_af.h"
#include "register/infer_data_slice_registry.h"
#include "register/infer_axis_slice_registry.h"
#include "register/op_impl_kernel_registry.h"
#include "graph/op_desc.h"

namespace af {
using InferShapeV2Func = uint32_t (*)(const af::Operator &op, const OpDescPtr &);
using InferDataTypeFunc = uint32_t (*)(const OpDescPtr &);
using InferShapeRangeFunc = uint32_t (*)(const af::Operator &op, const OpDescPtr &);
using InferFormatV2Func = uint32_t (*)(const af::Operator &, const OpDescPtr &);
using IsInferFormatV2RegisteredFunc = bool (*)(const OpDescPtr &);
using IsInferShapeV2RegisteredFunc = bool (*)(const OpDescPtr &);

struct InferValueRangePara {
 public:
  InferValueRangePara() = default;
  InferValueRangePara(const WHEN_CALL call, const bool cpu_kernel, const InferValueRangeFunc func) {
    is_initialized = true;
    use_cpu_kernel = cpu_kernel;
    when_call = call;
    infer_value_func = func;
  }
  friend class OpDescImpl;
  friend class InferValueRangePass;
  friend class OpDescUtilsEx;
  ~InferValueRangePara() = default;

 private:
  bool is_initialized = false;
  bool use_cpu_kernel = false;
  WHEN_CALL when_call = INPUT_IS_DYNAMIC;
  InferValueRangeFunc infer_value_func = nullptr;
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OperatorFactoryImpl {
 public:
  static Operator CreateOperator(const std::string &operator_name, const std::string &operator_type);

  static graphStatus GetOpsTypeList(std::vector<std::string> &all_ops);

  static bool IsExistOp(const std::string &operator_type);

  static InferShapeFunc GetInferShapeFunc(const std::string &operator_type);

  static InferShapeV2Func GetInferShapeV2Func();

  static InferDataTypeFunc GetInferDataTypeFunc();

  static InferShapeRangeFunc GetInferShapeRangeFunc();

  static InferFormatFunc GetInferFormatFunc(const std::string &operator_type);

  static InferValueRangePara GetInferValueRangePara(const std::string &operator_type);

  static VerifyFunc GetVerifyFunc(const std::string &operator_type);

  static InferDataSliceFunc GetInferDataSliceFunc(const std::string &operator_type);

  static InferAxisSliceFunc GetInferAxisSliceFunc(const std::string &operator_type);

  static InferAxisTypeInfoFunc GetInferAxisTypeInfoFunc(const std::string &operator_type);

  static void SetRegisterOverridable(const bool &is_overridable);

  static graphStatus RegisterOperatorCreator(const std::string &operator_type, OpCreator const &op_creator);

  static graphStatus RegisterOperatorCreator(const std::string &operator_type, OpCreatorV2 const &op_creator);

  static graphStatus RegisterInferShapeFunc(const std::string &operator_type, InferShapeFunc const infer_shape_func);

  static void RegisterInferShapeV2Func(InferShapeV2Func const infer_shape_func);

  static void RegisterInferDataTypeFunc(InferDataTypeFunc const infer_data_type_func);

  static void RegisterInferShapeRangeFunc(InferShapeRangeFunc const infer_shape_range_func);

  static graphStatus RegisterInferFormatFunc(const std::string &operator_type, InferFormatFunc const infer_format_func);

  static graphStatus RegisterVerifyFunc(const std::string &operator_type, VerifyFunc const verify_func);

  static graphStatus RegisterInferDataSliceFunc(const std::string &operator_type,
                                                InferDataSliceFunc const infer_data_slice_func);

  static graphStatus RegisterInferValueRangeFunc(const std::string &operator_type);

  static graphStatus RegisterInferValueRangeFunc(const std::string &operator_type, const WHEN_CALL when_call,
                                                 const bool use_cpu_kernel,
                                                 const InferValueRangeFunc &infer_value_range_func);

  static graphStatus RegisterInferAxisSliceFunc(const std::string &operator_type,
                                                const InferAxisSliceFunc &infer_axis_slice_func);

  static graphStatus RegisterInferAxisTypeInfoFunc(const std::string &operator_type,
                                                   const InferAxisTypeInfoFunc &infer_axis_type_info_func);

  static void RegisterInferFormatV2Func(InferFormatV2Func const infer_format_func);

  static InferFormatV2Func GetInferFormatV2Func();

  static void RegisterIsInferFormatV2RegisteredFunc(
      IsInferFormatV2RegisteredFunc const is_infer_format_v2_registered_func);

  static IsInferFormatV2RegisteredFunc GetIsInferFormatV2RegisteredFunc();

  static void RegisterIsInferShapeV2RegisteredFunc(
      IsInferShapeV2RegisteredFunc const is_infer_shape_v2_registered_func);

  static IsInferShapeV2RegisteredFunc GetIsInferShapeV2RegisteredFunc();

  static void ReleaseRegInfo();

  static std::shared_ptr<std::map<std::string, OpCreator>> operator_creators_;
  // Kept for ABI compatibility with older clients. New registrations use operator_creators_.
  static std::shared_ptr<std::map<std::string, OpCreatorV2>> operator_creators_v2_;
  static std::shared_ptr<std::map<std::string, InferShapeFunc>> operator_infershape_funcs_;
  static std::shared_ptr<std::map<std::string, InferFormatFunc>> operator_inferformat_funcs_;
  static std::shared_ptr<std::map<std::string, VerifyFunc>> operator_verify_funcs_;
  static std::shared_ptr<std::map<std::string, InferDataSliceFunc>> operator_infer_data_slice_funcs_;
  static std::shared_ptr<std::map<std::string, InferValueRangePara>> operator_infer_value_range_paras_;
  static std::shared_ptr<std::map<std::string, InferAxisSliceFunc>> operator_infer_axis_slice_funcs_;
  static std::shared_ptr<std::map<std::string, InferAxisTypeInfoFunc>> operator_infer_axis_type_info_funcs_;
  static InferShapeV2Func operator_infer_shape_v2_func_;
  static InferDataTypeFunc operator_infer_datatype_func_;
  static InferShapeRangeFunc operator_infer_shape_range_func_;
  static InferFormatV2Func operator_infer_format_v2_func_;
  static IsInferFormatV2RegisteredFunc is_infer_format_v2_registered_func_;
  static IsInferShapeV2RegisteredFunc is_infer_shape_v2_registered_func_;
};
}  // namespace af

#endif  // INC_GRAPH_OPERATOR_FACTORY_IMPL_H_
