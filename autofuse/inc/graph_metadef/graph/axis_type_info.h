/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_AXIS_TYPE_INFO_H_
#define INC_GRAPH_AXIS_TYPE_INFO_H_

#include <vector>
#include <map>
#include <string>

#include "graph/ge_error_codes_af.h"

namespace af {
using CutInfo = std::pair<int64_t, std::vector<int64_t>>;

enum class AxisType {
  UNSPLIT = 0,
  ELEMENTWISE,
  REDUCESUM,
  REDUCEMAX,
  REDUCEMIN,
  REDUCEMEAN,
  TRANSPOSE,
  SLIDINGWINDOW,
  REDUCEGATHER,
  SLIDINGWINDOWGRAD,
  ELEMENTWITHSHAPEVALUE,
  REDUCEPROD,
  ELEMENTWISEWITHFACTOR,
};

class AxisTypeInfo {
 public:
  AxisTypeInfo() = default;
  ~AxisTypeInfo() = default;
  void SetAxisType(const AxisType axis_type) {
    axis_type_ = axis_type;
  }
  AxisType GetAxisType() const {
    return axis_type_;
  }
  void AddInputCutInfo(CutInfo &input_cut_info);
  void AddOutputCutInfo(CutInfo &output_cut_info);
  graphStatus GetInputCutInfo(const size_t index, CutInfo &input_cut_info) const;
  graphStatus GetOutputCutInfo(const size_t index, CutInfo &output_cut_info) const;
  const std::vector<CutInfo> &GetRelateInputs() const {
    return relate_inputs_;
  }
  const std::vector<CutInfo> &GetRelateOutputs() const {
    return relate_outputs_;
  }
  void SetRelateInputs(const std::vector<CutInfo> &inputs_info) {
    relate_inputs_ = inputs_info;
  }
  void SetRelateOutputs(const std::vector<CutInfo> &outputs_info) {
    relate_outputs_ = outputs_info;
  }
  const std::vector<CutInfo> &GetOriRelateInputs() const {
    return ori_relate_inputs_;
  }
  const std::vector<CutInfo> &GetOriRelateOutputs() const {
    return ori_relate_outputs_;
  }
  void SetOriRelateInputs(const std::vector<CutInfo> &inputs_info) {
    ori_relate_inputs_ = inputs_info;
  }
  void SetOriRelateOutputs(const std::vector<CutInfo> &outputs_info) {
    ori_relate_outputs_ = outputs_info;
  }
  void SetAxisTypes(const std::vector<AxisType> &axis_types) {
    axis_types_ = axis_types;
  }
  const std::vector<AxisType> &GetAxisTypes() const {
    return axis_types_;
  }
  void AddInputValueCutInfo(const CutInfo &cut_info);
  void AddOutputValueCutInfo(const CutInfo &cut_info);
  graphStatus GetInputValueCutInfo(const size_t index, CutInfo &cut_info) const;
  graphStatus GetOutputValueCutInfo(const size_t index, CutInfo &cut_info) const;
  const std::vector<CutInfo> &GetRelateInputValues() const {
    return relate_input_values_;
  }
  const std::vector<CutInfo> &GetRelateOutputValues() const {
    return relate_output_values_;
  }
  void SetAdditionInfo(const std::map<std::string, std::string> &addition_info) {
    addition_info_ = addition_info;
  }
  const std::map<std::string, std::string> &GetAdditionInfo() const {
    return addition_info_;
  }

 private:
  static graphStatus DoGetCutInfo(const std::vector<CutInfo> &cut_infos, const size_t index, CutInfo &cut_info);

  AxisType axis_type_ = AxisType::UNSPLIT;
  std::vector<CutInfo> relate_inputs_;
  std::vector<CutInfo> relate_outputs_;
  std::vector<AxisType> axis_types_;
  std::vector<CutInfo> relate_input_values_;
  std::vector<CutInfo> relate_output_values_;
  std::vector<CutInfo> ori_relate_inputs_;   // backup relate_inputs_
  std::vector<CutInfo> ori_relate_outputs_;  // backup relate_outputs_
  std::map<std::string, std::string> addition_info_;
};
}  // namespace af

#endif  // INC_GRAPH_AXIS_TYPE_INFO_H_
