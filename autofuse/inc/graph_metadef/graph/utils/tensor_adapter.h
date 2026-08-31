/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_UTILS_TENSOR_ADAPTER_H_
#define INC_GRAPH_UTILS_TENSOR_ADAPTER_H_

#include <memory>
#include "graph/ge_tensor.h"
#include "graph/tensor_af.h"
#include "graph/ge_attr_value.h"

namespace af {
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TensorAdapter {
 public:
  static GeTensorDesc TensorDesc2GeTensorDesc(const TensorDesc &tensor_desc);
  static TensorDesc GeTensorDesc2TensorDesc(const GeTensorDesc &ge_tensor_desc);
  static Tensor GeTensor2Tensor(const ConstGeTensorPtr &ge_tensor);

  static ConstGeTensorPtr AsGeTensorPtr(const Tensor &tensor);  // Share value
  static GeTensorPtr AsGeTensorPtr(Tensor &tensor);             // Share value
  static const GeTensor AsGeTensor(const Tensor &tensor);       // Share value
  static GeTensor AsGeTensor(Tensor &tensor);                   // Share value
  static const Tensor AsTensor(const GeTensor &ge_tensor);      // Share value
  static Tensor AsTensor(GeTensor &ge_tensor);                  // Share value
  static GeTensor AsGeTensorShared(const Tensor &tensor);
  static GeTensor NormalizeGeTensor(const GeTensor &tensor);
  static void NormalizeGeTensorDesc(GeTensorDesc &tensor_desc);
  static const GeTensor *AsBareGeTensorPtr(const Tensor &tensor);
};
}  // namespace af
#endif  // INC_GRAPH_UTILS_TENSOR_ADAPTER_H_
