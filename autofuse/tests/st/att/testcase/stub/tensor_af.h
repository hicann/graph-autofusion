/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_AF_EXTERNAL_GRAPH_TENSOR_H_
#define INC_AF_EXTERNAL_GRAPH_TENSOR_H_

#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include <utility>

static const int64_t UNKNOWN_DIM_NUM = -2;

namespace ge {

class ShapeImpl {
 public:
  ShapeImpl() = default;
  ~ShapeImpl() = default;
  explicit ShapeImpl(const std::vector<int64_t> &dims) {
    bool is_unknown_dim_num = false;
    for (const auto &dim : dims) {
      if (dim == UNKNOWN_DIM_NUM) {
        is_unknown_dim_num = true;
        break;
      }
    }
    dims_ = is_unknown_dim_num ? std::vector<int64_t>({UNKNOWN_DIM_NUM}) : dims;
  }

 private:
  std::vector<int64_t> dims_;
  friend class Shape;
};

template <typename T, typename... Args>
static inline std::shared_ptr<T> ComGraphMakeShared(Args &&...args) {
  using T_nc = typename std::remove_const<T>::type;
  std::shared_ptr<T> ret = nullptr;
  try {
    ret = std::make_shared<T_nc>(std::forward<Args>(args)...);
  } catch (const std::bad_alloc &) {
    ret = nullptr;
  }
  return ret;
}

class Shape {
 public:
  Shape();
  ~Shape() = default;
  Shape(const std::vector<int64_t> &dims) {
    impl_ = ComGraphMakeShared<ShapeImpl>(dims);
  }

  std::vector<int64_t> GetDims() const {
    const std::vector<int64_t> dims;
    if (impl_ != nullptr) {
      return impl_->dims_;
    }
    return dims;
  }
  /**
   * 获取shape的各个维度的dim值的乘积
   * @return
   * 如果dim值包含-1或者-2，那么size直接返回-1, 含义是unknown shape
   * 如果dim值包含0，那么size直接返回0，含义是空tensor
   * 如果dim值的个数为0，那么size直接返回0，含义是标量
   * 如果dim值的乘积产生了int64的溢出，那么size直接返回0，含义是乘积溢出
   */
  int64_t GetShapeSize() const;

 private:
  std::shared_ptr<ShapeImpl> impl_;
};
}  // namespace ge

#endif  // INC_AF_EXTERNAL_GRAPH_TENSOR_H_
