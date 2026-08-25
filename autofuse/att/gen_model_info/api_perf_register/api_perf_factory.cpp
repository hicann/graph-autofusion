/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "api_perf_factory.h"
namespace att {
ApiPerfFactory &ApiPerfFactory::Instance() {
  static ApiPerfFactory instance;
  return instance;
}

std::unique_ptr<ApiPerf> ApiPerfFactory::Create(const std::string &class_name) {
  ApiPerfCreatorFun creator;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = creator_map_.find(class_name);
    if (iter == creator_map_.end()) {
      GELOGW("Cannot find node type %s in inner map.", class_name.c_str());
      return nullptr;
    }
    creator = iter->second;
  }
  GELOGD("Create ApiPerf success, class_name: %s", class_name.c_str());
  return creator(class_name);
}
}  // namespace att
