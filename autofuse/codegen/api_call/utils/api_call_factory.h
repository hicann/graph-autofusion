/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AUTOFUSE_API_CALL_FACTORY_H__
#define __AUTOFUSE_API_CALL_FACTORY_H__

#include <string>
#include <memory>
#include <functional>
#include <map>
#include <mutex>
#include <tuple>
#include "codegen_kernel.h"
#include "common_utils.h"
namespace codegen {
using ApiCallCreatorFun = std::function<ApiCall *(const std::string &)>;

class ApiCallFactory {
 public:
  static ApiCallFactory &Instance() {
    static ApiCallFactory instance;
    return instance;
  }

  ApiCall *Create(const std::string &class_name, const std::string &api_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = creator_map_.find(class_name);
    if (iter == creator_map_.end()) {
      GELOGW("Cannot find node type %s in inner map.", class_name.c_str());
      return nullptr;
    }
    auto &func = creator_map_[class_name];
    return func(api_name);
  }

  ApiCallFactory(const ApiCallFactory &) = delete;
  ApiCallFactory &operator=(const ApiCallFactory &) = delete;

  class Registerar {
   public:
    Registerar(const std::string class_name, const ApiCallCreatorFun &func) noexcept {
      ApiCallFactory::Instance().RegisterCreator(class_name, func);
    }

    ~Registerar() = default;
  };

 private:
  friend class ApiCallRegistryStub;  // for test
  ApiCallFactory() = default;

  ~ApiCallFactory() = default;
  void RegisterCreator(const std::string class_name, const ApiCallCreatorFun &func) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = creator_map_.find(class_name);
    if (iter != creator_map_.end()) {
      GELOGD("ApiCallFactory::RegisterCreator: %s creator already exists", class_name.c_str());
      return;
    }
    creator_map_[class_name] = func;
  }

  std::map<std::string, ApiCallCreatorFun> creator_map_;
  std::mutex mutex_;
};

template <typename T>
class ApiCallRegister {
 public:
  // 构造函数：自动注册类
  explicit ApiCallRegister(const std::string &className) {
    // 正确用法：使用模板参数T创建对象，而非字符串className
    ApiCallCreatorFun creator = [](const std::string &name) {
      return new (std::nothrow) T(name);  // 直接使用模板类型T
    };
    ApiCallFactory::Registerar(className, creator);
  }
};

inline ApiCall *CreateApiCallObject(const ascir::NodeView &node) {
  auto impl = ascgen_utils::GetAscIrCodegenImpl(node->GetType());
  if (impl == nullptr) {
    return nullptr;
  }

  GELOGD("create api call, type:%s, api_call_name:%s, api_name:%s.", node->GetTypePtr(), impl->GetApiCallName().c_str(),
         impl->GetApiName().c_str());
  return ApiCallFactory::Instance().Create(impl->GetApiCallName(), impl->GetApiName());
}

}  // namespace codegen
#endif  // __AUTOFUSE_API_CALL_FACTORY_H__
