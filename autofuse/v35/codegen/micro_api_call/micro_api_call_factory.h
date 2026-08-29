/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AUTOFUSE_MICRO_API_CALL_FACTORY_H__
#define __AUTOFUSE_MICRO_API_CALL_FACTORY_H__

#include <string>
#include <memory>
#include <functional>
#include <map>
#include <mutex>
#include <tuple>
#include "micro_api_call.h"
#include "common_utils.h"
#include "ascir/ascir_codegen_v2.h"

namespace codegen {
using MicroApiCallCreatorFun = std::function<MicroApiCall *(const std::string &)>;

class MicroApiCallFactory {
 public:
  static MicroApiCallFactory &Instance() {
    static MicroApiCallFactory instance;
    return instance;
  }

  MicroApiCall *Create(const std::string &class_name, const std::string &micro_api_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = creator_map_.find(class_name);
    if (iter == creator_map_.end()) {
      GELOGW("Cannot find node type %s in inner map.", class_name.c_str());
      return nullptr;
    }
    auto &func = creator_map_[class_name];
    return func(micro_api_name);
  }

  MicroApiCallFactory(const MicroApiCallFactory &) = delete;
  MicroApiCallFactory &operator=(const MicroApiCallFactory &) = delete;

  class Register {
   public:
    Register(const std::string &class_name, const MicroApiCallCreatorFun &func) noexcept {
      MicroApiCallFactory::Instance().RegisterCreator(class_name, func);
    }

    ~Register() = default;
  };

 private:
  friend class MicroApiCallRegistryStub;  // for test
  MicroApiCallFactory() = default;

  ~MicroApiCallFactory() = default;
  void RegisterCreator(const std::string &class_name, const MicroApiCallCreatorFun &func) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = creator_map_.find(class_name);
    if (iter != creator_map_.end()) {
      GELOGD("MicroApiCallFactory::RegisterCreator: %s creator already exists", class_name.c_str());
      return;
    }
    creator_map_[class_name] = func;
  }

  std::map<std::string, MicroApiCallCreatorFun> creator_map_;
  std::mutex mutex_;
};

template <typename T>
class MicroApiCallRegister {
 public:
  // 构造函数：自动注册类
  explicit MicroApiCallRegister(const std::string &className) {
    // 正确用法：使用模板参数T创建对象，而非字符串className
    MicroApiCallCreatorFun creator = [](const std::string &name) {
      return new (std::nothrow) T(name);  // 直接使用模板类型T
    };
    MicroApiCallFactory::Register(className, creator);
  }
};

inline MicroApiCall *CreateMicroApiCallObject(const ascir::NodeView &node) {
  auto ascir_codegen_impl = ascgen_utils::GetAscIrCodegenImpl(node->GetType());
  auto impl = dynamic_cast<af::ascir::AscIrCodegenV2 *>(ascir_codegen_impl.get());
  if (impl == nullptr) {
    return nullptr;
  }

  GELOGD("create micro api call, type:%s, micro_api_call_name:%s, micro_api_name:%s.", node->GetTypePtr(),
         impl->GetMicroApiCallName().c_str(), impl->GetMicroApiName().c_str());
  return MicroApiCallFactory::Instance().Create(impl->GetMicroApiCallName(), impl->GetMicroApiName());
}

}  // namespace codegen
#endif  // __AUTOFUSE_MICRO_API_CALL_FACTORY_H__
