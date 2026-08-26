/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __API_PERF_REGISTER_API_PERF_FACTORY_H__
#define __API_PERF_REGISTER_API_PERF_FACTORY_H__

#include <string>
#include <memory>
#include <functional>
#include <map>
#include <mutex>
#include <tuple>
#include <utility>
#include "api_perf.h"
#include "common_utils.h"
namespace att {
using ApiPerfCreatorFun = std::function<std::unique_ptr<ApiPerf>(const std::string &)>;
Perf GetPerfFunc(const std::string &op_type);
class __attribute__((visibility("default"))) ApiPerfFactory {
 public:
  static ApiPerfFactory &Instance();

  std::unique_ptr<ApiPerf> Create(const std::string &class_name);

  ApiPerfFactory(const ApiPerfFactory &) = delete;
  ApiPerfFactory &operator=(const ApiPerfFactory &) = delete;

  class Registerar {
   public:
    Registerar(const std::string &api_name, ApiPerfCreatorFun creator) noexcept {
      ApiPerfFactory::Instance().RegisterCreator(api_name, std::move(creator));
    }

    ~Registerar() = default;
  };

 private:
  ApiPerfFactory() = default;

  ~ApiPerfFactory() = default;
  void RegisterCreator(const std::string &api_name, ApiPerfCreatorFun creator) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = creator_map_.find(api_name);
    if (iter != creator_map_.end()) {
      GELOGD("ApiCallFactory::RegisterCreator: %s creator already exist", api_name.c_str());
      return;
    }
    creator_map_[api_name] = std::move(creator);
    GELOGD("RegisterCreator success, api_name: %s", api_name.c_str());
  }

  std::map<std::string, ApiPerfCreatorFun> creator_map_;
  std::mutex mutex_;
};

template <typename T>
class ApiPerfRegister {
 public:
  // 构造函数：自动注册类
  explicit ApiPerfRegister(const std::string &api_name, Perf perf_func, MicroPerfFunc micro_perf_func,
                           const PerfParamTable *perf_param,
                           const TilingScheduleConfigTable *tiling_schedule_config_table) {
    // 正确用法：使用模板参数T创建对象
    ApiPerfCreatorFun creator = [perf_func, micro_perf_func, perf_param,
                                 tiling_schedule_config_table](const std::string &name) {
      return std::make_unique<T>(name, perf_func, micro_perf_func, perf_param, tiling_schedule_config_table);
    };
    ApiPerfFactory::Registerar(api_name, std::move(creator));
  }

  explicit ApiPerfRegister(const std::string &api_name, const std::string &perf_func_name,
                           MicroPerfFunc micro_perf_func, const PerfParamTable *perf_param,
                           const TilingScheduleConfigTable *tiling_schedule_config_table) {
    ApiPerfCreatorFun creator = [perf_func_name, micro_perf_func, perf_param,
                                 tiling_schedule_config_table](const std::string &name) {
      return std::make_unique<T>(name, GetPerfFunc(perf_func_name), micro_perf_func, perf_param,
                                 tiling_schedule_config_table);
    };
    ApiPerfFactory::Registerar(api_name, std::move(creator));
  }
};

}  // namespace att
#endif  // __API_PERF_REGISTER_API_PERF_FACTORY_H__
