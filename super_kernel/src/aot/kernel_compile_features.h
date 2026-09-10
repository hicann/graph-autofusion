/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef KERNEL_COMPILE_FEATURES_H
#define KERNEL_COMPILE_FEATURES_H

#include <regex>
#include <string>
#include <string_view>
#include <vector>
#include "kernel_spec_options.h"

namespace sk {
namespace static_compile {
namespace detail {

class ModelSettings final {
 public:
  bool Init(const aclspecOptions *options);
  uint64_t CompileJobs() const {
    return jobs_;
  }

 private:
  uint64_t jobs_{1};
};

class KernelMatcher final {
 public:
  bool Init(const std::vector<std::string> &patterns);
  bool Matches(std::string_view entry) const;

 private:
  std::vector<std::regex> patterns_;
};

class KernelFilter final {
 public:
  bool Init(const aclspecOptions *options);
  bool FilterSpecialize(std::string_view entry) const;

 private:
  bool hasAllowList_{false};
  KernelMatcher allow_;
  KernelMatcher deny_;
};

class CompileOptions final {
 public:
  bool Init(const aclspecOptions *options);
  std::vector<std::string> Basic(std::string_view entry) const;
  std::vector<std::string> Sk(std::string_view entry) const;

 private:
  struct Rule {
    KernelMatcher matcher;
    std::vector<std::string> arguments;
  };
  bool InitBasicOptions(const aclspecOption *options, uint64_t count);
  bool InitSkOptions(const aclskOptions *options);
  std::vector<Rule> basicRules_;
  std::vector<std::string> globalSkArguments_;
};

}  // namespace detail
}  // namespace static_compile
}  // namespace sk
#endif
