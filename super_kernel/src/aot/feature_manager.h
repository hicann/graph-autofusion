/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef STATIC_NORMAL_FEATURE_MANAGER_H
#define STATIC_NORMAL_FEATURE_MANAGER_H

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>
#include "kernel_spec_options.h"

namespace sk {
namespace static_compile {

enum class Feature : uint32_t { CompileJobs, CompileEnabled, BasicCompileOptions, SkCompileOptions, Count };

namespace detail {
// Supported business values, independent of model/kernel query scope.
using FeatureValue = std::variant<uint64_t, bool, std::vector<std::string>>;
template <Feature F>
struct FeatureTraits;
template <>
struct FeatureTraits<Feature::CompileJobs> {
  using Result = uint64_t;
  static constexpr bool kKernelScoped = false;
};
template <>
struct FeatureTraits<Feature::CompileEnabled> {
  using Result = bool;
  static constexpr bool kKernelScoped = true;
};
template <>
struct FeatureTraits<Feature::BasicCompileOptions> {
  using Result = std::vector<std::string>;
  static constexpr bool kKernelScoped = true;
};
template <>
struct FeatureTraits<Feature::SkCompileOptions> : FeatureTraits<Feature::BasicCompileOptions> {};
}  // namespace detail

// Init owns a snapshot; callers may release input data after Init returns.
// Init/destruction require exclusive access. Successful Get calls may run concurrently.
class FeatureManager final {
 public:
  FeatureManager();
  ~FeatureManager();
  FeatureManager(const FeatureManager &) = delete;
  FeatureManager &operator=(const FeatureManager &) = delete;

  // Invalid configuration returns false. Exceptions propagate; either failure preserves the old state.
  [[nodiscard]] bool Init(const aclspecOptions *options);

  template <Feature F>
  typename detail::FeatureTraits<F>::Result Get() const {
    static_assert(!detail::FeatureTraits<F>::kKernelScoped, "This feature requires a kernel entry");
    return std::get<typename detail::FeatureTraits<F>::Result>(QueryFeature(F, std::nullopt));
  }

  // Uninitialized queries throw logic_error; an empty kernelEntry throws invalid_argument.
  template <Feature F>
  typename detail::FeatureTraits<F>::Result Get(std::string_view kernelEntry) const {
    static_assert(detail::FeatureTraits<F>::kKernelScoped, "This feature has model scope");
    return std::get<typename detail::FeatureTraits<F>::Result>(QueryFeature(F, kernelEntry));
  }

 private:
  detail::FeatureValue QueryFeature(Feature feature, std::optional<std::string_view> entry) const;
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace static_compile
}  // namespace sk
#endif
