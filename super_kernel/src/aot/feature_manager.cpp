/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "feature_manager.h"

#include <array>
#include <stdexcept>
#include <utility>
#include <variant>
#include "kernel_compile_features.h"

namespace sk {
namespace static_compile {

class FeatureManager::Impl {
  struct Provider {
    virtual ~Provider() = default;
    virtual bool Init(const aclspecOptions *options) = 0;
  };

  template <typename P>
  struct Holder final : Provider {
    P value;
    bool Init(const aclspecOptions *options) override {
      return value.Init(options);
    }
  };

  using Query = detail::FeatureValue (*)(const Provider &, std::string_view);
  struct Binding {
    size_t slot{0};
    bool kernelScoped{false};
    Query query{nullptr};
  };
  struct Factory {
    const void *type;
    std::unique_ptr<Provider> (*create)();
  };
  struct State {
    std::vector<std::unique_ptr<Provider>> providers;
  };

  template <typename P>
  static const void *TypeKey() {
    static const char key = 0;
    return &key;
  }

 public:
  Impl() {
    Register<Feature::CompileJobs, detail::ModelSettings, &detail::ModelSettings::CompileJobs>();
    Register<Feature::CompileEnabled, detail::KernelFilter, &detail::KernelFilter::FilterSpecialize>();
    Register<Feature::BasicCompileOptions, detail::CompileOptions, &detail::CompileOptions::Basic>();
    Register<Feature::SkCompileOptions, detail::CompileOptions, &detail::CompileOptions::Sk>();
  }

  bool Init(const aclspecOptions *options) {
    for (const auto &binding : bindings_) {
      if (binding.query == nullptr || binding.slot >= factories_.size()) {
        throw std::logic_error("Incomplete static compile feature registration");
      }
    }
    auto next = std::make_unique<State>();
    next->providers.reserve(factories_.size());
    for (const auto &factory : factories_) {
      auto provider = factory.create();
      if (!provider->Init(options)) {
        return false;
      }
      next->providers.push_back(std::move(provider));
    }
    // Commit only after every provider succeeds; adapters always resolve against this current state.
    state_.swap(next);
    return true;
  }

  detail::FeatureValue QueryFeature(Feature feature, std::optional<std::string_view> entry) const {
    if (!state_) {
      throw std::logic_error("FeatureManager must be initialized before Get");
    }
    const auto index = static_cast<size_t>(feature);
    if (index >= bindings_.size()) {
      throw std::logic_error("Invalid static compile feature binding");
    }
    const auto &binding = bindings_[index];
    if (binding.query == nullptr || binding.slot >= state_->providers.size() ||
        binding.kernelScoped != entry.has_value()) {
      throw std::logic_error("Static compile feature binding signature mismatch");
    }
    if (entry && entry->empty()) {
      throw std::invalid_argument("Static compile kernel entry must not be empty");
    }
    return binding.query(*state_->providers[binding.slot], entry.value_or(std::string_view{}));
  }

 private:
  template <Feature F, typename P, auto Method>
  void Register() {
    using Traits = detail::FeatureTraits<F>;
    using Result = typename Traits::Result;
    static_assert(std::is_default_constructible_v<P>, "Provider must be default constructible");
    static_assert(std::is_same_v<decltype(&P::Init), bool (P::*)(const aclspecOptions *)>,
                  "Provider must implement bool Init(const aclspecOptions *)");
    auto &binding = bindings_.at(static_cast<size_t>(F));
    if (binding.query != nullptr) {
      throw std::logic_error("Duplicate static compile feature registration");
    }
    size_t slot = 0;
    for (; slot < factories_.size(); ++slot) {
      if (factories_[slot].type == TypeKey<P>()) {
        break;
      }
    }
    // Basic and SK share one CompileOptions provider and one initialization per model.
    if (slot == factories_.size()) {
      factories_.push_back({TypeKey<P>(), []() -> std::unique_ptr<Provider> { return std::make_unique<Holder<P>>(); }});
    }
    binding.slot = slot;
    binding.kernelScoped = Traits::kKernelScoped;
    if constexpr (Traits::kKernelScoped) {
      static_assert(std::is_same_v<decltype(Method), Result (P::*)(std::string_view) const>,
                    "Kernel query signature must agree with FeatureTraits");
      binding.query = +[](const Provider &provider, std::string_view entry) -> detail::FeatureValue {
        return detail::FeatureValue(std::in_place_type<Result>,
                                    (static_cast<const Holder<P> &>(provider).value.*Method)(entry));
      };
    } else {
      static_assert(std::is_same_v<decltype(Method), Result (P::*)() const>,
                    "Model query signature must agree with FeatureTraits");
      binding.query = +[](const Provider &provider, std::string_view) -> detail::FeatureValue {
        return detail::FeatureValue(std::in_place_type<Result>,
                                    (static_cast<const Holder<P> &>(provider).value.*Method)());
      };
    }
  }

  std::array<Binding, static_cast<size_t>(Feature::Count)> bindings_{};
  std::vector<Factory> factories_;
  std::unique_ptr<State> state_;
};

FeatureManager::FeatureManager() : impl_(std::make_unique<Impl>()) {}
FeatureManager::~FeatureManager() = default;
bool FeatureManager::Init(const aclspecOptions *options) {
  return impl_->Init(options);
}
detail::FeatureValue FeatureManager::QueryFeature(Feature feature, std::optional<std::string_view> entry) const {
  return impl_->QueryFeature(feature, entry);
}

}  // namespace static_compile
}  // namespace sk
