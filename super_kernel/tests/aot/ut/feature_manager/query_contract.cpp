/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "feature_manager.h"

using sk::static_compile::Feature;
using sk::static_compile::FeatureManager;

void CheckFeatureQueryContract(FeatureManager &manager) {
#if defined(QUERY_MODEL_WITH_ENTRY)
  (void)manager.Get<Feature::CompileJobs>("add");
#elif defined(QUERY_KERNEL_WITHOUT_ENTRY)
  (void)manager.Get<Feature::BasicCompileOptions>();
#elif defined(QUERY_COUNT)
  (void)manager.Get<Feature::Count>();
#elif defined(IGNORE_INIT)
  manager.Init(nullptr);
#else
  static_assert(std::is_same_v<decltype(manager.Get<Feature::CompileJobs>()), uint64_t>);
  static_assert(std::is_same_v<decltype(manager.Get<Feature::CompileEnabled>("add")), bool>);
  static_assert(std::is_same_v<decltype(manager.Get<Feature::BasicCompileOptions>("add")), std::vector<std::string> >);
  static_assert(std::is_same_v<decltype(manager.Get<Feature::SkCompileOptions>("add")), std::vector<std::string> >);
  (void)manager.Get<Feature::CompileJobs>();
  (void)manager.Get<Feature::CompileEnabled>("add");
  (void)manager.Get<Feature::BasicCompileOptions>("add");
  (void)manager.Get<Feature::SkCompileOptions>("add");
#endif
}
