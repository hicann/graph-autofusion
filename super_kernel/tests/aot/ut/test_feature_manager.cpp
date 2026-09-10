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

#include <algorithm>
#include <future>
#include <limits>
#include <stdexcept>
#include <thread>
#include "gtest/gtest.h"

namespace sk {
namespace static_compile {
namespace {
using Arguments = std::vector<std::string>;

class FeatureManagerTest : public testing::Test {
 protected:
  FeatureManager manager;
  aclspecOptions options{};
};

TEST_F(FeatureManagerTest, DefaultsAndModelJobs) {
  ASSERT_TRUE(manager.Init(nullptr));
  EXPECT_EQ(manager.Get<Feature::CompileJobs>(), std::max(1U, std::thread::hardware_concurrency()));
  EXPECT_TRUE(manager.Get<Feature::CompileEnabled>("addcustom"));
  EXPECT_TRUE(manager.Get<Feature::BasicCompileOptions>("addcustom").empty());
  EXPECT_TRUE(manager.Get<Feature::SkCompileOptions>("addcustom").empty());
  options.jobs = std::numeric_limits<uint64_t>::max();
  ASSERT_TRUE(manager.Init(&options));
  EXPECT_EQ(manager.Get<Feature::CompileJobs>(), options.jobs);
}

TEST_F(FeatureManagerTest, InvalidQueryUsageThrows) {
  EXPECT_THROW(manager.Get<Feature::CompileJobs>(), std::logic_error);
  EXPECT_THROW(manager.Get<Feature::BasicCompileOptions>("add"), std::logic_error);
  ASSERT_TRUE(manager.Init(nullptr));
  EXPECT_THROW(manager.Get<Feature::CompileEnabled>(""), std::invalid_argument);
  EXPECT_THROW(manager.Get<Feature::BasicCompileOptions>(""), std::invalid_argument);
  EXPECT_THROW(manager.Get<Feature::SkCompileOptions>(""), std::invalid_argument);
}

TEST_F(FeatureManagerTest, RegexFullMatchSupportsSuffixAndAlternation) {
  const char *args[] = {"-O2", "-DVALUE=1"};
  aclspecOption basic{R"((addcustom|mulcustom)(_[a-z0-9]+)?)", 2, args};
  options.specOptionCount = 1;
  options.specOptions = &basic;
  ASSERT_TRUE(manager.Init(&options));
  for (const char *name : {"addcustom", "addcustom_tiling", "addcustom_123", "mulcustom_9"}) {
    EXPECT_EQ(manager.Get<Feature::BasicCompileOptions>(name), (Arguments{"-O2", "-DVALUE=1"}));
  }
  for (const char *name : {"xaddcustom", "AddCustom", "addcustom_", "addcustom_123_x"}) {
    EXPECT_TRUE(manager.Get<Feature::BasicCompileOptions>(name).empty());
  }
}

TEST_F(FeatureManagerTest, MatchingRulesPreserveArgvOrderDuplicatesAndSpaces) {
  const char *args1[] = {"-mllvm", "first", "-O2"};
  const char *args2[] = {"-mllvm", "second", "-DNAME=two words", "-O2"};
  aclspecOption basic[] = {{"add.*", 3, args1}, {"addcustom", 4, args2}};
  options.specOptionCount = 2;
  options.specOptions = basic;
  ASSERT_TRUE(manager.Init(&options));
  EXPECT_EQ(manager.Get<Feature::BasicCompileOptions>("addcustom"),
            (Arguments{"-mllvm", "first", "-O2", "-mllvm", "second", "-DNAME=two words", "-O2"}));
}

TEST_F(FeatureManagerTest, InputIsDeepCopiedAndResultDoesNotAliasState) {
  {
    std::string pattern = "add.*";
    std::string arg = "-O2";
    const char *args[] = {arg.c_str()};
    aclspecOption basic{pattern.c_str(), 1, args};
    options.specOptionCount = 1;
    options.specOptions = &basic;
    ASSERT_TRUE(manager.Init(&options));
  }
  auto result = manager.Get<Feature::BasicCompileOptions>("addcustom");
  result[0] = "-O0";
  EXPECT_EQ(manager.Get<Feature::BasicCompileOptions>("addcustom"), (Arguments{"-O2"}));
}

TEST_F(FeatureManagerTest, FilterPatternsAreDeepCopied) {
  {
    std::string pattern = "add.*";
    const char *names[] = {pattern.c_str()};
    aclKernelSpecFilter allow{ACL_KERNEL_SPEC_FILTER_ALLOW, 1, names};
    aclKernelSpecFilter *filters[] = {&allow};
    options.filterCount = 1;
    options.filter = filters;
    ASSERT_TRUE(manager.Init(&options));
  }
  EXPECT_TRUE(manager.Get<Feature::CompileEnabled>("add_123"));
  EXPECT_FALSE(manager.Get<Feature::CompileEnabled>("mul_123"));
}

TEST_F(FeatureManagerTest, AllowRulesMergeIncludingEmptyAllow) {
  const char *names1[] = {"add(_.*)?"};
  const char *names2[] = {"matmul_[0-9]+"};
  aclKernelSpecFilter allow1{ACL_KERNEL_SPEC_FILTER_ALLOW, 1, names1};
  aclKernelSpecFilter allow2{ACL_KERNEL_SPEC_FILTER_ALLOW, 1, names2};
  aclKernelSpecFilter empty{ACL_KERNEL_SPEC_FILTER_ALLOW, 0, nullptr};
  aclKernelSpecFilter *filters[] = {&empty, &allow1, &allow2};
  options.filterCount = 3;
  options.filter = filters;
  ASSERT_TRUE(manager.Init(&options));
  EXPECT_TRUE(manager.Get<Feature::CompileEnabled>("add_tiling"));
  EXPECT_TRUE(manager.Get<Feature::CompileEnabled>("matmul_123"));
  EXPECT_FALSE(manager.Get<Feature::CompileEnabled>("relu"));
  options.filterCount = 1;
  ASSERT_TRUE(manager.Init(&options));
  EXPECT_FALSE(manager.Get<Feature::CompileEnabled>("add_tiling"));
}

TEST_F(FeatureManagerTest, DenyRulesAndEmptyDeny) {
  const char *names[] = {"add.*", "relu"};
  aclKernelSpecFilter deny{ACL_KERNEL_SPEC_FILTER_DENY, 2, names};
  aclKernelSpecFilter *filters[] = {&deny};
  options.filterCount = 1;
  options.filter = filters;
  ASSERT_TRUE(manager.Init(&options));
  EXPECT_FALSE(manager.Get<Feature::CompileEnabled>("addcustom"));
  EXPECT_FALSE(manager.Get<Feature::CompileEnabled>("relu"));
  EXPECT_TRUE(manager.Get<Feature::CompileEnabled>("xrelu"));
  deny.kernelEntryCount = 0;
  deny.kernelEntries = nullptr;
  ASSERT_TRUE(manager.Init(&options));
  EXPECT_TRUE(manager.Get<Feature::CompileEnabled>("relu"));
}

TEST_F(FeatureManagerTest, MixedAllowDenyAlwaysRejectedInEitherOrder) {
  aclKernelSpecFilter allow{ACL_KERNEL_SPEC_FILTER_ALLOW, 0, nullptr};
  aclKernelSpecFilter deny{ACL_KERNEL_SPEC_FILTER_DENY, 0, nullptr};
  aclKernelSpecFilter *filters[] = {&allow, &deny};
  options.filterCount = 2;
  options.filter = filters;
  EXPECT_FALSE(manager.Init(&options));
  std::swap(filters[0], filters[1]);
  EXPECT_FALSE(manager.Init(&options));
  EXPECT_THROW(manager.Get<Feature::CompileJobs>(), std::logic_error);
}

TEST_F(FeatureManagerTest, InvalidBasicPatternAndArgumentRejected) {
  aclspecOption basic{};
  options.specOptionCount = 1;
  options.specOptions = &basic;
  const char *invalidPatterns[] = {nullptr, "", "[", "**add**"};
  for (const char *pattern : invalidPatterns) {
    basic.kernelName = pattern;
    EXPECT_FALSE(manager.Init(&options));
  }
  basic.kernelName = "add.*";
  basic.compileOptionCount = 1;
  EXPECT_FALSE(manager.Init(&options));
  const char *args[] = {nullptr};
  basic.compileOption = args;
  EXPECT_FALSE(manager.Init(&options));
  args[0] = "";
  EXPECT_FALSE(manager.Init(&options));
  basic.compileOptionCount = 0;
  ASSERT_TRUE(manager.Init(&options));
}

TEST_F(FeatureManagerTest, InvalidFilterFieldsRejected) {
  aclKernelSpecFilter *filters[] = {nullptr};
  options.filterCount = 1;
  options.filter = filters;
  EXPECT_FALSE(manager.Init(&options));
  aclKernelSpecFilter filter{};
  filters[0] = &filter;
  filter.mode = static_cast<aclKernelSpecFilterMode>(9);
  EXPECT_FALSE(manager.Init(&options));
  filter.mode = ACL_KERNEL_SPEC_FILTER_ALLOW;
  filter.kernelEntryCount = 1;
  EXPECT_FALSE(manager.Init(&options));
  const char *names[] = {nullptr};
  filter.kernelEntries = names;
  EXPECT_FALSE(manager.Init(&options));
  names[0] = "";
  EXPECT_FALSE(manager.Init(&options));
  names[0] = "[";
  EXPECT_FALSE(manager.Init(&options));
}

TEST_F(FeatureManagerTest, CountPointerMismatchAndOverflowRejectedBeforeReading) {
  options.specOptionCount = 1;
  EXPECT_FALSE(manager.Init(&options));
  options.specOptionCount = 0;
  options.filterCount = 1;
  EXPECT_FALSE(manager.Init(&options));
  options.filterCount = 0;
  aclskOptions sk{nullptr, 1};
  options.skOptions = &sk;
  EXPECT_FALSE(manager.Init(&options));
  options.skOptions = nullptr;
  aclspecOption basic{".*", 0, nullptr};
  options.specOptions = &basic;
  options.specOptionCount = std::numeric_limits<uint64_t>::max();
  EXPECT_FALSE(manager.Init(&options));
  options.specOptionCount = 1;
  const char *args[] = {"-O2"};
  basic.compileOption = args;
  basic.compileOptionCount = std::numeric_limits<uint64_t>::max();
  EXPECT_FALSE(manager.Init(&options));
}

TEST_F(FeatureManagerTest, SupportedSkSwitchesProduceGlobalArguments) {
  aclskOption input[3]{};
  input[0].optionType = aclskOptionType::DEBUG_SYNC_ALL;
  input[0].debugSync.debugSyncAll = 1;
  input[1].optionType = aclskOptionType::DEBUG_PER_OP_MAX_CORE_NUM;
  input[1].debugPerOpMaxCoreNum.enableDebugPerOpMaxCoreNum = 1;
  input[2].optionType = aclskOptionType::DCCI_DISABLE_ON_KERNEL;
  aclskOptions sk{input, 3};
  options.skOptions = &sk;
  ASSERT_TRUE(manager.Init(&options));
  const Arguments expected{"--enable-super-kernel", "-D__DEBUG_SYNC_ALL__",
                           "-D__ENABLE_SUPER_KERNEL_INNER_CORE_SYNC_CHECK__", "--cce-no-dcache-flush"};
  EXPECT_EQ(manager.Get<Feature::SkCompileOptions>("addcustom"), expected);
  EXPECT_EQ(manager.Get<Feature::SkCompileOptions>("matmul_123"), expected);
  input[0].debugSync.debugSyncAll = 0;
  EXPECT_EQ(manager.Get<Feature::SkCompileOptions>("addcustom"), expected);
  EXPECT_TRUE(manager.Get<Feature::BasicCompileOptions>("addcustom").empty());
}

TEST_F(FeatureManagerTest, DcciPresenceIgnoresKernelListForModelScope) {
  char name[] = "only_this_kernel";
  char *names[] = {name};
  aclskOption input{};
  input.optionType = aclskOptionType::DCCI_DISABLE_ON_KERNEL;
  input.disableKernelDcci = {names, 1};
  aclskOptions sk{&input, 1};
  options.skOptions = &sk;
  ASSERT_TRUE(manager.Init(&options));
  EXPECT_EQ(manager.Get<Feature::SkCompileOptions>("another_kernel"),
            (Arguments{"--enable-super-kernel", "--cce-no-dcache-flush"}));
}

TEST_F(FeatureManagerTest, DisabledSkDoesNotEnableSuperKernel) {
  aclskOption input[2]{};
  input[0].optionType = aclskOptionType::DEBUG_SYNC_ALL;
  input[1].optionType = aclskOptionType::DEBUG_PER_OP_MAX_CORE_NUM;
  aclskOptions sk{input, 2};
  options.skOptions = &sk;
  ASSERT_TRUE(manager.Init(&options));
  EXPECT_TRUE(manager.Get<Feature::SkCompileOptions>("addcustom").empty());
  input[1].debugPerOpMaxCoreNum.enableDebugPerOpMaxCoreNum = 2;
  EXPECT_FALSE(manager.Init(&options));
  input[1].debugPerOpMaxCoreNum.enableDebugPerOpMaxCoreNum = 0;
  input[0].debugSync.debugSyncAll = 2;
  EXPECT_FALSE(manager.Init(&options));
}

TEST_F(FeatureManagerTest, DuplicateSupportedSkRejectedEvenWhenDisabled) {
  aclskOption input[2]{};
  aclskOptions sk{input, 2};
  options.skOptions = &sk;
  for (auto type : {aclskOptionType::DEBUG_SYNC_ALL, aclskOptionType::DEBUG_PER_OP_MAX_CORE_NUM,
                    aclskOptionType::DCCI_DISABLE_ON_KERNEL}) {
    input[0].optionType = type;
    input[1].optionType = type;
    EXPECT_FALSE(manager.Init(&options));
  }
}

TEST_F(FeatureManagerTest, UnsupportedSkIgnoredButInvalidEnumRejected) {
  aclskOption input[2]{};
  input[0].optionType = aclskOptionType::PRELOAD_CODE;
  input[1].optionType = aclskOptionType::PRELOAD_CODE;
  aclskOptions sk{input, 2};
  options.skOptions = &sk;
  ASSERT_TRUE(manager.Init(&options));
  EXPECT_TRUE(manager.Get<Feature::SkCompileOptions>("add").empty());
  for (uint32_t invalid : {6U, static_cast<uint32_t>(aclskOptionType::SK_OPTION_MAX), UINT32_MAX}) {
    input[0].optionType = static_cast<aclskOptionType>(invalid);
    EXPECT_FALSE(manager.Init(&options));
  }
}

TEST_F(FeatureManagerTest, AllKnownUnsupportedSkOptionsRemainAccepted) {
  aclskOption input{};
  aclskOptions sk{&input, 1};
  options.skOptions = &sk;
  for (auto type :
       {aclskOptionType::PRELOAD_CODE, aclskOptionType::SPLIT_MODE, aclskOptionType::STREAM_FUSION,
        aclskOptionType::KERNEL_MAP, aclskOptionType::AUTO_OP_PARALLEL, aclskOptionType::DCCI_BEFORE_KERNEL_START,
        aclskOptionType::DEBUG_OP_EXEC_TRACE, aclskOptionType::DEBUG_CROSS_CORE_SYNC_CHECK,
        aclskOptionType::OPT_EXTEND_OPTION, aclskOptionType::DEBUG_EXTEND_OPTION,
        aclskOptionType::DCCI_AFTER_KERNEL_END, aclskOptionType::AGGRESSIVE_OPT_STRATEGIES,
        aclskOptionType::UBUF_LOCK_IGNORE_KERNEL, aclskOptionType::EARLY_START}) {
    input.optionType = type;
    ASSERT_TRUE(manager.Init(&options)) << static_cast<uint32_t>(type);
    EXPECT_TRUE(manager.Get<Feature::SkCompileOptions>("add").empty());
  }
}

TEST_F(FeatureManagerTest, SkArgumentOrderFollowsInputRatherThanDescriptorOrder) {
  aclskOption input[3]{};
  input[0].optionType = aclskOptionType::DEBUG_PER_OP_MAX_CORE_NUM;
  input[0].debugPerOpMaxCoreNum.enableDebugPerOpMaxCoreNum = 1;
  input[1].optionType = aclskOptionType::DCCI_DISABLE_ON_KERNEL;
  input[2].optionType = aclskOptionType::DEBUG_SYNC_ALL;
  input[2].debugSync.debugSyncAll = 1;
  aclskOptions sk{input, 3};
  options.skOptions = &sk;
  ASSERT_TRUE(manager.Init(&options));
  EXPECT_EQ(manager.Get<Feature::SkCompileOptions>("add"),
            (Arguments{"--enable-super-kernel", "-D__ENABLE_SUPER_KERNEL_INNER_CORE_SYNC_CHECK__",
                       "--cce-no-dcache-flush", "-D__DEBUG_SYNC_ALL__"}));
}

TEST_F(FeatureManagerTest, FailedReinitializationPreservesAllPreviousProviders) {
  const char *args[] = {"-O2"};
  aclspecOption basic{"add.*", 1, args};
  const char *names[] = {"add.*"};
  aclKernelSpecFilter allow{ACL_KERNEL_SPEC_FILTER_ALLOW, 1, names};
  aclKernelSpecFilter *filters[] = {&allow};
  aclskOption input{};
  input.optionType = aclskOptionType::DEBUG_SYNC_ALL;
  input.debugSync.debugSyncAll = 1;
  aclskOptions sk{&input, 1};
  options = {4, 1, &basic, &sk, 1, filters};
  ASSERT_TRUE(manager.Init(&options));
  options.jobs = 16;
  options.filterCount = 0;
  args[0] = "-O0";
  input.debugSync.debugSyncAll = 2;
  EXPECT_FALSE(manager.Init(&options));
  EXPECT_EQ(manager.Get<Feature::CompileJobs>(), 4U);
  EXPECT_FALSE(manager.Get<Feature::CompileEnabled>("mul"));
  EXPECT_EQ(manager.Get<Feature::BasicCompileOptions>("add"), (Arguments{"-O2"}));
  EXPECT_EQ(manager.Get<Feature::SkCompileOptions>("add"),
            (Arguments{"--enable-super-kernel", "-D__DEBUG_SYNC_ALL__"}));
  ASSERT_TRUE(manager.Init(nullptr));
  EXPECT_TRUE(manager.Get<Feature::CompileEnabled>("mul"));
  EXPECT_TRUE(manager.Get<Feature::BasicCompileOptions>("add").empty());
  EXPECT_TRUE(manager.Get<Feature::SkCompileOptions>("add").empty());
}

TEST_F(FeatureManagerTest, ConcurrentReadQueriesHaveIndependentResults) {
  const char *args[] = {"-O2"};
  aclspecOption basic{"add_[0-9]+", 1, args};
  options.specOptions = &basic;
  options.specOptionCount = 1;
  ASSERT_TRUE(manager.Init(&options));
  std::vector<std::future<bool>> readers;
  for (size_t i = 0; i < 8; ++i) {
    readers.push_back(std::async(std::launch::async, [&] {
      for (size_t j = 0; j < 100; ++j) {
        if (manager.Get<Feature::BasicCompileOptions>("add_123") != Arguments{"-O2"} ||
            !manager.Get<Feature::CompileEnabled>("add_123")) {
          return false;
        }
      }
      return true;
    }));
  }
  for (auto &reader : readers) {
    EXPECT_TRUE(reader.get());
  }
}
}  // namespace
}  // namespace static_compile
}  // namespace sk
