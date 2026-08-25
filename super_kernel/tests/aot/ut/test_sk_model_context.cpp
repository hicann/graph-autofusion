/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_sk_model_context.cpp
 * \brief Unit tests for sk_model_context.h (per-model identity, sk_meta layout, SkModelContext)
 */

#include <gtest/gtest.h>
#include <array>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#include <atomic>
#include <fstream>
#include <future>
#include <string>
#include <set>
#include <mutex>

#define private public
#include "sk_log.h"
#undef private
#include "sk_model_context.h"

namespace {

constexpr const char *MODEL_LABEL_PREFIX = "model_";
constexpr const char *UNKNOWN_MODEL_ID = "unknown";

std::string UnknownModelLabel() {
  return std::string(MODEL_LABEL_PREFIX) + UNKNOWN_MODEL_ID;
}

// 测试用的 aclmdlRI 模拟句柄
inline aclmdlRI MakeFakeModel(uintptr_t v) {
  return reinterpret_cast<aclmdlRI>(v);
}

std::string ExpectedModelId(uint32_t rtsModelId, uint64_t callCount) {
  return std::to_string(rtsModelId) + "_" + std::to_string(callCount);
}

std::string ExpectedModelLabel(uint32_t rtsModelId, uint64_t callCount) {
  return "model_" + ExpectedModelId(rtsModelId, callCount);
}

// 清理 sk_meta 测试目录（递归）
void CleanupSkMetaDirs() {
  pid_t pid = getpid();
  std::string pidDir = "sk_meta/" + std::to_string(pid);
  // best-effort 清理；遗留的子目录留给后续测试或人工处理
  rmdir(pidDir.c_str());
  rmdir("sk_meta");
}

}  // namespace

// ==================== 测试 Fixture ====================

class TestSkModelContext : public testing::Test {
 protected:
  void SetUp() override {
    CleanupSkMetaDirs();
  }

  void TearDown() override {
    CleanupSkMetaDirs();
  }
};

// ==================== Public entry counter behavior ====================

TEST_F(TestSkModelContext, SkModelContext_RepeatedInvocationBumpsCounter) {
  constexpr uint32_t modelId = 0xA001;
  aclmdlRI model = MakeFakeModel(modelId);

  {
    SkModelContext firstGuard(model);
    EXPECT_EQ(GetCurrentModelId(), ExpectedModelId(modelId, 1));
    EXPECT_EQ(GetCurrentModelLabel(), ExpectedModelLabel(modelId, 1));
  }

  {
    SkModelContext secondGuard(model);
    EXPECT_EQ(GetCurrentModelId(), ExpectedModelId(modelId, 2));
    EXPECT_EQ(GetCurrentModelLabel(), ExpectedModelLabel(modelId, 2));
  }

  EXPECT_EQ(GetCurrentModelId(), UNKNOWN_MODEL_ID);
}

TEST_F(TestSkModelContext, SkModelContext_SameRtsModelIdDifferentAddressSharesCounter) {
  constexpr uint32_t modelId = 0xA002;
  aclmdlRI firstModel = MakeFakeModel(0x100000000ULL | modelId);
  aclmdlRI secondModel = MakeFakeModel(0x200000000ULL | modelId);

  {
    SkModelContext firstGuard(firstModel);
    EXPECT_EQ(GetCurrentModelId(), ExpectedModelId(modelId, 1));
  }
  {
    SkModelContext secondGuard(secondModel);
    EXPECT_EQ(GetCurrentModelId(), ExpectedModelId(modelId, 2));
  }
}

TEST_F(TestSkModelContext, SkModelContext_ConcurrentInvocationsProduceUniqueIds) {
  constexpr uint32_t modelId = 0xA003;
  aclmdlRI model = MakeFakeModel(modelId);
  const int numThreads = 4;
  const int guardCountPerThread = 25;
  const int totalGuardCount = numThreads * guardCountPerThread;
  std::vector<std::thread> threads;
  std::set<std::string> observedIds;
  std::mutex observedIdsMutex;
  for (int t = 0; t < numThreads; ++t) {
    threads.emplace_back([model, guardCountPerThread, &observedIds, &observedIdsMutex]() {
      for (int i = 0; i < guardCountPerThread; ++i) {
        SkModelContext guard(model);
        std::lock_guard<std::mutex> lock(observedIdsMutex);
        observedIds.insert(GetCurrentModelId());
      }
    });
  }
  for (auto &th : threads) th.join();

  EXPECT_EQ(observedIds.size(), static_cast<size_t>(totalGuardCount));
  EXPECT_NE(observedIds.find(ExpectedModelId(modelId, 1)), observedIds.end());
  EXPECT_NE(observedIds.find(ExpectedModelId(modelId, totalGuardCount)), observedIds.end());
}

// ==================== Thread-local frozen id ====================

TEST_F(TestSkModelContext, CurrentModelContext_DefaultValue) {
  EXPECT_EQ(GetCurrentModelId(), UNKNOWN_MODEL_ID);
  EXPECT_EQ(GetCurrentModelLabel(), UnknownModelLabel());
}

TEST_F(TestSkModelContext, CurrentModelContext_ThreadLocalIsolated) {
  constexpr uint32_t mainModelId = 0xA006;
  constexpr uint32_t workerModelId = 0xA007;
  SkModelContext mainGuard(MakeFakeModel(mainModelId));
  std::string workerBeforeGuard;
  std::string workerInsideGuard;
  std::thread worker([&workerBeforeGuard, &workerInsideGuard]() {
    workerBeforeGuard = GetCurrentModelLabel();
    SkModelContext workerGuard(MakeFakeModel(workerModelId));
    workerInsideGuard = GetCurrentModelLabel();
  });
  worker.join();
  EXPECT_EQ(workerBeforeGuard, UnknownModelLabel());
  EXPECT_EQ(workerInsideGuard, ExpectedModelLabel(workerModelId, 1));
  EXPECT_EQ(GetCurrentModelLabel(), ExpectedModelLabel(mainModelId, 1));
}

// ==================== SkModelContext ====================

TEST_F(TestSkModelContext, SkModelContext_FreezesUniqueIdAndBumpsCounter) {
  constexpr uint32_t modelId = 0xA008;
  aclmdlRI model = MakeFakeModel(modelId);
  EXPECT_EQ(GetCurrentModelId(), UNKNOWN_MODEL_ID);
  EXPECT_EQ(GetCurrentModelLabel(), UnknownModelLabel());

  {
    SkModelContext guard(model);
    EXPECT_EQ(GetCurrentModelId(), ExpectedModelId(modelId, 1));
    EXPECT_EQ(GetCurrentModelLabel(), ExpectedModelLabel(modelId, 1));
  }

  // RAII 析构后恢复为默认上下文
  EXPECT_EQ(GetCurrentModelId(), UNKNOWN_MODEL_ID);
  EXPECT_EQ(GetCurrentModelLabel(), UnknownModelLabel());
}

TEST_F(TestSkModelContext, SkModelContext_NestedScopeDoesNotRestoreOuterContext) {
  constexpr uint32_t outerModelId = 0xA009;
  constexpr uint32_t innerModelId = 0xA00A;
  aclmdlRI outer = MakeFakeModel(outerModelId);
  aclmdlRI inner = MakeFakeModel(innerModelId);

  {
    SkModelContext outerGuard(outer);
    EXPECT_EQ(GetCurrentModelId(), ExpectedModelId(outerModelId, 1));
    EXPECT_EQ(GetCurrentModelLabel(), ExpectedModelLabel(outerModelId, 1));
    {
      SkModelContext innerGuard(inner);
      EXPECT_EQ(GetCurrentModelId(), ExpectedModelId(innerModelId, 1));
      EXPECT_EQ(GetCurrentModelLabel(), ExpectedModelLabel(innerModelId, 1));
    }
    EXPECT_EQ(GetCurrentModelId(), UNKNOWN_MODEL_ID);
    EXPECT_EQ(GetCurrentModelLabel(), UnknownModelLabel());
  }
  EXPECT_EQ(GetCurrentModelId(), UNKNOWN_MODEL_ID);
  EXPECT_EQ(GetCurrentModelLabel(), UnknownModelLabel());
}

TEST_F(TestSkModelContext, SkModelContext_RepeatedInvocationDisambiguates) {
  constexpr uint32_t modelId = 0xA00B;
  aclmdlRI model = MakeFakeModel(modelId);
  std::string first;
  std::string second;
  {
    SkModelContext g1(model);
    first = GetCurrentModelLabel();
  }
  {
    SkModelContext g2(model);
    second = GetCurrentModelLabel();
  }
  EXPECT_EQ(first, ExpectedModelLabel(modelId, 1));
  EXPECT_EQ(second, ExpectedModelLabel(modelId, 2));
  EXPECT_NE(first, second);  // 反复调用同一 handle，id 不冲突
}

TEST_F(TestSkModelContext, SkModelContext_SameRtsModelIdDifferentAddressDisambiguates) {
  constexpr uint32_t modelId = 0xA00C;
  aclmdlRI firstModel = MakeFakeModel(0x100000000ULL | modelId);
  aclmdlRI secondModel = MakeFakeModel(0x200000000ULL | modelId);
  std::string first;
  std::string second;
  {
    SkModelContext guard(firstModel);
    first = GetCurrentModelId();
  }
  {
    SkModelContext guard(secondModel);
    second = GetCurrentModelId();
  }
  EXPECT_EQ(first, ExpectedModelId(modelId, 1));
  EXPECT_EQ(second, ExpectedModelId(modelId, 2));
}

TEST_F(TestSkModelContext, SkModelContext_NullModelStillSetsFrozenId) {
  {
    SkModelContext guard(nullptr);
    EXPECT_EQ(GetCurrentModelId(), UNKNOWN_MODEL_ID);
    // nullptr 不递增 counter，但仍设置一个稳定的 frozen id
    EXPECT_EQ(GetCurrentModelLabel(), UnknownModelLabel());
  }
  EXPECT_EQ(GetCurrentModelId(), UNKNOWN_MODEL_ID);
  EXPECT_EQ(GetCurrentModelLabel(), UnknownModelLabel());
}

// ==================== SanitizePathComponent ====================

TEST_F(TestSkModelContext, SanitizePathComponent_NoInvalidCharsUnchanged) {
  EXPECT_EQ(SanitizePathComponent("model_42_1"), "model_42_1");
  EXPECT_EQ(SanitizePathComponent("simple_name"), "simple_name");
  EXPECT_EQ(SanitizePathComponent(""), "");
}

TEST_F(TestSkModelContext, SanitizePathComponent_ReplacesAllInvalidChars) {
  // 9 个非法字符全替换为下划线
  EXPECT_EQ(SanitizePathComponent("a/b\\c:d*e?f\"g<h>i|j"), "a_b_c_d_e_f_g_h_i_j");
}

TEST_F(TestSkModelContext, SanitizePathComponent_PreservesNormalChars) {
  std::string s = "abc-123.XYZ_test";
  EXPECT_EQ(SanitizePathComponent(s), s);
}

TEST_F(TestSkModelContext, SanitizePathComponent_AllInvalidProducesAllUnderscores) {
  EXPECT_EQ(SanitizePathComponent("/\\:*?\"<>|"), "_________");
}

// ==================== GetSkMetaBasePath / GetSkMetaPath ====================

TEST_F(TestSkModelContext, GetSkMetaBasePath_HasPidSuffix) {
  std::string base = GetSkMetaBasePath();
  std::string expected = "sk_meta/" + std::to_string(getpid());
  EXPECT_EQ(base, expected);
}

TEST_F(TestSkModelContext, GetSkMetaPath_UnknownModelUsesUnknownLabel) {
  std::string path = GetSkMetaPath(UnknownModelLabel());
  EXPECT_EQ(path, GetSkMetaBasePath() + "/" + UnknownModelLabel());
}

TEST_F(TestSkModelContext, GetSkMetaPath_UsesFrozenIdWhenActive) {
  constexpr uint32_t modelId = 0xA010;
  aclmdlRI model = MakeFakeModel(modelId);
  SkModelContext guard(model);
  EXPECT_EQ(GetSkMetaPath(GetCurrentModelLabel()), GetSkMetaBasePath() + "/" + ExpectedModelLabel(modelId, 1));
}

TEST_F(TestSkModelContext, GetSkMetaPath_UsesExplicitModelLabel) {
  constexpr uint32_t modelId = 0xA011;
  EXPECT_EQ(GetSkMetaPath(ExpectedModelLabel(modelId, 0)), GetSkMetaBasePath() + "/" + ExpectedModelLabel(modelId, 0));
}

// ==================== CreateDirectoryRecursive ====================

TEST_F(TestSkModelContext, CreateDirectoryRecursive_EmptyPathReturnsFalse) {
  EXPECT_FALSE(CreateDirectoryRecursive(""));
}

TEST_F(TestSkModelContext, CreateDirectoryRecursive_CreatesNestedPath) {
  std::string base = "sk_meta/" + std::to_string(getpid()) + "/nested_a/nested_b";
  EXPECT_TRUE(CreateDirectoryRecursive(base));

  struct stat st;
  EXPECT_EQ(stat(base.c_str(), &st), 0);
  EXPECT_TRUE(S_ISDIR(st.st_mode));

  // 清理
  rmdir(base.c_str());
  std::string parent = "sk_meta/" + std::to_string(getpid()) + "/nested_a";
  rmdir(parent.c_str());
}

TEST_F(TestSkModelContext, CreateDirectoryRecursive_IdempotentOnExisting) {
  std::string path = "sk_meta/" + std::to_string(getpid()) + "/idem";
  EXPECT_TRUE(CreateDirectoryRecursive(path));
  // 二次调用应仍成功
  EXPECT_TRUE(CreateDirectoryRecursive(path));
  rmdir(path.c_str());
}

// ==================== CreateSkMetaDirectory ====================

TEST_F(TestSkModelContext, CreateSkMetaDirectory_UnknownModelCreatesUnknownSubdir) {
  std::string path = CreateSkMetaDirectory(UnknownModelLabel());
  EXPECT_FALSE(path.empty());
  EXPECT_EQ(path, GetSkMetaBasePath() + "/" + UnknownModelLabel());

  struct stat st;
  EXPECT_EQ(stat(path.c_str(), &st), 0);
  EXPECT_TRUE(S_ISDIR(st.st_mode));

  rmdir(path.c_str());
}

TEST_F(TestSkModelContext, CreateSkMetaDirectory_UsesFrozenIdWhenActive) {
  constexpr uint32_t modelId = 0xA012;
  aclmdlRI model = MakeFakeModel(modelId);
  SkModelContext guard(model);

  std::string path = CreateSkMetaDirectory(GetCurrentModelLabel());
  EXPECT_EQ(path, GetSkMetaBasePath() + "/" + ExpectedModelLabel(modelId, 1));

  struct stat st;
  EXPECT_EQ(stat(path.c_str(), &st), 0);
  EXPECT_TRUE(S_ISDIR(st.st_mode));

  rmdir(path.c_str());
}

TEST_F(TestSkModelContext, CreateSkMetaDirectory_IdempotentOnSameModelHandle) {
  aclmdlRI model = MakeFakeModel(0xA013);
  SkModelContext guard(model);

  std::string first = CreateSkMetaDirectory(GetCurrentModelLabel());
  std::string second = CreateSkMetaDirectory(GetCurrentModelLabel());
  // 同一个 guard 范围内，frozen id 不变，路径相同且都成功
  EXPECT_EQ(first, second);
  EXPECT_FALSE(first.empty());

  rmdir(first.c_str());
}

// ==================== 端到端：guard + 路径一致性 ====================

TEST_F(TestSkModelContext, GuardedScope_PathStaysStableAcrossCalls) {
  aclmdlRI model = MakeFakeModel(0xA014);
  SkModelContext guard(model);

  std::string id1 = GetCurrentModelLabel();
  std::string path1 = GetSkMetaPath(id1);
  std::string id2 = GetCurrentModelLabel();
  std::string path2 = GetSkMetaPath(id2);

  EXPECT_EQ(id1, id2);
  EXPECT_EQ(path1, path2);
}

TEST_F(TestSkModelContext, BuildModelLabel_UsesCanonicalPrefixAndUnknownFallback) {
  EXPECT_EQ(BuildModelLabel("12_3"), "model_12_3");
  EXPECT_EQ(BuildModelLabel(""), UnknownModelLabel());
}

TEST_F(TestSkModelContext, LogContextGuard_RoutesNestedModelsAndRestoresPreviousState) {
  const std::string originalLabel = "model_route_original";
  const std::string outerLabel = "model_route_outer";
  const std::string innerLabel = "model_route_inner";

  sk::logger::LoggerConfig config;
  config.enabled = true;
  config.modelLabel = originalLabel;
  ASSERT_TRUE(sk::logger::FileLogger::Instance().Initialize(config));

  const std::string originalHandle = "model_" + originalLabel;
  sk::logger::FileLogger::SetCurrentModelLabel(originalLabel);
  ASSERT_TRUE(sk::logger::FileHandleManager::Instance().SwitchToFile(originalHandle));

  {
    sk::logger::LogContextGuard outerContext(outerLabel);
    EXPECT_EQ(sk::logger::FileLogger::GetCurrentModelLabel(), outerLabel);
    EXPECT_EQ(sk::logger::FileHandleManager::Instance().GetCurrentHandle(), "default");

    {
      sk::logger::LogContextGuard innerContext(innerLabel);
      EXPECT_EQ(sk::logger::FileLogger::GetCurrentModelLabel(), innerLabel);
      EXPECT_EQ(sk::logger::FileHandleManager::Instance().GetCurrentHandle(), "default");
    }

    EXPECT_EQ(sk::logger::FileLogger::GetCurrentModelLabel(), outerLabel);
    EXPECT_EQ(sk::logger::FileHandleManager::Instance().GetCurrentHandle(), "default");
  }

  EXPECT_EQ(sk::logger::FileLogger::GetCurrentModelLabel(), originalLabel);
  EXPECT_EQ(sk::logger::FileHandleManager::Instance().GetCurrentHandle(), originalHandle);
  sk::logger::FileLogger::Instance().SetEnabled(false);
  sk::logger::FileHandleManager::Instance().SwitchToDefault();
}

TEST_F(TestSkModelContext, LogContextGuard_LoggerDisabledDoesNotChangeModelContext) {
  const std::string originalLabel = "model_route_disabled_original";
  sk::logger::FileLogger::SetCurrentModelLabel(originalLabel);
  sk::logger::FileLogger::Instance().SetEnabled(false);
  sk::logger::FileHandleManager::Instance().SwitchToDefault();

  {
    sk::logger::LogContextGuard logContext("model_route_disabled_target");
    EXPECT_EQ(sk::logger::FileLogger::GetCurrentModelLabel(), originalLabel);
    EXPECT_EQ(sk::logger::FileHandleManager::Instance().GetCurrentHandle(), "default");
  }

  EXPECT_EQ(sk::logger::FileLogger::GetCurrentModelLabel(), originalLabel);
  EXPECT_EQ(sk::logger::FileHandleManager::Instance().GetCurrentHandle(), "default");
}

TEST_F(TestSkModelContext, LogContextGuard_ConcurrentThreadsKeepModelContextsIsolated) {
  sk::logger::LoggerConfig config;
  config.enabled = true;
  config.modelLabel = "model_route_thread_config";
  ASSERT_TRUE(sk::logger::FileLogger::Instance().Initialize(config));

  std::atomic<uint32_t> readyCount{0U};
  std::array<bool, 2> routeChecks{false, false};
  std::array<bool, 2> restoreChecks{false, false};
  std::vector<std::thread> workers;
  workers.reserve(routeChecks.size());
  for (size_t i = 0; i < routeChecks.size(); ++i) {
    workers.emplace_back([i, &readyCount, &routeChecks, &restoreChecks]() {
      const std::string previousLabel = "model_route_thread_previous_" + std::to_string(i);
      const std::string targetLabel = "model_route_thread_target_" + std::to_string(i);
      sk::logger::FileLogger::SetCurrentModelLabel(previousLabel);
      sk::logger::FileHandleManager::Instance().SwitchToDefault();
      {
        sk::logger::LogContextGuard logContext(targetLabel);
        readyCount.fetch_add(1U, std::memory_order_release);
        while (readyCount.load(std::memory_order_acquire) < routeChecks.size()) {
          std::this_thread::yield();
        }
        routeChecks[i] = sk::logger::FileLogger::GetCurrentModelLabel() == targetLabel &&
                         sk::logger::FileHandleManager::Instance().GetCurrentHandle() == "default";
      }
      restoreChecks[i] = sk::logger::FileLogger::GetCurrentModelLabel() == previousLabel &&
                         sk::logger::FileHandleManager::Instance().GetCurrentHandle() == "default";
    });
  }

  for (auto &worker : workers) {
    worker.join();
  }
  EXPECT_TRUE(routeChecks[0]);
  EXPECT_TRUE(routeChecks[1]);
  EXPECT_TRUE(restoreChecks[0]);
  EXPECT_TRUE(restoreChecks[1]);
  sk::logger::FileLogger::Instance().SetEnabled(false);
}

TEST_F(TestSkModelContext, FileLoggerConcurrentConfigurationAccessCompletesWithFinalState) {
  sk::logger::LoggerConfig config;
  config.enabled = false;
  config.modelLabel = "concurrent_config";
  ASSERT_TRUE(sk::logger::FileLogger::Instance().Initialize(config));

  constexpr uint32_t threadCount = 8;
  constexpr uint32_t iterations = 1000;
  std::atomic<uint32_t> ready{0};
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  for (uint32_t threadIndex = 0; threadIndex < threadCount; ++threadIndex) {
    threads.emplace_back([&, threadIndex]() {
      ready.fetch_add(1, std::memory_order_relaxed);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (uint32_t i = 0; i < iterations; ++i) {
        sk::logger::FileLogger::Instance().SetEnabled((i + threadIndex) % 2U == 0U);
        sk::logger::FileLogger::Instance().SetMinLevel((i % 2U == 0U) ? sk::logger::LogLevel::DEBUG
                                                                      : sk::logger::LogLevel::WARNING);
        sk::logger::FileLogger::Instance().SetModelLabel("concurrent_model_" + std::to_string(threadIndex));
        (void)sk::logger::FileLogger::Instance().IsEnabled();
        sk::logger::FileLogger::Instance().WriteLogIfEnabled(sk::logger::LogLevel::INFO, __FUNCTION__, __FILE__,
                                                             __LINE__, "iteration=%u", i);
      }
    });
  }

  while (ready.load(std::memory_order_acquire) != threadCount) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (auto &thread : threads) {
    thread.join();
  }

  sk::logger::FileLogger::Instance().SetEnabled(false);
  EXPECT_FALSE(sk::logger::FileLogger::Instance().IsEnabled());
  sk::logger::FileLogger::Instance().SetEnabled(true);
  EXPECT_TRUE(sk::logger::FileLogger::Instance().IsEnabled());
  sk::logger::FileLogger::Instance().SetEnabled(false);
}

TEST_F(TestSkModelContext, FileLoggerInitializationLockDoesNotBlockLogConfigSnapshot) {
  auto &logger = sk::logger::FileLogger::Instance();
  sk::logger::LoggerConfig config;
  config.enabled = true;
  config.modelLabel = "initialization_lock_snapshot";
  ASSERT_TRUE(logger.Initialize(config));

  std::unique_lock<std::mutex> initializationLock(logger.initializationMutex_);
  auto writeFuture = std::async(std::launch::async, [&logger]() {
    logger.WriteLogIfEnabled(sk::logger::LogLevel::INFO, __FUNCTION__, __FILE__, __LINE__,
                             "log while initialization is serialized");
  });

  EXPECT_EQ(writeFuture.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  initializationLock.unlock();
  writeFuture.get();
  logger.SetEnabled(false);
}

TEST_F(TestSkModelContext, FileHandleManagerRegisterFailureDoesNotReenterMutex) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        alarm(1);
        sk::logger::FileLogger::Instance().SetEnabled(true);
        sk::logger::FileLogger::Instance().SetMinLevel(sk::logger::LogLevel::ERROR);
        bool registered = sk::logger::FileHandleManager::Instance().RegisterFile(
            "reentrant_register_failure", "/proc/graph_autofusion_missing/super_kernel.log");
        _exit(registered ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(TestSkModelContext, FileHandleManagerSwitchFailureDoesNotReenterMutex) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        alarm(1);
        sk::logger::FileLogger::Instance().SetEnabled(true);
        sk::logger::FileLogger::Instance().SetMinLevel(sk::logger::LogLevel::ERROR);
        bool switched = sk::logger::FileHandleManager::Instance().SwitchToFile("missing_handle_for_reentrant_log");
        _exit(switched ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(TestSkModelContext, FileHandleManagerLogDoesNotAcquireLoggerConfigMutex) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        alarm(1);
        auto &logger = sk::logger::FileLogger::Instance();
        auto &handleManager = sk::logger::FileHandleManager::Instance();
        logger.SetEnabled(true);
        logger.SetMinLevel(sk::logger::LogLevel::ERROR);
        std::lock_guard<decltype(handleManager.mutex_)> managerLock(handleManager.mutex_);
        std::lock_guard<std::mutex> configLock(logger.mutex_);
        logger.WriteLogIfEnabled(sk::logger::LogLevel::ERROR, __FUNCTION__, __FILE__, __LINE__,
                                 "log while manager and config are locked");
        _exit(0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(TestSkModelContext, FileHandleManagerFailureIsWrittenToFileAfterUnlock) {
  const std::string modelLabel = "file_handle_failure_file_sink";
  const std::string handleName = "model_" + modelLabel;
  const std::string logPath = GetSkMetaBasePath() + "/" + modelLabel + "/super_kernel.log";
  sk::logger::LoggerConfig config;
  config.enabled = true;
  config.modelLabel = modelLabel;
  sk::logger::FileLogger::SetCurrentModelLabel(modelLabel);
  ASSERT_TRUE(sk::logger::FileLogger::Instance().Initialize(config));

  EXPECT_FALSE(sk::logger::FileHandleManager::Instance().SwitchToFile("missing_handle_for_file_sink"));
  sk::logger::FileLogger::Instance().SetEnabled(false);
  sk::logger::FileHandleManager::Instance().CloseFile(handleName);
  sk::logger::FileHandleManager::Instance().SwitchToDefault();

  std::ifstream file(logPath);
  ASSERT_TRUE(file.good());
  bool foundError = false;
  std::string line;
  while (std::getline(file, line)) {
    if (line.find("File handle not found: missing_handle_for_file_sink") != std::string::npos) {
      foundError = true;
      break;
    }
  }
  EXPECT_TRUE(foundError);
  file.close();
  std::remove(logPath.c_str());
  rmdir((GetSkMetaBasePath() + "/" + modelLabel).c_str());
  sk::logger::FileLogger::SetCurrentModelLabel("");
}

TEST_F(TestSkModelContext, LogContextGuard_FileAndModelContextsRestoreInNestedOrder) {
  const std::string originalLabel = "model_context_original";
  const std::string nestedLabel = "model_context_nested";
  sk::logger::LoggerConfig config;
  config.enabled = true;
  config.modelLabel = originalLabel;
  ASSERT_TRUE(sk::logger::FileLogger::Instance().Initialize(config));
  sk::logger::FileLogger::SetCurrentModelLabel(originalLabel);
  sk::logger::FileHandleManager::Instance().SwitchToDefault();

  auto fileContext = sk::logger::FileLogger::Instance().CreateContext("nested_context.log", originalLabel);
  ASSERT_NE(fileContext, nullptr);
  const std::string fileHandle = sk::logger::FileHandleManager::Instance().GetCurrentHandle();
  EXPECT_NE(fileHandle, "default");

  {
    sk::logger::LogContextGuard modelContext(nestedLabel);
    EXPECT_EQ(sk::logger::FileLogger::GetCurrentModelLabel(), nestedLabel);
    EXPECT_EQ(sk::logger::FileHandleManager::Instance().GetCurrentHandle(), "default");
  }

  EXPECT_EQ(sk::logger::FileLogger::GetCurrentModelLabel(), originalLabel);
  EXPECT_EQ(sk::logger::FileHandleManager::Instance().GetCurrentHandle(), fileHandle);
  fileContext.reset();
  EXPECT_EQ(sk::logger::FileLogger::GetCurrentModelLabel(), originalLabel);
  EXPECT_EQ(sk::logger::FileHandleManager::Instance().GetCurrentHandle(), "default");
  sk::logger::FileLogger::Instance().SetEnabled(false);
}

TEST_F(TestSkModelContext, LogContextGuard_MoveTransfersContextOwnership) {
  const std::string originalLabel = "model_move_original";
  const std::string targetLabel = "model_move_target";
  sk::logger::LoggerConfig config;
  config.enabled = true;
  config.modelLabel = originalLabel;
  ASSERT_TRUE(sk::logger::FileLogger::Instance().Initialize(config));
  sk::logger::FileLogger::SetCurrentModelLabel(originalLabel);
  sk::logger::FileHandleManager::Instance().SwitchToDefault();

  {
    sk::logger::LogContextGuard source(targetLabel);
    sk::logger::LogContextGuard moved(std::move(source));
    EXPECT_FALSE(source.IsActive());
    EXPECT_TRUE(moved.IsActive());
    EXPECT_EQ(sk::logger::FileLogger::GetCurrentModelLabel(), targetLabel);
  }
  EXPECT_EQ(sk::logger::FileLogger::GetCurrentModelLabel(), originalLabel);
  EXPECT_EQ(sk::logger::FileHandleManager::Instance().GetCurrentHandle(), "default");

  {
    sk::logger::FileLogger::Instance().SetEnabled(false);
    sk::logger::LogContextGuard moved("model_move_inactive");
    sk::logger::FileLogger::Instance().SetEnabled(true);
    sk::logger::LogContextGuard source(targetLabel);
    moved = std::move(source);
    EXPECT_FALSE(source.IsActive());
    EXPECT_TRUE(moved.IsActive());
  }
  EXPECT_EQ(sk::logger::FileLogger::GetCurrentModelLabel(), originalLabel);
  EXPECT_EQ(sk::logger::FileHandleManager::Instance().GetCurrentHandle(), "default");
  sk::logger::FileLogger::Instance().SetEnabled(false);
}

TEST_F(TestSkModelContext, LogContextUsesExplicitModelLabel) {
  constexpr uint32_t modelId = 0xA016;
  aclmdlRI model = MakeFakeModel(modelId);
  sk::logger::FileLogger::Instance().SetEnabled(true);

  {
    SkModelContext guard(model);
    sk::logger::LoggerConfig config;
    config.enabled = true;
    config.modelLabel = GetCurrentModelLabel();
    ASSERT_TRUE(sk::logger::FileLogger::Instance().Initialize(config));

    const std::string modelLabel = GetCurrentModelLabel();
    SK_LOG_CONTEXT("explicit_label.log", modelLabel);
    SK_LOGI("log with explicit model label");

    std::string path = GetSkMetaBasePath() + "/" + modelLabel + "/explicit_label.log";
    std::ifstream file(path);
    EXPECT_TRUE(file.good());
  }

  sk::logger::FileLogger::Instance().SetEnabled(false);
}
