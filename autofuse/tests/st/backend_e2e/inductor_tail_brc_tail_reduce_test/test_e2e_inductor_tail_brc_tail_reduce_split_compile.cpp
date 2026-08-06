/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <future>
#include <gtest/gtest.h>

#include "../common/inductor_split_compile_common.h"
#include <sstream>
#include <string>
#include <sys/wait.h>

#include "../common/inductor_split_compile_config.h"

#ifndef PGO_TILING_DEF_FILE
#define PGO_TILING_DEF_FILE ""
#endif
#ifndef PGO_HOST_CODE_FILE
#define PGO_HOST_CODE_FILE ""
#endif
#ifndef PGO_DEVICE_CODE_FILE
#define PGO_DEVICE_CODE_FILE ""
#endif
namespace {

using autofuse::tests::FileExists;
using autofuse::tests::ReadFile;
std::string FindGenerationDir(const std::string &tiling_so) {
  const std::filesystem::path output(tiling_so);
  const std::string prefix = output.filename().string() + ".pgo.";
  std::string result;
  for (const auto &entry : std::filesystem::directory_iterator(output.parent_path())) {
    const auto name = entry.path().filename().string();
    if (entry.is_directory() && name.rfind(prefix, 0U) == 0U) {
      if (!result.empty()) return {};
      result = entry.path().string();
    }
  }
  return result;
}

constexpr const char *kGraphName = "inductor_tail_brc_tail_reduce";

int RunKernelCompile(const std::string &tiling_def, const std::string &device_code, const std::string &output_file,
                     const std::string &work_dir, const std::string &tiling_repr) {
  return autofuse::tests::RunKernelCompile(tiling_def, device_code, output_file, work_dir, {kGraphName, tiling_repr});
}

}  // namespace
class TestBackendInductorTailBrcTailReduceSplitCompile : public testing::Test {};

void PrepareInputs(std::string &tiling_def, std::string &host_code, std::string &device_code) {
  tiling_def = ReadFile(TILING_DEF_FILE);
  host_code = ReadFile(HOST_CODE_FILE);
  device_code = ReadFile(DEVICE_CODE_FILE);
  ASSERT_FALSE(tiling_def.empty()) << "tiling_def empty";
  ASSERT_FALSE(host_code.empty()) << "host_code empty";
  ASSERT_FALSE(device_code.empty()) << "device_code empty";
}

void CompileAndVerifyKernels(const std::string &tiling_def, const std::string &device_code,
                             const std::string &tiling_repr) {
  const std::string static_dir = OUTPUT_DIR "/device_static";
  const std::string kernel_static = OUTPUT_DIR "/inductor_tail_brc_tail_reduce_static.so";
  const std::string dynamic_dir = OUTPUT_DIR "/device_dynamic";
  const std::string kernel_dynamic = OUTPUT_DIR "/inductor_tail_brc_tail_reduce_dynamic.so";
  autofuse::tests::ParallelCompileAndVerifySo(tiling_def, device_code, tiling_repr, static_dir, kernel_static,
                                              dynamic_dir, kernel_dynamic, RunKernelCompile);

  const std::string static_src = ReadFile(static_dir + "/device/inductor_tail_brc_tail_reduce_op_kernel.cpp");
  EXPECT_NE(static_src.find("constexpr AutofuseTilingData t = AutofuseTilingData{"), std::string::npos)
      << "static kernel should have constexpr tiling";
  EXPECT_EQ(static_src.find("const AutofuseTilingData t;"), std::string::npos)
      << "static kernel should not have non-const tiling";

  const std::string dynamic_src = ReadFile(dynamic_dir + "/device/inductor_tail_brc_tail_reduce_op_kernel.cpp");
  EXPECT_NE(dynamic_src.find("AutofuseTilingData t)"), std::string::npos)
      << "dynamic kernel should have tiling parameter";
  EXPECT_EQ(dynamic_src.find("constexpr AutofuseTilingData t = AutofuseTilingData{"), std::string::npos)
      << "dynamic kernel should not have constexpr tiling";
}

TEST_F(TestBackendInductorTailBrcTailReduceSplitCompile, SplitCompileChainWorks) {
  std::string tiling_def, host_code, device_code;
  PrepareInputs(tiling_def, host_code, device_code);

  const std::string host_bin = OUTPUT_DIR "/inductor_tail_brc_tail_reduce_host.so";
  ASSERT_EQ(
      autofuse::tests::RunHostCompile(tiling_def, host_code, host_bin, "inductor_tail_brc_tail_reduce", "-Werror"), 0);
  ASSERT_TRUE(FileExists(host_bin)) << "host so not found: " << host_bin;
  ASSERT_TRUE(autofuse::tests::HasCxx11AbiSymbols(host_bin)) << "host so should use ABI=1: " << host_bin;
  const std::string tiling_repr_file = OUTPUT_DIR "/tiling_repr.txt";
  ASSERT_EQ(autofuse::tests::RunHostHelper(host_bin, tiling_repr_file), 0);
  std::string tiling_repr = ReadFile(tiling_repr_file);
  ASSERT_FALSE(tiling_repr.empty());

  CompileAndVerifyKernels(tiling_def, device_code, tiling_repr);
}

TEST_F(TestBackendInductorTailBrcTailReduceSplitCompile, PgoMultiGroupReduceRCoreRunnerCompiles) {
  const std::string tiling_def = ReadFile(PGO_TILING_DEF_FILE);
  const std::string host_code = ReadFile(PGO_HOST_CODE_FILE);
  const std::string device_code = ReadFile(PGO_DEVICE_CODE_FILE);
  ASSERT_FALSE(tiling_def.empty() || host_code.empty() || device_code.empty());
  ASSERT_NE(device_code.find("SyncAll();"), std::string::npos);

  const std::string pgo_dir = OUTPUT_DIR "/pgo_tail_reduce";
  std::filesystem::remove_all(pgo_dir);
  const std::string tiling_so = pgo_dir + "/tiling.so";
  ASSERT_EQ(autofuse::tests::RunHostCompile(tiling_def, host_code, tiling_so, kGraphName, "-Werror"), 0);
  const std::string generation_dir = FindGenerationDir(tiling_so);
  ASSERT_FALSE(generation_dir.empty());
  const std::string generation_tiling = generation_dir + "/tiling.so";
  const std::string runner = generation_dir + "/tiling.so.pgo_runner";
  const std::string kernel = generation_dir + "/tiling.so.pgo_kernel.aicore_binary_elf_v1";
  ASSERT_TRUE(FileExists(generation_tiling) && FileExists(runner) && FileExists(kernel));
  const std::string runner_source =
      ReadFile(OUTPUT_DIR "/host_out/host/inductor_tail_brc_tail_reduce_tiling_func_PgoRunner.cpp");
  ASSERT_FALSE(runner_source.empty());
  EXPECT_NE(runner_source.find("size_t output0_size = 128;"), std::string::npos);
  EXPECT_EQ(runner_source.find("Invalid or symbolic PGO output0 memory size"), std::string::npos);
}
