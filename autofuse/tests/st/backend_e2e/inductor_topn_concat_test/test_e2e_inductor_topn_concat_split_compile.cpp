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
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <set>

#include "../common/inductor_split_compile_common.h"
#include "../common/inductor_split_compile_config.h"
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <vector>

namespace {

using autofuse::tests::FileExists;
using autofuse::tests::ReadFile;

std::set<std::string> CollectSystemHeaders(const std::string &source) {
  std::set<std::string> headers;
  std::stringstream stream(source);
  std::string line;
  while (std::getline(stream, line)) {
    if (line.rfind("#include <", 0) == 0) {
      const size_t begin = sizeof("#include <") - 1;
      const size_t end = line.find('>', begin);
      if (end != std::string::npos) {
        headers.insert(line.substr(begin, end - begin));
      }
      continue;
    }
    if (line.rfind("#include \"", 0) == 0) {
      continue;
    }
    if (!headers.empty()) {
      break;
    }
  }
  return headers;
}

void ExpectSystemHeaders(const std::string &source, const std::vector<std::string> &expected) {
  EXPECT_EQ(CollectSystemHeaders(source), std::set<std::string>(expected.begin(), expected.end()));
}

void VerifyTilingFuncSystemHeaders(const std::string &host_dir) {
  const std::string normal_group =
      ReadFile(host_dir + "/inductor_topn_concat_tiling_func_asc_graph0_schedule_result0_g0.cpp");
  ExpectSystemHeaders(normal_group, {"algorithm", "array", "cfloat", "cmath", "cstddef", "cstdint", "cstdlib", "map",
                                     "memory", "new", "string", "unordered_map", "vector"});
  EXPECT_NE(normal_group.find("#include \"autofuse_tiling_func_api.h\""), std::string::npos);

  const std::string reuse_group =
      ReadFile(host_dir + "/inductor_topn_concat_tiling_func_asc_graph0_schedule_result1_g1.cpp");
  ExpectSystemHeaders(reuse_group, {"cstdint", "unordered_map"});

  const std::string tail = ReadFile(host_dir + "/inductor_topn_concat_tiling_func_schedule_group_tail.cpp");
  ExpectSystemHeaders(
      tail, {"algorithm", "array", "cfloat", "cstddef", "cstdint", "functional", "unordered_map", "utility", "vector"});

  const std::string solver = ReadFile(host_dir + "/inductor_topn_concat_tiling_func_solver_func.cpp");
  ExpectSystemHeaders(solver, {"algorithm", "cmath", "cstddef", "cstdint", "functional", "utility", "vector"});

  const std::string entry = ReadFile(host_dir + "/inductor_topn_concat_tiling_func_tiling_def_and_tiling_const.cpp");
  ExpectSystemHeaders(entry, {"algorithm", "cfloat", "cmath", "cstddef", "cstdint", "map", "ostream", "sstream",
                              "string", "unordered_map", "utility", "vector"});
}

constexpr const char *kGraphName = "inductor_topn_concat";

int RunKernelCompile(const std::string &tiling_def, const std::string &device_code, const std::string &output_file,
                     const std::string &work_dir, const std::string &tiling_repr) {
  return autofuse::tests::RunKernelCompile(tiling_def, device_code, output_file, work_dir, {kGraphName, tiling_repr});
}

}  // namespace
class TestBackendInductorTopnConcatSplitCompile : public testing::Test {};

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
  const std::string kernel_static = OUTPUT_DIR "/inductor_topn_concat_static.so";
  const std::string dynamic_dir = OUTPUT_DIR "/device_dynamic";
  const std::string kernel_dynamic = OUTPUT_DIR "/inductor_topn_concat_dynamic.so";
  autofuse::tests::ParallelCompileAndVerifySo(tiling_def, device_code, tiling_repr, static_dir, kernel_static,
                                              dynamic_dir, kernel_dynamic, RunKernelCompile);

  const std::string static_src = ReadFile(static_dir + "/device/inductor_topn_concat_op_kernel.cpp");
  EXPECT_NE(static_src.find("constexpr AutofuseTilingData t = AutofuseTilingData{"), std::string::npos)
      << "static kernel should have constexpr tiling";
  EXPECT_EQ(static_src.find("const AutofuseTilingData t;"), std::string::npos)
      << "static kernel should not have non-const tiling";

  const std::string dynamic_src = ReadFile(dynamic_dir + "/device/inductor_topn_concat_op_kernel.cpp");
  EXPECT_NE(dynamic_src.find("AutofuseTilingData t)"), std::string::npos)
      << "dynamic kernel should have tiling parameter";
  EXPECT_EQ(dynamic_src.find("constexpr AutofuseTilingData t = AutofuseTilingData{"), std::string::npos)
      << "dynamic kernel should not have constexpr tiling";
}

TEST_F(TestBackendInductorTopnConcatSplitCompile, SplitCompileChainWorks) {
  std::string tiling_def, host_code, device_code;
  PrepareInputs(tiling_def, host_code, device_code);

  const std::string host_bin = OUTPUT_DIR "/inductor_topn_concat_host.so";
  ASSERT_EQ(autofuse::tests::RunHostCompile(tiling_def, host_code, host_bin, "inductor_topn_concat", "-Werror"), 0);
  ASSERT_TRUE(FileExists(host_bin)) << "host so not found: " << host_bin;
  ASSERT_TRUE(autofuse::tests::HasCxx11AbiSymbols(host_bin)) << "host so should use ABI=1: " << host_bin;
  VerifyTilingFuncSystemHeaders(OUTPUT_DIR "/host_out/host");
  const std::string tiling_repr_file = OUTPUT_DIR "/tiling_repr.txt";
  ASSERT_EQ(autofuse::tests::RunHostHelper(host_bin, tiling_repr_file), 0);
  std::string tiling_repr = ReadFile(tiling_repr_file);
  ASSERT_FALSE(tiling_repr.empty());

  CompileAndVerifyKernels(tiling_def, device_code, tiling_repr);
}
