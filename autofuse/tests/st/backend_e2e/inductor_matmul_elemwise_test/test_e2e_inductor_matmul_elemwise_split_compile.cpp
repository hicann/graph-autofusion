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
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <sys/wait.h>

#include "../common/inductor_split_compile_common.h"
#include "../common/inductor_split_compile_config.h"

namespace {

using autofuse::tests::FileExists;
using autofuse::tests::ReadFile;

constexpr const char *kGraphName = "inductor_matmul_elemwise";

int RunKernelCompile(const std::string &tiling_def, const std::string &device_code, const std::string &output_file,
                     const std::string &work_dir, const std::string &tiling_repr) {
  return autofuse::tests::RunKernelCompile(tiling_def, device_code, output_file, work_dir, {kGraphName, tiling_repr});
}

struct DlHandle {
  void *ptr = nullptr;
  explicit DlHandle(void *p) : ptr(p) {}
  ~DlHandle() {
    if (ptr) dlclose(ptr);
  }
  operator bool() const {
    return ptr != nullptr;
  }
};

}  // namespace
class TestBackendInductorMatmulElemwiseSplitCompile : public testing::Test {};

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
  const std::string kernel_static = OUTPUT_DIR "/inductor_matmul_elemwise_static.so";
  ASSERT_EQ(RunKernelCompile(tiling_def, device_code, kernel_static, static_dir, tiling_repr), 0);
  ASSERT_TRUE(FileExists(kernel_static)) << "static device so not found: " << kernel_static;

  const std::string dynamic_dir = OUTPUT_DIR "/device_dynamic";
  const std::string kernel_dynamic = OUTPUT_DIR "/inductor_matmul_elemwise_dynamic.so";
  ASSERT_EQ(RunKernelCompile(tiling_def, device_code, kernel_dynamic, dynamic_dir, ""), 0);
  ASSERT_TRUE(FileExists(kernel_dynamic)) << "dynamic device so not found: " << kernel_dynamic;

  DlHandle static_handle(dlopen(kernel_static.c_str(), RTLD_NOW | RTLD_LOCAL));
  ASSERT_TRUE(static_handle) << "dlopen static device failed: " << dlerror();
  EXPECT_NE(dlsym(static_handle.ptr, "AutofuseLaunch"), nullptr) << "AutofuseLaunch missing in static so";

  DlHandle dynamic_handle(dlopen(kernel_dynamic.c_str(), RTLD_NOW | RTLD_LOCAL));
  ASSERT_TRUE(dynamic_handle) << "dlopen dynamic device failed: " << dlerror();
  EXPECT_NE(dlsym(dynamic_handle.ptr, "AutofuseLaunch"), nullptr) << "AutofuseLaunch missing in dynamic so";

  const std::string static_src = ReadFile(static_dir + "/device/inductor_matmul_elemwise_op_kernel.cpp");
  EXPECT_NE(static_src.find("constexpr CVAutofuseTilingData kConstTilingData"), std::string::npos)
      << "static CV kernel should have constexpr CV tiling";
  EXPECT_NE(static_src.find("kConstMatmulTilingBytes"), std::string::npos)
      << "static CV kernel should embed matmul tiling bytes";
  EXPECT_EQ(static_src.find("CVAutofuseTilingData t)"), std::string::npos)
      << "static CV kernel should not pass dynamic CV tiling";

  const std::string dynamic_src = ReadFile(dynamic_dir + "/device/inductor_matmul_elemwise_op_kernel.cpp");
  EXPECT_NE(dynamic_src.find("CVAutofuseTilingData t)"), std::string::npos)
      << "dynamic CV kernel should have CV tiling parameter";
  EXPECT_EQ(dynamic_src.find("constexpr CVAutofuseTilingData kConstTilingData"), std::string::npos)
      << "dynamic CV kernel should not have constexpr CV tiling";
  EXPECT_EQ(dynamic_src.find("AscirCompileAndLaunch"), std::string::npos)
      << "CV inductor kernel should not restore old AscirCompileAndLaunch entry";
}

TEST_F(TestBackendInductorMatmulElemwiseSplitCompile, SplitCompileChainWorks) {
  std::string tiling_def, host_code, device_code;
  PrepareInputs(tiling_def, host_code, device_code);

  EXPECT_NE(tiling_def.find("CVAutofuseTilingData"), std::string::npos);
  EXPECT_NE(host_code.find("CallCubeTiling"), std::string::npos);
  EXPECT_EQ(host_code.find("// AUTOFUSE_SPLIT_FILE_BEGIN: TilingDataLog"), std::string::npos);
  EXPECT_EQ(host_code.find("GenerateTopnSolutions"), std::string::npos);
  EXPECT_EQ(host_code.find("AscirCompileAndLaunch"), std::string::npos);

  const std::string host_bin = OUTPUT_DIR "/inductor_matmul_elemwise_host.so";
  ASSERT_EQ(autofuse::tests::RunHostCompile(tiling_def, host_code, host_bin, "inductor_matmul_elemwise", "-Werror"), 0);
  ASSERT_TRUE(FileExists(OUTPUT_DIR "/host_out/host/autofuse_tiling_func_log.h"));
  ASSERT_TRUE(FileExists(host_bin)) << "host so not found: " << host_bin;
  ASSERT_TRUE(autofuse::tests::HasCxx11AbiSymbols(host_bin)) << "host so should use ABI=1: " << host_bin;
  const std::string tiling_repr_file = OUTPUT_DIR "/tiling_repr.txt";
  ASSERT_EQ(autofuse::tests::RunHostHelper(host_bin, tiling_repr_file), 0);
  std::string tiling_repr = ReadFile(tiling_repr_file);
  ASSERT_FALSE(tiling_repr.empty());

  CompileAndVerifyKernels(tiling_def, device_code, tiling_repr);
}
