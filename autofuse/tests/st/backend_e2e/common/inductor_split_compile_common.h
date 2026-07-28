/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_TESTS_ST_BACKEND_E2E_COMMON_INDUCTOR_SPLIT_COMPILE_COMMON_H_
#define AUTOFUSE_TESTS_ST_BACKEND_E2E_COMMON_INDUCTOR_SPLIT_COMPILE_COMMON_H_

#include <dlfcn.h>
#include <gtest/gtest.h>

#include <fstream>
#include <functional>
#include <future>
#include <string>

namespace autofuse::tests {

struct SplitCompileDlHandle {
  void *ptr = nullptr;
  explicit SplitCompileDlHandle(void *p) : ptr(p) {}
  ~SplitCompileDlHandle() {
    if (ptr) dlclose(ptr);
  }
  operator bool() const {
    return ptr != nullptr;
  }
};

namespace detail {
inline bool SoFileExists(const std::string &path) {
  std::ifstream f(path);
  return f.good();
}
}  // namespace detail

template <typename CompileFn>
void ParallelCompileAndVerifySo(const std::string &tiling_def, const std::string &device_code,
                                const std::string &tiling_repr, const std::string &static_dir,
                                const std::string &kernel_static, const std::string &dynamic_dir,
                                const std::string &kernel_dynamic, CompileFn compile_kernel) {
  auto dynamic_compile = std::async(std::launch::async, [&] {
    return compile_kernel(tiling_def, device_code, kernel_dynamic, dynamic_dir, std::string(""));
  });
  int static_ret = compile_kernel(tiling_def, device_code, kernel_static, static_dir, tiling_repr);
  int dynamic_ret = dynamic_compile.get();
  ASSERT_EQ(static_ret, 0);
  ASSERT_EQ(dynamic_ret, 0);
  ASSERT_TRUE(detail::SoFileExists(kernel_static)) << "static device so not found: " << kernel_static;
  ASSERT_TRUE(detail::SoFileExists(kernel_dynamic)) << "dynamic device so not found: " << kernel_dynamic;

  SplitCompileDlHandle static_handle(dlopen(kernel_static.c_str(), RTLD_NOW | RTLD_LOCAL));
  ASSERT_TRUE(static_handle) << "dlopen static device failed: " << dlerror();
  EXPECT_NE(dlsym(static_handle.ptr, "AutofuseLaunch"), nullptr) << "AutofuseLaunch missing in static so";

  SplitCompileDlHandle dynamic_handle(dlopen(kernel_dynamic.c_str(), RTLD_NOW | RTLD_LOCAL));
  ASSERT_TRUE(dynamic_handle) << "dlopen dynamic device failed: " << dlerror();
  EXPECT_NE(dlsym(dynamic_handle.ptr, "AutofuseLaunch"), nullptr) << "AutofuseLaunch missing in dynamic so";
}

}  // namespace autofuse::tests

#endif  // AUTOFUSE_TESTS_ST_BACKEND_E2E_COMMON_INDUCTOR_SPLIT_COMPILE_COMMON_H_
