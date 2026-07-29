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

#include <cstdio>
#include <cstdint>
#include <fstream>
#include <functional>
#include <future>
#include <cstdlib>
#include <sstream>
#include <string>
#include <sys/wait.h>

#include "inductor_split_compile_config.h"

namespace autofuse::tests {

inline std::string ReadFile(const std::string &path) {
  std::ifstream in(path);
  std::stringstream buf;
  buf << in.rdbuf();
  return buf.str();
}

inline bool WriteFile(const std::string &path, const std::string &content) {
  std::ofstream out(path);
  if (!out.is_open()) return false;
  out << content;
  return true;
}

inline int RunCommand(const std::string &cmd) {
  int status = std::system(cmd.c_str());
  if (WIFEXITED(status)) return WEXITSTATUS(status);
  return -1;
}

inline bool FileExists(const std::string &path) {
  std::ifstream f(path);
  return f.good();
}

inline bool HasCxx11AbiSymbols(const std::string &path) {
  return RunCommand("nm -D " + path + " 2>/dev/null | c++filt | grep -q 'std::__cxx11'") == 0;
}

inline std::string PythonPreamble(const std::string &pyautofuse_dir, const std::string &autofuse_python_dir,
                                  const std::string &ascend_home_path) {
  return "import sys, os, traceback\n"
         "pkg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autofuse_pkg')\n"
         "os.makedirs(pkg_dir, exist_ok=True)\n"
         "autofuse_dir = os.path.join(pkg_dir, 'autofuse')\n"
         "if os.path.islink(autofuse_dir) or os.path.isfile(autofuse_dir):\n"
         "    os.unlink(autofuse_dir)\n"
         "os.makedirs(autofuse_dir, exist_ok=True)\n"
         "for name in os.listdir('" +
         autofuse_python_dir +
         "'):\n"
         "    src = os.path.join('" +
         autofuse_python_dir +
         "', name)\n"
         "    dst = os.path.join(autofuse_dir, name)\n"
         "    if not os.path.lexists(dst):\n"
         "        os.symlink(src, dst)\n"
         "pyautofuse_src = os.path.join('" +
         pyautofuse_dir +
         "', 'pyautofuse.so')\n"
         "if not os.path.exists(pyautofuse_src):\n"
         "    raise FileNotFoundError(pyautofuse_src)\n"
         "pyautofuse_dst = os.path.join(autofuse_dir, 'pyautofuse.so')\n"
         "if os.path.lexists(pyautofuse_dst):\n"
         "    os.unlink(pyautofuse_dst)\n"
         "os.symlink(pyautofuse_src, pyautofuse_dst)\n"
         "sys.path.insert(0, pkg_dir)\n"
         "import autofuse.ascendc_compile as _ac\n"
         "_ac.ASCEND_PATH = '" +
         ascend_home_path + "'\n";
}

inline std::string PythonPreamble() {
  return PythonPreamble(PYAUTOFUSE_DIR, AUTOFUSE_PYTHON_DIR, ASCEND_HOME_PATH);
}

inline std::string HostCompileScript(const std::string &graph_name, const std::string &output_file,
                                     const std::string &output_dir, const std::string &compile_options,
                                     const std::string &extra_body = "") {
  const std::string compile_options_arg = compile_options.empty() ? "        '--soc_version=Ascend910B'])\n"
                                                                  : "        '--soc_version=Ascend910B',\n"
                                                                    "        '--compile_options=" +
                                                                        compile_options + "'])\n";
  return "try:\n"
         "    from autofuse.compile_adapter import host_compile\n"
         "    import os\n"
         "    os.makedirs('" +
         output_dir +
         "/host_out', exist_ok=True)\n"
         "    td = open('" +
         output_dir +
         "/host_tiling_def.h').read()\n"
         "    hc = open('" +
         output_dir +
         "/host_impl.cpp').read()\n"
         "    host_compile(td, hc, [\n"
         "        '--graph_name=" +
         graph_name +
         "',\n"
         "        '--output_file=" +
         output_file +
         "',\n"
         "        '--output_path=" +
         output_dir + "/host_out',\n" + compile_options_arg + extra_body +
         "except Exception:\n"
         "    traceback.print_exc()\n"
         "    sys.exit(1)\n";
}

inline std::string KernelCompileScript(const std::string &graph_name, const std::string &output_file,
                                       const std::string &work_dir, const std::string &repr_arg) {
  return "try:\n"
         "    from autofuse.compile_adapter import kernel_compile\n"
         "    import os\n"
         "    os.makedirs('" +
         work_dir +
         "', exist_ok=True)\n"
         "    td = open('" +
         work_dir +
         "/device_tiling_def.h').read()\n"
         "    dc = open('" +
         work_dir +
         "/device_impl.cpp').read()\n"
         "    argv = ['--graph_name=" +
         graph_name +
         "',\n"
         "            '--output_file=" +
         output_file +
         "',\n"
         "            '--output_path=" +
         work_dir +
         "',\n"
         "            '--soc_version=Ascend910B',\n"
         "            '--compile_options=-D_GLIBCXX_USE_CXX11_ABI=0']\n"
         "    kernel_compile(td, dc, argv" +
         repr_arg +
         ")\n"
         "except Exception:\n"
         "    traceback.print_exc()\n"
         "    sys.exit(1)\n";
}

inline int RunPythonScript(const std::string &script_path, const std::string &script,
                           const std::string &ascend_home_path, const std::string &stage) {
  WriteFile(script_path, script);
  const std::string cmd = "ASCEND_HOME_PATH=" + ascend_home_path + " python3 " + script_path + " 2>&1";
  const int ret = RunCommand(cmd);
  if (ret != 0) {
    std::printf("%s failed, ret=%d\n", stage.c_str(), ret);
  }
  return ret;
}

inline int RunHostCompile(const std::string &tiling_def, const std::string &host_code, const std::string &output_file,
                          const std::string &graph_name, const std::string &compile_options) {
  WriteFile(std::string(OUTPUT_DIR) + "/host_tiling_def.h", tiling_def);
  WriteFile(std::string(OUTPUT_DIR) + "/host_impl.cpp", host_code);
  const std::string script_path = std::string(OUTPUT_DIR) + "/run_host_compile.py";
  return RunPythonScript(script_path,
                         PythonPreamble() + HostCompileScript(graph_name, output_file, OUTPUT_DIR, compile_options),
                         ASCEND_HOME_PATH, "host_compile");
}

struct HostHelperOptions {
  std::string input_configs_json = HOST_INPUT_CONFIGS_JSON;
  int64_t topn = HOST_TOPN;
  std::string perf_order = HOST_PERF_ORDER;
  bool check_z0t_positive = false;
};

inline int RunHostHelper(const std::string &host_bin, const std::string &tiling_repr_file,
                         const HostHelperOptions &options = {}) {
  const std::string input_configs_file = std::string(OUTPUT_DIR) + "/host_input_configs.json";
  WriteFile(input_configs_file, options.input_configs_json);
  std::string cmd = std::string(HOST_HELPER_BIN) + " --host-so " + host_bin + " --tiling-repr-out " + tiling_repr_file +
                    " --input-configs " + input_configs_file + " --topn " + std::to_string(options.topn) +
                    " --perf-order " + options.perf_order;
  if (!std::string(HOST_DYNAMIC_SHAPE_ARGS).empty()) {
    cmd += " --dynamic-shape-args " + std::string(HOST_DYNAMIC_SHAPE_ARGS);
  }
  if (HOST_VERIFY_EMPTY_CONFIG != 0) {
    cmd += " --verify-empty-config";
  }
  if (options.check_z0t_positive) {
    cmd += " --check-z0t-positive";
  }
  const int ret = RunCommand(cmd + " 2>&1");
  if (ret != 0) {
    std::printf("host helper failed, ret=%d\n", ret);
  }
  return ret;
}

struct KernelCompileOptions {
  std::string graph_name;
  std::string tiling_repr;
};

inline int RunKernelCompile(const std::string &tiling_def, const std::string &device_code,
                            const std::string &output_file, const std::string &work_dir,
                            const KernelCompileOptions &options) {
  RunCommand("mkdir -p " + work_dir);
  WriteFile(work_dir + "/device_tiling_def.h", tiling_def);
  WriteFile(work_dir + "/device_impl.cpp", device_code);
  std::string repr_arg;
  if (!options.tiling_repr.empty()) {
    WriteFile(work_dir + "/tiling_repr.txt", options.tiling_repr);
    repr_arg = ", tiling_repr=open('" + work_dir + "/tiling_repr.txt').read()";
  }
  const std::string script_path = work_dir + "/run_kernel_compile.py";
  const std::string script =
      PythonPreamble() + KernelCompileScript(options.graph_name, output_file, work_dir, repr_arg);
  return RunPythonScript(script_path, script, ASCEND_HOME_PATH, "kernel_compile(" + work_dir + ")");
}

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
