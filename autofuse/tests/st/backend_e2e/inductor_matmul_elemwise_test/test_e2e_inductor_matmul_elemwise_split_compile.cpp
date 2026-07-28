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

#ifndef HOST_CODE_FILE
#define HOST_CODE_FILE ""
#endif
#ifndef DEVICE_CODE_FILE
#define DEVICE_CODE_FILE ""
#endif
#ifndef OUTPUT_DIR
#define OUTPUT_DIR ""
#endif
#ifndef HOST_HELPER_BIN
#define HOST_HELPER_BIN ""
#endif
#ifndef HOST_DYNAMIC_SHAPE_ARGS
#define HOST_DYNAMIC_SHAPE_ARGS ""
#endif
#ifndef HOST_INPUT_CONFIGS_JSON
#define HOST_INPUT_CONFIGS_JSON "[]"
#endif
#ifndef HOST_TOPN
#define HOST_TOPN 4
#endif
#ifndef HOST_PERF_ORDER
#define HOST_PERF_ORDER "ascending-skip-first"
#endif
#ifndef HOST_VERIFY_EMPTY_CONFIG
#define HOST_VERIFY_EMPTY_CONFIG 0
#endif

namespace {

std::string ReadFile(const std::string &path) {
  std::ifstream in(path);
  std::stringstream buf;
  buf << in.rdbuf();
  return buf.str();
}

bool WriteFile(const std::string &path, const std::string &content) {
  std::ofstream out(path);
  if (!out.is_open()) return false;
  out << content;
  return true;
}

int RunCommand(const std::string &cmd) {
  int status = std::system(cmd.c_str());
  if (WIFEXITED(status)) return WEXITSTATUS(status);
  return -1;
}

bool FileExists(const std::string &path) {
  std::ifstream f(path);
  return f.good();
}

bool HasCxx11AbiSymbols(const std::string &path) {
  return RunCommand("nm -D " + path + " 2>/dev/null | c++filt | grep -q 'std::__cxx11'") == 0;
}

#ifndef PYAUTOFUSE_DIR
#define PYAUTOFUSE_DIR ""
#endif
#ifndef AUTOFUSE_PYTHON_DIR
#define AUTOFUSE_PYTHON_DIR ""
#endif
#ifndef ASCEND_HOME_PATH
#define ASCEND_HOME_PATH ""
#endif

std::string PythonPreamble() {
  return "import sys, os, traceback\n"
         "pkg_dir = os.path.join('" +
         std::string(OUTPUT_DIR) +
         "', 'autofuse_pkg')\n"
         "os.makedirs(pkg_dir, exist_ok=True)\n"
         "autofuse_dir = os.path.join(pkg_dir, 'autofuse')\n"
         "if os.path.islink(autofuse_dir) or os.path.isfile(autofuse_dir):\n"
         "    os.unlink(autofuse_dir)\n"
         "os.makedirs(autofuse_dir, exist_ok=True)\n"
         "for name in os.listdir('" +
         std::string(AUTOFUSE_PYTHON_DIR) +
         "'):\n"
         "    src = os.path.join('" +
         std::string(AUTOFUSE_PYTHON_DIR) +
         "', name)\n"
         "    dst = os.path.join(autofuse_dir, name)\n"
         "    if not os.path.lexists(dst):\n"
         "        os.symlink(src, dst)\n"
         "pyautofuse_src = os.path.join('" +
         std::string(PYAUTOFUSE_DIR) +
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
         std::string(ASCEND_HOME_PATH) + "'\n";
}

int RunHostCompile(const std::string &tiling_def, const std::string &host_code, const std::string &output_file) {
  WriteFile(OUTPUT_DIR "/host_tiling_def.h", tiling_def);
  WriteFile(OUTPUT_DIR "/host_impl.cpp", host_code);

  std::string script_path = std::string(OUTPUT_DIR) + "/run_host_compile.py";
  WriteFile(script_path, PythonPreamble() +
                             "try:\n"
                             "    from autofuse.compile_adapter import host_compile\n"
                             "    import os\n"
                             "    os.makedirs('" +
                             std::string(OUTPUT_DIR) +
                             "/host_out', exist_ok=True)\n"
                             "    td = open('" +
                             std::string(OUTPUT_DIR) +
                             "/host_tiling_def.h').read()\n"
                             "    hc = open('" +
                             std::string(OUTPUT_DIR) +
                             "/host_impl.cpp').read()\n"
                             "    host_compile(td, hc, [\n"
                             "        '--graph_name=inductor_matmul_elemwise',\n"
                             "        '--output_file=" +
                             output_file +
                             "',\n"
                             "        '--output_path=" +
                             std::string(OUTPUT_DIR) +
                             "/host_out',\n"
                             "        '--soc_version=Ascend910B',\n"
                             "        '--compile_options=-Werror'])\n"
                             "except Exception:\n"
                             "    traceback.print_exc()\n"
                             "    sys.exit(1)\n");

  std::string cmd = "ASCEND_HOME_PATH=" + std::string(ASCEND_HOME_PATH) + " python3 " + script_path + " 2>&1";
  int ret = RunCommand(cmd);
  if (ret != 0) printf("host_compile failed, ret=%d\n", ret);
  return ret;
}

int RunHostHelper(const std::string &host_bin, const std::string &tiling_repr_file) {
  const std::string input_configs_file = OUTPUT_DIR "/host_input_configs.json";
  WriteFile(input_configs_file, HOST_INPUT_CONFIGS_JSON);
  std::string cmd = std::string(HOST_HELPER_BIN) + " --host-so " + host_bin + " --tiling-repr-out " + tiling_repr_file +
                    " --input-configs " + input_configs_file + " --topn " + std::to_string(HOST_TOPN) +
                    " --perf-order " + std::string(HOST_PERF_ORDER);
  if (!std::string(HOST_DYNAMIC_SHAPE_ARGS).empty()) {
    cmd += " --dynamic-shape-args " + std::string(HOST_DYNAMIC_SHAPE_ARGS);
  }
  if (HOST_VERIFY_EMPTY_CONFIG != 0) {
    cmd += " --verify-empty-config";
  }
  cmd += " 2>&1";
  int ret = RunCommand(cmd);
  if (ret != 0) printf("host helper failed, ret=%d\n", ret);
  return ret;
}

int RunKernelCompile(const std::string &tiling_def, const std::string &device_code, const std::string &output_file,
                     const std::string &work_dir, const std::string &tiling_repr) {
  std::string mkdir_cmd = "mkdir -p " + work_dir;
  RunCommand(mkdir_cmd);
  WriteFile(work_dir + "/device_tiling_def.h", tiling_def);
  WriteFile(work_dir + "/device_impl.cpp", device_code);

  std::string repr_arg;
  if (!tiling_repr.empty()) {
    WriteFile(work_dir + "/tiling_repr.txt", tiling_repr);
    repr_arg = ", tiling_repr=open('" + work_dir + "/tiling_repr.txt').read()";
  }

  std::string script_path = work_dir + "/run_kernel_compile.py";
  WriteFile(script_path, PythonPreamble() +
                             "try:\n"
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
                             "    argv = ['--graph_name=inductor_matmul_elemwise',\n"
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
                             "    sys.exit(1)\n");

  std::string cmd = "ASCEND_HOME_PATH=" + std::string(ASCEND_HOME_PATH) + " python3 " + script_path + " 2>&1";
  int ret = RunCommand(cmd);
  if (ret != 0) printf("kernel_compile failed, ret=%d, work_dir=%s\n", ret, work_dir.c_str());
  return ret;
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
  EXPECT_EQ(host_code.find("GenerateTopnSolutions"), std::string::npos);
  EXPECT_EQ(host_code.find("AscirCompileAndLaunch"), std::string::npos);

  const std::string host_bin = OUTPUT_DIR "/inductor_matmul_elemwise_host.so";
  ASSERT_EQ(RunHostCompile(tiling_def, host_code, host_bin), 0);
  ASSERT_TRUE(FileExists(host_bin)) << "host so not found: " << host_bin;
  ASSERT_TRUE(HasCxx11AbiSymbols(host_bin)) << "host so should use ABI=1: " << host_bin;
  const std::string tiling_repr_file = OUTPUT_DIR "/tiling_repr.txt";
  ASSERT_EQ(RunHostHelper(host_bin, tiling_repr_file), 0);
  std::string tiling_repr = ReadFile(tiling_repr_file);
  ASSERT_FALSE(tiling_repr.empty());

  CompileAndVerifyKernels(tiling_def, device_code, tiling_repr);
}
