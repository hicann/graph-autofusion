/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen_tiling.h"

#include "backend/backend_spec.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"

namespace codegen {
using namespace ascgen_utils;

namespace {
bool IsNeedFfts() {
  const auto backend_spec = optimize::BackendSpec::GetInstance();
  GE_ASSERT_NOTNULL(backend_spec);
  return backend_spec->pgo_spec.need_ffts;
}

void AppendPgoLogDefs(std::stringstream &ss) {
  ss << "constexpr int32_t PGO_MODULE_NAME = static_cast<int32_t>(" << GE_MODULE_NAME << ");" << std::endl;
  ss << "#define PGO_LOG_PREFIX \"%\" PRIu64 \" %s:[PGO][\" PGO_GRAPH_NAME \"] \"" << std::endl;
  ss << "#define DLOGD(fmt, ...) do { dlog_debug(PGO_MODULE_NAME, PGO_LOG_PREFIX fmt, PgoGetTid(), &__FUNCTION__[0U], "
        "##__VA_ARGS__); } while (false)"
     << std::endl;
  ss << "#define DLOGI(fmt, ...) do { dlog_info(PGO_MODULE_NAME, PGO_LOG_PREFIX fmt, PgoGetTid(), &__FUNCTION__[0U], "
        "##__VA_ARGS__); } while (false)"
     << std::endl;
  ss << "#define DLOGW(fmt, ...) do { dlog_warn(PGO_MODULE_NAME, PGO_LOG_PREFIX fmt, PgoGetTid(), &__FUNCTION__[0U], "
        "##__VA_ARGS__); } while (false)"
     << std::endl;
  ss << "#define DLOGE(fmt, ...) do { dlog_error(PGO_MODULE_NAME, PGO_LOG_PREFIX fmt, PgoGetTid(), &__FUNCTION__[0U], "
        "##__VA_ARGS__); } while (false)"
     << std::endl;
}
}  // namespace

void TilingLib::GenPgoHeaders(std::stringstream &ss) const {
  ss << "#include <cinttypes>" << std::endl;
  ss << "#include <unistd.h>" << std::endl;
  ss << "#include <fcntl.h>" << std::endl;
  ss << "#include <sys/file.h>" << std::endl;
  ss << "#include <sys/syscall.h>" << std::endl;
  ss << "#include <sys/wait.h>" << std::endl;
  ss << "#include <dlfcn.h>" << std::endl << std::endl;

  ss << "#include <algorithm>" << std::endl;
  ss << "#include <chrono>" << std::endl;
  ss << "#include <cfloat>" << std::endl;
  ss << "#include <cstdint>" << std::endl;
  ss << "#include <cerrno>" << std::endl;
  ss << "#include <cstring>" << std::endl;
  ss << "#include <securec.h>" << std::endl;
  ss << "#include <fstream>" << std::endl;
  ss << "#include <map>" << std::endl;
  ss << "#include <string>" << std::endl;
  ss << "#include <thread>" << std::endl;
  ss << "#include <unordered_map>" << std::endl;
  ss << "#include <vector>" << std::endl << std::endl;

  ss << "#include \"acl/acl.h\"" << std::endl;
  ss << "#include \"dlog_pub.h\"" << std::endl;
  ss << "#include \"mspti.h\"" << std::endl;
  ss << "#include \"tiling/platform/platform_ascendc.h\"" << std::endl << std::endl;

  ss << "#include \"autofuse_tiling_data.h\"" << std::endl << std::endl;
  ss << PGOTensorArgsDef();
}

void TilingLib::GenDynamicLibraryLoaderCode(std::stringstream &ss) const {
  ss << "static void *handle = nullptr;" << std::endl;
  ss << "static bool initialized = false;" << std::endl;
  ss << R"(
__attribute__((constructor)) void Init() {
  if (initialized) return;
  handle = dlopen(kernel_file, RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    DLOGE("Failed to load %s: %s", kernel_file, dlerror());
    return;
  }
  DLOGD("Kernel api lib %s load succeed", kernel_file);
  initialized = true;
})" << std::endl;
  ss << R"(
__attribute__((destructor)) void DeInit() {
  if (handle) {
    dlclose(handle);
    handle = nullptr;
  }
  initialized = false;
})" << std::endl;
  ss << R"(
inline void *GetFunc(const char *func_name) {
  if (handle == nullptr) {
    return nullptr;
  }
  void *func = dlsym(handle, func_name);
  if (func == nullptr) {
    DLOGE("Failed to load wrapper api func: %s", dlerror());
  }
  return func;
})" << std::endl;
}

void TilingLib::GenPgoCardLock(std::stringstream &ss) const {
  ss << R"(
class CardLock {
public:
  CardLock(const char *path) {
    fd_ = open(path, O_RDWR | O_CREAT, 0666);
    if (fd_ == -1) {
      DLOGE("open lock file: %s", std::strerror(errno));
      std::exit(1);
    }
    if (flock(fd_, LOCK_EX) == -1) {
      DLOGE("flock LOCK_EX: %s", std::strerror(errno));
      std::exit(1);
    }
  }

  ~CardLock() {
    if (fd_ != -1) {
      if (flock(fd_, LOCK_UN) == -1) {
        DLOGW("flock LOCK_UN: %s", std::strerror(errno));
      }
      close(fd_);
    }
  }

  CardLock(const CardLock&) = delete;
  CardLock& operator=(const CardLock&) = delete;

private:
  int fd_{-1};
};
)" << std::endl;
}

void TilingLib::GenPgoSaveTilingKey(std::stringstream &ss) const {
  ss << R"(void PgoSaveTilingKey(const AutofuseTilingData &tiling_data, double best_perf, std::ofstream &out_file) {
  const size_t tiling_bytes = sizeof(tiling_data);
  const size_t tiling_bytes_align = (tiling_bytes + sizeof(int32_t) - 1) / sizeof(int32_t);
  std::vector<int32_t> tiling_i32(tiling_bytes_align, 0);
  memcpy_s(tiling_i32.data(), tiling_i32.size() * sizeof(int32_t), &tiling_data, tiling_bytes);
  for (size_t idx = 0; idx < tiling_i32.size(); ++idx) {
    out_file << tiling_i32[idx] << " ";
  }
  out_file << "# " << best_perf << std::endl;
})" << std::endl;
}

void TilingLib::GenPgoAppendSearchTilingData(std::stringstream &ss) const {
  GenPgoSaveTilingKey(ss);
  ss << "void AppendPgoSearchTilingData(const AutofuseTilingData &tiling_data, double best_perf, std::ios::openmode "
        "mode = std::ios::app) {"
     << std::endl;
  ss << "  DLOGD(\"AppendPgoSearchTilingData to file: %s\", search_file);" << std::endl;
  ss << "  std::ofstream out_file(search_file, mode);" << std::endl;
  ss << "  if (!out_file.is_open()) {" << std::endl;
  ss << "    DLOGE(\"Failed to open file:%s\", search_file);" << std::endl;
  ss << "    return;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  PgoSaveTilingKey(tiling_data, best_perf, out_file);" << std::endl;
  ss << "  out_file.close();" << std::endl;
  ss << std::endl;

  ss << "  int fd = ::open(search_file, O_WRONLY);" << std::endl;
  ss << "  if (fd < 0) {" << std::endl;
  ss << "    DLOGE(\"Failed to open file:%s\", search_file);" << std::endl;
  ss << "    return;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  if (::fsync(fd) < 0) {" << std::endl;
  ss << "    DLOGW(\"Failed to fsync file:%s\", search_file);" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ::close(fd);" << std::endl;
  ss << std::endl;
  ss << "  return;" << std::endl;
  ss << "}" << std::endl;
}

void TilingLib::GenPgoKernelLaunchOpArgs(const ascir::FusedScheduledResult &fused_schedule_result,
                                         std::stringstream &ss) const {
  ss << "struct AivKernelLaunchOpArgs {" << std::endl;
  ss << PGOSearchStructInputOutputDef(fused_schedule_result);
  ss << "  uint64_t workspace_addr;" << std::endl;
  ss << "  uint64_t tiling_addr;" << std::endl;
  ss << "};" << std::endl;

  ss << "struct MixKernelLaunchOpArgs {" << std::endl;
  if (IsNeedFfts()) {
    ss << "  uint64_t ffts;" << std::endl;
  }
  ss << PGOSearchStructInputOutputDef(fused_schedule_result);
  ss << "  uint64_t workspace_addr;" << std::endl;
  ss << "  uint64_t tiling_addr;" << std::endl;
  ss << "};" << std::endl;

  ss << "void *g_workspace = nullptr;" << std::endl;
}

void TilingLib::GenPgoCheckTilingIsMix(const ascir::FusedScheduledResult &fused_schedule_result,
                                       std::stringstream &ss) const {
  ss << "bool IsMixTiling(const AutofuseTilingData &t) {" << std::endl;
  ss << "  if constexpr (!g_is_mix_operator) {" << std::endl;
  ss << "    return false;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  if (!g_is_static_kernel) {" << std::endl;
  ss << "    return true;" << std::endl;
  ss << "  }" << std::endl;
  if (!ascgen_utils::IsSingleGroup(fused_schedule_result)) {
    for (size_t graph_id = 0U; graph_id < fused_schedule_result.node_idx_to_scheduled_results.size(); graph_id++) {
      ss << "  if (!g_mix_graph" << graph_id << "_tiling_keys.empty() && std::find(g_mix_graph" << graph_id
         << "_tiling_keys.begin(), g_mix_graph" << graph_id << "_tiling_keys.end(), t.graph" << graph_id
         << "_tiling_key) != g_mix_graph" << graph_id << "_tiling_keys.end()) {" << std::endl;
      ss << "    return true;" << std::endl;
      ss << "  }" << std::endl;
    }
  }
  ss << "  return false;" << std::endl;
  ss << "}" << std::endl;
}

void TilingLib::GenPgoLaunchParamsInit(const ascir::FusedScheduledResult &fused_schedule_result,
                                       std::stringstream &ss) const {
  ss << "aclError LaunchParamsInit(PgoTensorArgs *tensor_args) {" << std::endl;
  ss << "  static void *ffts = nullptr;" << std::endl;
  ss << "  aclError ret = ACL_SUCCESS;" << std::endl;
  ss << "  constexpr uint32_t kPgoInputCount = " << fused_schedule_result.input_nodes.size() << "U;" << std::endl;
  ss << "  constexpr uint32_t kPgoOutputCount = " << PGOSearchFuncGetOutputCount(fused_schedule_result) << "U;"
     << std::endl;
  ss << "  if (tensor_args == nullptr || tensor_args->input_num < kPgoInputCount ||" << std::endl;
  ss << "      tensor_args->output_num < kPgoOutputCount ||" << std::endl;
  ss << "      (kPgoInputCount > 0U && tensor_args->inputs == nullptr) ||" << std::endl;
  ss << "      (kPgoOutputCount > 0U && tensor_args->outputs == nullptr)) {" << std::endl;
  ss << "    DLOGE(\"invalid pgo tensor args\");" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << PGOSearchFuncInputOutputStructAssignDef(fused_schedule_result, "  g_launch_params.aiv_args");
  ss << "  g_launch_params.aiv_args.tiling_addr = reinterpret_cast<uint64_t>(g_tiling_device_addr);" << std::endl;
  if (IsNeedFfts()) {
    ss << "  ret = aclrtGetHardwareSyncAddr(&ffts);" << std::endl;
    ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
    ss << "    DLOGE(\"acl get hardware sync addr failed, ERROR: %d\", ret);" << std::endl;
    ss << "    return FAILED;" << std::endl;
    ss << "  }" << std::endl;
    ss << "  g_launch_params.mix_args.ffts = reinterpret_cast<uint64_t>(ffts);" << std::endl;
  }
  ss << PGOSearchFuncInputOutputStructAssignDef(fused_schedule_result, "  g_launch_params.mix_args");
  ss << "  g_launch_params.mix_args.tiling_addr = reinterpret_cast<uint64_t>(g_tiling_device_addr);" << std::endl;
  ss << "  ret = aclrtMalloc(&g_launch_params.aiv_args_device, sizeof(AivKernelLaunchOpArgs), "
        "ACL_MEM_MALLOC_HUGE_FIRST);"
     << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"acl malloc aiv args device failed, ERROR: %d\", ret);" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ret = aclrtMalloc(&g_launch_params.mix_args_device, sizeof(MixKernelLaunchOpArgs), "
        "ACL_MEM_MALLOC_HUGE_FIRST);"
     << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"acl malloc mix args device failed, ERROR: %d\", ret);" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  return ACL_SUCCESS;" << std::endl;
  ss << "}" << std::endl;
}

void TilingLib::GenPgoLaunchParamsDeInit(std::stringstream &ss) const {
  ss << "void LaunchParamsDeInit() {" << std::endl;
  ss << "  if (g_launch_params.aiv_args_device != nullptr) {" << std::endl;
  ss << "    auto ret = aclrtFree(g_launch_params.aiv_args_device);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGW(\"acl free aiv args device failed, ERROR: %d\", ret);" << std::endl;
  ss << "    }" << std::endl;
  ss << "    g_launch_params.aiv_args_device = nullptr;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  if (g_launch_params.mix_args_device != nullptr) {" << std::endl;
  ss << "    auto ret = aclrtFree(g_launch_params.mix_args_device);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGW(\"acl free mix args device failed, ERROR: %d\", ret);" << std::endl;
  ss << "    }" << std::endl;
  ss << "    g_launch_params.mix_args_device = nullptr;" << std::endl;
  ss << "  }" << std::endl;
  ss << "}" << std::endl;
}

void TilingLib::GenPgoUpdateLaunchParams(std::stringstream &ss) const {
  ss << "aclError UpdateLaunchParam(const AutofuseTilingData &tiling_data) {" << std::endl;
  ss << "  if (IsMixTiling(tiling_data)) {" << std::endl;
  ss << "    auto ret = aclrtMemcpy((void *)g_launch_params.mix_args.tiling_addr, sizeof(AutofuseTilingData), (void "
        "*)&tiling_data, "
     << "sizeof(AutofuseTilingData), ACL_MEMCPY_HOST_TO_DEVICE);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGE(\"memcpy tiling data to device failed, ERROR: %d\", ret);" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    g_launch_params.mix_args.workspace_addr = reinterpret_cast<uint64_t>(g_workspace);" << std::endl;
  ss << "    ret = aclrtMemcpy(g_launch_params.mix_args_device, sizeof(g_launch_params.mix_args), (void "
        "*)&g_launch_params.mix_args, sizeof(g_launch_params.mix_args), ACL_MEMCPY_HOST_TO_DEVICE);"
     << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGE(\"memcpy mix_args to device failed, ERROR: %d\", ret);" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  } else {" << std::endl;
  ss << "    auto ret = aclrtMemcpy((void *)g_launch_params.aiv_args.tiling_addr, sizeof(AutofuseTilingData), (void "
        "*)&tiling_data, "
     << "sizeof(AutofuseTilingData), ACL_MEMCPY_HOST_TO_DEVICE);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGE(\"memcpy tiling data to device failed, ERROR: %d\", ret);" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    g_launch_params.aiv_args.workspace_addr = reinterpret_cast<uint64_t>(g_workspace);" << std::endl;
  ss << "    ret = aclrtMemcpy(g_launch_params.aiv_args_device, sizeof(g_launch_params.aiv_args), (void "
        "*)&g_launch_params.aiv_args, sizeof(g_launch_params.aiv_args), ACL_MEMCPY_HOST_TO_DEVICE);"
     << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGE(\"memcpy aiv_args to device failed, ERROR: %d\", ret);" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  return ACL_SUCCESS;" << std::endl;
  ss << "}" << std::endl;
}

void TilingLib::GenPgoLaunchParams(const ascir::FusedScheduledResult &fused_schedule_result,
                                   std::stringstream &ss) const {
  ss << "struct LaunchParams {" << std::endl;
  ss << "  AivKernelLaunchOpArgs aiv_args;" << std::endl;
  ss << "  void *aiv_args_device;" << std::endl;
  ss << "  MixKernelLaunchOpArgs mix_args;" << std::endl;
  ss << "  void *mix_args_device;" << std::endl;
  ss << "} g_launch_params;" << std::endl;

  GenPgoLaunchParamsInit(fused_schedule_result, ss);
  GenPgoLaunchParamsDeInit(ss);
  GenPgoUpdateLaunchParams(ss);
}

void TilingLib::GenPgoToolFunction(const ascir::FusedScheduledResult &fused_schedule_result, const std::string &pgo_dir,
                                   std::stringstream &ss) const {
  std::string graph_name = CamelToLowerSneak(fused_schedule_result.fused_graph_name.GetString());
  ss << "namespace {" << std::endl;
  ss << "constexpr bool g_is_mix_operator = " << (IsMixKernelTaskType(fused_schedule_result) ? "true;" : "false;")
     << std::endl;
  ss << "static bool g_is_static_kernel = false;" << std::endl;
  GenPgoMixTilingTable(fused_schedule_result, ss);
  GenPgoCheckTilingIsMix(fused_schedule_result, ss);
  ss << "static std::string g_kernel_name;" << std::endl;
  ss << "static std::string g_kernel_o_file;" << std::endl;
  ss << "static std::string g_npu_lock_file;" << std::endl;
  ss << "#define PGO_GRAPH_NAME \"" << graph_name << "\"" << std::endl;
  ss << "const char *pgo_dir = \"" << pgo_dir << "\";" << std::endl;
  ss << "const char *config_file = \"" << pgo_dir << "/" << graph_name << "_config.txt" << "\";" << std::endl;
  ss << "const char *search_file = \"" << pgo_dir << "/" << graph_name << "_search.txt" << "\";" << std::endl;
  ss << "const char *kernel_file = \"" << pgo_dir << "/lib" << graph_name << ".so" << "\";" << std::endl;
  ss << "#define SUCCESS 0" << std::endl;
  ss << "#define FAILED 1" << std::endl;

  ss << "inline uint64_t PgoGetTid() {" << std::endl;
  ss << "  return static_cast<uint64_t>(syscall(__NR_gettid));" << std::endl;
  ss << "}" << std::endl;
  AppendPgoLogDefs(ss);

  GenPgoCardLock(ss);
  GenPgoAppendSearchTilingData(ss);
  GenPgoKernelLaunchOpArgs(fused_schedule_result, ss);

  GenDynamicLibraryLoaderCode(ss);

  ss << "aclrtStream g_stream;" << std::endl;
  ss << PGOSearchTensorInputOutputDef(fused_schedule_result) << std::endl;
  ss << "void *g_tiling_device_addr = nullptr;" << std::endl;

  GenPgoLaunchParams(fused_schedule_result, ss);

  ss << "struct ResLimit {" << std::endl;
  ss << "  uint32_t valid_num = 0;" << std::endl;
  ss << "  uint32_t aiv_num = 0;" << std::endl;
  ss << "  uint32_t aic_num = 0;" << std::endl;
  ss << "  uint32_t ub_size = 0;" << std::endl;
  ss << "  uint32_t resv[10];" << std::endl;
  ss << "};" << std::endl;
  ss << "ResLimit g_res_limit = {1, {}};" << std::endl;
  ss << "inline bool IsEqual(double a, double b) {" << std::endl;
  ss << "  const double epsilon = 1e-8;" << std::endl;
  ss << "  double abs = (a > b) ? (a - b) : (b - a);" << std::endl;
  ss << "  return abs < epsilon;" << std::endl;
  ss << "}" << std::endl;
  ss << "} // namespace" << std::endl;
}

void TilingLib::GenPgoWrapperParmCall(const ascir::FusedScheduledResult &fused_schedule_result,
                                      std::stringstream &ss) const {
  ss << "  if (tiling_data == nullptr) {" << std::endl;
  ss << "    DLOGE(\"tiling_data is null\");" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  uint32_t block_dim = tiling_data->block_dim;" << std::endl;
  ss << "  aclError ret = ACL_SUCCESS;" << std::endl;
  ss << "  int64_t tiling_key = 0;" << std::endl;
  if (CanUseTilingKey(fused_schedule_result)) {
    ss << "  if (find_best_tiling_key_fn != nullptr) {" << std::endl;
    ss << "    tiling_key = find_best_tiling_key_fn(*tiling_data);" << std::endl;
    ss << "    if (tiling_key == -1) {" << std::endl;
    ss << "      DLOGE(\"find best tiling key failed\");" << std::endl;
    ss << "      return FAILED;" << std::endl;
    ss << "    }" << std::endl;
    ss << "  } else {" << std::endl;
    ss << "    DLOGE(\"find best tiling key func is null\");" << std::endl;
    ss << "    return FAILED;" << std::endl;
    ss << "  }" << std::endl;
  }
}

void TilingLib::GenPgoWrapperKernelLaunch(std::stringstream &ss) const {
  ss << "  if (IsMixTiling(*tiling_data)) {" << std::endl;
  const auto backend_spce = optimize::BackendSpec::GetInstance();
  const bool use_local_memory = (backend_spce != nullptr && backend_spce->set_local_memory_size > 0);
  ss << (use_local_memory ? "    ret = aclrtLaunchKernelV2(func_handles[tiling_key], block_dim, "
                            "g_launch_params.mix_args_device, sizeof(g_launch_params.mix_args), &kernel_cfg, g_stream);"
                          : "    ret = aclrtLaunchKernelV2(func_handles[tiling_key], block_dim, "
                            "g_launch_params.mix_args_device, sizeof(g_launch_params.mix_args), nullptr, g_stream);")
     << std::endl;
  ss << "  } else {" << std::endl;
  ss << (use_local_memory ? "    ret = aclrtLaunchKernelV2(func_handles[tiling_key], block_dim, "
                            "g_launch_params.aiv_args_device, sizeof(g_launch_params.aiv_args), &kernel_cfg, g_stream);"
                          : "    ret = aclrtLaunchKernelV2(func_handles[tiling_key], block_dim, "
                            "g_launch_params.aiv_args_device, sizeof(g_launch_params.aiv_args), nullptr, g_stream);")
     << std::endl;
  ss << "  }" << std::endl;
  ss << "  auto ret_async = aclrtSynchronizeStream(g_stream);" << std::endl;
}

void TilingLib::GenPgoWrapper(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const {
  ss << "typedef uint64_t (*GetTilingKeyCountType)(void);" << std::endl;
  ss << "GetTilingKeyCountType get_tiling_key_count_fn = "
        "reinterpret_cast<GetTilingKeyCountType>(GetFunc(\"GetTilingKeyCount\"));"
     << std::endl;
  if (CanUseTilingKey(fused_schedule_result)) {
    ss << "typedef int64_t (*FindBestTilingKeyType)(AutofuseTilingData &t);" << std::endl;
    ss << "FindBestTilingKeyType find_best_tiling_key_fn = "
          "reinterpret_cast<FindBestTilingKeyType>(GetFunc(\"FindBestTilingKey\"));"
       << std::endl;
  }
  ss << "int WrapperOnlyLaunch(uint32_t workspace_size, AutofuseTilingData *tiling_data) {" << std::endl;
  ss << "  static bool inited = false;" << std::endl;
  ss << "  static aclrtBinHandle bin_handle = nullptr;" << std::endl;
  const auto backend_spce = optimize::BackendSpec::GetInstance();
  if (backend_spce != nullptr && backend_spce->set_local_memory_size > 0) {
    ss << "  static aclrtLaunchKernelCfg kernel_cfg{};" << std::endl;
    ss << "  static aclrtLaunchKernelAttr local_memory_size_attr{};" << std::endl;
  }
  ss << "  if (get_tiling_key_count_fn == nullptr) {" << std::endl;
  ss << "    DLOGE(\"get_tiling_key_count_fn is nullptr\");" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  static uint64_t tiling_key_count = get_tiling_key_count_fn();" << std::endl;
  ss << "  static std::vector<aclrtFuncHandle> func_handles(tiling_key_count);" << std::endl;

  GenPgoWrapperParmCall(fused_schedule_result, ss);
  GenPgoLaunchKernelInit(ss);
  GenPgoWrapperKernelLaunch(ss);
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"aclrtLaunchKernelV2 failed, ERROR: %d\", ret);" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  if (ret_async != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"aclrtSynchronizeStream failed, ERROR: %d\", ret_async);" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  return ret;" << std::endl;
  ss << "}" << std::endl << std::endl;
}

void TilingLib::GenPgoProfilingConstants(std::stringstream &ss) const {
  ss << "#define ALIGN_SIZE (8)" << std::endl;
  ss << "#define ALIGN_BUFFER(buffer, align) \\" << std::endl;
  ss << "    (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : "
     << "(buffer))" << std::endl;
  ss << "constexpr size_t group_size = 1000ULL;" << std::endl;
  ss << "static std::map<uint64_t, msptiActivity*> g_profiling_map;" << std::endl;
  ss << "constexpr uint64_t loop = 20;" << std::endl;
  ss << "constexpr int max_flush_times = 5;" << std::endl;
  ss << "constexpr size_t mspti_buffer_size = 16ULL * 1024 * 1024;" << std::endl;
  ss << "static double best_perf = DBL_MAX;" << std::endl;
}

void TilingLib::GenPgoMsptiStringTable(std::stringstream &ss) const {
  ss << R"(
static const char* GetActivityKindString(msptiActivityKind kind) {
  static const std::unordered_map<msptiActivityKind, const char*> STRING_MAP = {
    {MSPTI_ACTIVITY_KIND_INVALID, "INVALID"},
    {MSPTI_ACTIVITY_KIND_MARKER, "MARKER"},
    {MSPTI_ACTIVITY_KIND_KERNEL, "KERNEL"},
    {MSPTI_ACTIVITY_KIND_API, "API"},
    {MSPTI_ACTIVITY_KIND_MEMORY, "MEMORY"},
    {MSPTI_ACTIVITY_KIND_MEMSET, "MEMSET"},
    {MSPTI_ACTIVITY_KIND_MEMCPY, "MEMCPY"},
    {MSPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION, "CORRELATION"}
  };
  auto it = STRING_MAP.find(kind);
  return it != STRING_MAP.end() ? it->second : "<unknown>";
})" << std::endl;
  ss << R"(
static const char* GetResultCodeString(msptiResult result) {
  static const std::unordered_map<msptiResult, const char*> STRING_MAP = {
    {MSPTI_SUCCESS, "SUCCESS"},
    {MSPTI_ERROR_INVALID_PARAMETER, "ERROR_INVALID_PARAMETER"},
    {MSPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED, "MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED"},
    {MSPTI_ERROR_DEVICE_OFFLINE, "DEVICE_OFFLINE"},
    {MSPTI_ERROR_QUEUE_EMPTY, "QUEUE_EMPTY"},
    {MSPTI_ERROR_INNER, "ERROR_INNER"}
  };

  auto it = STRING_MAP.find(result);
  return it != STRING_MAP.end() ? it->second : "<unknown>";
})" << std::endl;
}

void TilingLib::GenPgoMsptiRequest(std::stringstream &ss) const {
  ss << R"(
void UserBufferRequest(uint8_t **buffer, size_t *size, size_t *records_num) {
  DLOGD("[mspti] UserBufferRequest...");
  uint8_t *mspti_buffer = reinterpret_cast<uint8_t *>(malloc(mspti_buffer_size + ALIGN_SIZE));
  if (mspti_buffer == nullptr) {
    DLOGE("[mspti] malloc mspti_buffer failed");
    *buffer = nullptr;
    *size = 0;
    *records_num = 0;
    return;
  }
  *buffer = ALIGN_BUFFER(mspti_buffer, ALIGN_SIZE);
  *size = mspti_buffer_size;
  *records_num = 0;
})" << std::endl;
}

void TilingLib::GenPgoMsptiComplete(std::stringstream &ss) const {
  ss << R"(
void UserBufferComplete(uint8_t *buffer, size_t size, size_t valid_size) {
  DLOGD("[mspti] UserBufferComplete, buf addr: %" PRIuPTR ", size: %zu, valid size: %zu", (uintptr_t)buffer, size, valid_size);
  if (valid_size > 0) {
    msptiActivity *mspti_record = NULL;
    msptiResult status = MSPTI_SUCCESS;
    do {
      status = msptiActivityGetNextRecord(buffer, valid_size, &mspti_record);
      if (status == MSPTI_SUCCESS) {
        if (mspti_record->kind == MSPTI_ACTIVITY_KIND_KERNEL) {
          msptiActivityKernel* kernelRecord = (msptiActivityKernel*)mspti_record;
          msptiActivity* pRecordCopy = (msptiActivity *)malloc(sizeof(msptiActivityKernel));
          memset(pRecordCopy, 0, sizeof(msptiActivityKernel));
          memcpy(pRecordCopy, kernelRecord, sizeof(msptiActivityKernel));
          g_profiling_map[kernelRecord->start] = pRecordCopy;

        } else {
          DLOGD("[mspti] [%s] ignored", GetActivityKindString(mspti_record->kind));
        }
      } else if (status == MSPTI_ERROR_MAX_LIMIT_REACHED) {
        break;
      } else {
        DLOGW("[mspti] Consume data fail error is %s", GetResultCodeString(status));
        break;
      }
    } while (1);
  }
  free(buffer);
})" << std::endl;
}

void TilingLib::GenPgoMsptiToolFunction(std::stringstream &ss) const {
  ss << R"(
void SetUpMspti(msptiSubscriberHandle* subscriber) {
  DLOGD("[mspti] setup mspti");
  msptiSubscribe(subscriber, nullptr, nullptr);
  msptiActivityRegisterCallbacks(UserBufferRequest, UserBufferComplete);
  msptiActivityEnable(MSPTI_ACTIVITY_KIND_KERNEL);
})" << std::endl;
  ss << R"(
void TearDownMspti(msptiSubscriberHandle *subscriber) {
  DLOGD("[mspti] tear down mspti");
  msptiUnsubscribe(*subscriber);
  msptiActivityFlushAll(1);
})" << std::endl;
}

void TilingLib::GenPgoMsptiProfiling(std::stringstream &ss) const {
  GenPgoProfilingConstants(ss);
  GenPgoMsptiStringTable(ss);
  GenPgoMsptiRequest(ss);
  GenPgoMsptiComplete(ss);
  GenPgoMsptiToolFunction(ss);
}

void TilingLib::GenPgoBatchCallback(std::stringstream &ss) const {
  ss << "  result = aclrtSynchronizeStream(g_stream);" << std::endl;
  ss << "  TearDownMspti(&subscriber);" << std::endl << std::endl;
  ss << "  int flush_count = 0;" << std::endl;
  ss << "  while (g_profiling_map.size() < batch_size * loop && flush_count < max_flush_times) {" << std::endl;
  ss << "    flush_count++;" << std::endl;
  ss << "    std::this_thread::sleep_for(std::chrono::milliseconds(10 * flush_count));" << std::endl;
  ss << "    msptiActivityFlushAll(1);" << std::endl;
  ss << "  }" << std::endl << std::endl;
  ss << "  if (g_profiling_map.size() < batch_size * loop) {" << std::endl;
  ss << "    DLOGE(\"ProfilingBatchProcess g_profiling_map size %zu is less than batch_size * loop %\" PRIu64 \"\", "
        "g_profiling_map.size(), batch_size * loop);"
     << std::endl;
  ss << "    for (auto &item : g_profiling_map) {" << std::endl;
  ss << "      free(item.second);" << std::endl;
  ss << "    }" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl << std::endl;
  ss << "  auto it = g_profiling_map.begin();" << std::endl;
  ss << "  for (uint64_t i = 0; i < batch_size; ++i) {" << std::endl;
  ss << "    uint64_t total_duration = 0;" << std::endl;
  ss << "    std::vector<uint64_t> durations;" << std::endl;
  ss << "    for (uint64_t j = 0; j < loop; ++j) {" << std::endl;
  ss << "      msptiActivityKernel* kernel = reinterpret_cast<msptiActivityKernel*>(it->second);" << std::endl;
  ss << "      durations.push_back(kernel->end - kernel->start);" << std::endl;
  ss << "      std::advance(it, 1);" << std::endl;
  ss << "    }" << std::endl;
  ss << "    std::sort(durations.begin(), durations.end(), std::greater<int>());" << std::endl;
  ss << "    for (size_t k = 1; k < 6; ++k) {" << std::endl;
  ss << "      total_duration += durations[k];" << std::endl;
  ss << "    }" << std::endl;
  ss << "    double average_duration = static_cast<double>(total_duration) / 5;" << std::endl;
  ss << "    (begin + i)->best_perf = average_duration;" << std::endl;
  ss << "    if (best_perf > average_duration) {" << std::endl;
  ss << "      best_perf = average_duration;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    DLOGD(\"average_duration:%f best_perf:%f count:%\" PRId64 \" batch_size:%\" PRIu64 \" flush_count:%d\", "
        "average_duration, best_perf, count, batch_size, flush_count);"
     << std::endl;
  ss << "  }" << std::endl;
  ss << "  for (auto &item : g_profiling_map) {" << std::endl;
  ss << "    free(item.second);" << std::endl;
  ss << "  }" << std::endl;
}

void TilingLib::GenPgoBatchProcess(std::stringstream &ss) const {
  ss << "int ProfilingBatchProcess(uint32_t workspace_size, std::vector<AutofuseTilingDataPerf>::iterator begin, "
        "std::vector<AutofuseTilingDataPerf>::iterator end) {"
     << std::endl;
  ss << "  uint64_t batch_size = end - begin;" << std::endl;
  ss << "  g_profiling_map.clear();" << std::endl;
  ss << "  msptiSubscriberHandle subscriber;" << std::endl;
  ss << "  SetUpMspti(&subscriber);" << std::endl << std::endl;
  ss << "  static int64_t count = 0;" << std::endl;
  ss << "  count++;" << std::endl << std::endl;
  ss << "  int64_t result = 0;" << std::endl;
  ss << "  for (auto it = begin; it != end; ++it) {" << std::endl;
  ss << "    it->best_perf = DBL_MAX;" << std::endl;
  ss << "    AutofuseTilingData &tiling_data = it->tiling_data;" << std::endl;
  ss << "    UpdateLaunchParam(tiling_data);" << std::endl;
  ss << "    for (uint64_t i = 0; i < loop; ++i) {" << std::endl;
  ss << "      result = WrapperOnlyLaunch(workspace_size, &tiling_data);" << std::endl;
  ss << "      if (result != 0) {" << std::endl;
  ss << "        DLOGE(\"ProfilingBatchProcess launch failed loop:%\" PRIu64 \"\", i);" << std::endl;
  ss << "        TearDownMspti(&subscriber);" << std::endl;
  ss << "        return -1;" << std::endl;
  ss << "      }" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl << std::endl;
  GenPgoBatchCallback(ss);
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl << std::endl;
}

void TilingLib::GenPgoGetProfilingBatch(const ascir::FusedScheduledResult &fused_schedule_result,
                                        std::stringstream &ss) const {
  ss << "extern \"C\" long int PGOGetProfilingBatch(" << PGOSearchFuncInputOutputCallBackDef(fused_schedule_result)
     << "void* stream, uint32_t workspace_size, std::vector<AutofuseTilingDataPerf> *profiles) {" << std::endl;
  ss << "  int case_num = profiles->size();" << std::endl;
  ss << "  DLOGI(\"PGOGetProfilingBatch case_num:%d\", case_num);" << std::endl;
  ss << "  if (workspace_size > 0) {" << std::endl;
  ss << "    auto ret = aclrtMalloc(&g_workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGE(\"malloc workspace failed, size: %u, ERROR: %d\", workspace_size, ret);" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  int64_t result = 0;" << std::endl;
  ss << "  auto it = profiles->begin();" << std::endl;
  ss << "  while (it != profiles->end()) {" << std::endl;
  ss << "    auto end_it = (it + group_size >= profiles->end()) ? profiles->end() : it + group_size;" << std::endl;
  ss << "    size_t start_index = std::distance(profiles->begin(), it);" << std::endl;
  ss << "    for (int i = 0; i < 3; i++) {" << std::endl;
  ss << "      result = ProfilingBatchProcess(workspace_size, it, end_it);" << std::endl;
  ss << "      if (result != 0) {" << std::endl;
  ss << "        DLOGW(\"ProfilingBatchProcess failed at start_index:%zu retry time:%d\", start_index, i);"
     << std::endl;
  ss << "      } else {" << std::endl;
  ss << "        break;" << std::endl;
  ss << "      }" << std::endl;
  ss << "    }" << std::endl;
  ss << "    it = end_it;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  if (g_workspace != nullptr) {" << std::endl;
  ss << "    auto ret = aclrtFree(g_workspace);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGE(\"free workspace failed, ERROR: %d\", ret);" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl << std::endl;
}

void TilingLib::GenPgoProfilingCallback(std::stringstream &ss) const {
  ss << "  result = aclrtSynchronizeStream(g_stream);" << std::endl;
  ss << "  if (result != 0) {" << std::endl;
  ss << "    DLOGE(\"sync stream failed\");" << std::endl;
  ss << "    TearDownMspti(&subscriber);" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  TearDownMspti(&subscriber);" << std::endl;
  ss << std::endl;
  ss << "  int flush_count = 0;" << std::endl;
  ss << "  while (g_profiling_map.size() < loop && flush_count < max_flush_times) {" << std::endl;
  ss << "    flush_count++;" << std::endl;
  ss << "    std::this_thread::sleep_for(std::chrono::milliseconds(10 * flush_count));" << std::endl;
  ss << "    msptiActivityFlushAll(1);" << std::endl;
  ss << "  }" << std::endl;
  ss << std::endl;
  ss << "  if (g_profiling_map.size() != loop) {" << std::endl;
  ss << "    DLOGE(\"map size %zu not equals to loop %\" PRIu64 \"\", g_profiling_map.size(), loop);" << std::endl;
  ss << "    for (auto &item : g_profiling_map) {" << std::endl;
  ss << "      free(item.second);" << std::endl;
  ss << "    }" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << std::endl;
  ss << "  uint64_t total_duration = 0;" << std::endl;
  ss << "  std::vector<uint64_t> durations;" << std::endl;
  ss << "  for (const auto &pair : g_profiling_map) {" << std::endl;
  ss << "    msptiActivityKernel* kernel = reinterpret_cast<msptiActivityKernel*>(pair.second);" << std::endl;
  ss << "    durations.push_back(kernel->end - kernel->start);" << std::endl;
  ss << "    DLOGD(\"kernel duration:%\" PRIu64 \"\", kernel->end - kernel->start);" << std::endl;
  ss << "  }" << std::endl;
  ss << "  std::sort(durations.begin(), durations.end(), std::greater<int>());" << std::endl;
  ss << "  for (size_t i = 1; i < 6; ++i) {" << std::endl;
  ss << "    total_duration += durations[i];" << std::endl;
  ss << "  }" << std::endl;
  ss << "  double average_duration = static_cast<double>(total_duration) / 5;" << std::endl;
  ss << "  *outCostTime = average_duration;" << std::endl;
  ss << std::endl;
  ss << "  if (best_perf > *outCostTime) {" << std::endl;
  ss << "    best_perf = *outCostTime;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  DLOGD(\"average_duration:%f best_perf:%f count:%\" PRId64 \" flush_count:%d\", *outCostTime, best_perf, "
        "count, flush_count);"
     << std::endl;
  ss << "  for (auto &item : g_profiling_map) {" << std::endl;
  ss << "    free(item.second);" << std::endl;
  ss << "  }" << std::endl;
}

void TilingLib::GenPgoGetProfiling(const ascir::FusedScheduledResult &fused_schedule_result,
                                   std::stringstream &ss) const {
  ss << "extern \"C\" long int PGOGetProfiling(" << PGOSearchFuncInputOutputCallBackDef(fused_schedule_result)
     << "void *stream, uint32_t workspace_size, AutofuseTilingData *tiling_data, double *outCostTime) {" << std::endl;
  ss << "  if (workspace_size > 0) {" << std::endl;
  ss << "    auto ret = aclrtMalloc(&g_workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGE(\"malloc workspace failed, size: %u, ERROR: %d\", workspace_size, ret);" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  g_profiling_map.clear();" << std::endl;
  ss << "  msptiSubscriberHandle subscriber;" << std::endl;
  ss << "  SetUpMspti(&subscriber);" << std::endl << std::endl;
  ss << "  int64_t result = -1;" << std::endl;
  ss << "  *outCostTime = DBL_MAX;" << std::endl;
  ss << "  static int64_t count = 0;" << std::endl;
  ss << "  count++;" << std::endl << std::endl;

  ss << "  UpdateLaunchParam(*tiling_data);" << std::endl;
  ss << "  for (uint64_t j = 0; j < loop; ++j) {" << std::endl;
  ss << "    result = WrapperOnlyLaunch(workspace_size, tiling_data);" << std::endl;
  ss << "    if (result != 0) {" << std::endl;
  ss << "      DLOGE(\"launch failed loop:%\" PRIu64 \"\", j);" << std::endl;
  ss << "      TearDownMspti(&subscriber);" << std::endl;
  ss << "      return -1;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl << std::endl;

  ss << "  if (g_workspace != nullptr) {" << std::endl;
  ss << "    auto ret = aclrtFree(g_workspace);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGE(\"free workspace failed, ERROR: %d\", ret);" << std::endl;
  ss << "      TearDownMspti(&subscriber);" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  GenPgoProfilingCallback(ss);
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl << std::endl;
}

void TilingLib::GenPgoFunc(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const {
  ss << "int pgo() {" << std::endl;
  ss << "  AutofuseTilingData tiling_data = {0};" << std::endl;
  ss << "  PgoTensorArgs *tensor_args = &g_pgo_tensor_args;" << std::endl;
  ss << "  uint32_t workspace_size = 0;" << std::endl;
  ss << "  uint32_t block_dim = 0;" << std::endl;
  ss << "  if (pgo_search_fn == nullptr) {" << std::endl;
  ss << "    DLOGE(\"pgo search func not found\");" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  int64_t result = pgo_search_fn((char*)search_file, (char *)config_file, &tiling_data, &workspace_size, "
        "&block_dim, &g_res_limit,"
     << PGOSearchFuncInputOutputCall(fused_schedule_result)
     << "&g_stream, reinterpret_cast<void*>(PGOGetProfiling), reinterpret_cast<void*>(PGOGetProfilingBatch));"
     << std::endl;
  ss << "  if (result != 0) {" << std::endl;
  ss << "    DLOGE(\"pgo search failed. ERROR: %\" PRId64 \"\", result);" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl << std::endl;
}

void TilingLib::GenPgoStaticFunc(const ascir::FusedScheduledResult &fused_schedule_result,
                                 std::stringstream &ss) const {
  ss << "int static_pgo(const char* config_file) {" << std::endl;
  ss << "  if (autofuse_tiling_with_config_fn == nullptr) {" << std::endl;
  ss << "    DLOGE(\"autofuse tiling with config func not found\");" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  AutofuseTilingData tiling_data = {0};" << std::endl;
  ss << "  PgoTensorArgs *tensor_args = &g_pgo_tensor_args;" << std::endl;
  ss << "  uint32_t workspace_size = 0;" << std::endl;
  ss << "  uint32_t block_dim = 0;" << std::endl;
  ss << "  int64_t result = autofuse_tiling_with_config_fn(config_file, &tiling_data, &workspace_size, &block_dim, "
        "&g_res_limit);"
     << std::endl;
  ss << "  if (result != 0) {" << std::endl;
  ss << "    DLOGE(\"autofuse tiling with config failed. ERROR: %\" PRId64 \"\", result);" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  double out_cost = DBL_MAX;" << std::endl;
  ss << "  for (int i = 0; i < max_flush_times; i++) {" << std::endl;
  ss << "    result = PGOGetProfiling(" << PGOSearchFuncInputOutputCall(fused_schedule_result)
     << "g_stream, workspace_size, &tiling_data, &out_cost);" << std::endl;
  ss << "    if (result != 0 || IsEqual(out_cost, DBL_MAX)) {" << std::endl;
  ss << "      DLOGW(\"get profiling failed.\");" << std::endl;
  ss << "    } else {" << std::endl;
  ss << "      break;" << std::endl;
  ss << "    }" << std::endl;
  ss << "  }" << std::endl;
  ss << "  AppendPgoSearchTilingData(tiling_data, out_cost);" << std::endl;
  ss << "  return 0;" << std::endl;
  ss << "}" << std::endl << std::endl;
}

void TilingLib::GenPgoProfiling(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const {
  GenPgoMsptiProfiling(ss);
  GenPgoBatchProcess(ss);
  GenPgoGetProfilingBatch(fused_schedule_result, ss);
  GenPgoGetProfiling(fused_schedule_result, ss);
  ss << "typedef int64_t (*PGOSearchType)(char *search_file, char *config_file, AutofuseTilingData *tiling_data, "
        "uint32_t *workspace_size, uint32_t *blockDim, void *resource_limit, "
     << PGOSearchFuncInputOutputCallBackDef(fused_schedule_result)
     << "void *stream, void *prof_callback, void *prof_batch_callback);" << std::endl;
  ss << "static PGOSearchType pgo_search_fn = reinterpret_cast<PGOSearchType>(GetFunc(\"PgoTilingSearch\"));"
     << std::endl;
  GenPgoFunc(fused_schedule_result, ss);
  ss << "typedef int64_t (*AutofuseTilingWithConfigType)(const char *config_file, AutofuseTilingData *tiling, uint32_t "
        "*"
     << "workspace_size, uint32_t *blockDim, ResLimit *res_limit);" << std::endl;
  ss << "static AutofuseTilingWithConfigType autofuse_tiling_with_config_fn = "
     << "reinterpret_cast<AutofuseTilingWithConfigType>(GetFunc(\"AutofuseTilingWithConfig\"));" << std::endl;
  GenPgoStaticFunc(fused_schedule_result, ss);
}

void TilingLib::GenPgoMain(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const {
  ss << "int main(int argc, char *argv[]) {" << std::endl;
  ss << "  if (argc != 6) {" << std::endl;
  ss << "    DLOGE(\"Usage: %s <type> <device_id> <aiv_num> <ub_size> <kernel_name>\", argv[0]);" << std::endl;
  ss << "    return -1;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  int32_t type = static_cast<int32_t>(atoi(argv[1]));" << std::endl;
  ss << "  int32_t device_id = static_cast<int32_t>(atoi(argv[2]));" << std::endl;
  ss << "  int32_t aiv_num = static_cast<int32_t>(atoi(argv[3]));" << std::endl;
  ss << "  int32_t ub_size = static_cast<int32_t>(atoi(argv[4]));" << std::endl;
  ss << "  g_kernel_name = argv[5];" << std::endl;
  ss << "  DLOGI(\"execute info : type: %d, device_id: %d, kernel_name: %s\", type, device_id, g_kernel_name.c_str());"
     << std::endl;
  ss << "  DLOGI(\"execute limit: aiv_num is %d, ub_size is %d\", aiv_num, ub_size);" << std::endl;
  ss << "  g_npu_lock_file = std::string(pgo_dir) + \"/npu_lock_\" + std::to_string(device_id) + \".lock\";"
     << std::endl;
  ss << "  g_kernel_o_file = std::string(pgo_dir) + \"/\" + g_kernel_name + \".o\";" << std::endl;
  ss << "  CardLock lock(g_npu_lock_file.c_str());" << std::endl;
  GenPgoEnvInit(fused_schedule_result, ss);
  ss << "  if (type == 0) {" << std::endl;
  ss << "    ret = pgo();" << std::endl;
  ss << "  } else if (type == 1) {" << std::endl;
  ss << "    g_is_static_kernel = true;" << std::endl;
  ss << "    ret = static_pgo(config_file);" << std::endl;
  ss << "  } else {" << std::endl;
  ss << "    DLOGE(\"Invalid type: %d\", type);" << std::endl;
  ss << "    ret = -1;" << std::endl;
  ss << "  }" << std::endl;
  GenPgoDeinit(fused_schedule_result, ss);
  ss << "  return ret;" << std::endl;
  ss << "}" << std::endl;
}

void TilingLib::GenPgoEnvInit(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const {
  ss << "  g_res_limit.aiv_num = aiv_num;" << std::endl;
  ss << "  g_res_limit.ub_size = ub_size;" << std::endl;
  ss << "  auto ret = aclInit(nullptr);" << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"acl init failed, ERROR: %d\", ret);" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ret = aclrtSetDevice(device_id);" << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"acl set device failed, device id: %d, ERROR: %d\", device_id, ret);" << std::endl;
  ss << "    aclFinalize();" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ret = aclrtCreateStream(&g_stream);" << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"acl create stream failed, ERROR: %d\", ret);" << std::endl;
  ss << "    aclrtResetDevice(device_id);" << std::endl;
  ss << "    aclFinalize();" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << PGOSearchTensorMallocDef(fused_schedule_result) << std::endl;
  ss << PGOSearchTensorArgsUpdateDef(fused_schedule_result);
  ss << "  ret = aclrtMalloc(&g_tiling_device_addr, sizeof(AutofuseTilingData), ACL_MEM_MALLOC_HUGE_FIRST);"
     << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"acl malloc tiling data failed, ERROR: %d\", ret);" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ret = LaunchParamsInit(&g_pgo_tensor_args);" << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
}

void TilingLib::GenPgoLaunchKernelInit(std::stringstream &ss) const {
  ss << "  if (!inited) {" << std::endl;
  ss << "    auto ret = aclrtBinaryLoadFromFile(g_kernel_o_file.c_str(), nullptr, &bin_handle);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGE(\"acl load binary from file failed, ERROR: %d\", ret);" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    if (g_is_static_kernel) {" << std::endl;
  ss << "      aclrtFuncHandle func_handle = nullptr;" << std::endl;
  ss << "      ret = aclrtBinaryGetFunction(bin_handle, (g_kernel_name + \"_\" + std::to_string(tiling_key)).c_str(), "
        "&func_handle);"
     << std::endl;
  ss << "      if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "        DLOGE(\"acl get function failed, ERROR: %d\", ret);" << std::endl;
  ss << "        return FAILED;" << std::endl;
  ss << "      }" << std::endl;
  ss << "      func_handles[tiling_key] = func_handle;" << std::endl;
  ss << "    } else {" << std::endl;
  ss << "      for (uint64_t i = 0; i < tiling_key_count; ++i) {" << std::endl;
  ss << "        aclrtFuncHandle func_handle = nullptr;" << std::endl;
  ss << "        ret = aclrtBinaryGetFunction(bin_handle, (g_kernel_name + \"_\" + std::to_string(i)).c_str(), "
        "&func_handle);"
     << std::endl;
  ss << "        if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "          DLOGE(\"acl get function failed, ERROR: %d\", ret);" << std::endl;
  ss << "          return FAILED;" << std::endl;
  ss << "        }" << std::endl;
  ss << "        func_handles[i] = func_handle;" << std::endl;
  ss << "      }" << std::endl;
  ss << "    }" << std::endl;
  const auto backend_spce = optimize::BackendSpec::GetInstance();
  if (backend_spce != nullptr && backend_spce->set_local_memory_size > 0) {
    ss << "    local_memory_size_attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;" << std::endl;
    ss << "    local_memory_size_attr.value.dynUBufSize = " << backend_spce->set_local_memory_size << ";" << std::endl;
    ss << "    kernel_cfg.numAttrs = 1;" << std::endl;
    ss << "    kernel_cfg.attrs = &local_memory_size_attr;" << std::endl;
  }
  ss << "    inited = true;" << std::endl;
  ss << "  }" << std::endl;
}

void TilingLib::GenPgoDeinit(const ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const {
  ss << "  LaunchParamsDeInit();" << std::endl;
  ss << PGOSearchTensorFreeDef(fused_schedule_result) << std::endl;
  ss << "  if (g_tiling_device_addr != nullptr) {" << std::endl;
  ss << "    ret = aclrtFree(g_tiling_device_addr);" << std::endl;
  ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "      DLOGE(\"acl free tiling data failed, ERROR: %d\", ret);" << std::endl;
  ss << "      return FAILED;" << std::endl;
  ss << "    }" << std::endl;
  ss << "    g_tiling_device_addr = nullptr;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ret = aclrtDestroyStream(g_stream);" << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"acl destroy stream failed, ERROR: %d\", ret);" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ret = aclrtResetDevice(device_id);" << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"acl reset device failed, device id: %d, ERROR: %d\", device_id, ret);" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ret = aclFinalize();" << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"acl finalize failed, ERROR: %d\", ret);" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  DeInit();" << std::endl;
}

}  // namespace codegen
