/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file acl_spec_optimizer.cpp
 * \brief Model kernel specialization, resource ID extraction, and task replacement with rollback.
 */

#include "acl_spec_optimizer.h"
#include "aclmodel_specialization.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <elf.h>
#include <new>
#include <stdexcept>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>

#include <boost/filesystem.hpp>
#include <boost/system/error_code.hpp>

#include "feature_manager.h"
#include "runtime/rt_external_kernel.h"
#include "securec.h"
#include "sk_common.h"
#include "sk_log.h"

using sk::static_compile::Feature;
using sk::static_compile::FeatureManager;

namespace {
namespace fs = boost::filesystem;

void BindOptionPointers(const std::vector<std::string> &storage, std::vector<const char *> &pointers) {
  pointers.clear();
  pointers.reserve(storage.size());
  for (const auto &option : storage) {
    pointers.push_back(option.c_str());
  }
}

}  // namespace

aclError aclmdlRISpecOptimize(aclmdlRI modelRI, aclmdlRISpecOptions *options) {
  try {
    FeatureManager featureManager;
    if (!featureManager.Init(options)) {
      SK_DLOGE("aclmdlRISpecOptimize invalid configuration: model=%p", modelRI);
      return ACL_ERROR_INVALID_PARAM;
    }
    AclSpecOptimizer optimizer;
    const aclError result = optimizer.Init(modelRI);
    if (result != ACL_SUCCESS) {
      return result;
    }
    return optimizer.Optimize(modelRI, featureManager);
  } catch (const std::bad_alloc &) {
    SK_DLOGE("aclmdlRISpecOptimize allocation exception: model=%p", modelRI);
    return ACL_ERROR_BAD_ALLOC;
  } catch (const std::length_error &) {
    SK_DLOGE("aclmdlRISpecOptimize length exception: model=%p", modelRI);
    return ACL_ERROR_BAD_ALLOC;
  } catch (const std::invalid_argument &) {
    SK_DLOGE("aclmdlRISpecOptimize invalid argument exception: model=%p", modelRI);
    return ACL_ERROR_INVALID_PARAM;
  } catch (const std::exception &error) {
    SK_DLOGE("aclmdlRISpecOptimize exception: model=%p reason=%s", modelRI, error.what());
    return ACL_ERROR_FAILURE;
  } catch (...) {
    SK_DLOGE("aclmdlRISpecOptimize unknown exception: model=%p", modelRI);
    return ACL_ERROR_FAILURE;
  }
}

aclError AclSpecOptimizer::Init(aclmdlRI model) {
  if (model == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  uint32_t modelId = 0;
  aclError result = aclmdlRIGetId(model, &modelId);
  if (result != ACL_SUCCESS) {
    SK_DLOGE("aclmdlRISpecOptimize model id unavailable: model=%p ret=%d", model, result);
    return result;
  }
  result = outputDirectory_.Create(modelId);
  if (result != ACL_SUCCESS) {
    return result;
  }
  return ACL_SUCCESS;
}

aclError AclSpecOptimizer::Optimize(aclmdlRI model, const FeatureManager &featureManager) const {
  std::vector<std::unique_ptr<KernelSpecRequest>> requests;
  aclError result = Collect(model, featureManager, outputDirectory_.Path(), requests);
  if (result != ACL_SUCCESS) {
    SK_DLOGE("aclmdlRISpecOptimize collection failed: model=%p ret=%d", model, result);
    return result;
  }
  requests.erase(
      std::remove_if(requests.begin(), requests.end(), [this](auto &request) { return !SpecializeRequest(*request); }),
      requests.end());
  if (requests.empty()) {
    return ACL_SUCCESS;
  }
  auto &resourceManager = ModelSpecResourceManager::GetInstance();
  result = resourceManager.RegisterModel(model, requests.size());
  if (result != ACL_SUCCESS) {
    return result;
  }
  result = Apply(model, requests);
  if (result == ACL_SUCCESS || result == ACL_ERROR_INTERNAL_ERROR) {
    for (auto &request : requests) {
      resourceManager.RetainBinary(model, request->binary);
    }
  }
  return result;
}

aclError AclSpecOptimizer::Collect(aclmdlRI model, const FeatureManager &featureManager,
                                   std::string_view outputDirectory,
                                   std::vector<std::unique_ptr<KernelSpecRequest>> &requests) const {
  if (model == nullptr || outputDirectory.empty()) {
    return ACL_ERROR_INVALID_PARAM;
  }
  uint32_t streamCount = 0;
  aclError result = aclmdlRIGetStreams(model, nullptr, &streamCount);
  if (result != ACL_SUCCESS) {
    return result;
  }
  std::vector<aclrtStream> streams(streamCount);
  if (streamCount != 0) {
    result = aclmdlRIGetStreams(model, streams.data(), &streamCount);
    if (result != ACL_SUCCESS) {
      return result;
    }
    if (streamCount > streams.size()) {
      return ACL_ERROR_FAILURE;
    }
  }
  std::vector<std::unique_ptr<KernelSpecRequest>> collected;
  for (uint32_t streamIndex = 0; streamIndex < streamCount; ++streamIndex) {
    if (streams[streamIndex] == nullptr) {
      return ACL_ERROR_FAILURE;
    }
    uint32_t taskCount = 0;
    result = aclmdlRIGetTasksByStream(streams[streamIndex], nullptr, &taskCount);
    if (result != ACL_SUCCESS) {
      return result;
    }
    if (taskCount == 0) {
      continue;
    }
    std::vector<aclmdlRITask> tasks(taskCount);
    result = aclmdlRIGetTasksByStream(streams[streamIndex], tasks.data(), &taskCount);
    if (result != ACL_SUCCESS) {
      return result;
    }
    if (taskCount > tasks.size()) {
      return ACL_ERROR_FAILURE;
    }
    for (uint32_t taskIndex = 0; taskIndex < taskCount; ++taskIndex) {
      aclmdlRITaskType type = ACL_MODEL_RI_TASK_DEFAULT;
      if (tasks[taskIndex] == nullptr || aclmdlRITaskGetType(tasks[taskIndex], &type) != ACL_SUCCESS) {
        SK_DLOGI("aclmdlRISpecOptimize task type unavailable: stream=%u task=%u", streamIndex, taskIndex);
        continue;
      }
      if (type != ACL_MODEL_RI_TASK_KERNEL) {
        continue;
      }
      auto request = std::make_unique<KernelSpecRequest>();
      if (TryBuildRequest(tasks[taskIndex], streamIndex, taskIndex, featureManager, outputDirectory, *request)) {
        collected.push_back(std::move(request));
      }
    }
  }
  for (auto &request : collected) {
    request->BindApiRequest();
  }
  requests.swap(collected);
  return ACL_SUCCESS;
}

bool AclSpecOptimizer::TryBuildRequest(aclmdlRITask task, uint32_t streamIndex, uint32_t taskIndex,
                                       const FeatureManager &featureManager, std::string_view outputDirectory,
                                       KernelSpecRequest &request) const {
  request.task = task;
  request.streamIndex = streamIndex;
  request.taskIndex = taskIndex;
  if (aclmdlRITaskGetParams(task, &request.originalParams) != ACL_SUCCESS) {
    SK_DLOGI("aclmdlRISpecOptimize keep dynamic task: stream=%u task=%u reason=task parameters unavailable",
             streamIndex, taskIndex);
    return false;
  }
  const auto &params = request.originalParams.kernelTaskParams;
  if (params.funcHandle == nullptr) {
    SK_DLOGI("aclmdlRISpecOptimize keep dynamic task: stream=%u task=%u reason=null function handle", streamIndex,
             taskIndex);
    return false;
  }
  std::array<char, MAX_SCOPE_NAME_LEN> name{};
  if (aclrtGetFunctionName(params.funcHandle, name.size(), name.data()) != ACL_SUCCESS || name[0] == '\0' ||
      std::memchr(name.data(), '\0', name.size()) == nullptr) {
    SK_DLOGI("aclmdlRISpecOptimize keep dynamic task: stream=%u task=%u reason=kernel entry unavailable", streamIndex,
             taskIndex);
    return false;
  }
  request.kernelEntry = name.data();
  aclrtBinHandle binary = nullptr;
  void *binaryData = nullptr;
  uint32_t binarySize = 0;
  if (aclrtFunctionGetBinary(params.funcHandle, &binary) != ACL_SUCCESS || binary == nullptr ||
      rtGetBinBuffer(binary, RT_BIN_HOST_ADDR, &binaryData, &binarySize) != RT_ERROR_NONE) {
    SK_DLOGI("aclmdlRISpecOptimize keep dynamic task: stream=%u task=%u reason=kernel binary unavailable", streamIndex,
             taskIndex);
    return false;
  }
  if (!ReadSpecResourceId(binaryData, binarySize, request.resourceId)) {
    SK_DLOGI("aclmdlRISpecOptimize keep dynamic task: stream=%u task=%u reason=no usable resource id", streamIndex,
             taskIndex);
    return false;
  }
  size_t count = 0;
  if (aclrtFunctionGetParamCount(params.funcHandle, &count) != ACL_SUCCESS || count == 0 || count > params.argsSize ||
      params.args == nullptr) {
    SK_DLOGI("aclmdlRISpecOptimize invalid arguments: stream=%u task=%u", streamIndex, taskIndex);
    return false;
  }
  // ACLRTC reads the argument values itself, so it needs host addresses; a captured kernel task
  // carries them on the device.
  request.hostArgs.resize(params.argsSize);
  if (aclrtMemcpy(request.hostArgs.data(), request.hostArgs.size(), params.args, params.argsSize,
                  ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
    SK_DLOGI("aclmdlRISpecOptimize keep dynamic task: stream=%u task=%u reason=arguments unavailable on host",
             streamIndex, taskIndex);
    return false;
  }
  request.argumentAddresses.reserve(count);
  request.argumentBytes.reserve(count);
  for (size_t index = 0; index < count; ++index) {
    size_t offset = 0;
    size_t size = 0;
    if (aclrtFunctionGetParamInfo(params.funcHandle, index, &offset, &size) != ACL_SUCCESS || size == 0 ||
        offset > params.argsSize || size > params.argsSize - offset) {
      SK_DLOGI("aclmdlRISpecOptimize invalid parameter range: stream=%u task=%u parameter=%zu", streamIndex, taskIndex,
               index);
      return false;
    }
    request.argumentAddresses.push_back(request.hostArgs.data() + offset);
    request.argumentBytes.push_back(size);
  }
  request.basicOptionStorage = featureManager.Get<Feature::BASIC_COMPILE_OPTIONS>(request.kernelEntry);
  request.skOptionStorage = featureManager.Get<Feature::SK_COMPILE_OPTIONS>(request.kernelEntry);
  request.outputElfPath = (fs::path(std::string(outputDirectory)) /
                           ("stream_" + std::to_string(streamIndex) + "_task_" + std::to_string(taskIndex) + ".elf"))
                              .string();
  return true;
}

bool AclSpecOptimizer::SpecializeRequest(KernelSpecRequest &request) const {
  request.BindApiRequest();
  aclError result = aclrtcKernelSpecialization(&request.apiRequest, request.outputElfPath.c_str());
  if (result != ACL_SUCCESS) {
    SK_DLOGW("aclmdlRISpecOptimize specialization failed: stream=%u task=%u ret=%d", request.streamIndex,
             request.taskIndex, result);
    return false;
  }
  aclrtBinHandle binary = nullptr;
  result = aclrtBinaryLoadFromFile(request.outputElfPath.c_str(), nullptr, &binary);
  request.binary.reset(binary);
  if (result != ACL_SUCCESS || binary == nullptr) {
    SK_DLOGW("aclmdlRISpecOptimize binary load failed: stream=%u task=%u ret=%d", request.streamIndex,
             request.taskIndex, result);
    return false;
  }
  result = aclrtBinaryGetFunction(binary, request.kernelEntry.c_str(), &request.specializedFunction);
  if (result != ACL_SUCCESS || request.specializedFunction == nullptr) {
    SK_DLOGW("aclmdlRISpecOptimize function lookup failed: stream=%u task=%u ret=%d", request.streamIndex,
             request.taskIndex, result);
    return false;
  }
  return true;
}

aclError AclSpecOptimizer::Apply(aclmdlRI model,
                                 const std::vector<std::unique_ptr<KernelSpecRequest>> &requests) const {
  if (requests.empty()) {
    return ACL_SUCCESS;
  }
  size_t changedCount = 0;
  aclError result = ACL_SUCCESS;
  for (const auto &request : requests) {
    auto params = request->originalParams;
    params.kernelTaskParams.funcHandle = request->specializedFunction;
    ++changedCount;
    result = aclmdlRITaskSetParams(request->task, &params);
    if (result != ACL_SUCCESS) {
      SK_DLOGE("aclmdlRISpecOptimize SetParams failed: model=%p task=%p ret=%d", model, request->task, result);
      break;
    }
  }
  const bool updateAttempted = result == ACL_SUCCESS;
  if (updateAttempted) {
    result = aclmdlRIUpdate(model);
  }
  if (result == ACL_SUCCESS) {
    return ACL_SUCCESS;
  }
  const aclError restoreError = Restore(model, requests, changedCount, updateAttempted);
  SK_DLOGE("aclmdlRISpecOptimize commit failed: model=%p ret=%d restoreRet=%d", model, result, restoreError);
  if (restoreError != ACL_SUCCESS) {
    SK_DLOGE("aclmdlRISpecOptimize model state uncertain; do not execute or optimize: model=%p", model);
    return ACL_ERROR_INTERNAL_ERROR;
  }
  return result == ACL_ERROR_INTERNAL_ERROR ? ACL_ERROR_FAILURE : result;
}

aclError AclSpecOptimizer::Restore(aclmdlRI model, const std::vector<std::unique_ptr<KernelSpecRequest>> &requests,
                                   size_t changedCount, bool updateAttempted) const {
  aclError restoreError = ACL_SUCCESS;
  for (size_t index = 0; index < changedCount; ++index) {
    auto params = requests[index]->originalParams;
    const aclError result = aclmdlRITaskSetParams(requests[index]->task, &params);
    if (result != ACL_SUCCESS) {
      SK_DLOGE("aclmdlRISpecOptimize restore task failed: model=%p task=%p ret=%d", model, requests[index]->task,
               result);
      if (restoreError == ACL_SUCCESS) {
        restoreError = result;
      }
    }
  }
  if (!updateAttempted) {
    return restoreError;
  }
  const aclError updateError = aclmdlRIUpdate(model);
  if (updateError != ACL_SUCCESS) {
    SK_DLOGE("aclmdlRISpecOptimize restore Update failed: model=%p ret=%d", model, updateError);
    if (restoreError == ACL_SUCCESS) {
      restoreError = updateError;
    }
  }
  return restoreError;
}

void KernelSpecRequest::BindApiRequest() {
  BindOptionPointers(basicOptionStorage, basicOptionPointers);
  BindOptionPointers(skOptionStorage, skOptionPointers);
  apiRequest.resourceId = resourceId.c_str();
  apiRequest.kernelEntry = kernelEntry.c_str();
  apiRequest.argsCount = argumentAddresses.size();
  apiRequest.argsAddr = argumentAddresses.data();
  apiRequest.argsBytes = argumentBytes.data();
  apiRequest.options = basicOptionPointers.data();
  apiRequest.optionCount = basicOptionPointers.size();
  apiRequest.skOptions = skOptionPointers.data();
  apiRequest.skOptionCount = skOptionPointers.size();
}

void KernelSpecRequest::UnloadBinary(aclrtBinHandle binary) {
  const aclError result = aclrtBinaryUnLoad(binary);
  if (result != ACL_SUCCESS) {
    SK_DLOGE("aclmdlRISpecOptimize unload binary failed: binary=%p ret=%d", binary, result);
  }
}

ModelSpecResourceManager &ModelSpecResourceManager::GetInstance() {
  static ModelSpecResourceManager instance;
  return instance;
}

aclError ModelSpecResourceManager::RegisterModel(aclmdlRI model, size_t binaryCount) {
  if (model == nullptr || binaryCount == 0) {
    return ACL_ERROR_INVALID_PARAM;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  auto entry = modelBinaries_.find(model);
  if (entry != modelBinaries_.end()) {
    auto &binaries = entry->second;
    if (binaryCount > binaries.max_size() - binaries.size()) {
      SK_DLOGE("aclmdlRISpecOptimize binary storage exhausted: model=%p", model);
      return ACL_ERROR_BAD_ALLOC;
    }
    binaries.reserve(binaries.size() + binaryCount);
    return ACL_SUCCESS;
  }
  std::vector<aclrtBinHandle> binaries;
  binaries.reserve(binaryCount);
  entry = modelBinaries_.emplace(model, std::move(binaries)).first;
  const aclError result = aclmdlRIDestroyRegisterCallback(model, OnModelDestroy, model);
  if (result != ACL_SUCCESS) {
    modelBinaries_.erase(entry);
    SK_DLOGE("aclmdlRISpecOptimize register destroy callback failed: model=%p ret=%d", model, result);
  }
  return result;
}

void ModelSpecResourceManager::RetainBinary(aclmdlRI model, KernelSpecRequest::Binary &binary) {
  std::lock_guard<std::mutex> lock(mutex_);
  modelBinaries_.at(model).push_back(binary.get());
  binary.release();
}

void ModelSpecResourceManager::OnModelDestroy(void *userData) {
  const aclmdlRI model = static_cast<aclmdlRI>(userData);
  if (model == nullptr) {
    SK_DLOGE("aclmdlRISpecOptimize destroy callback has invalid model=%p", userData);
    return;
  }
  std::vector<aclrtBinHandle> binaries;
  {
    auto &manager = GetInstance();
    std::lock_guard<std::mutex> lock(manager.mutex_);
    const auto entry = manager.modelBinaries_.find(model);
    if (entry == manager.modelBinaries_.end()) {
      return;
    }
    binaries.swap(entry->second);
    manager.modelBinaries_.erase(entry);
  }
  for (const auto binary : binaries) {
    KernelSpecRequest::UnloadBinary(binary);
  }
}

aclError SpecOutputDirectory::Create(uint32_t modelId) {
  if (created_) {
    return ACL_ERROR_INVALID_PARAM;
  }
  boost::system::error_code error;
  const char *temporaryRoot = std::getenv("TMPDIR");
  auto root =
      temporaryRoot != nullptr && temporaryRoot[0] != '\0' ? fs::path(temporaryRoot) : fs::temp_directory_path(error);
  if (error) {
    SK_DLOGE("aclmdlRISpecOptimize temporary root unavailable: modelId=%u ret=%d", modelId, error.value());
    return ACL_ERROR_FAILURE;
  }
  root = fs::absolute(root, error);
  if (error) {
    return ACL_ERROR_FAILURE;
  }
  path_ = (root / ("acl_spec_optimize_" + std::to_string(getpid()) + "_" + std::to_string(syscall(SYS_gettid)) + "_" +
                   std::to_string(modelId) + "_XXXXXX"))
              .string();
  if (mkdtemp(path_.data()) == nullptr) {
    SK_DLOGE("aclmdlRISpecOptimize directory creation failed: modelId=%u errno=%d", modelId, errno);
    path_.clear();
    return ACL_ERROR_FAILURE;
  }
  created_ = true;
  return ACL_SUCCESS;
}

SpecOutputDirectory::~SpecOutputDirectory() {
  if (!created_) {
    return;
  }
  try {
    boost::system::error_code error;
    fs::remove_all(path_, error);
    if (error) {
      SK_DLOGW("aclmdlRISpecOptimize directory cleanup failed: path=%s ret=%d", path_.c_str(), error.value());
    }
  } catch (...) {
    SK_DLOGW("aclmdlRISpecOptimize directory cleanup failed: path=%s", path_.c_str());
  }
}

bool ReadSpecResourceId(const void *binary, size_t binarySize, std::string &resourceId) {
  if (binary == nullptr || binarySize < sizeof(Elf64_Ehdr)) {
    SK_DLOGW("aclmdlRISpecOptimize cannot read resource id: binary is null or smaller than an ELF64 header");
    return false;
  }
  const auto *bytes = static_cast<const unsigned char *>(binary);
  const auto containsRange = [binarySize](uint64_t offset, uint64_t length) {
    return offset <= binarySize && length <= binarySize - offset;
  };
  Elf64_Ehdr header{};
  if (memcpy_s(&header, sizeof(header), bytes, sizeof(header)) != EOK) {
    SK_DLOGW("aclmdlRISpecOptimize cannot read resource id: the ELF header could not be copied");
    return false;
  }
  if (std::memcmp(header.e_ident, ELFMAG, SELFMAG) != 0 || header.e_ident[EI_CLASS] != ELFCLASS64 ||
      header.e_ident[EI_DATA] != ELFDATA2LSB || header.e_ident[EI_VERSION] != EV_CURRENT ||
      header.e_version != EV_CURRENT || header.e_ehsize != sizeof(header) || header.e_shentsize != sizeof(Elf64_Shdr) ||
      header.e_shnum == 0 || header.e_shstrndx == SHN_UNDEF || header.e_shstrndx >= header.e_shnum ||
      header.e_shoff < sizeof(header) ||
      !containsRange(header.e_shoff, static_cast<uint64_t>(header.e_shnum) * sizeof(Elf64_Shdr))) {
    SK_DLOGW(
        "aclmdlRISpecOptimize cannot read resource id: not a little-endian ELF64 object with a usable section table");
    return false;
  }
  const auto readSection = [&](size_t index, Elf64_Shdr &section) {
    const auto *source = bytes + header.e_shoff + index * sizeof(section);
    return memcpy_s(&section, sizeof(section), source, sizeof(section)) == EOK;
  };
  Elf64_Shdr names{};
  if (!readSection(header.e_shstrndx, names)) {
    SK_DLOGW("aclmdlRISpecOptimize cannot read resource id: the section name table could not be copied");
    return false;
  }
  if (names.sh_type != SHT_STRTAB || !containsRange(names.sh_offset, names.sh_size)) {
    SK_DLOGW("aclmdlRISpecOptimize cannot read resource id: section name table is not a readable string table");
    return false;
  }
  const auto *nameData = reinterpret_cast<const char *>(bytes + names.sh_offset);
  std::string_view found;
  for (size_t index = 0; index < header.e_shnum; ++index) {
    Elf64_Shdr section{};
    if (!readSection(index, section)) {
      SK_DLOGW("aclmdlRISpecOptimize cannot read resource id: a section header could not be copied");
      return false;
    }
    if (section.sh_type == SHT_NULL || section.sh_type == SHT_NOBITS || section.sh_name >= names.sh_size) {
      continue;
    }
    const char *name = nameData + section.sh_name;
    const auto *end = static_cast<const char *>(std::memchr(name, '\0', names.sh_size - section.sh_name));
    if (end == nullptr || std::string_view(name, end - name) != SPEC_RESOURCE_ID_SECTION_NAME) {
      continue;
    }
    if (section.sh_type != SHT_PROGBITS || !containsRange(section.sh_offset, section.sh_size)) {
      SK_DLOGW("aclmdlRISpecOptimize cannot read resource id: section .ascend.meta is not readable progbits");
      return false;
    }
    const auto *data = bytes + section.sh_offset;
    size_t offset = 0;
    while (offset < section.sh_size) {
      if (section.sh_size - offset < sizeof(uint16_t) * 2) {
        SK_DLOGW("aclmdlRISpecOptimize cannot read resource id: trailing bytes are too short for a TLV head");
        return false;
      }
      uint16_t type = 0;
      uint16_t length = 0;
      if (memcpy_s(&type, sizeof(type), data + offset, sizeof(type)) != EOK ||
          memcpy_s(&length, sizeof(length), data + offset + sizeof(type), sizeof(length)) != EOK) {
        SK_DLOGW("aclmdlRISpecOptimize cannot read resource id: a TLV head could not be copied");
        return false;
      }
      offset += sizeof(type) + sizeof(length);
      if (length > section.sh_size - offset) {
        SK_DLOGW("aclmdlRISpecOptimize cannot read resource id: a TLV value runs past the end of .ascend.meta");
        return false;
      }
      if (type == SPEC_RESOURCE_ID_TLV_TYPE) {
        if (!found.empty()) {
          SK_DLOGW("aclmdlRISpecOptimize cannot read resource id: more than one resource id entry");
          return false;
        }
        if (length != SPEC_RESOURCE_ID_TLV_LENGTH) {
          SK_DLOGW("aclmdlRISpecOptimize cannot read resource id: resource id entry has an unexpected length");
          return false;
        }
        const std::string_view value(reinterpret_cast<const char *>(data + offset), length);
        for (const char digit : value) {
          if (!((digit >= '0' && digit <= '9') || (digit >= 'a' && digit <= 'f'))) {
            SK_DLOGW("aclmdlRISpecOptimize cannot read resource id: resource id is not lowercase hexadecimal");
            return false;
          }
        }
        found = value;
      }
      offset += length;
    }
  }
  if (found.empty()) {
    return false;
  }
  resourceId.assign(found);
  return true;
}
