/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_INC_EXE_GRAPH_RUNTIME_DFX_INFO_FILLER_H
#define METADEF_INC_EXE_GRAPH_RUNTIME_DFX_INFO_FILLER_H

#include <string>
#include <vector>
#include <graph/ge_error_codes_af.h>

namespace gert {
enum class NodeProfInfoType : uint32_t {
  kOriginalNode,
  kCmoPreFetch,
  kCmoInvalidate,
  kCmoWriteBack,
  kMixVectorCore,
  kNodeTypeMax
};

class ProfilingInfoWrapper {
 public:
  virtual ~ProfilingInfoWrapper() = default;

  virtual void SetBlockDim(uint32_t block_dim) {
    (void)block_dim;
  }

  virtual void SetBlockDim(const uint32_t block_dim, const NodeProfInfoType prof_info_type) {
    (void)block_dim;
    (void)prof_info_type;
  }

  virtual void SetMixLaunchEnable(const bool mix_launch_enable) {
    (void)mix_launch_enable;
  }

  virtual void SetLaunchTimeStamp(const uint64_t begin_time, const uint64_t end_time,
                                  const NodeProfInfoType prof_info_type) {
    (void)begin_time;
    (void)end_time;
    (void)prof_info_type;
  }

  virtual void SetBlockDimForAtomic(uint32_t block_dim) {
    (void)block_dim;
  }

  virtual ge::graphStatus FillShapeInfo(const std::vector<std::vector<int64_t>> &input_shapes,
                                        const std::vector<std::vector<int64_t>> &output_shapes) {
    (void)input_shapes;
    (void)output_shapes;
    return ge::GRAPH_SUCCESS;
  }
};

class DataDumpInfoWrapper {
 public:
  virtual ~DataDumpInfoWrapper() = default;
  virtual ge::graphStatus CreateFftsCtxInfo(uint32_t thread_id, uint32_t context_id) = 0;
  virtual ge::graphStatus AddFftsCtxAddr(uint32_t thread_id, bool is_input, uint64_t address, uint64_t size) = 0;
  virtual void AddWorkspace(uintptr_t addr, int64_t bytes) = 0;
  virtual bool SetStrAttr(const std::string &name, const std::string &value) = 0;
};

class ExceptionDumpInfoWrapper {
 public:
  virtual ~ExceptionDumpInfoWrapper() = default;
  virtual void SetTilingData(uintptr_t addr, size_t size) = 0;
  virtual void SetTilingKey(uint32_t key) = 0;
  virtual void SetHostArgs(uintptr_t addr, size_t size) = 0;
  virtual void SetDeviceArgs(uintptr_t addr, size_t size) = 0;
  virtual void AddWorkspace(uintptr_t addr, int64_t bytes) = 0;
};
}  // namespace gert

#endif
