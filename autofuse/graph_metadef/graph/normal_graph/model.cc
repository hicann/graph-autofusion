/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/model.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <sys/types.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include "graph/debug/ge_attr_define.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph_metadef/common/ge_common/util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/model_serialize.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "mmpa/mmpa_api.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/ge_ir_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "common/checker.h"
#include "proto/af_ir.pb.h"

namespace {
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;

const int32_t DEFAULT_VERSION = 1;
const int32_t ACCESS_PERMISSION_BITS = 256;  // 0400;
static af::ModelSerialize SERIALIZE;
}  // namespace

namespace af {
static std::string GetStrError() {
  constexpr size_t kMaxErrLen = 128U;
  char_t err_buf[kMaxErrLen + 1U] = {};
  const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrLen);
  return FormatErrnoReason(mmGetErrorCode(), err_msg);
}

void Model::Init() {
  (void)AttrUtils::SetInt(this, ATTR_MODEL_MEMORY_SIZE, 0);
  (void)AttrUtils::SetInt(this, ATTR_MODEL_P2P_MEMORY_SIZE, 0);
  (void)AttrUtils::SetInt(this, ATTR_MODEL_STREAM_NUM, 0);
  (void)AttrUtils::SetInt(this, ATTR_MODEL_EVENT_NUM, 0);
  (void)AttrUtils::SetInt(this, ATTR_MODEL_LABEL_NUM, 0);
  (void)AttrUtils::SetInt(this, ATTR_MODEL_WEIGHT_SIZE, 0);
  (void)AttrUtils::SetStr(this, ATTR_MODEL_TARGET_TYPE, TARGET_TYPE_MINI);
  version_ = 0U;
}

Model::Model() : AttrHolder() {
  Init();
}

Model::Model(const std::string &name, const std::string &custom_version)
    : AttrHolder(), name_(name), version_(static_cast<uint32_t>(DEFAULT_VERSION)), platform_version_(custom_version) {
  Init();
}

Model::Model(const char_t *name, const char_t *custom_version)
    : Model(std::string(name == nullptr ? "" : name), std::string(custom_version == nullptr ? "" : custom_version)) {}

std::string Model::GetName() const {
  return name_;
}

void Model::SetName(const std::string &name) {
  name_ = name;
}

uint32_t Model::GetVersion() const {
  return version_;
}

std::string Model::GetPlatformVersion() const {
  return platform_version_;
}

void Model::SetGraph(const ComputeGraphPtr &graph) {
  graph_ = graph;
}

const ComputeGraphPtr Model::GetGraph() const {
  return graph_;
}

graphStatus Model::Save(Buffer &buffer, const bool is_dump) const {
  buffer = SERIALIZE.SerializeModel(*this, is_dump);
  return (buffer.GetSize() > 0U) ? GRAPH_SUCCESS : GRAPH_FAILED;
}

graphStatus Model::SaveWithoutSeparate(Buffer &buffer, const bool is_dump) const {
  std::string path;
  buffer = SERIALIZE.SerializeModel(*this, path, false, is_dump);
  return (buffer.GetSize() > 0U) ? GRAPH_SUCCESS : GRAPH_FAILED;
}

graphStatus Model::Save(Buffer &buffer, const std::string &path, const bool is_dump) const {
  buffer = SERIALIZE.SerializeModel(*this, path, true, is_dump);
  return (buffer.GetSize() > 0U) ? GRAPH_SUCCESS : GRAPH_FAILED;
}

graphStatus Model::SaveSeparateModel(Buffer &buffer, const std::string &path, const bool is_dump) const {
  buffer = SERIALIZE.SerializeSeparateModel(*this, path, is_dump);
  return (buffer.GetSize() > 0U) ? GRAPH_SUCCESS : GRAPH_FAILED;
}

graphStatus Model::Save(proto::ModelDef &model_def, const bool is_dump) const {
  return SERIALIZE.SerializeModel(*this, is_dump, model_def);
}

void Model::SetAttr(const ProtoAttrMap &attrs) {
  attrs_ = attrs;
}

graphStatus Model::Load(const uint8_t *data, size_t len, Model &model) {
  return SERIALIZE.UnserializeModel(data, len, model) ? GRAPH_SUCCESS : GRAPH_FAILED;
}

graphStatus Model::LoadWithMultiThread(const uint8_t *data, size_t len, Model &model) {
  return SERIALIZE.UnserializeModel(data, len, model, true) ? GRAPH_SUCCESS : GRAPH_FAILED;
}

graphStatus Model::Load(af::proto::ModelDef &model_def, const std::string &path) {
  return SERIALIZE.UnserializeModel(model_def, *this, path) ? GRAPH_SUCCESS : GRAPH_FAILED;
}

graphStatus Model::Load(af::proto::ModelDef &model_def) {
  return SERIALIZE.UnserializeModel(model_def, *this) ? GRAPH_SUCCESS : GRAPH_FAILED;
}

graphStatus Model::SaveToFile(const std::string &file_name, const bool force_separate) const {
  Buffer buffer;
  std::string dir_path;
  std::string file;
  SplitFilePath(file_name, dir_path, file);
  if (!dir_path.empty()) {
    GE_ASSERT_TRUE((CreateDir(dir_path) == 0), "Create directory failed, path: %s.", file_name.c_str());
  } else {
    GE_ASSERT_SUCCESS(GetAscendWorkPath(dir_path));
    if (dir_path.empty()) {
      dir_path = "./";
    }
  }
  std::string real_path = RealPath(dir_path.c_str());
  GE_ASSERT_TRUE(!real_path.empty(), "Path: %s is empty", file_name.c_str());
  real_path = real_path + "/" + file;

  graphStatus ret = GRAPH_SUCCESS;
  if (!force_separate) {
    ret = (*this).Save(buffer, real_path);
  } else {
    ret = (*this).SaveSeparateModel(buffer, real_path);
  }
  if (ret != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "[Save][Data] to file:%s fail.", file_name.c_str());
    GELOGE(ret, "[Save][Data] to file:%s fail.", file_name.c_str());
    return ret;
  }
  // Write file
  if (buffer.GetData() != nullptr) {
    af::proto::ModelDef ge_proto;
    const std::string str(PtrToPtr<uint8_t, char_t>(buffer.GetData()), buffer.GetSize());
    if (!ge_proto.ParseFromString(str)) {
      return GRAPH_FAILED;
    }
    const int32_t fd = mmOpen2(&real_path[0],
                               static_cast<int32_t>(static_cast<uint32_t>(M_WRONLY) | static_cast<uint32_t>(M_CREAT) |
                                                    static_cast<uint32_t>(O_TRUNC)),
                               static_cast<uint32_t>(ACCESS_PERMISSION_BITS));
    if (fd < 0) {
      const std::string reason = GetStrError();
      REPORT_INNER_ERR_MSG("E18888", "open file:%s failed, reason:%s", &real_path[0], reason.c_str());
      GELOGE(GRAPH_FAILED, "[Open][File] %s failed, error:%s ", &real_path[0], reason.c_str());
      return GRAPH_FAILED;
    }
    const bool result = ge_proto.SerializeToFileDescriptor(fd);
    if (!result) {
      REPORT_INNER_ERR_MSG("E18888", "SerializeToFileDescriptor failed, file:%s.", &real_path[0]);
      GELOGE(GRAPH_FAILED, "[Call][SerializeToFileDescriptor] failed, file:%s.", &real_path[0]);
      if (mmClose(fd) != 0) {
        const std::string reason = GetStrError();
        REPORT_INNER_ERR_MSG("E18888", "close file:%s fail, reason:%s.", &real_path[0], reason.c_str());
        GELOGE(GRAPH_FAILED, "[Close][File] %s fail, error:%s.", &real_path[0], reason.c_str());
        return GRAPH_FAILED;
      }
      return GRAPH_FAILED;
    }
    if (mmClose(fd) != 0) {
      const std::string reason = GetStrError();
      REPORT_INNER_ERR_MSG("E18888", "close file:%s fail, reason:%s.", &real_path[0], reason.c_str());
      GELOGE(GRAPH_FAILED, "[Close][File] %s fail, error:%s.", &real_path[0], reason.c_str());
      return GRAPH_FAILED;
    }
    if (!result) {
      REPORT_INNER_ERR_MSG("E18888", "SerializeToFileDescriptor failed, file:%s.", &real_path[0]);
      GELOGE(GRAPH_FAILED, "[Call][SerializeToFileDescriptor] failed, file:%s.", &real_path[0]);
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

bool Model::IsValid() const {
  return graph_ != nullptr;
}

graphStatus Model::LoadFromFile(const std::string &file_name) {
  char_t real_path[MMPA_MAX_PATH] = {};
  if (strnlen(file_name.c_str(), sizeof(real_path)) >= sizeof(real_path)) {
    return GRAPH_FAILED;
  }
  const INT32 result = mmRealPath(file_name.c_str(), &real_path[0], MMPA_MAX_PATH);
  if (result != EN_OK) {
    const std::string reason = GetStrError();
    REPORT_INNER_ERR_MSG("E18888", "get realpath failed for %s, reason:%s.", file_name.c_str(), reason.c_str());
    GELOGE(GRAPH_FAILED, "[Get][RealPath] failed for %s, error:%s.", file_name.c_str(), reason.c_str());
    return GRAPH_FAILED;
  }
  const int32_t fd = mmOpen(&real_path[0], M_RDONLY);
  if (fd < 0) {
    const std::string reason = GetStrError();
    REPORT_INNER_ERR_MSG("E18888", "open file:%s failed, reason:%s", &real_path[0], reason.c_str());
    GELOGE(GRAPH_FAILED, "[Open][File] %s failed, error:%s", &real_path[0], reason.c_str());
    return GRAPH_FAILED;
  }

  af::proto::ModelDef model_def;
  const bool ret = model_def.ParseFromFileDescriptor(fd);
  if (!ret) {
    REPORT_INNER_ERR_MSG("E18888", "ParseFromFileDescriptor failed, file:%s.", &real_path[0]);
    GELOGE(GRAPH_FAILED, "[Call][ParseFromFileDescriptor] failed, file:%s.", &real_path[0]);
    if (mmClose(fd) != 0) {
      const std::string reason = GetStrError();
      REPORT_INNER_ERR_MSG("E18888", "close file:%s fail, reason:%s.", &real_path[0], reason.c_str());
      GELOGE(GRAPH_FAILED, "[Close][File] %s fail. error:%s", &real_path[0], reason.c_str());
      return GRAPH_FAILED;
    }
    return GRAPH_FAILED;
  }
  if (mmClose(fd) != 0) {
    const std::string reason = GetStrError();
    REPORT_INNER_ERR_MSG("E18888", "close file:%s fail, reason:%s.", &real_path[0], reason.c_str());
    GELOGE(GRAPH_FAILED, "[Close][File] %s fail. error:%s", &real_path[0], reason.c_str());
    return GRAPH_FAILED;
  }
  if (!ret) {
    REPORT_INNER_ERR_MSG("E18888", "ParseFromFileDescriptor failed, file:%s.", &real_path[0]);
    GELOGE(GRAPH_FAILED, "[Call][ParseFromFileDescriptor] failed, file:%s.", &real_path[0]);
    return GRAPH_FAILED;
  }
  std::string path(real_path);
  return Load(model_def, file_name);
}

ProtoAttrMap &Model::MutableAttrMap() {
  return attrs_;
}

ConstProtoAttrMap &Model::GetAttrMap() const {
  return attrs_;
}
}  // namespace af

#ifdef __cplusplus
extern "C" {
#endif

af::Status GeApiWrapper_ModelSaveToString(const af::Graph &graph, const std::string &node_name,
                                          std::string &model_str) {
  std::string model_name = "onnx_compute_model_" + node_name;
  af::Buffer model_buf;
  af::Model onnx_model(model_name.c_str(), "");
  onnx_model.SetGraph(af::GraphUtilsEx::GetComputeGraph(graph));
  GE_ASSERT_SUCCESS(onnx_model.Save(model_buf, false), "[GEOP] node:%s Onnx Model Serialized Failed.",
                    node_name.c_str());
  model_str = std::string(reinterpret_cast<const char *>(model_buf.GetData()), model_buf.GetSize());
  return af::SUCCESS;
}

#ifdef __cplusplus
}
#endif
