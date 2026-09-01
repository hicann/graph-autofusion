/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_MODEL_H_
#define INC_GRAPH_MODEL_H_

#include <memory>
#include <string>
#include "graph/attr_store.h"
#include "graph/detail/attributes_holder.h"
#include "graph/ge_attr_value.h"
#include "graph/compute_graph.h"

namespace af {
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Model : public AttrHolder {
 public:
  Model();

  ~Model() override = default;

  Model(const std::string &name, const std::string &custom_version);

  Model(const char_t *name, const char_t *custom_version);

  std::string GetName() const;

  void SetName(const std::string &name);

  uint32_t GetVersion() const;

  void SetVersion(const uint32_t version) {
    version_ = version;
  }

  std::string GetPlatformVersion() const;

  void SetPlatformVersion(const std::string version) {
    platform_version_ = version;
  }

  const ComputeGraphPtr GetGraph() const;

  void SetGraph(const ComputeGraphPtr &graph);

  void SetAttr(const ProtoAttrMap &attrs);

  using AttrHolder::GetAllAttrNames;
  using AttrHolder::GetAllAttrs;
  using AttrHolder::GetAttr;
  using AttrHolder::HasAttr;
  using AttrHolder::SetAttr;

  graphStatus Save(Buffer &buffer, const bool is_dump = false) const;
  graphStatus Save(proto::ModelDef &model_def, const bool is_dump = false) const;
  graphStatus SaveWithoutSeparate(Buffer &buffer, const bool is_dump = false) const;
  graphStatus SaveToFile(const std::string &file_name, const bool force_separate = false) const;
  // Model will be rewrite
  static graphStatus Load(const uint8_t *data, size_t len, Model &model);
  /**
   * 兼容旧调用方的模型加载接口。当前统一采用串行反序列化。
   * @param data 模型序列化后的内容指针
   * @param len 模型序列化后的内容长度
   * @param model 模型加载后的承载对象
   * @return 成功返回GRAPH_SUCCESS, 失败返回GRAPH_FAILED
   */
  static graphStatus LoadWithMultiThread(const uint8_t *data, size_t len, Model &model);
  graphStatus Load(proto::ModelDef &model_def);
  graphStatus LoadFromFile(const std::string &file_name);

  bool IsValid() const;

 protected:
  ConstProtoAttrMap &GetAttrMap() const override;
  ProtoAttrMap &MutableAttrMap() override;

 private:
  void Init();
  graphStatus Load(proto::ModelDef &model_def, const std::string &path);
  graphStatus Save(Buffer &buffer, const std::string &path, const bool is_dump = false) const;
  graphStatus SaveSeparateModel(Buffer &buffer, const std::string &path, const bool is_dump = false) const;
  AttrStore attrs_;
  friend class ModelSerializeImp;
  friend class GraphDebugImp;
  friend class OnnxUtils;
  friend class ModelHelper;
  friend class ModelBuilder;
  std::string name_;
  uint32_t version_;
  std::string platform_version_{""};
  ComputeGraphPtr graph_;
};
using ModelPtr = std::shared_ptr<Model>;
}  // namespace af

#endif  // INC_GRAPH_MODEL_H_
