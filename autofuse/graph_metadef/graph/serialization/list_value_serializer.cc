/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "list_value_serializer.h"
#include <vector>
#include <string>
#include <functional>

#include "framework/common/debug/ge_log.h"
#include "graph/types_af.h"
#include "graph/utils/attr_utils.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "tensor_desc_serializer.h"
#include "tensor_serializer.h"
#include "named_attrs_serializer.h"
#include "graph_serializer.h"
#include "graph/ge_tensor.h"
#include "graph/def_types.h"

namespace af {
using ComputeGraphPtr = std::shared_ptr<ComputeGraph>;
using GeTensorPtr = std::shared_ptr<GeTensor>;
using ListValue = GeIrAttrDef::ListValue;
using std::placeholders::_1;
using std::placeholders::_2;

graphStatus ListValueSerializer::Serialize(const AnyValue &av, GeIrAttrDef &def) {
  const static std::map<AnyValue::ValueType, std::function<graphStatus(const AnyValue &, GeIrAttrDef &)>>
      type_serializer_map = {
          {AnyValue::VT_LIST_INT, std::bind(&ListValueSerializer::SerializeListInt, _1, _2)},
          {AnyValue::VT_LIST_FLOAT, std::bind(&ListValueSerializer::SerializeListFloat, _1, _2)},
          {AnyValue::VT_LIST_BOOL, std::bind(&ListValueSerializer::SerializeListBool, _1, _2)},
          {AnyValue::VT_LIST_BYTES, std::bind(&ListValueSerializer::SerializeListBuffer, _1, _2)},
          {AnyValue::VT_LIST_DATA_TYPE, std::bind(&ListValueSerializer::SerializeListDataType, _1, _2)},
          {AnyValue::VT_LIST_STRING, std::bind(&ListValueSerializer::SerializeListString, _1, _2)},
          {AnyValue::VT_LIST_NAMED_ATTRS, std::bind(&ListValueSerializer::SerializeListNamedAttrs, _1, _2)},
          {AnyValue::VT_LIST_TENSOR_DESC, std::bind(&ListValueSerializer::SerializeListGeTensorDesc, _1, _2)},
          {AnyValue::VT_LIST_TENSOR, std::bind(&ListValueSerializer::SerializeListGeTensor, _1, _2)},
          {AnyValue::VT_LIST_GRAPH, std::bind(&ListValueSerializer::SerializeListGraphDef, _1, _2)},
      };

  const auto iter = type_serializer_map.find(av.GetValueType());
  if (iter == type_serializer_map.end()) {
    GELOGE(GRAPH_FAILED, "Value type [%d] is not supported.", static_cast<int32_t>(av.GetValueType()));
    return GRAPH_FAILED;
  }
  return iter->second(av, def);
}
graphStatus ListValueSerializer::Deserialize(const GeIrAttrDef &def, AnyValue &av) {
  const static std::map<ListValue::ListValueType, std::function<graphStatus(const GeIrAttrDef &def, AnyValue &av)>>
      type_deserializer_map = {
          {ListValue::VT_LIST_INT, std::bind(&ListValueSerializer::DeserializeListInt, _1, _2)},
          {ListValue::VT_LIST_FLOAT, std::bind(&ListValueSerializer::DeserializeListFloat, _1, _2)},
          {ListValue::VT_LIST_STRING, std::bind(&ListValueSerializer::DeserializeListString, _1, _2)},
          {ListValue::VT_LIST_BYTES, std::bind(&ListValueSerializer::DeserializeListBuffer, _1, _2)},
          {ListValue::VT_LIST_BOOL, std::bind(&ListValueSerializer::DeserializeListBool, _1, _2)},
          {ListValue::VT_LIST_DATA_TYPE, std::bind(&ListValueSerializer::DeserializeListDataType, _1, _2)},
          {ListValue::VT_LIST_NAMED_ATTRS, std::bind(&ListValueSerializer::DeserializeListNamedAttrs, _1, _2)},
          {ListValue::VT_LIST_TENSOR_DESC, std::bind(&ListValueSerializer::DeserializeListGeTensorDesc, _1, _2)},
          {ListValue::VT_LIST_TENSOR, std::bind(&ListValueSerializer::DeserializeListGeTensor, _1, _2)},
          {ListValue::VT_LIST_GRAPH, std::bind(&ListValueSerializer::DeserializeListGraphDef, _1, _2)},
      };

  const auto iter = type_deserializer_map.find(def.list().val_type());
  if (iter == type_deserializer_map.end()) {
    GELOGE(GRAPH_FAILED, "Value type [%d] is not supported.", static_cast<int32_t>(def.list().val_type()));
    return GRAPH_FAILED;
  }
  return iter->second(def, av);
}

graphStatus ListValueSerializer::SerializeListInt(const AnyValue &av, GeIrAttrDef &def) {
  std::vector<int64_t> list_val;
  const graphStatus ret = av.GetValue(list_val);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to get list_int attr.");
    return GRAPH_FAILED;
  }
  const auto mutable_list = def.mutable_list();
  GE_CHECK_NOTNULL(mutable_list);
  mutable_list->clear_i();
  for (const auto val : list_val) {
    mutable_list->add_i(val);
  }
  mutable_list->set_val_type(GeIrAttrDef::ListValue::VT_LIST_INT);
  return GRAPH_SUCCESS;
}

graphStatus ListValueSerializer::SerializeListString(const AnyValue &av, GeIrAttrDef &def) {
  std::vector<std::string> list_val;
  const graphStatus ret = av.GetValue(list_val);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to get list_string attr.");
    return GRAPH_FAILED;
  }
  const auto mutable_list = def.mutable_list();
  GE_CHECK_NOTNULL(mutable_list);
  mutable_list->clear_s();
  for (const auto &val : list_val) {
    mutable_list->add_s(val);
  }
  mutable_list->set_val_type(GeIrAttrDef::ListValue::VT_LIST_STRING);
  return GRAPH_SUCCESS;
}

graphStatus ListValueSerializer::SerializeListFloat(const AnyValue &av, GeIrAttrDef &def) {
  std::vector<float> list_val;
  const graphStatus ret = av.GetValue(list_val);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to get list_float attr.");
    return GRAPH_FAILED;
  }
  const auto mutable_list = def.mutable_list();
  GE_CHECK_NOTNULL(mutable_list);
  mutable_list->clear_f();
  for (const auto val : list_val) {
    mutable_list->add_f(val);
  }
  mutable_list->set_val_type(GeIrAttrDef::ListValue::VT_LIST_FLOAT);

  return GRAPH_SUCCESS;
}

graphStatus ListValueSerializer::SerializeListBool(const AnyValue &av, GeIrAttrDef &def) {
  std::vector<bool> list_val;
  const graphStatus ret = av.GetValue(list_val);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to get list_bool attr.");
    return GRAPH_FAILED;
  }
  const auto mutable_list = def.mutable_list();
  GE_CHECK_NOTNULL(mutable_list);
  mutable_list->clear_b();
  for (const auto val : list_val) {
    mutable_list->add_b(val);
  }
  mutable_list->set_val_type(GeIrAttrDef::ListValue::VT_LIST_BOOL);

  return GRAPH_SUCCESS;
}

graphStatus ListValueSerializer::SerializeListGeTensorDesc(const AnyValue &av, GeIrAttrDef &def) {
  std::vector<af::GeTensorDesc> list_val;
  const graphStatus ret = av.GetValue(list_val);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to get list_tensor_desc attr.");
    return GRAPH_FAILED;
  }
  const auto mutable_list = def.mutable_list();
  GE_CHECK_NOTNULL(mutable_list);
  mutable_list->clear_td();
  for (const auto &val : list_val) {
    const auto attr_proto = mutable_list->add_td();
    GE_CHECK_NOTNULL(attr_proto);
    GeTensorSerializeUtils::GeTensorDescAsProto(val, attr_proto);
  }

  mutable_list->set_val_type(GeIrAttrDef::ListValue::VT_LIST_TENSOR_DESC);

  return GRAPH_SUCCESS;
}

graphStatus ListValueSerializer::SerializeListGeTensor(const AnyValue &av, GeIrAttrDef &def) {
  std::vector<GeTensor> list_val;
  const graphStatus ret = av.GetValue(list_val);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to get list_tensor attr_value.");
    return GRAPH_FAILED;
  }
  const auto mutable_list = def.mutable_list();
  GE_CHECK_NOTNULL(mutable_list);
  mutable_list->clear_t();
  for (const auto &val : list_val) {
    const auto attr_proto = mutable_list->add_t();
    GE_CHECK_NOTNULL(attr_proto);
    GeTensorSerializeUtils::GeTensorAsProto(val, attr_proto);
  }

  mutable_list->set_val_type(GeIrAttrDef::ListValue::VT_LIST_TENSOR);

  return GRAPH_SUCCESS;
}
graphStatus ListValueSerializer::SerializeListBuffer(const AnyValue &av, GeIrAttrDef &def) {
  std::vector<Buffer> list_val;
  const graphStatus ret = av.GetValue(list_val);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to get list_buffer attr.");
    return GRAPH_FAILED;
  }
  const auto mutable_list = def.mutable_list();
  GE_CHECK_NOTNULL(mutable_list);
  mutable_list->clear_bt();
  for (auto val : list_val) {
    if ((val.GetData() != nullptr) && (val.size() > 0U)) {
      mutable_list->add_bt(val.GetData(), val.GetSize());
    }
  }
  mutable_list->set_val_type(GeIrAttrDef::ListValue::VT_LIST_BYTES);

  return GRAPH_SUCCESS;
}

graphStatus ListValueSerializer::SerializeListGraphDef(const AnyValue &av, GeIrAttrDef &def) {
  std::vector<proto::GraphDef> list_val;
  const graphStatus ret = av.GetValue(list_val);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to get list_graph attr_value.");
    return GRAPH_FAILED;
  }

  const auto mutable_list = def.mutable_list();
  GE_CHECK_NOTNULL(mutable_list);
  mutable_list->clear_g();
  for (const auto &graph : list_val) {
    const auto mutable_graph = mutable_list->add_g();
    GE_CHECK_NOTNULL(mutable_graph);
    *mutable_graph = graph;
  }

  mutable_list->set_val_type(GeIrAttrDef::ListValue::VT_LIST_GRAPH);

  return GRAPH_SUCCESS;
}

graphStatus ListValueSerializer::SerializeListNamedAttrs(const AnyValue &av, GeIrAttrDef &def) {
  std::vector<af::NamedAttrs> list_val;
  const graphStatus ret = av.GetValue(list_val);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to get list_named_attr attr.");
    return GRAPH_FAILED;
  }
  const auto mutable_list = def.mutable_list();
  GE_CHECK_NOTNULL(mutable_list);
  mutable_list->clear_na();
  const auto attr_serializer = AttrSerializerRegistry::GetInstance().GetSerializer(GetTypeId<af::NamedAttrs>());
  const auto named_attr_serializer = dynamic_cast<NamedAttrsSerializer *>(attr_serializer);
  GE_CHECK_NOTNULL(named_attr_serializer);

  for (const auto &val : list_val) {
    const auto attr_proto = mutable_list->add_na();
    GE_CHECK_NOTNULL(attr_proto);
    if (named_attr_serializer->Serialize(val, attr_proto) != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "NamedAttr [%s] serialize failed.", val.GetName().c_str());
      return GRAPH_FAILED;
    }
  }

  mutable_list->set_val_type(GeIrAttrDef::ListValue::VT_LIST_NAMED_ATTRS);

  return GRAPH_SUCCESS;
}
graphStatus ListValueSerializer::SerializeListDataType(const AnyValue &av, GeIrAttrDef &def) {
  std::vector<ge::DataType> list_val;
  const graphStatus ret = av.GetValue(list_val);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to get list_datatype attr.");
    return GRAPH_FAILED;
  }
  const auto mutable_list = def.mutable_list();
  GE_CHECK_NOTNULL(mutable_list);
  mutable_list->clear_dt();
  for (const auto val : list_val) {
    mutable_list->add_dt(static_cast<proto::DataType>(val));
  }
  mutable_list->set_val_type(GeIrAttrDef::ListValue::VT_LIST_DATA_TYPE);

  return GRAPH_SUCCESS;
}

graphStatus ListValueSerializer::DeserializeListInt(const GeIrAttrDef &def, AnyValue &av) {
  std::vector<int64_t> values(static_cast<size_t>(def.list().i_size()));
  for (auto idx = 0; idx < def.list().i_size(); ++idx) {
    values[static_cast<size_t>(idx)] = def.list().i(idx);
  }
  return av.SetValue(std::move(values));
}

graphStatus ListValueSerializer::DeserializeListString(const GeIrAttrDef &def, AnyValue &av) {
  std::vector<std::string> values(static_cast<size_t>(def.list().s_size()));
  for (auto idx = 0; idx < def.list().s_size(); ++idx) {
    values[static_cast<size_t>(idx)] = def.list().s(idx);
  }
  return av.SetValue(std::move(values));
}

graphStatus ListValueSerializer::DeserializeListFloat(const GeIrAttrDef &def, AnyValue &av) {
  std::vector<float> values(static_cast<size_t>(def.list().f_size()));
  for (auto idx = 0; idx < def.list().f_size(); ++idx) {
    values[static_cast<size_t>(idx)] = def.list().f(idx);
  }

  return av.SetValue(std::move(values));
}

graphStatus ListValueSerializer::DeserializeListBool(const GeIrAttrDef &def, AnyValue &av) {
  std::vector<bool> values(static_cast<size_t>(def.list().b_size()));
  for (auto idx = 0; idx < def.list().b_size(); ++idx) {
    values[static_cast<size_t>(idx)] = def.list().b(idx);
  }
  return av.SetValue(std::move(values));
}

graphStatus ListValueSerializer::DeserializeListGeTensorDesc(const GeIrAttrDef &def, AnyValue &av) {
  std::vector<af::GeTensorDesc> values(static_cast<size_t>(def.list().td_size()));
  for (auto idx = 0; idx < def.list().td_size(); ++idx) {
    GeTensorSerializeUtils::AssembleGeTensorDescFromProto(&def.list().td(idx), values[static_cast<size_t>(idx)]);
  }

  return av.SetValue(std::move(values));
}

graphStatus ListValueSerializer::DeserializeListGeTensor(const GeIrAttrDef &def, AnyValue &av) {
  std::vector<GeTensor> values(static_cast<size_t>(def.list().t_size()));
  for (auto idx = 0; idx < def.list().t_size(); ++idx) {
    GeTensorSerializeUtils::AssembleGeTensorFromProto(&def.list().t(idx), values[static_cast<size_t>(idx)]);
  }

  return av.SetValue(std::move(values));
}

graphStatus ListValueSerializer::DeserializeListBuffer(const GeIrAttrDef &def, AnyValue &av) {
  std::vector<Buffer> values(static_cast<size_t>(def.list().bt_size()));
  for (auto idx = 0; idx < def.list().bt_size(); ++idx) {
    values[static_cast<size_t>(idx)] =
        Buffer::CopyFrom(PtrToPtr<const char_t, const uint8_t>(def.list().bt(idx).data()), def.list().bt(idx).size());
  }

  return av.SetValue(std::move(values));
}
graphStatus ListValueSerializer::DeserializeListGraphDef(const GeIrAttrDef &def, AnyValue &av) {
  std::vector<proto::GraphDef> values(static_cast<size_t>(def.list().g_size()));
  for (auto idx = 0; idx < def.list().g_size(); ++idx) {
    values[static_cast<size_t>(idx)] = def.list().g(idx);
  }
  return av.SetValue(std::move(values));
}

graphStatus ListValueSerializer::DeserializeListNamedAttrs(const GeIrAttrDef &def, AnyValue &av) {
  const auto attr_deserializer = AttrSerializerRegistry::GetInstance().GetDeserializer(GeIrAttrDef::ValueCase::kFunc);
  const auto named_attr_deserializer = dynamic_cast<NamedAttrsSerializer *>(attr_deserializer);
  GE_CHECK_NOTNULL(named_attr_deserializer);

  std::vector<af::NamedAttrs> values(static_cast<size_t>(def.list().na_size()));
  for (auto idx = 0; idx < def.list().na_size(); ++idx) {
    if (named_attr_deserializer->Deserialize(def.list().na(idx), values[static_cast<size_t>(idx)]) != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "NamedAttr [%s] deserialize failed.", def.list().na(idx).name().c_str());
      return GRAPH_FAILED;
    }
  }

  return av.SetValue(std::move(values));
}
graphStatus ListValueSerializer::DeserializeListDataType(const GeIrAttrDef &def, AnyValue &av) {
  std::vector<ge::DataType> values(static_cast<size_t>(def.list().dt_size()));
  for (auto idx = 0; idx < def.list().dt_size(); ++idx) {
    values[static_cast<size_t>(idx)] = static_cast<DataType>(def.list().dt(idx));
  }

  return av.SetValue(std::move(values));
}

REG_GEIR_SERIALIZER(list_int, ListValueSerializer, GetTypeId<std::vector<int64_t>>(), GeIrAttrDef::kList);
REG_GEIR_SERIALIZER(list_str, ListValueSerializer, GetTypeId<std::vector<std::string>>(), GeIrAttrDef::kList);
REG_GEIR_SERIALIZER(list_float, ListValueSerializer, GetTypeId<std::vector<float>>(), GeIrAttrDef::kList);
REG_GEIR_SERIALIZER(list_bool, ListValueSerializer, GetTypeId<std::vector<bool>>(), GeIrAttrDef::kList);
REG_GEIR_SERIALIZER(list_tensor_desc, ListValueSerializer, GetTypeId<std::vector<GeTensorDesc>>(), GeIrAttrDef::kList);
REG_GEIR_SERIALIZER(list_tensor, ListValueSerializer, GetTypeId<std::vector<GeTensor>>(), GeIrAttrDef::kList);
REG_GEIR_SERIALIZER(list_buffer, ListValueSerializer, GetTypeId<std::vector<Buffer>>(), GeIrAttrDef::kList);
REG_GEIR_SERIALIZER(list_graph_def, ListValueSerializer, GetTypeId<std::vector<proto::GraphDef>>(), GeIrAttrDef::kList);
REG_GEIR_SERIALIZER(list_named_attr, ListValueSerializer, GetTypeId<std::vector<af::NamedAttrs>>(), GeIrAttrDef::kList);
REG_GEIR_SERIALIZER(list_data_type, ListValueSerializer, GetTypeId<std::vector<ge::DataType>>(), GeIrAttrDef::kList);
}  // namespace af
