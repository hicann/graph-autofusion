/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_ASCENDC_IR_DEF_H
#define METADEF_CXX_ASCENDC_IR_DEF_H

#include <string>
#include <memory>
#include "attr_store.h"
#include "graph/symbolizer/symbolic.h"
#include "graph/expression/const_values.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/ge_error_codes_af.h"
#include "graph/ge_tensor.h"
#include "ascendc_ir/ascendc_ir_check.h"
#include "proto/ascendc_ir.pb.h"
#include "serialization/attr_serializer_registry.h"
#include "attribute_group/af_attr_group_base.h"
#include "graph/operator_af.h"
#include "graph/anchor.h"
#include "proto/af_ir.pb.h"
// proto调整后，兼容老代码
namespace ascendc_ir {
namespace proto {
using AscIrAttrDef = af::proto::AscIrAttrDef;
using AscNodeAttrGroupsDef = af::proto::AscNodeAttrGroupsDef;
using AscGraphAttrGroupsDef = af::proto::AscGraphAttrGroupsDef;
using AscTensorAttrGroupsDef = af::proto::AscTensorAttrGroupsDef;
}  // namespace proto
}  // namespace ascendc_ir
namespace af {
#ifdef AUTOFUSE_USE_GE_METADEF
using AfAttrGroupsBase = af::AttrGroupsBase;
using AttrHolder = ge::AttrHolder;
using ge::DataType;
using ge::DT_BF16;
using ge::DT_BOOL;
using ge::DT_COMPLEX128;
using ge::DT_COMPLEX64;
using ge::DT_DOUBLE;
using ge::DT_FLOAT;
using ge::DT_FLOAT16;
using ge::DT_HIFLOAT8;
using ge::DT_INT16;
using ge::DT_INT32;
using ge::DT_INT4;
using ge::DT_INT64;
using ge::DT_INT8;
using ge::DT_QINT16;
using ge::DT_QINT32;
using ge::DT_QINT8;
using ge::DT_QUINT16;
using ge::DT_QUINT8;
using ge::DT_RESOURCE;
using ge::DT_STRING;
using ge::DT_UINT16;
using ge::DT_UINT32;
using ge::DT_UINT64;
using ge::DT_UINT8;
using ge::DT_UNDEFINED;
using ge::Expression;
using ge::FAILED;
using ge::GeTensorDesc;
using ge::GRAPH_FAILED;
using ge::GRAPH_SUCCESS;
using ge::Operator;
using ge::OutDataAnchor;
using ge::SUCCESS;
using ge::Symbol;
namespace sym {
using ge::sym::Abs;
using ge::sym::Add;
using ge::sym::Align;
using ge::sym::AlignWithPositiveInteger;
using ge::sym::Ceiling;
using ge::sym::Coeff;
using ge::sym::Div;
using ge::sym::Eq;
using ge::sym::Floor;
using ge::sym::Ge;
using ge::sym::Gt;
using ge::sym::kSymbolOne;
using ge::sym::kSymbolZero;
using ge::sym::Le;
using ge::sym::Log;
using ge::sym::LogicalAnd;
using ge::sym::LogicalOr;
using ge::sym::Lt;
using ge::sym::Max;
using ge::sym::Min;
using ge::sym::Mod;
using ge::sym::Mul;
using ge::sym::Ne;
using ge::sym::Neg;
using ge::sym::Not;
using ge::sym::Pow;
using ge::sym::Rational;
using ge::sym::Sub;
}  // namespace sym
#else
using AfAttrGroupsBase = AttrGroupsBase;
class OutDataAnchor;
class Operator;
#endif
namespace {
constexpr int64_t kIdNone = -1;
const std::string kDataIndex = "index";
}  // namespace
struct SizeVar {
  using Type = enum : int32_t {
    kSizeTypeVar = 0,
    kSizeTypeConst = 1,
  };

  // [HI] 符号`id`，从0开始，TODO：待删除
  int64_t id{};

  // [HI] 符号名，图内唯一，符号名被用于全图的表达式，TODO：待删除
  std::string name;

  // [HI] 如果符号是常量，`const_value`表示常量的值，TODO：待删除，使用expr中的内容
  int64_t const_value{};

  // [HI] 符号的类型，TODO：待删除，使用expr中的内容
  Type type;

  // [HI] TODO 这里只能使用Symbol创建，不允许使用Expression
  explicit SizeVar(Expression expr_other) : type(kSizeTypeVar), expr(std::move(expr_other)) {}

  // [HI] 符号，expr中的符号名图内唯一，符号名被用于全图的表达式
  Expression expr;
};
using SizeVarPtr = std::shared_ptr<SizeVar>;

struct Axis {
  using Type = enum : int32_t {
    kAxisTypeOriginal,
    kAxisTypeBlockOuter,  // outer axis after split by multicore
    kAxisTypeBlockInner,  // inner axis after split by multicore
    kAxisTypeTileOuter,   // outer axis after split by one core
    kAxisTypeTileInner,   // inner axis after split by one core
    kAxisTypeMerged,
    kAxisTypeInvalid
  };

  int64_t id{kIdNone};  // axis id

  // [HI] 轴的名字，图内唯一
  std::string name;  // axis name

  // [HI] 轴的类型
  Type type{kAxisTypeInvalid};

  // [I] 是否为`block`轴
  bool bind_block{false};

  // [HI] 轴的大小
  Expression size;

  // [I] TODO 轴的对齐要求，详细说明不同的值分别是什么含义
  Expression align{Symbol(-1)};

  // [I] 当轴为被切分轴时，
  std::vector<int64_t> from;

  // [I] 如果轴是被切分出来的，`split_pair`表示切分出来的另一个轴的`id`
  int64_t split_pair_other_id{kIdNone};
  // 自动融合场景的默认值，手写场景可以做配置，供ATT使用
  bool allow_oversize_axis{false};
  bool allow_unaligned_tail{true};
};
using AxisPtr = std::shared_ptr<Axis>;
using AxisId = int64_t;
enum class TransType : int64_t { kSplit = 0, kMerge, kValid };
struct OneTransInfo {
  TransType trans_type;
  std::vector<AxisPtr> src_axis;
  std::vector<AxisPtr> dst_axis;
};
using TransInfoRoadOfGraph = std::vector<OneTransInfo>;

enum class ComputeType : int32_t {
  kComputeLoad,
  kComputeStore,
  kComputeReduceStore,
  kComputeElewise,
  kComputeBroadcast,
  kComputeReduce,
  kComputeTranspose,
  kComputeConcat,
  kComputeGather,
  kComputeCube,
  kComputeSplit,
  kComputeInvalid,
};

enum class ComputeUnit : int32_t {
  kUnitNone,
  kUnitMTE1,
  kUnitMTE2,
  kUnitMTE3,
  kUnitScalar,
  kUnitVector,
  kUnitCube,
  kUnitInvalid,
};

enum class ApiType : int32_t {
  kAPITypeBuffer,   // Workspace/Data/Constant/IndexExpr/Output
  kAPITypeCompute,  // Load/Store/ReduceStore/Elewise/BroadCast/Reduce/Transpose
  kAPITypeInvalid,
};

enum class ExecuteCondition : int32_t {
  kNoCache = 0,                         // 不缓存
  kCacheBlockSplitFusedBroadcastAxis,   // 缓存，条件是合轴后的广播轴拆分到T和t中
  kCacheBlockSplitOriginBroadcastAxis,  // 缓存，条件是合轴后分到T中的原始轴都是广播轴
  kConditionInvalid,
};

struct ApiInfo {
  // [I] `api`的类型
  ApiType type = ApiType::kAPITypeInvalid;

  // [I] `api`的计算类型
  ComputeType compute_type = ComputeType::kComputeInvalid;

  // [I] `api`的计算单元
  ComputeUnit unit = ComputeUnit::kUnitInvalid;
};

struct SchedInfo {
  // [HI] 已废弃(2026-07-15)，仅为兼容性保留，经过兼容周期后可删除
  int64_t exec_order{kIdNone};

  // [HI] 节点所处的多层嵌套循环的轴`id`，按循环表示从外层到内层的轴`id`
  std::vector<int64_t> axis;

  // [I] 节点进行`api`计算的最内层循环，这个轴以内的部分将被映射为`api`的参数长度，这个轴以外的循环将会展开
  int64_t loop_axis{kIdNone};

  // [I] 节点的执行时的附加条件，目前主要用于是否对`api`结果进行缓存
  ExecuteCondition exec_condition{ExecuteCondition::kNoCache};
};

// 图级调度信息。当前仅用于接收外部统一调度轴，loop_axis 保留给内部流程使用。
// 与节点级 SchedInfo 分开定义，避免将节点级 exec_condition 提升到图级。
struct GraphSchedInfo {
  std::vector<int64_t> axis;
  int64_t loop_axis{kIdNone};
};

class AscIrAttrDefBase {
 public:
  AscIrAttrDefBase() = default;
  virtual ~AscIrAttrDefBase() = default;
  graphStatus Serialize(ascendc_ir::proto::AscIrAttrDef &asc_ir_attr_def);
  graphStatus Deserialize(const ascendc_ir::proto::AscIrAttrDef &asc_ir_attr_def);
  std::unique_ptr<AscIrAttrDefBase> Clone();
  ;
  template <typename T>
  graphStatus GetAttrValue(const std::string &attr_name, T &attr_value) {
    auto *const v = attr_store_.GetAnyValue(attr_name);
    if (v == nullptr) {
      GELOGW("Attr %s has not been set.", attr_name.c_str());
      return af::GRAPH_FAILED;
    }
    if (v->Get<T>() == nullptr) {
      GELOGW("Attr %s is set, however maybe type is not fit.", attr_name.c_str());
      return af::GRAPH_FAILED;
    }
    attr_value = *(v->Get<T>());
    return af::GRAPH_SUCCESS;
  }
  template <typename T>
  T *DownCastTo() {
    // 子类没有成员，所以可以这样搞
    static_assert(std::is_base_of<AscIrAttrDefBase, T>::value, "Template parameter must be derived from IrAttrDefBase");
    return reinterpret_cast<T *>(this);
  }
  template <typename T>
  const T *DownCastTo() const {
    static_assert(std::is_base_of<AscIrAttrDefBase, T>::value, "Template parameter must be derived from IrAttrDefBase");
    return reinterpret_cast<const T *>(this);
  }

 protected:
  AttrStore::CustomDefinedAttrStore attr_store_;
};

enum class AllocType : int32_t {
  kAllocTypeGlobal,
  kAllocTypeL1,
  kAllocTypeL2,
  kAllocTypeBuffer,
  kAllocTypeQueue,
  kAllocTypeInvalid,
};

enum class MemHardware : int32_t {
  kMemHardwareGM,
  kMemHardwareUB,
  kMemHardwareInvalid,
};

enum class Position : int32_t {
  kPositionGM,
  kPositionVecIn,
  kPositionVecOut,
  kPositionVecCalc,
  kPositionInvalid,
};

struct MemAttr {
  int64_t tensor_id = kIdNone;
  AllocType alloc_type = AllocType::kAllocTypeGlobal;
  Position position = Position::kPositionGM;
  MemHardware hardware = MemHardware::kMemHardwareGM;
  // TODO 待删除
  std::vector<int64_t> buf_ids;
  // TODO 待删除
  std::string name;
  // reuse_id配合que_id表达que的共用和复用
  // que_id相同，一个reuse_id对应一组tensor, 该组中的多个tensor共用该que_id,
  // tensor使用该que的offset由使用者自己计算和维护 que_id相同，多个reuse_id对应多组tensor，每组tensor间复用该que_id
  int64_t reuse_id = kIdNone;
};

struct MemQueAttr {
  int64_t id = kIdNone;
  int64_t depth{-1};
  int64_t buf_num{-1};
  // TODO 待删除
  std::string name{""};
};

struct MemBufAttr {
  int64_t id = kIdNone;
  // TODO 待删除
  std::string name{""};
};

struct MemOptAttr {
  int64_t reuse_id = kIdNone;  // TODO 待删除, 正式方案放在MemAttr
  int64_t ref_tensor = kIdNone;
  int64_t merge_scope = kIdNone;
};

struct TmpBufDesc {
  Expression size;
  int64_t life_time_axis_id = -1;  // -1: 生命周期为API级别, >= 0: loop级别
};

struct TmpBuffer {
  TmpBufDesc buf_desc;
  MemAttr mem{};
  int64_t id = kIdNone;
};

class AscNodeAttr : public AfAttrGroupsBase {
 public:
  // [HI] 节点名，图内唯一
  std::string name;
  // [HI] 节点类型
  std::string type;
  // 调度信息
  SchedInfo sched{};
  ApiInfo api{};
  // Ir定义的属性，跟具体Ir有关
  std::unique_ptr<AscIrAttrDefBase> ir_attr{nullptr};
  std::vector<TmpBuffer> tmp_buffers;
  AscNodeAttr() = default;
  ~AscNodeAttr() override = default;
  graphStatus SerializeAttr(ascendc_ir::proto::AscNodeAttrGroupsDef &asc_node_group) const;
  graphStatus DeserializeAttr(const ascendc_ir::proto::AscNodeAttrGroupsDef &asc_node_group);
  graphStatus Serialize(proto::AttrGroupDef &attr_group_def) override;
  graphStatus Deserialize(const proto::AttrGroupDef &attr_group_def, AttrHolder *attr_holder) override;
  AscNodeAttr &operator=(const AscNodeAttr &other);
  AscNodeAttr(const AscNodeAttr &other)
      : name(other.name),
        type(other.type),
        sched(other.sched),
        api(other.api),
        ir_attr(other.ir_attr ? other.ir_attr->Clone() : nullptr),
        tmp_buffers(other.tmp_buffers) {}
  // 没有注册ir属性时，调用这个接口
  static AscNodeAttr *Create(af::Operator &op);

  // 注册了ir属性时，调用这个接口
  template <typename IrAttrDef>
  static AscNodeAttr *Create(af::Operator &op) {
    static_assert(
        std::is_base_of<AscIrAttrDefBase, IrAttrDef>::value && !std::is_same<IrAttrDef, AscIrAttrDefBase>::value,
        "Template parameter must be derived from IrAttrDefBase");
    return CreateImplWithIrAttrInit<IrAttrDef>(op);
  }
  std::unique_ptr<AfAttrGroupsBase> CloneAf() override;

 private:
  static AscNodeAttr *CreateImpl(af::Operator &op);
  template <typename IrAttrDef>
  static AscNodeAttr *CreateImplWithIrAttrInit(af::Operator &op) {
    auto attr_group = CreateImpl(op);
    GE_ASSERT_NOTNULL(attr_group);
#ifdef AUTOFUSE_USE_GE_METADEF
    attr_group->ir_attr = std::move(ge::ComGraphMakeUnique<IrAttrDef>());
#else
    attr_group->ir_attr = std::move(af::ComGraphMakeUnique<IrAttrDef>());
#endif
    GE_ASSERT_NOTNULL(attr_group->ir_attr);
    return attr_group;
  }
};

class AscDataIrAttrDef : public AscIrAttrDefBase {
  // 子类不应该有自己的成员，只需要有对应的set,get函数
 public:
  ~AscDataIrAttrDef() override = default;
  graphStatus GetIndex(int64_t &index) const;
  graphStatus SetIndex(int64_t index);
};

enum class AscGraphType : int64_t {
  kHintGraph = 0,
  kImplGraph,
};

class AscGraphAttr : public AfAttrGroupsBase {
 public:
  // TODO 待确认正式方案
  int64_t tiling_key = -1;

  // [HI] 图上的轴
  std::vector<AxisPtr> axis;

  // 图级调度信息。axis 在进入优化流程前可同步到各节点的 sched.axis。
  GraphSchedInfo sched{};

  // TODO 待正式方案后删除
  TransInfoRoadOfGraph trans_info_road;

  // [HI] 图上的符号，TODO：未来不需要这个数据结构了，改成Expression即可
  std::vector<SizeVarPtr> size_vars;
  AscGraphType type{AscGraphType::kHintGraph};
  graphStatus SerializeAttr(ascendc_ir::proto::AscGraphAttrGroupsDef &asc_graph_group);
  graphStatus DeserializeAttr(const ascendc_ir::proto::AscGraphAttrGroupsDef &asc_graph_group);
  std::unique_ptr<AfAttrGroupsBase> CloneAf() override;
  graphStatus Serialize(proto::AttrGroupDef &attr_group_def) override;
  graphStatus Deserialize(const proto::AttrGroupDef &attr_group_def, AttrHolder *attr_holder) override;
};

class AscTensorDataType {
 public:
  operator af::DataType() const {
    if (tensor_desc_ == nullptr) {
      GELOGE(af::FAILED, "tensor_desc_ is null");
      return af::DT_UNDEFINED;
    }
    return tensor_desc_->GetDataType();
  };
  void operator=(const af::DataType &other) {
    if (tensor_desc_ == nullptr) {
      GELOGE(af::FAILED, "tensor_desc_ is null");
      return;
    }
    tensor_desc_->SetDataType(other);
  };
  AscTensorDataType &operator=(const AscTensorDataType &other) {
    if (this == &other) {
      return *this;
    }
    if ((tensor_desc_ != nullptr) && (other.tensor_desc_ != nullptr)) {
      tensor_desc_->SetDataType(static_cast<af::DataType>(other));
    }
    if ((tensor_desc_ == nullptr) && (other.tensor_desc_ != nullptr)) {
      // 浅拷贝，兼容已存在的用法，调用者需要保证声明周期有效
      tensor_desc_ = other.tensor_desc_;
    }
    return *this;
  }
  AscTensorDataType(const AscTensorDataType &other) {
    if ((tensor_desc_ != nullptr) && (other.tensor_desc_ != nullptr)) {
      tensor_desc_->SetDataType(static_cast<af::DataType>(other));
    }
    if ((tensor_desc_ == nullptr) && (other.tensor_desc_ != nullptr)) {
      // 浅拷贝，兼容已存在的用法，调用者需要保证声明周期有效
      tensor_desc_ = other.tensor_desc_;
    }
  }
  AscTensorDataType() = default;

 private:
  friend struct AscNodeOutputs;
  friend class AscTensorAttr;
  friend class AscGraphUtils;
  af::GeTensorDesc *tensor_desc_{nullptr};
};

class AscTensorAttr : public AfAttrGroupsBase {
  friend class AscGraphUtils;

 public:
  // [HI] 该`Tensor`的数据类型
  AscTensorDataType dtype;

  // [HI] 该`Tensor`中包含的轴的`id`
  std::vector<int64_t> axis;

  // [HI] `repeat[i]`表示该`Tensor`包含的第`i`个轴的大小的符号表达式
  std::vector<Expression> repeats;

  // [HI] `stride[i]`表示该`Tensor`包含的第`i`个轴，在索引时的步长
  std::vector<Expression> strides;

  // [I] `buffer`中存储哪些轴的内容
  std::vector<int64_t> vectorized_axis;

  // [I] `buffer`中存储的内容，按轴索引时的步长
  std::vector<Expression> vectorized_strides;
  MemAttr mem{};
  MemQueAttr que{};
  MemBufAttr buf{};
  MemOptAttr opt{};
  static AscTensorAttr &GetTensorAttr(af::Operator *op, const uint32_t index);
  static AscTensorAttr &GetTensorAttr(const af::OutDataAnchor &output);
  static AscTensorAttr *GetTensorAttrPtr(af::Operator *op, const uint32_t index);
  static AscTensorAttr *GetTensorAttrPtr(const af::OutDataAnchor &output);
  // Bridge for GE stub: accepts raw OpDesc pointer (void*) to avoid cross-library
  // symbol-mangling issues caused by AUTOFUSE_USE_GE_METADEF.
  static AscTensorAttr *GetOrCreateFromOpDescRaw(void *op_desc_raw, uint32_t index);
  graphStatus SerializeAttr(ascendc_ir::proto::AscTensorAttrGroupsDef &asc_tensor_group);
  graphStatus DeserializeAttr(const ascendc_ir::proto::AscTensorAttrGroupsDef &asc_tensor_group,
                              af::GeTensorDesc *tensor_desc);
  std::unique_ptr<AfAttrGroupsBase> CloneAf() override;
  graphStatus Serialize(proto::AttrGroupDef &attr_group_def) override;
  graphStatus Deserialize(const proto::AttrGroupDef &attr_group_def, AttrHolder *attr_holder) override;
};
}  // namespace af

namespace ge {
using af::AllocType;
using af::ApiInfo;
using af::ApiType;
using af::AscDataIrAttrDef;
using af::AscGraphAttr;
using af::AscGraphType;
using af::AscIrAttrDefBase;
using af::AscNodeAttr;
using af::AscTensorAttr;
using af::AscTensorDataType;
using af::Axis;
using af::AxisId;
using af::AxisPtr;
using af::ComputeType;
using af::ComputeUnit;
using af::ExecuteCondition;
using af::GraphSchedInfo;
using af::MemAttr;
using af::MemBufAttr;
using af::MemHardware;
using af::MemOptAttr;
using af::MemQueAttr;
using af::OneTransInfo;
using af::Position;
using af::SchedInfo;
using af::SizeVar;
using af::SizeVarPtr;
using af::TmpBufDesc;
using af::TmpBuffer;
using af::TransInfoRoadOfGraph;
using af::TransType;
}  // namespace ge

#endif  // METADEF_CXX_ASCENDC_IR_DEF_H
