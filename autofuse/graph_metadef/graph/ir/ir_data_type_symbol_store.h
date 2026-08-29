/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_GRAPH_IR_DATA_TYPE_SYMBOL_STORE_H_
#define METADEF_CXX_GRAPH_IR_DATA_TYPE_SYMBOL_STORE_H_

#include <string>
#include <set>
#include <unordered_map>
#include "graph/types_af.h"
#include "graph/ge_error_codes_af.h"
#include "graph/type/tensor_type_impl.h"
#include "graph/op_desc.h"
#include "graph/type/sym_dtype.h"

namespace af {
/**
 * @brief 建立IR上输入输出和data type symbol的映射
 */
class IRDataTypeSymbolStore {
 public:
  IRDataTypeSymbolStore() = default;
  ~IRDataTypeSymbolStore() = default;

  // 类型推导入口函数，op的输入dtype和属性视作符号实际值的上下文，推导结果原地更新到op上
  graphStatus InferDtype(const OpDescPtr &op) const;
  bool IsSupportSymbolicInferDtype() const;
  bool IsSupportOrderedSymbolicInferDtype() const;
  ;

  // 创建一个Sym，用于Sym出现早于定义的情况
  SymDtype *GetOrCreateSymbol(const std::string &origin_sym_id);

  // 设置输入对应的Sym
  SymDtype *SetInputSymbol(const std::string &ir_input, IrInputType input_type, const std::string &sym_id);

  // 声明一个Sym的取值范围或计算方式，只有具有声明的Sym支持类型推导和范围校验
  SymDtype *DeclareSymbol(const std::string &sym_id, const TensorType &types);
  SymDtype *DeclareSymbol(const std::string &sym_id, const ListTensorType &types);
  SymDtype *DeclareSymbol(const std::string &sym_id, const Promote &types);
  SymDtype *DeclareSymbol(const std::string &sym_id, const OrderedTensorTypeList &types);

  // 设置输出对应的Sym
  SymDtype *SetOutputSymbol(const std::string &ir_output, IrOutputType output_type, const std::string &sym_id);

  graphStatus GetPromoteIrInputList(std::vector<std::vector<size_t>> &promote_index_list);

  std::list<std::shared_ptr<SymDtype>> GetSymbols() const {
    return syms_;
  }

  std::map<std::string, SymDtype *> GetNamedSymbols() const {
    return named_syms_;
  }

  std::vector<SymDtype *> GetOutSymbols() const {
    return output_syms_;
  }

 private:
  size_t num_ir_inputs = 0U;
  // 全部的Sym，包含T方式声明的Sym，以及Legacy注册的输入输出无效Sym，在DATATYPE声明前，无法确定Sym是否有效
  std::list<std::shared_ptr<SymDtype>> syms_;  // For copy constructor
  // 通过DATATYPE声明的Sym为命名Sym，支持符号共享、计算和范围校验
  std::map<std::string, SymDtype *> named_syms_;
  // 每个输出对应一个Sym
  std::vector<SymDtype *> output_syms_;
  std::vector<std::pair<std::string, IrOutputType>> output_name_and_types_;
};
}  // namespace af
#endif  // METADEF_CXX_GRAPH_IR_DATA_TYPE_SYMBOL_STORE_H_
