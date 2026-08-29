/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_REGISTER_OP_LIB_REGISTER_IMPL_H_
#define METADEF_CXX_REGISTER_OP_LIB_REGISTER_IMPL_H_

#include <mutex>
#include <set>
#include <string>
#include "register/op_lib_register.h"
#include "graph/ge_error_codes_af.h"

namespace af {
class OpLibRegisterImpl {
 public:
  void SetVendorName(const std::string &vendor_name) {
    vendor_name_ = vendor_name;
  }
  void SetInitFunc(const OpLibRegister::OpLibInitFunc init_func) {
    init_func_ = init_func;
  }
  std::string GetVendorName() const {
    return vendor_name_;
  }
  OpLibRegister::OpLibInitFunc GetInitFunc() const {
    return init_func_;
  }

 private:
  std::string vendor_name_;
  OpLibRegister::OpLibInitFunc init_func_ = nullptr;
};

class OpLibRegistry {
 public:
  static OpLibRegistry &GetInstance();
  ~OpLibRegistry();
  void RegisterInitFunc(OpLibRegisterImpl &register_impl);
  graphStatus PreProcessForCustomOp();
  const char_t *InitAndGetCustomOpLibPath();

 private:
  void ClearHandles();
  graphStatus GetAllCustomOpApiSoPaths(const std::string &custom_opp_path,
                                       std::vector<std::string> &so_real_paths) const;
  graphStatus CallInitFunc(const std::string &custom_opp_path, const std::vector<std::string> &so_real_paths);

  std::mutex mu_;
  std::vector<std::pair<std::string, OpLibRegister::OpLibInitFunc>> vendor_funcs_;
  std::set<std::string> vendor_names_set_;
  std::vector<void *> handles_;
  bool is_init_ = false;
  std::string op_lib_paths_;
};
}  // namespace af
#endif  // METADEF_CXX_REGISTER_OP_LIB_REGISTER_IMPL_H_
