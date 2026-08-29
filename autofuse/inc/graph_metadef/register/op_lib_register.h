/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_REGISTER_OP_LIB_REGISTER_H
#define INC_EXTERNAL_REGISTER_OP_LIB_REGISTER_H
#include "graph/compiler_def.h"
#include "graph/types_af.h"
#include "graph/ascend_string_af.h"

namespace af {
class OpLibRegisterImpl;
class OpLibRegister {
 public:
  explicit OpLibRegister(const char_t *vendor_name);
  OpLibRegister(OpLibRegister &&other) noexcept;
  OpLibRegister(const OpLibRegister &other);
  OpLibRegister &operator=(const OpLibRegister &) = delete;
  OpLibRegister &operator=(OpLibRegister &&) = delete;
  ~OpLibRegister();

  using OpLibInitFunc = uint32_t (*)(ge::AscendString &);
  OpLibRegister &RegOpLibInit(OpLibInitFunc func);

 private:
  std::unique_ptr<OpLibRegisterImpl> impl_;
};
}  // namespace af

#define REGISTER_OP_LIB(vendor_name) REGISTER_OP_LIB_UNIQ_HELPER(vendor_name, __COUNTER__)

#define REGISTER_OP_LIB_UNIQ_HELPER(vendor_name, counter) REGISTER_OP_LIB_UNIQ(vendor_name, counter)

#define REGISTER_OP_LIB_UNIQ(vendor_name, counter) \
  static ge::OpLibRegister VAR_UNUSED g_##vendor_name##counter = ge::OpLibRegister(#vendor_name)

#endif  // INC_EXTERNAL_REGISTER_OP_LIB_REGISTER_H
