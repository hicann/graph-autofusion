/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_INC_EXTERNAL_GRAPH_ARG_DESC_INFO_H
#define METADEF_INC_EXTERNAL_GRAPH_ARG_DESC_INFO_H

#include <vector>
#include <cstdint>
#include <memory>
#include "graph/ge_error_codes_af.h"
#include "graph/ascend_string_af.h"

namespace af {
enum class ArgDescType {
  kIrInput = 0,     // 输入
  kIrOutput,        // 输出
  kWorkspace,       // workspace地址
  kTiling,          // tiling地址
  kHiddenInput,     // ir上不表达的额外输入
  kCustomValue,     // 自定义内容
  kIrInputDesc,     // 具有描述信息的输入地址
  kIrOutputDesc,    // 具有描述信息的输入地址
  kInputInstance,   // 实例化的输入
  kOutputInstance,  // 实例化的输出
  kEnd
};

enum class HiddenInputSubType {
  kHcom,  // 用于通信的hiddenInput，mc2算子使用
  kEnd
};

class ArgDescInfoImpl;
using ArgDescInfoImplPtr = std::unique_ptr<ArgDescInfoImpl>;
class ArgDescInfo {
 public:
  /**
   * 构造ArgDescInfo对象，ArgDescInfo对象主要用于描述args中某一个地址所表达的含义
   * @param arg_type 当前args地址的类型
   * @param ir_index 当前args地址对应算子的ir索引
   * @param is_folded 当前args地址是否为二级指针 (即是否将多个地址折叠到一个二级指针中设置给args，设置为true)
   */
  explicit ArgDescInfo(ArgDescType arg_type, int32_t ir_index = -1, bool is_folded = false);
  ~ArgDescInfo();
  ArgDescInfo(const ArgDescInfo &other);
  ArgDescInfo(ArgDescInfo &&other) noexcept;
  ArgDescInfo &operator=(const ArgDescInfo &other);
  ArgDescInfo &operator=(ArgDescInfo &&other) noexcept;
  /**
   * 构造一个CustomValue类型的ArgDescInfo对象
   * @param custom_value 自定义内容
   * @return ArgDescInfo对象
   */
  static ArgDescInfo CreateCustomValue(uint64_t custom_value);
  /**
   * 构造一个HiddenInput类型的ArgDescInfo对象
   * @param hidden_type hidden输入的类型
   * @return ArgDescInfo对象
   */
  static ArgDescInfo CreateHiddenInput(HiddenInputSubType hidden_type);
  /**
   * 获取当前ArgDescInfo的类型
   * @return 当ArgDescType非法时，返回kEnd，合法时，返回此arg地址的类型（未设置时的默认值为kEnd）
   */
  ArgDescType GetType() const;
  /**
   * 获取自定义内容的值，只有当type为kCustomValue时，才能获取到内容
   * @return 当ArgDescType非法时，返回uint64_max, 合法时，返回自定义内容（未设置时的默认值为0）
   */
  uint64_t GetCustomValue() const;
  /**
   * 设置自定义内容，只有当type为kCustomValue时，才能设置此字段
   * @param custom_value 自定义内容
   * @return SUCCESS: 设置成功 其他：ArgDescInfo非法或者type为非kCustomValue
   */
  graphStatus SetCustomValue(uint64_t custom_value);
  /**
   * 获取hidden输入的type，只有当type为kHiddenInput时，才能获取到内容
   * @return 当ArgDescType非法时，返回kEnd, 合法时，返回hidden输入的type（未设置时的默认值为kEnd）
   */
  HiddenInputSubType GetHiddenInputSubType() const;
  /**
   * 设置hidden输入的type，只有当type为kHiddenInput时，才能设置此字段
   * @param hidden_type hidden输入的type
   * @return SUCCESS: 设置成功 其他：ArgDescInfo非法或者type为非kHiddenInput
   */
  graphStatus SetHiddenInputSubType(HiddenInputSubType hidden_type);
  /**
   * 获取当前arg地址对应的ir索引
   * @return 返回ir索引（未设置时的默认值为-1）
   */
  int32_t GetIrIndex() const;
  /**
   * 设置当前arg地址对应的ir索引
   * @param ir_index ir索引
   */
  void SetIrIndex(int32_t ir_index);
  /**
   * 判断当前arg地址是否为二级指针
   * @return true: 是二级指针; false：不是二级指针（未设置时的默认值为false）
   */
  bool IsFolded() const;
  /**
   * 设置当前arg地址是否为二级指针
   * @param is_folded：是否为二级指针
   */
  void SetFolded(bool is_folded);

 private:
  friend class ArgsFormatSerializer;
  ArgDescInfo() = delete;
  explicit ArgDescInfo(ArgDescInfoImplPtr &&impl);
  std::unique_ptr<ArgDescInfoImpl> impl_;
};

class ArgsFormatSerializer {
 public:
  /**
   * 序列化argsFormat，argsFormat是由若干个ArgDescInfo组成，每一个ArgDescInfo表达当前args地址的信息
   * @param args_format args_format信息
   * @return 成功：返回序列化后的argsFormat; 失败：空字符串
   */
  static AscendString Serialize(const std::vector<ArgDescInfo> &args_format);
  /**
   * 将一个argsFormat的序列化字符串反序列化
   * @param args_str args_format的序列化字符串
   * @return 成功：返回反序列化后的argsFormat; 失败：空vector
   */
  static std::vector<ArgDescInfo> Deserialize(const AscendString &args_str);
};
}  // namespace af
#endif  // METADEF_INC_EXTERNAL_GRAPH_ARG_DESC_INFO_H
