/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_OPERATOR_LEVEL_CACHE_GEN_H_
#define ATT_OPERATOR_LEVEL_CACHE_GEN_H_

#include "tiling_cache_code_gen.h"
#include "base/model_info.h"
#include "ge_common_af/ge_api_types_af.h"

namespace att {
namespace cache {

/**
 * @brief 殗子级缓存代码生成器
 * 派责生成TilingCacheContext和算子级缓存相关代码
 */
class OperatorLevelCacheGen : public TilingCacheCodeGen {
 public:
  OperatorLevelCacheGen() = default;
  ~OperatorLevelCacheGen() override = default;

  /**
   * @brief 生成FixedSizeHashMap模板类定义
   * @param code_printer 代码打印器
   * @return af::Status
   */
  af::Status GenFixedSizeHashMapDef(ge::CodePrinter &code_printer) override;

  /**
   * @brief 生成Context类私有Hash函数
   * @return af::Status
   */
  static std::string GenContextHashFunction();

  /**
   * @brief 生成算子级缓存函数定义
   * @param code_printer 代码打印器
   * @param tiling_data_type_name TilingData类型名称
   * @return af::Status
   */
  static af::Status GenOperatorCacheFunctions(ge::CodePrinter &code_printer, const std::string &tiling_data_type_name);

  /**
   * @brief 生成TilingCacheContext类定义
   * @param code_printer 代码打印器
   * @param tiling_data_type_name TilingData类型名称
   * @return af::Status
   */
  static af::Status GenTilingCacheContext(ge::CodePrinter &code_printer, const std::string &tiling_data_type_name);

  /**
   * @brief 生成TilingCacheContext静态成员定义（必须在cpp文件中）
   * @param code_printer 代码打印器
   * @return af::Status
   */
  static af::Status GenTilingCacheContextStaticDefs(ge::CodePrinter &code_printer);

  /**
   * @brief 生成算子级缓存类型定义
   * @param code_printer 代码打印器
   * @param tiling_data_type_name TilingData类型名称
   * @return af::Status
   */
  static af::Status GenOperatorCacheTypes(ge::CodePrinter &code_printer, const std::string &tiling_data_type_name);
  /**
   *
   * @param code_printer 代码打印器（函数体）
   * @param tiling_model_info Tiling模型信息
   * @param config 生成器配置
   * @return af::Status
   */
  static af::Status GenSaveCacheCalls(ge::CodePrinter &code_printer, const TilingModelInfo &tiling_model_info,
                                      const TilingCodeGenConfig &config);

  /**
   * @brief 生成缓存初始化和查询代码
   * @param code_printer 代码打印器（函数体）
   * @param tiling_model_info Tiling模型信息
   * @param config 生成器配置
   * @return af::Status
   */
  static af::Status GenInitAndQueryCacheCode(ge::CodePrinter &code_printer, const TilingModelInfo &tiling_model_info,
                                             const TilingCodeGenConfig &config);

  /**
   * @brief 生成Context类代码
   * @param tiling_data_type_name TilingData类型名称
   * @return af::Status
   */
  static std::string GenContextClass(const std::string &tiling_data_type_name);

  /**
   * @brief 生成Context类结构体
   * @return Context类结构体字符串
   */
  static std::string GenContextClassStructure();

  /**
   * @brief 生成Context类公共方法
   * @return af::Status
   */
  static std::string GenContextClassPublicMethods();

  /**
   * @brief 生成Context类缓存操作方法
   * @param tiling_data_type_name TilingData类型名称
   * @return af::Status
   */
  static std::string GenContextCacheOperations(const std::string &tiling_data_type_name);

  /**
   * @brief 生成FindOperatorCache实现代码
   * @param tiling_data_type_name TilingData类型名称
   * @return 生成的代码字符串
   */
  static std::string GenFindOperatorCacheImpl(const std::string &tiling_data_type_name);

  /**
   * @brief 生成SaveOperatorCache实现代码
   * @param tiling_data_type_name TilingData类型名称
   * @return 生成的代码字符串
   */
  static std::string GenSaveOperatorCacheImpl(const std::string &tiling_data_type_name);
};
}  // namespace cache

}  // namespace att

#endif  // ATT_OPERATOR_LEVEL_CACHE_GEN_H_
