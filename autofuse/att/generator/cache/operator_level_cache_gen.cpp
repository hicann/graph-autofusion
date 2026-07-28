/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "operator_level_cache_gen.h"
#include <set>
#include <utility>
#include "common/code_printer.h"
#include "generator/preprocess/args_manager.h"
#include "util/base_types_printer.h"

namespace att {
namespace cache {
namespace {
std::vector<std::pair<std::string, std::string>> GetVarAccessors(const TilingModelInfo &tiling_model_info) {
  std::vector<std::pair<std::string, std::string>> var_accessors;
  std::set<std::string> visited_var_names;
  std::set<std::string> all_groups_prefix;
  for (const auto &model_info : tiling_model_info) {
    all_groups_prefix.insert(model_info.schedule_group_ident.GetGroupPrefix());
  }
  for (const auto &model_info : tiling_model_info) {
    ArgsManager args_manager(model_info);
    GE_ASSERT_TRUE(args_manager.Process(false), "Args manager process failed.");
    auto input_vars = args_manager.GetInputVars();
    bool is_unique_group = (all_groups_prefix.size() == 1);
    std::string group_prefix =
        is_unique_group ? "" : (model_info.schedule_group_ident.GetItemPrefix() + "_tiling_data.");

    for (const auto &var : input_vars) {
      std::string var_name = Str(var);
      if (visited_var_names.find(var_name) == visited_var_names.end()) {
        visited_var_names.insert(var_name);
        std::string accessor = "tiling_data." + group_prefix + "get_" + var_name + "()";
        var_accessors.emplace_back(var_name, accessor);
      }
    }
  }
  return var_accessors;
}

std::string GenShapeKeyToStringCode(const std::string &key_name) {
  std::stringstream ss;
  ss << "[&" << key_name << "]()->std::string {\n";
  ss << "  std::string out;\n";
  ss << "  for (size_t i = 0; i < " << key_name << ".size(); ++i) {\n";
  ss << "    if (i != 0) {\n";
  ss << "      out.append(\",\");\n";
  ss << "    }\n";
  ss << "    out.append(std::to_string(" << key_name << "[i]));\n";
  ss << "  }\n";
  ss << "  return out;\n";
  ss << "}.operator()().c_str()";
  return ss.str();
}
}  // namespace

af::Status OperatorLevelCacheGen::GenFixedSizeHashMapDef(ge::CodePrinter &code_printer) {
  // 生成FixedSizeHashMap模板类定义
  std::string hashmap_code = GenHashMapTemplate();
  code_printer.AddLine(hashmap_code);
  return af::SUCCESS;
}

af::Status OperatorLevelCacheGen::GenTilingCacheContext(ge::CodePrinter &code_printer,
                                                        const std::string &tiling_data_type_name) {
  // 生成TilingCacheContext类
  std::string context_class = GenContextClass(tiling_data_type_name);
  code_printer.AddLine(context_class);
  code_printer.AddLine("");
  return af::SUCCESS;
}

af::Status OperatorLevelCacheGen::GenTilingCacheContextStaticDefs(ge::CodePrinter &code_printer) {
  // 生成TilingCacheContext静态成员变量定义（必须在cpp文件中）
  code_printer.AddLine(R"(
// TilingCacheContext 静态成员变量定义
thread_local std::unique_ptr<OperatorLevelCache> TilingCacheContext::operator_cache_;
thread_local bool TilingCacheContext::initialized_ = false;
thread_local std::array<uint64_t, kOperatorCacheCapacity> TilingCacheContext::access_counts_;

)");
  return af::SUCCESS;
}

af::Status OperatorLevelCacheGen::GenOperatorCacheTypes(ge::CodePrinter &code_printer,
                                                        const std::string &tiling_data_type_name) {
  // 第一级：算子级缓存（使用kInputShapeSize）
  code_printer.AddLine("using OperatorLevelCache = FixedSizeHashMap<kInputShapeSize, kOperatorCacheCapacity, " +
                       tiling_data_type_name + ">;");
  code_printer.AddLine("");

  return af::SUCCESS;
}

af::Status OperatorLevelCacheGen::GenOperatorCacheFunctions(ge::CodePrinter &code_printer,
                                                            const std::string &tiling_data_type_name) {
  // 生成算子级缓存函数（使用R"()"格式以提高性能）
  std::string find_func = R"(
bool FindOperatorCache(std::array<uint32_t, kInputShapeSize>& input_shapes, )" +
                          tiling_data_type_name +
                          R"(& tiling_data, OperatorLevelCache& cache) {
  const auto* result = cache.Find(input_shapes);
  if (result != nullptr) {
    tiling_data = *result;
    return true;
  }
  return false;
}
)";

  std::string save_func = R"(
bool SaveOperatorCache(std::array<uint32_t, kInputShapeSize>& input_shapes, const )" +
                          tiling_data_type_name +
                          R"(& tiling_data, OperatorLevelCache& cache) {
  return cache.Insert(input_shapes, tiling_data);
}
)";

  code_printer.AddLine(find_func);
  code_printer.AddLine(save_func);

  return af::SUCCESS;
}

af::Status OperatorLevelCacheGen::GenSaveCacheCalls(ge::CodePrinter &code_printer,
                                                    const TilingModelInfo &tiling_model_info,
                                                    const TilingCodeGenConfig &config) {
  if (!config.cache_enabled_at_compile_time) {
    return af::SUCCESS;
  }
  const auto var_accessors = GetVarAccessors(tiling_model_info);
  if (var_accessors.empty()) {
    // 静态Shape场景：使用空key进行缓存
    GELOGI("Static shape detected, using empty key for operator level cache, model[%s].",
           tiling_model_info[0].graph_name.c_str());
    code_printer.AddLine("  // 静态Shape场景：使用空key缓存");
    code_printer.AddLine("  std::array<uint32_t, kInputShapeSize> empty_shapes = {};");
    // 缓存保存失败不影响GetTiling的整体结果，所以直接调用而不使用ret |=
    code_printer.AddLine("  (void)TilingCacheContext::SaveOperatorCache(empty_shapes, tiling_data);");
    return af::SUCCESS;
  }
  // 缓存保存失败不影响GetTiling的整体结果，所以直接调用而不使用ret |=
  code_printer.AddLine("  (void)TilingCacheContext::SaveOperatorCache(input_shapes, tiling_data);");
  return af::SUCCESS;
}

af::Status OperatorLevelCacheGen::GenInitAndQueryCacheCode(ge::CodePrinter &code_printer,
                                                           const TilingModelInfo &tiling_model_info,
                                                           const TilingCodeGenConfig &config) {
  if (!config.cache_enabled_at_compile_time) {
    return af::SUCCESS;
  }
  const auto var_accessors = GetVarAccessors(tiling_model_info);
  if (var_accessors.empty()) {
    // 静态Shape场景：使用空key进行缓存查询
    GELOGI("Static shape detected, using empty key for operator level cache query, model[%s].",
           tiling_model_info[0].graph_name.c_str());
    code_printer.AddLine("  // 静态Shape场景：算子级缓存查询（空key）");
    code_printer.AddLine("  std::array<uint32_t, kInputShapeSize> input_shapes = {};");
    code_printer.AddLine("  if (TilingCacheContext::FindOperatorCache(input_shapes) != nullptr) {");
    code_printer.AddLine(
        "      memcpy(&tiling_data, TilingCacheContext::FindOperatorCache(input_shapes), sizeof(tiling_data));");
    code_printer.AddLine("    OP_LOGI(OP_NAME, \"Operator level cache hit (static shape)\");");
    code_printer.AddLine("    return true;");
    code_printer.AddLine("  }");
    code_printer.AddLine("");
    return af::SUCCESS;
  }

  code_printer.AddLine("  // 第一级：算子级缓存查询，收集所有原始轴");
  std::string array_init = "  std::array<uint32_t, kInputShapeSize> input_shapes = {";
  for (size_t i = 0; i < var_accessors.size(); ++i) {
    if (i > 0) {
      array_init += ", ";
    }
    array_init += var_accessors[i].second;
  }
  array_init += "};";
  code_printer.AddLine(array_init);

  code_printer.AddLine("  if (TilingCacheContext::FindOperatorCache(input_shapes) != nullptr) {");
  code_printer.AddLine(
      "      memcpy(&tiling_data, TilingCacheContext::FindOperatorCache(input_shapes), sizeof(tiling_data));");
  code_printer.AddLine(std::string("    OP_LOGI(OP_NAME, \"Operator level cache hit, input_shapes[%s]\", ") +
                       GenShapeKeyToStringCode("input_shapes") + ");");
  code_printer.AddLine("    return true;");
  code_printer.AddLine("  }");
  code_printer.AddLine("");

  return af::SUCCESS;
}

std::string OperatorLevelCacheGen::GenContextClass(const std::string &tiling_data_type_name) {
  std::stringstream ss;

  ss << R"(
/**
 * @brief Tiling缓存上下文类
 * 线程级别的缓存上下文，使用thread_local存储，无需线程ID
 */
class TilingCacheContext {
)" << GenContextClassStructure()
     << GenContextClassPublicMethods() << GenContextCacheOperations(tiling_data_type_name) << GenContextHashFunction()
     << R"(
};
)";

  return ss.str();
}

std::string OperatorLevelCacheGen::GenContextClassStructure() {
  std::stringstream ss;

  ss << R"(
private:
  // 第一级：算子级缓存（thread_local，使用unique_ptr避免栈溢出）
  // 注意：使用kInputShapeSize大小的key，以支持不同数量的输入变量
  static thread_local std::unique_ptr<OperatorLevelCache> operator_cache_;
  static thread_local bool initialized_;
  // 访问计数（用于LRU老化）
  static thread_local std::array<uint64_t, kOperatorCacheCapacity> access_counts_;
)";

  return ss.str();
}

std::string OperatorLevelCacheGen::GenContextClassPublicMethods() {
  std::stringstream ss;

  ss << R"(
public:

  // 获取算子级缓存实例
  static OperatorLevelCache& GetOperatorCache() {
    if (!initialized_) {
      initialized_ = true;
      operator_cache_ = std::make_unique<OperatorLevelCache>();
      // 初始化访问计数
      for (size_t i = 0; i < kOperatorCacheCapacity; ++i) {
        access_counts_[i] = 0;
      }
    }
    return *operator_cache_;
  }

  // 清除算子级缓存
  static void ClearOperatorCache() {
    operator_cache_.reset();
    initialized_ = false;
  }
)";

  return ss.str();
}

std::string OperatorLevelCacheGen::GenFindOperatorCacheImpl(const std::string &tiling_data_type_name) {
  std::stringstream ss;
  ss << R"(
  // 查询算子级缓存（更新访问计数）
  static )"
     << tiling_data_type_name << R"(* FindOperatorCache(const std::array<uint32_t, kInputShapeSize>& shape_key) {
    )"
     << tiling_data_type_name
     << R"(* result = GetOperatorCache().Find(shape_key);
    if (result != nullptr) {
      OP_LOGI(OP_NAME, "[Operator Cache] HIT! key=[%s]", )" +
            GenShapeKeyToStringCode("shape_key") + R"();
      // 更新访问计数
      size_t hash = Hash(shape_key);
      size_t index = hash % kOperatorCacheCapacity;
      access_counts_[index]++;
    } else {
      OP_LOGI(OP_NAME, "[Operator Cache] MISS! key=[%s]", )" +
            GenShapeKeyToStringCode("shape_key") + R"();
    }
    return result;
  }
)";
  return ss.str();
}

std::string OperatorLevelCacheGen::GenSaveOperatorCacheImpl(const std::string &tiling_data_type_name) {
  std::stringstream ss;
  ss << R"(
  // 插入算子级缓存（带LRU老化）
  static bool SaveOperatorCache(const std::array<uint32_t, kInputShapeSize>& shape_key,
                                const )"
     << tiling_data_type_name
     << R"(& tiling_data) {
    auto& cache = GetOperatorCache();

    // 1. 尝试直接插入
    if (cache.Insert(shape_key, tiling_data)) {
      OP_LOGI(OP_NAME, "[Operator Cache] SAVE SUCCESS: key=[%s]", )" +
            GenShapeKeyToStringCode("shape_key") + R"();
      return true;
    }
    OP_LOGI(OP_NAME, "[Operator Cache] SAVE FAILED (cache full), key=[%s]", )" +
            GenShapeKeyToStringCode("shape_key") + R"();

    // 2. 缓存满，执行LRU老化
    if (cache.Size() >= kOperatorCacheCapacity) {
      // 找到访问计数最小的条目
      size_t min_index = 0;
      uint64_t min_count = access_counts_[0];
      for (size_t i = 1; i < kOperatorCacheCapacity; ++i) {
        if (access_counts_[i] < min_count) {
          min_index = i;
          min_count = access_counts_[i];
        }
      }

      OP_LOGI(OP_NAME, "[Operator Cache] Clearing cache (LRU), min_count=%lu", min_count);
      // 淘汰最少使用的条目（简化处理：清空后重新插入）
      cache.Clear();
      for (size_t i = 0; i < kOperatorCacheCapacity; ++i) {
        access_counts_[i] = 0;
      }

      // 重新插入
      return cache.Insert(shape_key, tiling_data);
    }

    return false;
  }
)";
  return ss.str();
}

std::string OperatorLevelCacheGen::GenContextCacheOperations(const std::string &tiling_data_type_name) {
  std::stringstream ss;
  ss << GenFindOperatorCacheImpl(tiling_data_type_name);
  ss << "\n";
  ss << GenSaveOperatorCacheImpl(tiling_data_type_name);
  return ss.str();
}

std::string OperatorLevelCacheGen::GenContextHashFunction() {
  std::stringstream ss;

  ss << R"(
private:
  // Hash函数
  static size_t Hash(const std::array<uint32_t, kInputShapeSize>& key) {
    size_t hash = 0;
    for (const auto& value : key) {
      constexpr uint32_t kHashPrime = 0x9e3779b9;  // 黄金比例的整数表示，用于hash混合
      hash ^= value + kHashPrime + (hash << 6) + (hash >> 2);
    }
    return hash;
  }
)";

  return ss.str();
}
}  // namespace cache
}  // namespace att
