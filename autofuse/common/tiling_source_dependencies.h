/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_COMMON_TILING_SOURCE_DEPENDENCIES_H_
#define AUTOFUSE_COMMON_TILING_SOURCE_DEPENDENCIES_H_

#include <set>
#include <string>
#if __has_include("ge_common_af/ge_api_error_codes_af.h")
#include "ge_common_af/ge_api_error_codes_af.h"
#else
#include "ge_common_af/ge_api_error_codes.h"
#endif

namespace autofuse {

enum class GeneratedHeaderId {
  kTilingData,
  kState,
  kLog,
  kPgo,
  kSolver,
  kApi,
};

struct SourceDependencies {
  std::set<std::string> system_headers;
  std::set<std::string> external_headers;
  std::set<std::string> cce_kt_excluded_external_headers;
  std::set<GeneratedHeaderId> generated_headers;
};

struct GeneratedCode {
  std::string body;
  SourceDependencies dependencies;
};

inline void RequireSystemHeader(SourceDependencies &dependencies, const std::string &header) {
  dependencies.system_headers.insert(header);
}

inline void RequireExternalHeader(SourceDependencies &dependencies, const std::string &header) {
  dependencies.external_headers.insert(header);
}

inline void RequireExternalHeaderUnlessCceKtTest(SourceDependencies &dependencies, const std::string &header) {
  dependencies.cce_kt_excluded_external_headers.insert(header);
}

inline void RequireGeneratedHeader(SourceDependencies &dependencies, GeneratedHeaderId header_id) {
  dependencies.generated_headers.insert(header_id);
}

inline void MergeDependencies(SourceDependencies &target, const SourceDependencies &source) {
  target.system_headers.insert(source.system_headers.begin(), source.system_headers.end());
  target.external_headers.insert(source.external_headers.begin(), source.external_headers.end());
  target.cce_kt_excluded_external_headers.insert(source.cce_kt_excluded_external_headers.begin(),
                                                 source.cce_kt_excluded_external_headers.end());
  target.generated_headers.insert(source.generated_headers.begin(), source.generated_headers.end());
}

inline void AppendGeneratedCode(GeneratedCode &target, const GeneratedCode &source) {
  target.body += source.body;
  MergeDependencies(target.dependencies, source.dependencies);
}

inline af::Status GetGeneratedHeaderFileName(GeneratedHeaderId header_id, std::string &file_name) {
  switch (header_id) {
    case GeneratedHeaderId::kTilingData:
      file_name = "autofuse_tiling_data.h";
      break;
    case GeneratedHeaderId::kState:
      file_name = "autofuse_tiling_func_state.h";
      break;
    case GeneratedHeaderId::kLog:
      file_name = "autofuse_tiling_func_log.h";
      break;
    case GeneratedHeaderId::kPgo:
      file_name = "autofuse_tiling_func_pgo.h";
      break;
    case GeneratedHeaderId::kSolver:
      file_name = "autofuse_tiling_func_solver.h";
      break;
    case GeneratedHeaderId::kApi:
      file_name = "autofuse_tiling_func_api.h";
      break;
    default:
      return af::FAILED;
  }
  return af::SUCCESS;
}

inline af::Status RenderIncludes(const SourceDependencies &dependencies, std::string &output) {
  output.clear();
  for (const auto &header : dependencies.system_headers) {
    output += "#include <" + header + ">\n";
  }
  for (const auto &header : dependencies.external_headers) {
    output += "#include \"" + header + "\"\n";
  }
  for (const auto header_id : dependencies.generated_headers) {
    std::string file_name;
    if (GetGeneratedHeaderFileName(header_id, file_name) != af::SUCCESS) {
      output.clear();
      return af::FAILED;
    }
    output += "#include \"" + file_name + "\"\n";
  }
  if (!dependencies.cce_kt_excluded_external_headers.empty()) {
    output += "#ifndef __CCE_KT_TEST__\n";
    for (const auto &header : dependencies.cce_kt_excluded_external_headers) {
      output += "#include \"" + header + "\"\n";
    }
    output += "#endif\n";
  }
  return af::SUCCESS;
}

inline af::Status RenderTranslationUnit(const GeneratedCode &code, std::string &output) {
  std::string includes;
  if (RenderIncludes(code.dependencies, includes) != af::SUCCESS) {
    return af::FAILED;
  }
  output = includes;
  if (!includes.empty() && !code.body.empty()) {
    output += "\n";
  }
  output += code.body;
  return af::SUCCESS;
}

inline af::Status RenderGeneratedHeader(const GeneratedCode &code, const std::string &guard, std::string &output) {
  if (!code.dependencies.generated_headers.empty()) {
    output.clear();
    return af::FAILED;
  }
  std::string includes;
  if (RenderIncludes(code.dependencies, includes) != af::SUCCESS) {
    return af::FAILED;
  }
  output = "#ifndef " + guard + "\n#define " + guard + "\n\n";
  if (!includes.empty()) {
    output += includes + "\n";
  }
  output += code.body + "\n#endif  // " + guard + "\n";
  return af::SUCCESS;
}

}  // namespace autofuse

#endif  // AUTOFUSE_COMMON_TILING_SOURCE_DEPENDENCIES_H_
