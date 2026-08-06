/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sk_model_context.h
 * \brief Host-only per-model identity: the unique modelId string and the
 *        sk_meta directory label derived from it.
 *
 * This is host-only on purpose: it pulls in dlog (via sk_log.h) and the host
 * filesystem APIs, neither of which exist in device (.asc) compilation. It must
 * therefore never be included from the device-visible sk_common.h.
 */

#ifndef SK_MODEL_CONTEXT_H
#define SK_MODEL_CONTEXT_H

#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>

#include "acl/acl.h"

std::string GetCurrentModelId();

std::string GetCurrentModelLabel();

std::string BuildModelLabel(const std::string &modelId);

// RAII scope installed at the aclskOptimize entry. Bumps the call counter and
// freezes the model identity once, so every downstream consumer reads one stable
// value rather than recomputing from the mutable counter. Nested/reentrant
// contexts are not supported by the current aclskOptimize flow.
class SkModelContext {
 public:
  explicit SkModelContext(aclmdlRI model);

  ~SkModelContext();

  SkModelContext(const SkModelContext &) = delete;
  SkModelContext &operator=(const SkModelContext &) = delete;

 private:
  friend std::string GetCurrentModelId();
  friend std::string GetCurrentModelLabel();

  std::string modelId_;
  std::string modelLabel_;
};

// ==================== sk_meta directory layout ====================

/**
 * @brief Sanitize path component by replacing invalid characters
 * @param component Path component to sanitize
 * @return Sanitized string safe for use as directory name
 */
inline std::string SanitizePathComponent(const std::string &component) {
  std::string result = component;
  for (char &pathChar : result) {
    if (pathChar == '/' || pathChar == '\\' || pathChar == ':' || pathChar == '*' || pathChar == '?' ||
        pathChar == '"' || pathChar == '<' || pathChar == '>' || pathChar == '|') {
      pathChar = '_';
    }
  }
  return result;
}

/**
 * @brief Get sk_meta base directory path (sk_meta/{pid})
 * @return sk_meta/{pid} path string
 *
 * This is the unified path generation function for sk_meta directory structure.
 * If the path structure needs to change in the future, only modify this function.
 */
inline std::string GetSkMetaBasePath() {
  return "sk_meta/" + std::to_string(getpid());
}

/**
 * @brief Get full sk_meta directory path (sk_meta/{pid}/{modelLabel})
 * @param modelLabel Frozen model label
 * @return Full path string
 *
 * This is the unified path generation function for sk_meta directory structure.
 * If the path structure needs to change in the future, only modify this function.
 *
 * @example
 *   std::string path = GetSkMetaPath(modelLabel);
 *   // Returns: "sk_meta/{pid}/model_{modelId}"
 */
inline std::string GetSkMetaPath(const std::string &modelLabel) {
  if (modelLabel.empty()) {
    return "";
  }
  return GetSkMetaBasePath() + "/" + SanitizePathComponent(modelLabel);
}

/**
 * @brief Create directory with full path (recursively create parent directories)
 * @param path Full directory path to create
 * @return true if directory exists or created successfully, false otherwise
 */
inline bool CreateDirectoryRecursive(const std::string &path) {
  if (path.empty()) {
    return false;
  }

  size_t pos = 0;
  do {
    pos = path.find('/', pos + 1);
    std::string subPath = path.substr(0, pos);

    if (subPath.empty()) {
      continue;
    }

    struct stat st;
    if (stat(subPath.c_str(), &st) != 0) {
      if (mkdir(subPath.c_str(), 0755) != 0 && errno != EEXIST) {
        return false;
      }
    }
  } while (pos != std::string::npos && pos < path.size());

  return true;
}

/**
 * @brief Create sk_meta directory structure: sk_meta/{pid}/{modelLabel}
 * @param modelLabel Frozen model label
 * @return Full path of created directory, empty string on failure
 *
 * This function creates the directory structure using the unified path generator:
 * - sk_meta/{pid} (always created)
 * - sk_meta/{pid}/{modelLabel} (created based on model pointer)
 *
 * @example
 *   std::string path = CreateSkMetaDirectory(modelLabel);
 *   // Creates: sk_meta/{pid}/model_{modelId}
 *   // Returns: "sk_meta/{pid}/model_{modelId}"
 */
inline std::string CreateSkMetaDirectory(const std::string &modelLabel) {
  std::string dirPath = GetSkMetaPath(modelLabel);
  if (!CreateDirectoryRecursive(dirPath)) {
    return "";
  }
  return dirPath;
}

#endif  // SK_MODEL_CONTEXT_H
