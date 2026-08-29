/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Source Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SK_FILE_GUARD_H
#define SK_FILE_GUARD_H

#include <cstdio>
#include <string>
#include "sk_log.h"

// RAII wrapper for FILE*, ensures exception safety
class FileGuard {
 public:
  explicit FileGuard(const char *filename = nullptr, const char *mode = "rb") {
    if (filename != nullptr) {
      filePtr = fopen(filename, mode);
    } else {
      filePtr = nullptr;
    }
  }
  ~FileGuard() {
    Close();
  }

  FileGuard(const FileGuard &) = delete;
  FileGuard &operator=(const FileGuard &) = delete;

  FileGuard(FileGuard &&other) noexcept : filePtr(other.filePtr) {
    other.filePtr = nullptr;
  }

  FILE *Get() const {
    return filePtr;
  }
  bool IsValid() const {
    return filePtr != nullptr;
  }
  operator bool() const {
    return IsValid();
  }

  // Open file
  bool Open(const char *filename, const char *mode) {
    Close();
    filePtr = fopen(filename, mode);
    return IsValid();
  }

  void Close() {
    if (filePtr != nullptr) {
      fclose(filePtr);
      filePtr = nullptr;
    }
  }

  // Release ownership of file handle, caller is responsible for closing
  FILE *Release() {
    FILE *fp = filePtr;
    filePtr = nullptr;
    return fp;
  }

 private:
  FILE *filePtr;
};

// Extract profiling path from MsprofGetPath
// Return path format:
// parent_dir/profiling/0002_3675077_20260321093945052_ascend_pt/PROF_000001_20260321093945081_03675077MGDRRBBN
static std::string GetBasePath() {
  const char *pathRaw = MsprofGetPath();
  if (pathRaw == nullptr || pathRaw[0] == '\0') {
    SK_DLOGE("[sk time profiling] MsprofGetPath returned empty path\n");
    SK_DLOGI("[sk time profiling] Profiler should start before the network starts, please check it\n");
    return "";
  }
  std::string path(pathRaw);

  SK_DLOGI("[sk time profiling] Output directory: %s\n", path.c_str());
  return path;
}

#endif  // SK_FILE_GUARD_H
