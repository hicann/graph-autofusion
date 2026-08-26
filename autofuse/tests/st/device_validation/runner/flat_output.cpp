/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flat_output.h"

#include <atomic>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>

#include <fstream>
#include <unordered_set>

namespace device_validation {
namespace {

FlatOutputTestHooks test_hooks;

std::atomic<uint64_t> temp_sequence{0};

void SetError(std::string *error, const char *message) {
  if (error != nullptr) {
    *error = message;
  }
}

bool CreateFlatTemporaryFile(const std::filesystem::path &directory, const std::filesystem::path &filename,
                             size_t output_index, std::vector<std::pair<std::filesystem::path, bool>> *cleanup_paths,
                             std::filesystem::path *temporary_path, int *temporary_fd, std::string *error) {
  std::filesystem::path candidate;
  constexpr int kMaxTemporaryPathAttempts = 100;
  for (int attempt = 0; attempt < kMaxTemporaryPathAttempts; ++attempt) {
    candidate = directory / (filename.string() + ".tmp." + std::to_string(temp_sequence.fetch_add(1)));
    cleanup_paths->emplace_back(candidate, false);
    const size_t cleanup_index = cleanup_paths->size() - 1;
    if (test_hooks.create_temp_file_at == static_cast<int>(output_index + 1)) {
      std::ofstream competing_temp(candidate, std::ios::binary);
      competing_temp.put('\x33');
      test_hooks.create_temp_file_at = 0;
    }
    if (test_hooks.create_temp_symlink_at == static_cast<int>(output_index + 1)) {
      const auto symlink_target = directory / (filename.string() + ".symlink-target");
      std::ofstream competing_target(symlink_target, std::ios::binary);
      competing_target.put('\x44');
      std::error_code symlink_error;
      std::filesystem::create_symlink(symlink_target.filename(), candidate, symlink_error);
      if (symlink_error) {
        SetError(error, "output file cannot be written");
        return false;
      }
      test_hooks.create_temp_symlink_at = 0;
    }
    *temporary_fd = open(candidate.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600);
    if (*temporary_fd >= 0) {
      (*cleanup_paths)[cleanup_index].second = true;
      break;
    }
    if (errno != EEXIST) {
      SetError(error, "output file cannot be written");
      return false;
    }
  }
  if (*temporary_fd < 0) {
    SetError(error, "output file cannot be written");
    return false;
  }
  *temporary_path = candidate;
  return true;
}

bool WriteTemporaryFileContents(int temporary_fd, const std::vector<uint8_t> &bytes, size_t output_index,
                                std::string *error) {
  size_t written = 0;
  while (written < bytes.size()) {
    const ssize_t count = write(temporary_fd, bytes.data() + written, bytes.size() - written);
    if (count > 0) {
      written += static_cast<size_t>(count);
    } else if (count < 0 && errno == EINTR) {
      continue;
    } else {
      close(temporary_fd);
      SetError(error, "output file cannot be written");
      return false;
    }
  }
  if (fsync(temporary_fd) != 0 || test_hooks.fail_flush_at == static_cast<int>(output_index + 1)) {
    close(temporary_fd);
    SetError(error, "output file cannot be written");
    return false;
  }
  if (close(temporary_fd) != 0 || test_hooks.fail_close_at == static_cast<int>(output_index + 1)) {
    SetError(error, "output file cannot be written");
    return false;
  }
  return true;
}

bool WriteFlatTemporaryFile(const std::filesystem::path &output_path, size_t output_index,
                            const std::vector<uint8_t> &bytes,
                            std::vector<std::pair<std::filesystem::path, bool>> *cleanup_paths,
                            std::filesystem::path *temporary_path, std::string *error) {
  const auto parent = output_path.parent_path();
  const auto directory = parent.empty() ? std::filesystem::path(".") : parent;
  int temporary_fd = -1;
  std::filesystem::path candidate;
  if (!CreateFlatTemporaryFile(directory, output_path.filename(), output_index, cleanup_paths, &candidate,
                               &temporary_fd, error))
    return false;
  *temporary_path = candidate;
  return WriteTemporaryFileContents(temporary_fd, bytes, output_index, error);
}

bool CommitFlatTemporaryFile(const std::filesystem::path &temporary_path, const std::filesystem::path &output_path,
                             size_t output_index, std::string *error) {
  std::error_code commit_error;
  if (test_hooks.create_target_at == static_cast<int>(output_index + 1)) {
    std::ofstream competing_target(output_path, std::ios::binary | std::ios::trunc);
    competing_target.put('\x55');
  }
  if (test_hooks.fail_rename_at == static_cast<int>(output_index + 1)) {
    commit_error = std::make_error_code(std::errc::io_error);
  } else {
    // Linux/POSIX same-directory hard links commit atomically without replacing an existing target.
    std::filesystem::create_hard_link(temporary_path, output_path, commit_error);
  }
  if (commit_error) {
    SetError(error, "output file cannot be written");
    return false;
  }
  std::error_code remove_error;
  std::filesystem::remove(temporary_path, remove_error);
  if (remove_error) {
    SetError(error, "output file cannot be written");
    return false;
  }
  return true;
}

}  // namespace

void SetFlatOutputTestHooks(const FlatOutputTestHooks &hooks) {
  test_hooks = hooks;
}

void ClearFlatOutputTestHooks() {
  test_hooks = {};
}

bool ValidateFlatOutputPaths(const std::vector<std::string> &output_files, const std::string &artifact_dir,
                             std::vector<std::filesystem::path> *output_paths, std::string *error) {
  std::unordered_set<std::string> unique_paths;
  for (const auto &output_file : output_files) {
    const std::filesystem::path output_path(output_file);
    const auto parent = output_path.parent_path();
    if (output_path.empty() || output_path.filename().empty()) {
      SetError(error, "output path is invalid");
      return false;
    }
    if (!parent.empty() && !std::filesystem::is_directory(parent)) {
      SetError(error, "output parent directory is unavailable");
      return false;
    }
    std::error_code containment_error;
    const auto root = std::filesystem::weakly_canonical(artifact_dir, containment_error);
    const auto canonical_parent = std::filesystem::weakly_canonical(parent.empty() ? "." : parent, containment_error);
    const auto relative_parent = canonical_parent.lexically_relative(root);
    if (containment_error || relative_parent.empty() || relative_parent == ".." ||
        relative_parent.string().compare(0, 3, "../") == 0) {
      SetError(error, "output path is outside artifact directory");
      return false;
    }
    if (!unique_paths.insert(output_path.lexically_normal().string()).second) {
      SetError(error, "output path is duplicated");
      return false;
    }
    std::error_code status_error;
    const auto status = std::filesystem::symlink_status(output_path, status_error);
    const bool output_exists = !status_error && status.type() != std::filesystem::file_type::not_found;
    const bool status_failed =
        status_error && status_error != std::make_error_code(std::errc::no_such_file_or_directory);
    if (output_exists || status_failed) {
      SetError(error, "output file already exists");
      return false;
    }
    output_paths->push_back(output_path);
  }
  return true;
}

bool WriteFlatOutputs(const std::vector<std::string> &output_files,
                      const std::vector<std::vector<uint8_t>> &output_bytes, const std::string &artifact_dir,
                      std::string *error) {
  if (output_files.size() != output_bytes.size()) {
    SetError(error, "output count does not match result");
    return false;
  }
  std::vector<std::filesystem::path> output_paths;
  if (!ValidateFlatOutputPaths(output_files, artifact_dir, &output_paths, error)) return false;
  std::vector<std::pair<std::filesystem::path, bool>> cleanup_paths;
  std::vector<std::filesystem::path> temporary_paths;
  const auto cleanup = [&cleanup_paths]() {
    for (const auto &[temporary_path, owned] : cleanup_paths) {
      if (!owned) {
        continue;
      }
      std::error_code remove_error;
      std::filesystem::remove(temporary_path, remove_error);
    }
  };

  for (size_t i = 0; i < output_paths.size(); ++i) {
    std::filesystem::path temporary_path;
    if (!WriteFlatTemporaryFile(output_paths[i], i, output_bytes[i], &cleanup_paths, &temporary_path, error)) {
      cleanup();
      return false;
    }
    temporary_paths.push_back(temporary_path);
  }

  // Outputs commit independently; a later failure does not roll back earlier links.
  for (size_t i = 0; i < output_paths.size(); ++i) {
    if (!CommitFlatTemporaryFile(temporary_paths[i], output_paths[i], i, error)) {
      cleanup();
      return false;
    }
  }
  return true;
}

}  // namespace device_validation
