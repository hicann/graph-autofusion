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
 * \file sk_log.cpp
 * \brief Super Kernel Log Module Implementation (Merged File Logger)
 */

#include "sk_log.h"
#include "sk_common.h"
#include "sk_model_context.h"
#include "securec.h"
#include "base/err_mgr.h"
#include <iostream>
#include <cstdarg>
#include <cstdlib>
#include <limits.h>

// ==================== Original Error Report Function ====================
constexpr size_t LIMIT_PREDEFINED_MESSAGE = 1024U;

void ReportErrorMessageInner(const std::string &code, const char *fmt, ...) {
  std::vector<char> buf(LIMIT_PREDEFINED_MESSAGE, '\0');
  va_list argList;
  va_start(argList, fmt);
  auto ret = vsnprintf_s(buf.data(), LIMIT_PREDEFINED_MESSAGE, LIMIT_PREDEFINED_MESSAGE - 1U, fmt, argList);
  if (ret == -1) {
    SK_DLOGE("Construct error message failed, maybe the length of error message exceed limits: %zu",
             LIMIT_PREDEFINED_MESSAGE);
  }
  va_end(argList);
  const std::vector<const char *> msgKey = {"message"};
  const std::vector<const char *> msgValue = {buf.data()};
  REPORT_PREDEFINED_ERR_MSG(code.c_str(), msgKey, msgValue);
}

// ==================== FileHandleManager Implementation ====================
namespace sk {
namespace logger {

bool FileHandleManager::RegisterFile(const std::string &name, const std::string &path) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (handles_.find(name) != handles_.end()) {
    return true;  // Already registered
  }

  FileHandleInfo handle;
  handle.filePath = path;
  handle.fileStream.open(path, std::ios::app | std::ios::out);

  if (!handle.fileStream.is_open()) {
    SK_DLOGE("Failed to open log file: %s", path.c_str());
    return false;
  }

  handle.createTime = std::chrono::system_clock::now();

  // Get current file size for existing files
  handle.fileStream.seekp(0, std::ios::end);
  handle.currentSize = static_cast<size_t>(handle.fileStream.tellp());

  handles_.emplace(name, std::move(handle));
  return true;
}

bool FileHandleManager::SwitchToFile(const std::string &name) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = handles_.find(name);
  if (it == handles_.end()) {
    SK_DLOGE("File handle not found: %s", name.c_str());
    return false;
  }

  currentHandle_ = name;
  return true;
}

void FileHandleManager::SwitchToDefault() {
  std::lock_guard<std::mutex> lock(mutex_);
  currentHandle_ = "default";
}

bool FileHandleManager::Write(const std::string &name, const std::string &content) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = handles_.find(name);
  if (it == handles_.end() || !it->second.fileStream.is_open()) {
    return false;
  }

  auto &handle = it->second;
  handle.fileStream << content;
  handle.fileStream.flush();

  handle.currentSize += content.size();
  handle.writeCount++;

  return true;
}

bool FileHandleManager::WriteToCurrent(const std::string &content) {
  return Write(currentHandle_, content);
}

std::string FileHandleManager::GetCurrentHandle() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return currentHandle_;
}

size_t FileHandleManager::GetFileSize(const std::string &name) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = handles_.find(name);
  if (it == handles_.end()) {
    return 0;
  }
  return it->second.currentSize;
}

void FileHandleManager::CloseFile(const std::string &name) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = handles_.find(name);
  if (it != handles_.end()) {
    if (it->second.fileStream.is_open()) {
      it->second.fileStream.flush();
      it->second.fileStream.close();
    }
    handles_.erase(it);
  }
}

bool FileHandleManager::InitializeDefault(const std::string &baseDir, pid_t pid, const std::string &modelLabel) {
  std::string dirPath = GetSkMetaBasePath() + "/" + SanitizePathComponent(modelLabel);
  if (!CreateDirectoryRecursive(dirPath)) {
    return false;
  }

  std::string handleName = "model_" + SanitizePathComponent(modelLabel);

  std::string defaultPath = dirPath + "/super_kernel.log";
  return RegisterFile(handleName, defaultPath);
}

// Thread-local current handle initialization
thread_local std::string FileHandleManager::currentHandle_ = "default";

// Thread-local current model label
thread_local std::string FileLogger::currentModelLabel_;

FileHandleManager::FileHandleManager() {}

FileHandleManager::~FileHandleManager() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto &pair : handles_) {
    if (pair.second.fileStream.is_open()) {
      pair.second.fileStream.flush();
      pair.second.fileStream.close();
    }
  }
}

// ==================== LogContextGuard Implementation ====================
LogContextGuard::LogContextGuard(const std::string &modelLabel) {
  auto &logger = FileLogger::Instance();
  if (!logger.IsInitialized() || !logger.IsEnabled()) {
    return;
  }

  previousModelLabel_ = FileLogger::GetCurrentModelLabel();
  previousHandle_ = FileHandleManager::Instance().GetCurrentHandle();
  FileLogger::SetCurrentModelLabel(modelLabel);
  FileHandleManager::Instance().SwitchToDefault();
  active_ = true;
}

LogContextGuard::LogContextGuard(const std::string &handleName, const std::string &filePath)
    : previousModelLabel_(FileLogger::GetCurrentModelLabel()),
      previousHandle_(FileHandleManager::Instance().GetCurrentHandle()) {
  if (!FileHandleManager::Instance().RegisterFile(handleName, filePath)) {
    SK_DLOGE("Failed to register log file: %s", handleName.c_str());
    return;
  }

  if (!FileHandleManager::Instance().SwitchToFile(handleName)) {
    SK_DLOGE("Failed to switch to log file: %s", handleName.c_str());
    return;
  }

  active_ = true;
}

LogContextGuard::~LogContextGuard() {
  Restore();
}

LogContextGuard::LogContextGuard(LogContextGuard &&other) noexcept
    : previousModelLabel_(std::move(other.previousModelLabel_)),
      previousHandle_(std::move(other.previousHandle_)),
      active_(other.active_) {
  other.active_ = false;
}

LogContextGuard &LogContextGuard::operator=(LogContextGuard &&other) noexcept {
  if (this != &other) {
    Restore();
    previousModelLabel_ = std::move(other.previousModelLabel_);
    previousHandle_ = std::move(other.previousHandle_);
    active_ = other.active_;
    other.active_ = false;
  }
  return *this;
}

void LogContextGuard::Restore() {
  if (!active_) {
    return;
  }

  FileLogger::SetCurrentModelLabel(previousModelLabel_);
  if (previousHandle_ == "default") {
    FileHandleManager::Instance().SwitchToDefault();
  } else if (!previousHandle_.empty()) {
    FileHandleManager::Instance().SwitchToFile(previousHandle_);
  }
  active_ = false;
}

// ==================== FileLogger Implementation ====================
FileLogger &FileLogger::Instance() {
  static FileLogger instance;
  return instance;
}

bool FileLogger::Initialize(const LoggerConfig &config) {
  std::lock_guard<std::mutex> lock(mutex_);

  std::string modelLabel = config.modelLabel;
  if (modelLabel.empty()) {
    modelLabel = GetCurrentModelLabel();
  }
  if (modelLabel.empty()) {
    modelLabel = config_.modelLabel;
  }

  config_ = config;
  config_.modelLabel = modelLabel;
  if (!config_.enabled) {
    SK_DLOGI("File logger is disabled");
    initialized_.store(true);
    return true;
  }

  // Use the current model label to create sk_meta directory structure.
  std::string logDir = GetSkMetaBasePath() + "/" + SanitizePathComponent(config_.modelLabel);
  if (!CreateDirectoryRecursive(logDir)) {
    logDir.clear();
  }
  if (logDir.empty() && config_.enabled) {
    SK_DLOGE("Failed to create sk_meta directory");
    return false;
  }

  // Extract PID from created directory
  pid_ = getpid();

  // 为新 model label 注册日志文件
  std::string modelLabelForLog = config_.modelLabel;
  std::string handleName = "model_" + SanitizePathComponent(modelLabelForLog);
  std::string defaultPath = logDir + "/super_kernel.log";

  if (!FileHandleManager::Instance().RegisterFile(handleName, defaultPath)) {
    SK_DLOGE("Failed to register log file for model: %s", modelLabelForLog.c_str());
    return false;
  }

  // 切换到新的日志文件
  FileHandleManager::Instance().SwitchToFile(handleName);

  initialized_.store(true);

  // Convert to absolute path for better visibility
  char absPath[PATH_MAX] = {0};  // Initialize buffer with zeros
  if (realpath(logDir.c_str(), absPath) != nullptr) {
    // Ensure null termination
    absPath[PATH_MAX - 1] = '\0';
    SK_DLOGI("File logger initialized: dir=%s, model=%s", absPath, modelLabelForLog.c_str());
  } else {
    SK_DLOGI("File logger initialized: dir=%s, model=%s", logDir.c_str(), modelLabelForLog.c_str());
  }
  return true;
}

std::string FileLogger::GetEffectiveModelLabel() const {
  if (!currentModelLabel_.empty()) {
    return currentModelLabel_;
  }
  return config_.modelLabel;
}

bool FileLogger::RegisterLogFile(const std::string &name, const std::string &subPath) {
  if (!initialized_.load() || !config_.enabled) {
    return false;
  }

  std::string modelLabel = GetEffectiveModelLabel();
  if (modelLabel.empty()) {
    return false;
  }

  std::string basePath = GetSkMetaBasePath() + "/" + SanitizePathComponent(modelLabel);
  std::string filePath = basePath + "/" + name;

  // If subPath provided, insert it before filename
  if (!subPath.empty()) {
    filePath = basePath + "/" + SanitizePathComponent(subPath) + "/" + name;
  }

  // Ensure directory exists using common utility
  size_t lastSlash = filePath.find_last_of('/');
  if (lastSlash != std::string::npos) {
    std::string dir = filePath.substr(0, lastSlash);
    if (!CreateDirectoryRecursive(dir)) {
      return false;
    }
  }

  std::string handleName = "model_" + SanitizePathComponent(modelLabel) + "_" + name;

  return FileHandleManager::Instance().RegisterFile(handleName, filePath);
}

bool FileLogger::SwitchToFile(const std::string &name) {
  if (!initialized_.load() || !config_.enabled) {
    return false;
  }
  return FileHandleManager::Instance().SwitchToFile(name);
}

void FileLogger::SwitchToDefault() {
  FileHandleManager::Instance().SwitchToDefault();
}

std::unique_ptr<LogContextGuard> FileLogger::CreateContext(const std::string &fileName, const std::string &modelLabel) {
  if (!initialized_.load() || !config_.enabled) {
    return nullptr;
  }

  if (modelLabel.empty()) {
    return nullptr;
  }

  std::string sanitizedModelLabel = SanitizePathComponent(modelLabel);
  std::string dirPath = GetSkMetaBasePath() + "/" + sanitizedModelLabel;
  if (!CreateDirectoryRecursive(dirPath)) {
    SK_DLOGE("Failed to create directory for context");
    return nullptr;
  }

  std::string filePath = dirPath + "/" + fileName;
  std::string handleName = "model_" + sanitizedModelLabel + "_" + fileName;

  return std::make_unique<LogContextGuard>(handleName, filePath);
}

void FileLogger::SetEnabled(bool enabled) {
  config_.enabled = enabled;
}

bool FileLogger::IsEnabled() const {
  return config_.enabled;
}

void FileLogger::SetMinLevel(LogLevel level) {
  config_.minLevel = level;
}

void FileLogger::SetModelLabel(const std::string &modelLabel) {
  config_.modelLabel = modelLabel;
}

bool FileLogger::IsInitialized() const {
  return initialized_.load();
}

const std::string &FileLogger::GetCurrentModelLabel() {
  return currentModelLabel_;
}

std::string FileLogger::FormatMessage(LogLevel level, const char *funcName, const char *fileName, int lineNum,
                                      const char *format, ...) {
  std::ostringstream oss;

  // Timestamp
  if (config_.enableTimestamp) {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    oss << "." << std::setfill('0') << std::setw(3) << ms.count();
    oss << " ";
  }

  // Process/Thread ID
  if (config_.enablePidTid) {
    oss << "[" << pid_ << ":" << std::this_thread::get_id() << "] ";
  }

  // Log level
  oss << "[" << LogLevelToString(level) << "] ";

  // File name and line number (extract just the filename from full path)
  if (fileName != nullptr) {
    const char *baseName = strrchr(fileName, '/');
    if (baseName != nullptr) {
      baseName++;  // Skip '/'
    } else {
      baseName = fileName;
    }
    oss << "[" << baseName << ":" << lineNum << "] ";
  }

  // Function name
  if (funcName != nullptr) {
    oss << "[" << funcName << "] ";
  }

  // Log content - use secure function
  va_list args;
  va_start(args, format);
  char buffer[4096];
  int ret = vsnprintf_s(buffer, sizeof(buffer), sizeof(buffer) - 1, format, args);
  va_end(args);

  if (ret < 0) {
    // Format error, use error message
    oss << "[FORMAT_ERROR] ";
    oss << "Failed to format log message\n";
  } else {
    oss << buffer << "\n";
  }

  return oss.str();
}

void FileLogger::WriteLog(const std::string &message) {
  if (message.empty()) {
    return;
  }

  // 获取当前 handle，判断是否在 LogContextGuard 上下文中
  std::string currentHandle = FileHandleManager::Instance().GetCurrentHandle();
  std::string targetHandle;

  // 如果当前 handle 是 "default" 或以 "model_" 开头但没有额外后缀，
  // 说明没有通过 LogContextGuard 切换到非 default 日志
  // 此时需要根据当前 model label 计算正确的 handle
  if (currentHandle == "default" || currentHandle.find('_') == currentHandle.rfind('_')) {
    std::string modelLabel = GetEffectiveModelLabel();
    targetHandle = "model_" + SanitizePathComponent(modelLabel);
  } else {
    // 在 LogContextGuard 上下文中，使用当前 handle
    targetHandle = currentHandle;
  }

  // Long log segmentation handling
  if (message.size() > config_.maxLineLength) {
    size_t offset = 0;
    size_t segmentNum = 0;

    while (offset < message.size()) {
      size_t length = std::min(config_.maxLineLength, message.size() - offset);
      std::string segment;

      if (segmentNum == 0) {
        segment = message.substr(offset, length);
      } else {
        segment = "[CONT:" + std::to_string(segmentNum) + "] " + message.substr(offset, length);
      }

      FileHandleManager::Instance().Write(targetHandle, segment);
      offset += length;
      segmentNum++;
    }
  } else {
    FileHandleManager::Instance().Write(targetHandle, message);
  }
}

}  // namespace logger
}  // namespace sk

// ==================== Global Initialization Helper Functions ====================

bool InitializeSkFileLogger(bool enabled, const std::string &modelLabel, sk::logger::LogLevel minLevel) {
  sk::logger::FileLogger::Instance().SetCurrentModelLabel(modelLabel);
  sk::logger::LoggerConfig config;
  config.enabled = enabled;
  config.modelLabel = modelLabel;
  config.minLevel = minLevel;

  if (!sk::logger::FileLogger::Instance().Initialize(config)) {
    // Disable file logging on initialization failure
    sk::logger::FileLogger::Instance().SetEnabled(false);
    return false;
  }
  return true;
}
