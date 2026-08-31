/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCGEN_COMMON_H__
#define __ASCGEN_COMMON_H__

#include <cstdio>
#include <cstdint>
#include <memory>
#include <unistd.h>
#include "dlog_pub.h"
#include "common/checker.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ASCGEN_LOG_HEADER "[AUTOFUSE]"
#define LOG_ERROR_HEADER "[ERROR]"
#define WARNING_ERROR_HEADER "[WARNING]"

#define ASCGEN_MODULE_NAME static_cast<int32_t>(GE)

using Status = uint32_t;
constexpr std::int32_t ASCCOMMON_SUC = 0;
constexpr std::int32_t ASCCOMMON_ERR = 1;

// 日志宏定义
#define LOG_PRINT(fmt, ...)                                                    \
  do {                                                                         \
    dlog_info(ASCGEN_MODULE_NAME, ASCGEN_LOG_HEADER fmt ".\n", ##__VA_ARGS__); \
  } while (0)

#define ERROR_PRINT(fmt, ...)                                                   \
  do {                                                                          \
    dlog_error(ASCGEN_MODULE_NAME, ASCGEN_LOG_HEADER fmt ".\n", ##__VA_ARGS__); \
  } while (0)

#define WARNING_PRINT(fmt, ...)                                                \
  do {                                                                         \
    dlog_warn(ASCGEN_MODULE_NAME, ASCGEN_LOG_HEADER fmt ".\n", ##__VA_ARGS__); \
  } while (0)

#define ASCCOMMON_CHK_STATUS_RET(expr, ...) \
  do {                                      \
    if ((expr) != ASCCOMMON_SUC) {          \
      ERROR_PRINT(__VA_ARGS__);             \
      return ASCCOMMON_ERR;                 \
    }                                       \
  } while (false)

#define ASCCOMMON_ASSERT_TRUE(ptr, ...) \
  do {                                  \
    if (!(ptr)) {                       \
      ERROR_PRINT(__VA_ARGS__);         \
      return ASCCOMMON_ERR;             \
    }                                   \
  } while (false)

#define PY_ASSERT_EQ(x, y)                                                                     \
  do {                                                                                         \
    const auto &xv = (x);                                                                      \
    const auto &yv = (y);                                                                      \
    if (xv != yv) {                                                                            \
      std::stringstream ss;                                                                    \
      ss << "Assert (" << #x << " == " << #y << ") failed, expect " << yv << " actual " << xv; \
      PyErr_Format(PyExc_TypeError, "%s", ss.str().c_str());                                   \
      GELOGE(af::FAILED, "%s", ss.str().c_str());                                              \
      return ::ErrorResult();                                                                  \
    }                                                                                          \
  } while (false)

#define PY_ASSERT(exp, ...)                                      \
  do {                                                           \
    if (!(exp)) {                                                \
      auto msg = CreateErrorMsg(__VA_ARGS__);                    \
      if (msg.empty()) {                                         \
        PyErr_Format(PyExc_TypeError, "Assert %s failed", #exp); \
        GELOGE(af::FAILED, "Assert %s failed", #exp);            \
      } else {                                                   \
        PyErr_Format(PyExc_TypeError, "%s", msg.data());         \
        GELOGE(af::FAILED, "Assert %s failed", msg.data());      \
      }                                                          \
      return ::ErrorResult();                                    \
    }                                                            \
  } while (false)
#define PY_ASSERT_NOTNULL(v, ...) PY_ASSERT(((v) != nullptr), __VA_ARGS__)
#define PY_ASSERT_SUCCESS(v, ...) PY_ASSERT(((v) == af::SUCCESS), __VA_ARGS__)
#define PY_ASSERT_GRAPH_SUCCESS(v, ...) PY_ASSERT(((v) == ge::GRAPH_SUCCESS), __VA_ARGS__)
#define PY_ASSERT_TRUE(exp, msg)                \
  do {                                          \
    if (!(exp)) {                               \
      PyErr_Clear();                            \
      PyErr_Format(PyExc_TypeError, "%s", msg); \
      GELOGE(af::FAILED, "%s", msg);            \
      return nullptr;                           \
    }                                           \
  } while (false)
#ifdef __cplusplus
}
#endif
#endif
