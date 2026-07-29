/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sk_scope_launch.cpp
 * \brief
 */

#include "sk_scope_launch.h"

#include <cstring>
#include <cstdlib>

#include "securec.h"
#include "sk_common.h"
#include "sk_log.h"

extern "C" void sk_scope_kernel_begin_do_dav_2201(void *stream, ScopeKernelArgs args);
extern "C" void sk_scope_kernel_end_do_dav_2201(void *stream, ScopeKernelArgs args);
extern "C" void sk_placeholder_kernel_do_dav_2201(void *stream, ScopeKernelArgs args);
extern "C" void sk_scope_kernel_begin_do_dav_3510(void *stream, ScopeKernelArgs args);
extern "C" void sk_scope_kernel_end_do_dav_3510(void *stream, ScopeKernelArgs args);
extern "C" void sk_placeholder_kernel_do_dav_3510(void *stream, ScopeKernelArgs args);

typedef void (*ScopeFuncImpl)(void *, ScopeKernelArgs);

namespace {

struct ScopeKernelImpls {
  ScopeFuncImpl begin;
  ScopeFuncImpl end;
  ScopeFuncImpl placeholder;
};

ScopeKernelImpls GetScopeKernelImpls() {
  const SkKernelArch arch = GetCurrentSkKernelArch();
  SK_DLOGI("Scope kernel implementation selected: arch=%s, symbolSuffix=%s", to_string(arch),
           GetSkKernelArchSymbolSuffix(arch));
  switch (arch) {
    case SkKernelArch::DAV_3510:
      return {sk_scope_kernel_begin_do_dav_3510, sk_scope_kernel_end_do_dav_3510, sk_placeholder_kernel_do_dav_3510};
    case SkKernelArch::DAV_2201:
    default:
      return {sk_scope_kernel_begin_do_dav_2201, sk_scope_kernel_end_do_dav_2201, sk_placeholder_kernel_do_dav_2201};
  }
}

aclError LaunchScopeKernelImpl(const char *scopeName, aclrtStream stream, ScopeFuncImpl scopeKernelImpl,
                               const char *markerType) {
  ScopeKernelArgs args;
  if (scopeName == nullptr) {
    scopeName = "default_sk_scope_name";
  }
  size_t len = strlen(scopeName);
  size_t maxLen = MAX_SCOPE_NAME_LEN - 1;
  if (len > maxLen) {
    SK_DLOGE("Failed to prepare scope kernel arguments: markerType=%s, scopeNameLength=%zu, maxLength=%zu", markerType,
             len, maxLen);
    return ACL_ERROR_INVALID_PARAM;
  }
  errno_t ret = memcpy_s(args.name, maxLen, scopeName, len);
  if (ret != 0) {
    SK_DLOGE("Failed to prepare scope kernel arguments: markerType=%s, memcpy_s ret=%d", markerType, ret);
    return ACL_ERROR_INVALID_PARAM;
  }
  args.name[len] = '\0';
  SK_DLOGI("Dispatch scope kernel: markerType=%s, scopeName=%s, stream=%p", markerType, scopeName, stream);
  scopeKernelImpl(stream, args);
  return ACL_SUCCESS;
}

}  // namespace

aclError LaunchScopeKernel(const char *scopeName, aclrtStream stream, bool isBegin) {
  ScopeKernelImpls impls = GetScopeKernelImpls();
  if (isBegin) {
    SK_DLOGI("Scope begin marker kernel dispatch started");
    aclError ret = LaunchScopeKernelImpl(scopeName, stream, impls.begin, "begin");
    if (ret != ACL_SUCCESS) {
      SK_DLOGE("Scope marker kernel processing failed: markerType=begin, ret=%d", ret);
      return ret;
    }
    ret = LaunchScopeKernelImpl(scopeName, stream, impls.placeholder, "placeholder");
    if (ret != ACL_SUCCESS) {
      SK_DLOGE("Scope marker kernel processing failed: markerType=placeholder, ret=%d", ret);
      return ret;
    }
    ret = LaunchScopeKernelImpl(scopeName, stream, impls.placeholder, "placeholder");
    if (ret != ACL_SUCCESS) {
      SK_DLOGE("Scope marker kernel processing failed: markerType=placeholder, ret=%d", ret);
      return ret;
    }
    SK_DLOGI("Scope begin marker kernels dispatched");
    return ret;
  } else {
    SK_DLOGI("Scope end marker kernel dispatch started");
    aclError ret = LaunchScopeKernelImpl(scopeName, stream, impls.placeholder, "placeholder");
    if (ret != ACL_SUCCESS) {
      SK_DLOGE("Scope marker kernel processing failed: markerType=placeholder, ret=%d", ret);
      return ret;
    }
    ret = LaunchScopeKernelImpl(scopeName, stream, impls.placeholder, "placeholder");
    if (ret != ACL_SUCCESS) {
      SK_DLOGE("Scope marker kernel processing failed: markerType=placeholder, ret=%d", ret);
      return ret;
    }
    ret = LaunchScopeKernelImpl(scopeName, stream, impls.end, "end");
    if (ret != ACL_SUCCESS) {
      SK_DLOGE("Scope marker kernel processing failed: markerType=end, ret=%d", ret);
      return ret;
    }
    SK_DLOGI("Scope end marker kernels dispatched");
    return ret;
  }
}
