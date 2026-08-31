/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_COMMON_FMK_ERROR_CODES_H_
#define INC_COMMON_FMK_ERROR_CODES_H_

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define GE_OBJECT_VISIBILITY
#else
#define GE_OBJECT_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define GE_OBJECT_VISIBILITY
#else
#define GE_OBJECT_VISIBILITY __attribute__((visibility("hidden")))
#endif
#endif

#include <map>
#include <string>

#include "common/ge_common/fmk_types.h"
#include "register/register_error_codes.h"
#include "ge_common_af/ge_error_codes_af.h"

// Each module uses the following four macros to define error codes:
#define DECLARE_ERRORNO_OMG(name, value) DECLARE_ERRORNO(SYSID_FWK, MODID_OMG, name, value)
#define DECLARE_ERRORNO_OME(name, value) DECLARE_ERRORNO(SYSID_FWK, MODID_OME, name, value)
#define DECLARE_ERRORNO_CALIBRATION(name, value) DECLARE_ERRORNO(SYSID_FWK, MODID_CALIBRATION, name, value)

#define DEF_ERRORNO(name, desc) const bool g_##name##_errorno = StatusFactory::Instance()->RegisterErrorNo(name, desc)

// Interface for Obtaining Error Code Description
#define GET_ERRORNO_STR(value) domi::StatusFactory::Instance()->GetErrDesc(value)

namespace domi {
constexpr int32_t MODID_OMG = 1;          // OMG module ID
constexpr int32_t MODID_OME = 2;          // OME module ID
constexpr int32_t MODID_CALIBRATION = 3;  // Calibration module ID

class GE_FUNC_VISIBILITY StatusFactory {
 public:
  static StatusFactory *Instance();

  bool RegisterErrorNo(const uint32_t err, const std::string &desc);

  std::string GetErrDesc(const uint32_t err);

 protected:
  StatusFactory() = default;
  virtual ~StatusFactory() = default;

 private:
  std::map<uint32_t, std::string> err_desc_;
};

// Common errocode
DECLARE_ERRORNO_COMMON(MEMALLOC_FAILED, 0);  // 50331648
DECLARE_ERRORNO_COMMON(CCE_FAILED, 2);       // 50331650
DECLARE_ERRORNO_COMMON(RT_FAILED, 3);        // 50331651
DECLARE_ERRORNO_COMMON(INTERNAL_ERROR, 4);   // 50331652
DECLARE_ERRORNO_COMMON(CSEC_ERROR, 5);       // 50331653
DECLARE_ERRORNO_COMMON(TEE_ERROR, 6);        // 50331653
DECLARE_ERRORNO_COMMON(UNSUPPORTED, 100);
DECLARE_ERRORNO_COMMON(OUT_OF_MEMORY, 101);

// Omg errorcode
DECLARE_ERRORNO_OMG(PARSE_MODEL_FAILED, 0);
DECLARE_ERRORNO_OMG(PARSE_WEIGHTS_FAILED, 1);
DECLARE_ERRORNO_OMG(NOT_INITIALIZED, 2);
DECLARE_ERRORNO_OMG(TIMEOUT, 3);

// Ome errorcode
DECLARE_ERRORNO_OME(MODEL_NOT_READY, 0);
DECLARE_ERRORNO_OME(PUSH_DATA_FAILED, 1);
DECLARE_ERRORNO_OME(DATA_QUEUE_ISFULL, 2);
}  // namespace domi

#endif  // INC_COMMON_FMK_ERROR_CODES_H_
