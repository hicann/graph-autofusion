/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_REGISTER_GRAPH_OPTIMIZE_REGISTER_ERROR_CODES_H_
#define INC_REGISTER_GRAPH_OPTIMIZE_REGISTER_ERROR_CODES_H_

#include <map>
#include <string>
#include <cstdint>

namespace fe {

/** Assigned SYS ID */
const uint8_t SYSID_FE = 3;

/** Common module ID */
const uint8_t FE_MODID_COMMON = 50;

/**  FE error code definition Macro
 *  Build error code
 */
#define FE_DEF_ERRORNO(sysid, modid, name, value, desc)                                                           \
  static constexpr fe::Status name = ((((static_cast<uint32_t>((0xFF) & (static_cast<uint8_t>(sysid)))) << 24) |  \
                                       ((static_cast<uint32_t>((0xFF) & (static_cast<uint8_t>(modid)))) << 16)) | \
                                      ((0xFFFF) & (static_cast<uint16_t>(value))));

using Status = uint32_t;

#define FE_DEF_ERRORNO_COMMON(name, value, desc) FE_DEF_ERRORNO(SYSID_FE, FE_MODID_COMMON, name, value, desc)

using Status = uint32_t;

FE_DEF_ERRORNO(0, 0, SUCCESS, 0, "success");
FE_DEF_ERRORNO(0xFF, 0xFF, FAILED, 0xFFFF, "failed");
FE_DEF_ERRORNO_COMMON(NOT_CHANGED, 201, "The nodes of the graph not changed.");
FE_DEF_ERRORNO_COMMON(PARAM_INVALID, 1, "Parameter's invalid!");
FE_DEF_ERRORNO_COMMON(GRAPH_FUSION_CYCLE, 301, "Graph is cycle after fusion!");

}  // namespace fe
#endif  // INC_REGISTER_GRAPH_OPTIMIZE_REGISTER_ERROR_CODES_H_
