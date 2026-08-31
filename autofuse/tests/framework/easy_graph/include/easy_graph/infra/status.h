/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HA25033D6_1564_4748_B2C8_4DE2C5A286DE
#define HA25033D6_1564_4748_B2C8_4DE2C5A286DE

#ifndef __cplusplus
#include <stdbool.h>
#endif
#include <stdint.h>
#include "easy_graph/eg.h"

EG_NS_BEGIN

typedef uint32_t Status;

#define EG_SUCC_STATUS(status) (EG_NS::Status) status
#define EG_FAIL_STATUS(status) (EG_NS::Status)(status | EG_RESERVED_FAIL)

/* OK */
#define EG_SUCCESS EG_SUCC_STATUS(0)

/* Error Status */
#define EG_RESERVED_FAIL (EG_NS::Status)0x80000000
#define EG_FAILURE EG_FAIL_STATUS(1)
#define EG_FATAL_BUG EG_FAIL_STATUS(2)
#define EG_TIMEDOUT EG_FAIL_STATUS(3)
#define EG_OUT_OF_RANGE EG_FAIL_STATUS(4)
#define EG_UNIMPLEMENTED EG_FAIL_STATUS(5)

static inline bool eg_status_is_ok(Status status) {
  return (status & EG_RESERVED_FAIL) == 0;
}

static inline bool eg_status_is_fail(Status status) {
  return !eg_status_is_ok(status);
}

#define __EG_FAILED(result) eg_status_is_fail(result)
#define __EG_OK(result) eg_status_is_ok(result)

EG_NS_END

#endif
