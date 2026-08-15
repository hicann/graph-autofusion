/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __TEST_E2E_LOAD_ABS_STORE_H__
#define __TEST_E2E_LOAD_ABS_STORE_H__

#include "ascendc_ir.h"

void LoadScalarAbsBrcStore_BeforeAutofuse(af::AscGraph &graph);
void LoadScalarAbsBrcStore_AfterAutofuse(af::AscGraph &graph);
void LoadScalarBrcStore_BeforeAutofuse2D(af::AscGraph &graph);
void LoadScalarBrcStore_AfterAutofuse2D(af::AscGraph &graph);
#endif
