/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_TESTS_ST_BACKEND_E2E_COMMON_INDUCTOR_SPLIT_COMPILE_CONFIG_H_
#define AUTOFUSE_TESTS_ST_BACKEND_E2E_COMMON_INDUCTOR_SPLIT_COMPILE_CONFIG_H_

#ifndef HOST_CODE_FILE
#define HOST_CODE_FILE ""
#endif
#ifndef DEVICE_CODE_FILE
#define DEVICE_CODE_FILE ""
#endif
#ifndef OUTPUT_DIR
#define OUTPUT_DIR ""
#endif
#ifndef HOST_HELPER_BIN
#define HOST_HELPER_BIN ""
#endif
#ifndef HOST_DYNAMIC_SHAPE_ARGS
#define HOST_DYNAMIC_SHAPE_ARGS ""
#endif
#ifndef HOST_INPUT_CONFIGS_JSON
#define HOST_INPUT_CONFIGS_JSON "[]"
#endif
#ifndef HOST_TOPN
#define HOST_TOPN 4
#endif
#ifndef HOST_VERIFY_EMPTY_CONFIG
#define HOST_VERIFY_EMPTY_CONFIG 0
#endif
#ifndef PYAUTOFUSE_DIR
#define PYAUTOFUSE_DIR ""
#endif
#ifndef AUTOFUSE_PYTHON_DIR
#define AUTOFUSE_PYTHON_DIR ""
#endif
#ifndef ASCEND_HOME_PATH
#define ASCEND_HOME_PATH ""
#endif

#endif  // AUTOFUSE_TESTS_ST_BACKEND_E2E_COMMON_INDUCTOR_SPLIT_COMPILE_CONFIG_H_
