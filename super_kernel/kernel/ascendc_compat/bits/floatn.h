/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _BITS_FLOATN_H
#define _BITS_FLOATN_H 1

/*
 * The bisheng compiler used for AscendC kernels may include the host glibc
 * floatn.h through CCE runtime headers.  On newer gcc/glibc environments,
 * the x86 float128 branch uses __float128 and __mode__(__TC__), which are not
 * supported by the AscendC compilation target.  SuperKernel ASC sources do
 * not depend on glibc float128 interfaces, so disable only that branch while
 * keeping the common floatn definitions from the host glibc.
 */
#define __HAVE_FLOAT128 0
#define __HAVE_DISTINCT_FLOAT128 0
#define __HAVE_FLOAT64X 1
#define __HAVE_FLOAT64X_LONG_DOUBLE 1

#include_next <bits/floatn-common.h>

#endif /* _BITS_FLOATN_H */
