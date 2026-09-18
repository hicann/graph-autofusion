#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import sys
import numpy as np

# for float32
relative_tol = 1e-5
absolute_tol = 1e-5
error_tol = 1e-5


def verify_result(output, golden, output_dtype):
    output = np.fromfile(output, dtype=output_dtype).reshape(-1)
    golden = np.fromfile(golden, dtype=output_dtype).reshape(-1)
    print("#################TEST RESULT#####################")
    print("output result:", output)
    print("golden result:", golden)
    different_element_results = np.isclose(
        output, golden, rtol=relative_tol, atol=absolute_tol, equal_nan=True
    )
    different_element_indexes = np.where(np.logical_not(different_element_results))[0]
    for index in range(len(different_element_indexes)):
        real_index = different_element_indexes[index]
        golden_data = golden[real_index]
        output_data = output[real_index]
        print(
            "data index: %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6f"
            % (
                real_index,
                golden_data,
                output_data,
                abs(output_data - golden_data) / golden_data,
            )
        )
        if index == 100:
            break
    error_ratio = float(different_element_indexes.size) / golden.size
    print("error ratio: %.4f, tolerance: %.4f" % (error_ratio, error_tol))
    return error_ratio <= error_tol


if __name__ == "__main__":
    try:
        if sys.argv[4] == "float16":
            print("output dtype is float16")
            output_dtype = np.float16
        elif sys.argv[4] == "uint8":
            print("output dtype is uint8")
            output_dtype = np.uint8
        elif sys.argv[4] == "int32":
            print("output dtype is int32")
            output_dtype = np.int32
        elif sys.argv[4] == "int64":
            print("output dtype is int64")
            output_dtype = np.int64
        elif sys.argv[4] == "float32":
            print("output dtype is float32")
            output_dtype = np.float32
        else:
            raise ValueError(
                "[ERROR]current output type(", sys.argv[4], ") is not supported"
            )
        res = verify_result(sys.argv[1], sys.argv[2], output_dtype)
        if not res:
            raise ValueError("[ERROR] testcase: ", sys.argv[3], ", Result: ERROR")
        else:
            print("testcase: ", sys.argv[3], ", Result: PASS")
    except Exception as e:
        print(e)
        sys.exit(1)
