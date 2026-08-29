#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
import re


def test_regex_patterns():
    test_log = """[Concat] [PROF]The value of s0t_size is 256 in graph0_result0_g0_0.
[Concat] [PROF]The value of s1Ts0Tb_size is 4096 in graph0_result0_g0_0.
[Concat] [PROF]The value of s1t_size is 1 in graph0_result0_g0_0.
[Concat] [PROF]The value of ub_size is 1024 in graph0_result0_g0_0.
[Concat] [PROF]The value of block_dim is 1 in graph0_result0_g0_0.
[Concat] [PROF]The value of q0_size is 1024 in graph0_result0_g0_0.
[Concat] [PROF]The value of AIV_MTE2 is 387906.099166 in graph0_result0_g0_0.
[Concat] [PROF]The value of AIV_MTE3 is 355058.291519 in graph0_result0_g0_0.
[Concat] [PROF]The objective value of the tiling data is 388206.099166 in graph0_result0_g0_0.
[Concat] [PROF]Among the templates, tiling case 0 of graph0_result0_g0 is the best choice."""

    print("Testing regex patterns...")
    print("=" * 80)

    operator_pattern = re.compile(r"\[([^\]]+)\]\s*\[PROF\]")
    operators = []
    for match in operator_pattern.finditer(test_log):
        operators.append(match.group(1))
    print(f"Operators found: {list(set(operators))}")

    tiling_value_pattern = re.compile(
        r"\[Concat\]\s*\[PROF\]The value of\s+(\w+)\s+is\s+([\d.]+)\s+in\s+graph0_result0_g0_0"
    )
    tiling_values = {}
    for match in tiling_value_pattern.finditer(test_log):
        field_name = match.group(1)
        value = float(match.group(2))
        tiling_values[field_name] = value
    print(f"Tiling values: {tiling_values}")

    objective_pattern = re.compile(
        r"\[Concat\]\s*\[PROF\]The objective value of the tiling data is\s+([\d.]+)\s+in\s+graph0_result0_g0_0"
    )
    match = objective_pattern.search(test_log)
    if match:
        print(f"Objective value: {match.group(1)}")

    group_case_pattern = re.compile(
        r"\[Concat\]\s*\[PROF\]Among the templates,\s*tiling case\s+(\d+)\s+of\s+graph0_result0_g(\d+)\s+is the best choice"
    )
    match = group_case_pattern.search(test_log)
    if match:
        print(f"Group: {match.group(2)}, Case: {match.group(1)}")

    print("\nAll regex patterns work correctly!")


if __name__ == "__main__":
    test_regex_patterns()
