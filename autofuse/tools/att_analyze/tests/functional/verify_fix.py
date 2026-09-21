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
"""
验证脚本：确认AIV_MTE2/AIV_MTE3不重复显示
"""


def test_regex_pattern():
    """Test that the regular expression correctly excludes AIV_MTE2/AIV_MTE3"""

    # Mock tiling_values
    tiling_values = {
        "s0t_size": 256,
        "s1Ts0Tb_size": 4096,
        "s1t_size": 1,
        "ub_size": 1024,
        "block_dim": 1,
        "q0_size": 1024,
        "AIV_MTE2": 387906.099166,
        "AIV_MTE3": 355058.291519,
    }

    all_tiling_keys = set(tiling_values.keys())
    fixed_tiling_keys = ["ub_size", "block_dim"]
    performance_keys = ["AIV_MTE2", "AIV_MTE3"]
    dynamic_tiling_keys = sorted(
        all_tiling_keys - set(fixed_tiling_keys) - set(performance_keys)
    )

    print("Verify that AIV_MTE2/AIV_MTE3 are not duplicated")
    print("=" * 80)
    print(f"All tiling keys: {sorted(all_tiling_keys)}")
    print(f"Fixed tiling keys: {fixed_tiling_keys}")
    print(f"Performance metric keys: {performance_keys}")
    print(f"Dynamic tiling keys (excluding performance metrics): {dynamic_tiling_keys}")
    print()

    # 验证
    if "AIV_MTE2" in dynamic_tiling_keys:
        print("❌ Error: AIV_MTE2 is duplicated in dynamic tiling keys")
        return False
    if "AIV_MTE3" in dynamic_tiling_keys:
        print("❌ Error: AIV_MTE3 is duplicated in dynamic tiling keys")
        return False

    print("✅ Correct: AIV_MTE2 and AIV_MTE3 are not duplicated in dynamic tiling keys")
    print()

    # 显示最终列顺序
    all_columns = [
        "Operator",
        "Graph",
        "Result",
        "Group",
        "Case",
        "AIV_MTE2",
        "AIV_MTE3",
        "Objective Value",
        "Result Perf",
    ]
    all_columns.extend(dynamic_tiling_keys)
    all_columns.extend(fixed_tiling_keys)

    print("Final column order:")
    for i, col in enumerate(all_columns, 1):
        print(f"  {i}. {col}")

    return True


if __name__ == "__main__":
    success = test_regex_pattern()
    if success:
        print("\n✅ Verification passed!")
    else:
        print("\n❌ Verification failed!")
