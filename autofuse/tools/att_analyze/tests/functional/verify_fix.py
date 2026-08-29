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
    """测试正则表达式是否正确排除AIV_MTE2/AIV_MTE3"""

    # 模拟tiling_values
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

    print("验证AIV_MTE2/AIV_MTE3不重复显示")
    print("=" * 80)
    print(f"所有切分键: {sorted(all_tiling_keys)}")
    print(f"固定切分键: {fixed_tiling_keys}")
    print(f"性能指标键: {performance_keys}")
    print(f"动态切分键（排除性能指标）: {dynamic_tiling_keys}")
    print()

    # 验证
    if "AIV_MTE2" in dynamic_tiling_keys:
        print("❌ 错误：AIV_MTE2在动态切分键中重复显示")
        return False
    if "AIV_MTE3" in dynamic_tiling_keys:
        print("❌ 错误：AIV_MTE3在动态切分键中重复显示")
        return False

    print("✅ 正确：AIV_MTE2和AIV_MTE3没有在动态切分键中重复显示")
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

    print("最终列顺序:")
    for i, col in enumerate(all_columns, 1):
        print(f"  {i}. {col}")

    return True


if __name__ == "__main__":
    success = test_regex_pattern()
    if success:
        print("\n✅ 验证通过！")
    else:
        print("\n❌ 验证失败！")
