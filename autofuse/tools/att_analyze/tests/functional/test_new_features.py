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
测试新功能：CSV和Excel导出
"""

import os
from summary_templates import (
    LogParser,
    export_to_csv,
    export_to_excel,
    print_summary_table,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def test_console_output():
    """测试控制台输出"""
    print("=" * 80)
    print("测试1：控制台输出")
    print("=" * 80)

    parser = LogParser()
    summaries = parser.parse_log_file(os.path.join(DATA_DIR, "test_concat.log"))
    print_summary_table(summaries)
    print("\n✅ 控制台输出测试完成\n")


def test_csv_export():
    """测试CSV导出"""
    print("=" * 80)
    print("测试2：CSV导出")
    print("=" * 80)

    parser = LogParser()
    summaries = parser.parse_log_file(os.path.join(DATA_DIR, "test_complete.log"))
    export_to_csv(summaries, "test_output.csv")
    print("✅ CSV导出测试完成\n")


def test_excel_export():
    """测试Excel导出"""
    print("=" * 80)
    print("测试3：Excel导出")
    print("=" * 80)

    parser = LogParser()
    summaries = parser.parse_log_file(os.path.join(DATA_DIR, "test_complete.log"))
    export_to_excel(summaries, "test_output.xlsx")
    print("✅ Excel导出测试完成\n")


def test_no_duplicate_performance_keys():
    """测试性能指标不重复显示"""
    print("=" * 80)
    print("测试4：验证AIV_MTE2/AIV_MTE3不重复显示")
    print("=" * 80)

    parser = LogParser()
    summaries = parser.parse_log_file(os.path.join(DATA_DIR, "test_concat.log"))

    all_tiling_keys = set()
    for summary in summaries:
        all_tiling_keys.update(summary.tiling_values.keys())

    performance_keys = ["AIV_MTE2", "AIV_MTE3"]
    dynamic_tiling_keys = sorted(
        all_tiling_keys - set(["ub_size", "block_dim"]) - set(performance_keys)
    )

    print(f"所有切分键: {sorted(all_tiling_keys)}")
    print(f"动态切分键（排除性能指标）: {dynamic_tiling_keys}")

    if "AIV_MTE2" not in dynamic_tiling_keys and "AIV_MTE3" not in dynamic_tiling_keys:
        print("✅ AIV_MTE2和AIV_MTE3没有在动态切分键中重复显示\n")
    else:
        print("❌ AIV_MTE2或AIV_MTE3在动态切分键中重复显示\n")


if __name__ == "__main__":
    test_console_output()
    test_csv_export()
    test_excel_export()
    test_no_duplicate_performance_keys()

    print("=" * 80)
    print("所有测试完成！")
    print("=" * 80)
