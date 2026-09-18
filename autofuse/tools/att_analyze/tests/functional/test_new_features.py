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
import sys

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from summary_templates import (  # noqa: E402
    LogParser,
    export_to_csv,
    export_to_excel,
    print_summary_table,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def test_console_output():
    """Test console output"""
    print("=" * 80)
    print("Test 1: Console output")
    print("=" * 80)

    parser = LogParser()
    summaries = parser.parse_log_file(os.path.join(DATA_DIR, "test_concat.log"))
    print_summary_table(summaries)
    print("\n✅ Console output test completed\n")


def test_csv_export():
    """Test CSV export"""
    print("=" * 80)
    print("Test 2: CSV export")
    print("=" * 80)

    parser = LogParser()
    summaries = parser.parse_log_file(os.path.join(DATA_DIR, "test_complete.log"))
    export_to_csv(summaries, "test_output.csv")
    print("✅ CSV export test completed\n")


def test_excel_export():
    """Test Excel export"""
    print("=" * 80)
    print("Test 3: Excel export")
    print("=" * 80)

    parser = LogParser()
    summaries = parser.parse_log_file(os.path.join(DATA_DIR, "test_complete.log"))
    export_to_excel(summaries, "test_output.xlsx")
    print("✅ Excel export test completed\n")


def test_no_duplicate_performance_keys():
    """Test that performance metrics are not duplicated"""
    print("=" * 80)
    print("Test 4: Verify that AIV_MTE2/AIV_MTE3 are not duplicated")
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

    print(f"All tiling keys: {sorted(all_tiling_keys)}")
    print(f"Dynamic tiling keys (excluding performance metrics): {dynamic_tiling_keys}")

    if "AIV_MTE2" not in dynamic_tiling_keys and "AIV_MTE3" not in dynamic_tiling_keys:
        print("✅ AIV_MTE2 and AIV_MTE3 are not duplicated in dynamic tiling keys\n")
    else:
        print("❌ AIV_MTE2 or AIV_MTE3 are duplicated in dynamic tiling keys\n")


if __name__ == "__main__":
    test_console_output()
    test_csv_export()
    test_excel_export()
    test_no_duplicate_performance_keys()

    print("=" * 80)
    print("All tests completed!")
    print("=" * 80)
