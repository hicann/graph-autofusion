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
from summary_templates import LogParser, print_summary_table


def test_concat():
    print("Testing test_concat.log:")
    print("=" * 80)
    parser = LogParser()
    summaries = parser.parse_log_file("test_concat.log")
    print(f"Found {len(summaries)} operators")
    for summary in summaries:
        print(f"Operator: {summary.operator_name}")
        print(
            f"Graph: {summary.graph}, Result: {summary.result}, Group: {summary.group}, Case: {summary.case}"
        )
        print(f"AIV_MTE2: {summary.aiv_mte2}, AIV_MTE3: {summary.aiv_mte3}")
        print(
            f"Objective Value: {summary.objective_value}, Result Performance: {summary.result_performance}"
        )
        print(f"Tiling Values: {summary.tiling_values}")
    print("\nSummary Table:")
    print_summary_table(summaries)
    print("\n")


def test_complete():
    print("Testing test_complete.log:")
    print("=" * 80)
    parser = LogParser()
    summaries = parser.parse_log_file("test_complete.log")
    print(f"Found {len(summaries)} operators")
    print_summary_table(summaries)
    print("\n")


def test_missing_graph():
    print("Testing test_missing_graph.log:")
    print("=" * 80)
    parser = LogParser()
    summaries = parser.parse_log_file("test_missing_graph.log")
    print(f"Found {len(summaries)} operators")
    print_summary_table(summaries)
    print("\n")


def test_missing_group():
    print("Testing test_missing_group.log:")
    print("=" * 80)
    parser = LogParser()
    summaries = parser.parse_log_file("test_missing_group.log")
    print(f"Found {len(summaries)} operators")
    print_summary_table(summaries)
    print("\n")


def test_missing_result_perf():
    print("Testing test_missing_result_perf.log:")
    print("=" * 80)
    parser = LogParser()
    summaries = parser.parse_log_file("test_missing_result_perf.log")
    print(f"Found {len(summaries)} operators")
    print_summary_table(summaries)
    print("\n")


if __name__ == "__main__":
    test_concat()
    test_complete()
    test_missing_graph()
    test_missing_group()
    test_missing_result_perf()
    print("All tests completed!")
