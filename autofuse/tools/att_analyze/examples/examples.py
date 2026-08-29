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
使用示例：演示如何使用日志解析工具
"""

from summary_templates import LogParser, print_summary_table


def example_1_cli_usage():
    """示例1：命令行调用（只展示命令，不会执行）"""
    print("示例1：命令行调用（只展示命令，不会执行）")
    print("=" * 80)
    print("请在仓库根目录执行以下命令，并将路径替换为自己的日志或产物目录：")
    commands = (
        "python3 autofuse/tools/att_analyze/src/att.py summary run.log -f csv -o summary.csv",
        "python3 autofuse/tools/att_analyze/src/att.py compare default.csv candidate.csv -f text -o compare.txt",
        "python3 autofuse/tools/att_analyze/src/att.py evidence run.log -o evidence/",
        "python3 autofuse/tools/att_analyze/src/att.py split-slog slog/ --op FlashAttentionScore --case r=1,g=0,c=2 -o split/",
        "python3 autofuse/tools/att_analyze/src/att.py perf-formula generated/ run.log --case r=0,g=0,c=1 -o perf/",
        "python3 autofuse/tools/att_analyze/src/att.py verify-tiling generated/ --scene tf --preset B --aiv-num 56 -o verify/",
    )
    for command in commands:
        print(f"$ {command}")
    print()


def example_2_basic_usage():
    """示例2：基本使用"""
    print("示例2：基本使用")
    print("=" * 80)

    parser = LogParser()
    summaries = parser.parse_log_file("test_concat.log")
    print_summary_table(summaries)
    print()


def example_3_multiple_operators():
    """示例3：多个算子"""
    print("示例3：多个算子")
    print("=" * 80)

    parser = LogParser()
    summaries = parser.parse_log_file("test_complete.log")
    print_summary_table(summaries)
    print()


def example_4_programmatic_access():
    """示例4：编程方式访问数据"""
    print("示例4：编程方式访问数据")
    print("=" * 80)

    parser = LogParser()
    summaries = parser.parse_log_file("test_concat.log")

    for summary in summaries:
        print(f"算子名称: {summary.operator_name}")
        print(
            f"选择的配置: graph{summary.graph}_result{summary.result}_g{summary.group}_case{summary.case}"
        )
        print("性能指标:")
        print(f"  - AIV_MTE2: {summary.aiv_mte2}")
        print(f"  - AIV_MTE3: {summary.aiv_mte3}")
        print(f"  - Objective Value: {summary.objective_value}")
        print(f"  - Result Performance: {summary.result_performance}")
        print("切分参数:")
        for key, value in summary.tiling_values.items():
            print(f"  - {key}: {value}")
    print()


def example_5_custom_processing():
    """示例5：自定义处理"""
    print("示例5：自定义处理")
    print("=" * 80)

    parser = LogParser()
    summaries = parser.parse_log_file("test_complete.log")

    # 找出性能最好的算子
    best_operator = None
    best_performance = float("inf")

    for summary in summaries:
        if summary.objective_value and summary.objective_value < best_performance:
            best_performance = summary.objective_value
            best_operator = summary

    if best_operator:
        print(f"性能最好的算子: {best_operator.operator_name}")
        print(f"Objective Value: {best_operator.objective_value}")
        print(
            f"选择的配置: graph{best_operator.graph}_result{best_operator.result}_g{best_operator.group}_case{best_operator.case}"
        )
    print()


def example_6_export_to_csv():
    """示例6：导出到CSV"""
    print("示例6：导出到CSV")
    print("=" * 80)

    parser = LogParser()
    summaries = parser.parse_log_file("test_complete.log")

    import csv

    # 收集所有切分参数
    all_tiling_keys = set()
    for summary in summaries:
        all_tiling_keys.update(summary.tiling_values.keys())

    fixed_tiling_keys = ["ub_size", "block_dim"]
    dynamic_tiling_keys = sorted(all_tiling_keys - set(fixed_tiling_keys))

    # 写入CSV文件
    with open("output.csv", "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
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
        fieldnames.extend(dynamic_tiling_keys)
        fieldnames.extend(fixed_tiling_keys)

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for summary in summaries:
            row = {
                "Operator": summary.operator_name,
                "Graph": summary.graph,
                "Result": summary.result,
                "Group": summary.group,
                "Case": summary.case,
                "AIV_MTE2": summary.aiv_mte2,
                "AIV_MTE3": summary.aiv_mte3,
                "Objective Value": summary.objective_value,
                "Result Perf": summary.result_performance,
            }

            for key in dynamic_tiling_keys:
                row[key] = summary.tiling_values.get(key, "")

            for key in fixed_tiling_keys:
                row[key] = summary.tiling_values.get(key, "")

            writer.writerow(row)

    print("数据已导出到 output.csv")
    print()


if __name__ == "__main__":
    example_1_cli_usage()
    example_2_basic_usage()
    example_3_multiple_operators()
    example_4_programmatic_access()
    example_5_custom_processing()
    example_6_export_to_csv()

    print("所有示例运行完成！")
