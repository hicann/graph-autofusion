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
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
import unittest
from core.log_parser import LogParser

MULTI_GROUP_LOG = """
[FlashAttentionScore] [PROF]Among the templates, tiling case 0 of graph0_result1_g0 is the best choice
[FlashAttentionScore] [PROF]Among the templates, tiling case 1 of graph0_result1_g1 is the best choice
[FlashAttentionScore] [PROF]Among all schedule results, graph0_result1 is the best choice
"""

RERUN_LOG = """
[FlashAttentionScore] [PROF]Among the templates, tiling case 0 of graph0_result1_g0 is the best choice
[FlashAttentionScore] [PROF]Among the templates, tiling case 2 of graph0_result1_g0 is the best choice
[FlashAttentionScore] [PROF]Among all schedule results, graph0_result1 is the best choice
"""

MULTI_GROUP_DETAIL_LOG = """
[FlashAttentionScore] [PROF]Among the templates, tiling case 0 of graph0_result1_g0 is the best choice
[FlashAttentionScore] [PROF]Among the templates, tiling case 1 of graph0_result1_g1 is the best choice
[FlashAttentionScore] [PROF]Among all schedule results, graph0_result1 is the best choice
[FlashAttentionScore] [PROF]The objective value of the tiling data is 100.0 in graph0_result1_g0_0
[FlashAttentionScore] [PROF]The objective value of the tiling data is 200.0 in graph0_result1_g1_1
[FlashAttentionScore] [PROF]The value of s0t_size is 16 in graph0_result1_g0_0
[FlashAttentionScore] [PROF]The value of s0t_size is 32 in graph0_result1_g1_1
[FlashAttentionScore] The value of graph0_result1 is 300.0
"""

MULTI_RESULT_DETAIL_LOG = """
[FlashAttentionScore] [PROF]Among the templates, tiling case 0 of graph0_result0_g0 is the best choice
[FlashAttentionScore] [PROF]Among the templates, tiling case 1 of graph0_result0_g1 is the best choice
[FlashAttentionScore] [PROF]Among the templates, tiling case 2 of graph0_result1_g0 is the best choice
[FlashAttentionScore] [PROF]Among the templates, tiling case 3 of graph0_result1_g1 is the best choice
[FlashAttentionScore] [PROF]Among all schedule results, graph0_result1 is the best choice
[FlashAttentionScore] [PROF]The objective value of the tiling data is 10.0 in graph0_result0_g0_0
[FlashAttentionScore] [PROF]The objective value of the tiling data is 11.0 in graph0_result0_g1_1
[FlashAttentionScore] [PROF]The objective value of the tiling data is 20.0 in graph0_result1_g0_2
[FlashAttentionScore] [PROF]The objective value of the tiling data is 21.0 in graph0_result1_g1_3
[FlashAttentionScore] The value of graph0_result0 is 100.0
[FlashAttentionScore] The value of graph0_result1 is 200.0
"""

MISSING_GROUP_DETAIL_LOG = """
[FlashAttentionScore] [PROF]Among all schedule results, graph0_result1 is the best choice
[FlashAttentionScore] [PROF]The objective value of the tiling data is 123.0 in graph0_result1_g-1_0
"""


class TestExtractAllGroupCases(unittest.TestCase):
    def setUp(self):
        self.parser = LogParser()

    def test_multi_group(self):
        result = self.parser.extract_all_group_cases(
            MULTI_GROUP_LOG, "FlashAttentionScore", 0, 1
        )
        self.assertEqual(result, {0: 0, 1: 1})

    def test_rerun_takes_last(self):
        result = self.parser.extract_all_group_cases(
            RERUN_LOG, "FlashAttentionScore", 0, 1
        )
        self.assertEqual(result, {0: 2})

    def test_not_found(self):
        result = self.parser.extract_all_group_cases("", "FlashAttentionScore", 0, 1)
        self.assertEqual(result, {})


class TestExtractAllGraphResults(unittest.TestCase):
    def setUp(self):
        self.parser = LogParser()

    def test_prefers_result_value_and_deduplicates(self):
        result = self.parser.extract_all_graph_results(
            MULTI_RESULT_DETAIL_LOG, "FlashAttentionScore"
        )
        self.assertEqual(result, [(0, 0), (0, 1)])

    def test_falls_back_to_template_lines(self):
        log = """
[FlashAttentionScore] [PROF]Among the templates, tiling case 0 of graph1_result2_g0 is the best choice
[FlashAttentionScore] [PROF]Among the templates, tiling case 1 of graph1_result3_g0 is the best choice
"""
        result = self.parser.extract_all_graph_results(log, "FlashAttentionScore")
        self.assertEqual(result, [(1, 2), (1, 3)])


class TestParseLogFileSummaryModes(unittest.TestCase):
    def setUp(self):
        self.parser = LogParser()

    def _write_temp_log(self, content: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".log")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        return path

    def test_default_returns_best_result_all_groups(self):
        path = self._write_temp_log(MULTI_GROUP_DETAIL_LOG)
        summaries = self.parser.parse_log_file(path)

        self.assertEqual(len(summaries), 2)
        summaries_by_group = {s.group: s for s in summaries}
        self.assertEqual(sorted(summaries_by_group.keys()), [0, 1])
        self.assertEqual(summaries_by_group[0].case, 0)
        self.assertEqual(summaries_by_group[1].case, 1)

    def test_all_results_all_groups_returns_multiple_results(self):
        path = self._write_temp_log(MULTI_RESULT_DETAIL_LOG)
        summaries = self.parser.parse_log_file(
            path, summary_mode="all_results_all_groups"
        )

        self.assertEqual(len(summaries), 4)
        pairs = sorted((s.result, s.group, s.case) for s in summaries)
        self.assertEqual(pairs, [(0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)])

    def test_default_only_returns_best_result_groups(self):
        path = self._write_temp_log(MULTI_RESULT_DETAIL_LOG)
        summaries = self.parser.parse_log_file(path)

        self.assertEqual(len(summaries), 2)
        self.assertTrue(all(s.result == 1 for s in summaries))

    def test_rerun_takes_last_case_per_group(self):
        path = self._write_temp_log("""
[FlashAttentionScore] [PROF]Among the templates, tiling case 0 of graph0_result1_g0 is the best choice
[FlashAttentionScore] [PROF]Among the templates, tiling case 2 of graph0_result1_g0 is the best choice
[FlashAttentionScore] [PROF]Among all schedule results, graph0_result1 is the best choice
[FlashAttentionScore] [PROF]The objective value of the tiling data is 222.0 in graph0_result1_g0_2
""")
        summaries = self.parser.parse_log_file(path)

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].group, 0)
        self.assertEqual(summaries[0].case, 2)
        self.assertEqual(summaries[0].objective_value, 222.0)

    def test_falls_back_when_group_missing(self):
        path = self._write_temp_log(MISSING_GROUP_DETAIL_LOG)
        summaries = self.parser.parse_log_file(path)

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].group, -1)
        self.assertEqual(summaries[0].case, 0)
        self.assertEqual(summaries[0].objective_value, 123.0)


if __name__ == "__main__":
    unittest.main()
