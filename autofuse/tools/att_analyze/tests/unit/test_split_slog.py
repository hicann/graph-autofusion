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
from commands.split_slog import SlogSplitter

COMPILER_LOG = """
[DFX]Begin to gen model info for asc graph 0, schedule result 1, schedule group 0, tiling_case size 2, graph name FlashAttentionScore.
[DFX]Begin to generate model info for graph FlashAttentionScore of tiling case id 0
some content for case 0
[DFX]End to generate model info for graph FlashAttentionScore of tiling case id 0
[DFX]Begin to generate model info for graph FlashAttentionScore of tiling case id 1
some content for case 1
[DFX]End to generate model info for graph FlashAttentionScore of tiling case id 1
[DFX]End to gen model info for graph0_result1_g0 tiling_case size 2, graph name FlashAttentionScore.
"""

RUNTIME_LOG = """
[FlashAttentionScore] [PROF]The value of AIV_MTE2 is 199.21 in graph0_result1_g0_0
[FlashAttentionScore] [PROF]The value of AIV_MTE3 is 168.89 in graph0_result1_g0_0
Finish calculating the tiling data for tiling_case_id 0.
[FlashAttentionScore] [PROF]The value of AIV_MTE2 is 210.5 in graph0_result1_g0_1
Finish calculating the tiling data for tiling_case_id 1.
[FlashAttentionScore] [PROF]Among the templates, tiling case 0 of graph0_result1_g0 is the best choice
[FlashAttentionScore] [PROF]Among all schedule results, graph0_result1 is the best choice
"""


class TestSlogSplitter(unittest.TestCase):
    def setUp(self):
        self.splitter = SlogSplitter()

    def test_parse_compiler_cases(self):
        groups = self.splitter.parse_compiler_model_info(COMPILER_LOG)
        self.assertIn((0, 1, 0), groups)
        cases = groups[(0, 1, 0)]
        self.assertEqual(len(cases), 2)
        self.assertIn("some content for case 0", cases[0])
        self.assertIn("some content for case 1", cases[1])

    def test_parse_runtime_cases(self):
        cases = self.splitter.parse_runtime_cases(RUNTIME_LOG, "FlashAttentionScore")
        self.assertIn((0, 1, 0), cases)
        self.assertEqual(len(cases[(0, 1, 0)]), 2)
        self.assertIn("199.21", cases[(0, 1, 0)][0])
        self.assertIn("210.5", cases[(0, 1, 0)][1])

    def test_write_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            groups = self.splitter.parse_compiler_model_info(COMPILER_LOG)
            self.splitter.write_compiler_output(groups, "FlashAttentionScore", tmpdir)
            case0_path = os.path.join(
                tmpdir,
                "FlashAttentionScore",
                "compiler",
                "graph0_result1",
                "g0",
                "case0.log",
            )
            self.assertTrue(os.path.exists(case0_path))
            case1_path = os.path.join(
                tmpdir,
                "FlashAttentionScore",
                "compiler",
                "graph0_result1",
                "g0",
                "case1.log",
            )
            self.assertTrue(os.path.exists(case1_path))


if __name__ == "__main__":
    unittest.main()
