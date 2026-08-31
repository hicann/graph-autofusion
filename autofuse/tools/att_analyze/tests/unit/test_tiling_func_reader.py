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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
import unittest
from core.tiling_func_reader import parse_perf_lines

PERF_LOG = """
[FlashAttentionScore] [PERF] Load_0[Load]: AIV_MTE2 = (base_cycles=172.2 + head_cost=27.01) = 199.21
[FlashAttentionScore] [PERF] Store_0[Store]: AIV_MTE3 = (block_cost=156.8 + head_cost=12.09) = 168.89
[FlashAttentionScore] [PERF] Add_0[Add]: AIV_VEC = base_cost=42.78
"""


class TestParsePerfLines(unittest.TestCase):
    def test_parse_multi_node(self):
        cases = parse_perf_lines(
            PERF_LOG, "FlashAttentionScore", result_id=1, group_id=0, case_id=0
        )
        self.assertEqual(len(cases), 1)
        info = cases[0]
        self.assertEqual(info.result_id, 1)
        self.assertEqual(len(info.nodes), 3)

    def test_node_sub_items(self):
        cases = parse_perf_lines(
            PERF_LOG, "FlashAttentionScore", result_id=1, group_id=0, case_id=0
        )
        load_node = next(n for n in cases[0].nodes if n.node_name == "Load_0")
        self.assertEqual(load_node.pipe_type, "AIV_MTE2")
        self.assertAlmostEqual(load_node.total, 199.21, places=1)
        self.assertEqual(len(load_node.sub_items), 2)
        base_cycles = next(s for s in load_node.sub_items if s.name == "base_cycles")
        self.assertAlmostEqual(base_cycles.value, 172.2, places=1)

    def test_bottleneck_detection(self):
        cases = parse_perf_lines(
            PERF_LOG, "FlashAttentionScore", result_id=1, group_id=0, case_id=0
        )
        # AIV_MTE2=199.21 > AIV_MTE3=168.89 > AIV_VEC=42.78，AIV_MTE2 是瓶颈
        load_node = next(n for n in cases[0].nodes if n.node_name == "Load_0")
        self.assertTrue(load_node.is_bottleneck)
        store_node = next(n for n in cases[0].nodes if n.node_name == "Store_0")
        self.assertFalse(store_node.is_bottleneck)

    def test_single_item_no_parens(self):
        cases = parse_perf_lines(
            PERF_LOG, "FlashAttentionScore", result_id=1, group_id=0, case_id=0
        )
        add_node = next(n for n in cases[0].nodes if n.node_name == "Add_0")
        self.assertEqual(len(add_node.sub_items), 1)
        self.assertAlmostEqual(add_node.total, 42.78, places=1)

    def test_no_match(self):
        cases = parse_perf_lines(
            PERF_LOG, "NonExistOp", result_id=0, group_id=0, case_id=0
        )
        self.assertEqual(cases, [])


if __name__ == "__main__":
    unittest.main()
