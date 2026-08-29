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
from core.tiling_func_reader import CasePerfInfo, NodePerfInfo, SubItem
from commands.perf_formula import render_svg, build_full_svg


def make_case(result_id=1, group_id=0, case_id=0):
    nodes = [
        NodePerfInfo(
            "Load_0",
            "Load",
            "AIV_MTE2",
            [SubItem("base_cycles", 172.2, 0.865), SubItem("head_cost", 27.01, 0.135)],
            total=199.21,
            is_bottleneck=True,
        ),
        NodePerfInfo(
            "Store_0",
            "Store",
            "AIV_MTE3",
            [SubItem("block_cost", 156.8, 0.928), SubItem("head_cost", 12.09, 0.072)],
            total=168.89,
            is_bottleneck=False,
        ),
    ]
    return CasePerfInfo(
        result_id=result_id,
        group_id=group_id,
        case_id=case_id,
        nodes=nodes,
        bottleneck_pipe="AIV_MTE2",
    )


class TestRenderOpSection(unittest.TestCase):
    """render_svg 返回 (svg_fragment, height) 元组，fragment 是 <g> 元素"""

    def test_produces_g_fragment(self):
        cases = [make_case()]
        frag, h = render_svg("FlashAttentionScore", cases, selected_case=(1, 0, 0))
        self.assertIn("<g", frag)
        self.assertNotIn("<svg", frag)
        self.assertIn("AIV_MTE2", frag)
        self.assertIn("base_cycles", frag)
        self.assertIn("199.21", frag)
        self.assertGreater(h, 0)

    def test_multi_group(self):
        cases = [make_case(group_id=0, case_id=0), make_case(group_id=1, case_id=0)]
        frag, h = render_svg("FlashAttentionScore", cases, selected_case=None)
        self.assertIn("group0", frag)
        self.assertIn("group1", frag)

    def test_bottleneck_marker(self):
        cases = [make_case()]
        frag, h = render_svg("FlashAttentionScore", cases, selected_case=(1, 0, 0))
        self.assertIn("red", frag.lower())


class TestBuildFullSvg(unittest.TestCase):
    """build_full_svg 将多个算子 section 合并为合法单 SVG"""

    def test_single_root_svg(self):
        sections = [
            ("<g>content</g>", 200),
            ("<g>content2</g>", 150),
        ]
        svg = build_full_svg(sections)
        self.assertEqual(svg.count("<svg"), 1)
        self.assertIn('height="350"', svg)
        self.assertIn("content", svg)
        self.assertIn("content2", svg)

    def test_empty_sections(self):
        svg = build_full_svg([])
        self.assertIn("<svg", svg)
        self.assertIn('height="0"', svg)


if __name__ == "__main__":
    unittest.main()
