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
from commands.case_filter import parse_case_arg


class TestParseCaseArg(unittest.TestCase):
    def test_result_only(self):
        f = parse_case_arg("r=0")
        self.assertEqual(f.results, [0])
        self.assertIsNone(f.groups)
        self.assertIsNone(f.cases)

    def test_multi_result(self):
        f = parse_case_arg("r=0,1")
        self.assertEqual(f.results, [0, 1])

    def test_full_spec(self):
        f = parse_case_arg("r=1,g=0,c=2")
        self.assertEqual(f.results, [1])
        self.assertEqual(f.groups, [0])
        self.assertEqual(f.cases, [2])

    def test_multi_case(self):
        f = parse_case_arg("g=0,c=0,1")
        self.assertEqual(f.groups, [0])
        self.assertEqual(f.cases, [0, 1])

    def test_longform(self):
        f = parse_case_arg("result=1,group=0,case=2")
        self.assertEqual(f.results, [1])
        self.assertEqual(f.groups, [0])
        self.assertEqual(f.cases, [2])

    def test_none_returns_none(self):
        self.assertIsNone(parse_case_arg(None))

    def test_unknown_dim_raises(self):
        with self.assertRaises(ValueError):
            parse_case_arg("x=1")

    def test_missing_dim_prefix_raises(self):
        with self.assertRaises(ValueError):
            parse_case_arg("1,2")


class TestCaseFilterMatch(unittest.TestCase):
    def test_matches_result(self):
        f = parse_case_arg("r=1")
        self.assertTrue(f.match(result_id=1, group_id=0, case_id=0))
        self.assertFalse(f.match(result_id=0, group_id=0, case_id=0))

    def test_matches_all_when_dim_not_set(self):
        f = parse_case_arg("r=1")
        self.assertTrue(f.match(result_id=1, group_id=99, case_id=99))

    def test_full_match(self):
        f = parse_case_arg("r=1,g=0,c=2")
        self.assertTrue(f.match(result_id=1, group_id=0, case_id=2))
        self.assertFalse(f.match(result_id=1, group_id=0, case_id=3))

    def test_group_only(self):
        f = parse_case_arg("g=0")
        self.assertTrue(f.match(result_id=99, group_id=0, case_id=99))
        self.assertFalse(f.match(result_id=99, group_id=1, case_id=99))


if __name__ == "__main__":
    unittest.main()
