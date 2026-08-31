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
from core.file_utils import find_log_files, ensure_output_dir, build_case_output_path


class TestFindLogFiles(unittest.TestCase):
    def test_finds_log_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "a.log"), "w").close()
            open(os.path.join(tmpdir, "b.txt"), "w").close()
            sub = os.path.join(tmpdir, "sub")
            os.makedirs(sub)
            open(os.path.join(sub, "c.log"), "w").close()
            result = find_log_files(tmpdir)
            self.assertEqual(len(result), 2)
            self.assertTrue(all(f.endswith(".log") for f in result))

    def test_single_file(self):
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            fname = f.name
        try:
            result = find_log_files(fname)
            self.assertEqual(result, [fname])
        finally:
            os.unlink(fname)

    def test_nonexistent(self):
        result = find_log_files("/nonexistent/path")
        self.assertEqual(result, [])


class TestEnsureOutputDir(unittest.TestCase):
    def test_creates_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "a", "b")
            result = ensure_output_dir(new_dir)
            self.assertTrue(os.path.isdir(new_dir))
            self.assertEqual(result, new_dir)

    def test_existing_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ensure_output_dir(tmpdir)
            self.assertEqual(result, tmpdir)


class TestBuildCaseOutputPath(unittest.TestCase):
    def test_path_structure(self):
        path = build_case_output_path("/out", "FlashAttn", 1, 0, 2)
        self.assertEqual(path, "/out/FlashAttn/graph0_result1/g0/case2.log")

    def test_different_values(self):
        path = build_case_output_path("/base", "MatMul", 0, 2, 1)
        self.assertEqual(path, "/base/MatMul/graph0_result0/g2/case1.log")


if __name__ == "__main__":
    unittest.main()
