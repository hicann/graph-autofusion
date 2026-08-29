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
import json
import tempfile
from contextlib import redirect_stdout
from io import StringIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
import unittest
from unittest.mock import MagicMock, patch
from commands.verify_tiling import (
    detect_scene,
    load_input_params,
    extract_inductor_artifacts,
    prepare_build_dir,
    validate_input_params,
)


class TestDetectScene(unittest.TestCase):
    def test_tf_scene(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "FlashAttn_tiling_func.cpp"), "w").close()
            self.assertEqual(detect_scene(d), "tf")

    def test_inductor_scene(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "output_code.py"), "w").close()
            self.assertEqual(detect_scene(d), "inductor")

    def test_unknown_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                detect_scene(d)


class TestLoadInputParams(unittest.TestCase):
    def test_preset_A(self):
        args = MagicMock()
        args.input_json = None
        args.preset = "A"
        params = load_input_params(args)
        self.assertEqual(params["dynamic_dims"], [])
        self.assertEqual(params["aiv_num"], 48)

    def test_preset_B(self):
        args = MagicMock()
        args.input_json = None
        args.preset = "B"
        params = load_input_params(args)
        self.assertEqual(params["dynamic_dims"], [1024, 512])
        self.assertEqual(params["aiv_num"], 56)

    def test_aiv_num_command_line_override(self):
        args = MagicMock()
        args.input_json = None
        args.preset = "B"
        args.aiv_num = 40
        params = load_input_params(args)
        self.assertEqual(params["aiv_num"], 40)

    def test_print_input_config_shows_aiv_num_and_source(self):
        from commands.verify_tiling import print_input_config

        output = StringIO()
        with redirect_stdout(output):
            print_input_config(
                {"aiv_num": 56, "ub_size": 262144, "dynamic_dims": [1024, 512]},
                "preset_B",
            )
        text = output.getvalue()
        self.assertIn("aiv_num=56", text)
        self.assertIn("source=preset_B", text)

    def test_custom_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"dynamic_dims": [256], "aiv_num": 10, "ub_size": 100000}, f)
            fname = f.name
        try:
            args = MagicMock()
            args.input_json = fname
            params = load_input_params(args)
            self.assertEqual(params["dynamic_dims"], [256])
        finally:
            os.unlink(fname)


FAKE_OUTPUT_CODE = """
fake_artifacts = {
    "tiling_def": "// tiling_def content",
    "host_impl": "// host_impl content",
    "device_impl": "// device_impl",
    "cpp_wrapper": "// cpp_wrapper",
}
"""


class TestExtractInductorArtifacts(unittest.TestCase):
    def test_extracts_tiling_def_and_host_impl(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(FAKE_OUTPUT_CODE)
            fname = f.name
        try:
            tiling_def, host_impl = extract_inductor_artifacts(fname)
            self.assertEqual(tiling_def, "// tiling_def content")
            self.assertEqual(host_impl, "// host_impl content")
        finally:
            os.unlink(fname)


class TestPrepareBuildDir(unittest.TestCase):
    def test_tf_scene_build_files_are_created_in_temp_dir(self):
        with (
            tempfile.TemporaryDirectory() as source_dir,
            tempfile.TemporaryDirectory() as tmp_dir,
        ):
            source_cpp = os.path.join(source_dir, "FlashAttn_tiling_func.cpp")
            source_header = os.path.join(source_dir, "autofuse_tiling_func_common.h")
            with open(source_cpp, "w") as f:
                f.write("// source")
            with open(source_header, "w") as f:
                f.write("// header")

            build_dir, _ = prepare_build_dir(source_dir, "tf", tmp_dir, {})

            self.assertEqual(build_dir, tmp_dir)
            self.assertTrue(
                os.path.isfile(os.path.join(tmp_dir, "FlashAttn_tiling_func.cpp"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(tmp_dir, "autofuse_tiling_func_common.h"))
            )
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, "CMakeLists.txt")))
            self.assertFalse(os.path.exists(os.path.join(source_dir, "CMakeLists.txt")))


class TestValidateInputParams(unittest.TestCase):
    def test_rejects_out_of_range_native_inputs(self):
        with self.assertRaises(ValueError):
            validate_input_params(
                {"dynamic_dims": [0], "aiv_num": 48, "ub_size": 196608}
            )
        with self.assertRaises(ValueError):
            validate_input_params({"dynamic_dims": [], "aiv_num": 0, "ub_size": 196608})
        with self.assertRaises(ValueError):
            validate_input_params({"dynamic_dims": [], "aiv_num": 48, "ub_size": 0})

    @patch("commands.verify_tiling.ctypes.CDLL")
    def test_execute_uses_five_argument_native_abi(self, load_library):
        library = MagicMock()
        library.GetTilingDataSize.return_value = 8

        def fill_outputs(*args):
            args[3]._obj.value = 7
            return 0

        library.AutofuseTiling.side_effect = fill_outputs
        load_library.return_value = library

        from commands.verify_tiling import execute_tiling

        result = execute_tiling(
            "fake.so",
            {
                "dynamic_dims": [1024],
                "aiv_num": 48,
                "ub_size": 196608,
                "abi": {"kind": "tf_dynamic", "shape_dims": 1, "block_dim_width": 32},
            },
            "tf",
        )

        self.assertEqual(result, {"block_dim": 7, "workspace_size": 0})
        self.assertEqual(len(library.AutofuseTiling.call_args.args), 6)


if __name__ == "__main__":
    unittest.main()
