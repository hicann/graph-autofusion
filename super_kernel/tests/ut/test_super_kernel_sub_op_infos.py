#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Minimal smoke test ensuring the UT harness executes."""

import os
import sys
import pytest
from unittest import mock
import importlib
from utils import compare_files
from superkernel.super_kernel_sub_op_infos import *

THIS_FILE_NAME = __file__
FILE_PATH = os.path.dirname(os.path.realpath(THIS_FILE_NAME))
UTILS_FILE_PATH = os.path.join(FILE_PATH, "../")
sys.path.append(UTILS_FILE_PATH)
SUPER_KERNEL_PATH = os.path.join(FILE_PATH, "../../src")
sys.path.append(SUPER_KERNEL_PATH)
GLODEN_FILE_PATH = os.path.join(FILE_PATH, "./golden")

sub_op_add_json = {
    "binFileName": "te_op_add",
    "binFileSuffix": ".o",
    "blockDim": 36,
    "kernelName": "te_op_add",
    "sha256": "12345",
    "workspace": {
        "num": 1,
        "size": [
            32
        ],
        "type": [
            0
        ]
    },
    "sub_operator_params": [
        "input_x",
        "input_y",
        "output_z",
        "workspace"
    ],
    "sub_operator_kernel_type": "KERNEL_TYPE_AIV_ONLY",
    "sub_operator_kernel_name": {
        "AiCore": {
            "func_name": "add_aiv",
            "obj_files": "add_aiv.o"
        },
        "dav-c220-cube": {
            "func_name": "add_aic",
            "obj_files": "add_aic.o"
        },
        "dav-c220-vec": {
            "func_name": "add_aiv",
            "obj_files": "add_aiv.o"
        }
    },
    "sub_operator_early_start_set_flag": False,
    "sub_operator_early_start_wait_flag": False,
    "timestamp_option": False,
    "debugBufSize": 0,
    "debugOptions": ""
}


class TestSuperKernelSubOpInfos:
    @staticmethod
    def setup_method():
        print(f"---------------SetUp---------------")

    @staticmethod
    def teardown_method():
        print(f"--------------TearDown-------------")

    @staticmethod
    def test_gen_profiling_for_notify():
        info_dict = {
            "bin_path": "",
            "json_path": "",
            "kernel_name": "add"
        }
        
        with mock.patch("json.load", return_value=sub_op_add_json):
            sub_op = SubOperatorInfos(0, info_dict, 0, {})
            sub_op.profiling_mode = SuperKernelProfilingMode.ProfilingDisable
            code_gen = sub_op.gen_profiling_for_notify(1, False)
            assert code_gen == ""
            
            sub_op.profiling_mode = SuperKernelProfilingMode.ProfilingEnable
            code_gen = sub_op.gen_profiling_for_notify(1, False)
            assert "RecordProfiling(1, 0x4, true);" in code_gen
            
            code_gen = sub_op.gen_profiling_for_notify(1, True)
            assert "RecordProfiling(1, 0x4, false);" in code_gen

    @staticmethod
    def test_gen_profiling_for_wait():
        info_dict = {
            "bin_path": "",
            "json_path": "",
            "kernel_name": "add"
        }
        with mock.patch("json.load", return_value=sub_op_add_json):
            sub_op = SubOperatorInfos(0, info_dict, 0, {})
            sub_op.profiling_mode = SuperKernelProfilingMode.ProfilingDisable
            code_gen = sub_op.gen_profiling_for_wait(1, False)
            assert code_gen == ""
            
            sub_op.profiling_mode = SuperKernelProfilingMode.ProfilingEnable
            code_gen = sub_op.gen_profiling_for_wait(1, False)
            assert "RecordProfiling(1, 12, true);\n" == code_gen
            
            code_gen = sub_op.gen_profiling_for_wait(1, True)
            assert "RecordProfiling(1, 12, false);\n" == code_gen

    @staticmethod
    def test_gen_notify_from_outside(tmp_dir):
        info_dict = {
            "bin_path": "",
            "json_path": "",
            "kernel_name": "add"
        }
        
        with mock.patch("json.load", return_value=sub_op_add_json):
            with mock.patch.object(CommonUtility, 'get_kernel_meta_dir', return_value=tmp_dir):
                sub_op = SubOperatorInfos(0, info_dict, 0, {})
                sub_op.send_event_list = [100, 101]
                sub_op.notify_param_offset = 5
                sub_op.kernel_name = "test_kernel"
                
                sub_op.kernel_type = KernelMetaType.KERNEL_TYPE_AIC_ONLY
                inner_event_id_set = []
                sub_op.gen_notify_from_outside(inner_event_id_set, False)
                goden_code = """\
if ASCEND_IS_AIC {
    // kernel=test_kernel, ev=100, param_offset=5
    NotifyFunc<true>(param_base[5]);
    // kernel=test_kernel, ev=101, param_offset=6
    NotifyFunc<true>(param_base[6]);
}
"""
                assert "NotifyFunc<true>" in sub_op.notify_block
                
                sub_op.kernel_type = KernelMetaType.KERNEL_TYPE_AIV_ONLY
                sub_op.gen_notify_from_outside(inner_event_id_set, False)
                goden_code = """\
if ASCEND_IS_AIV {
    // kernel=test_kernel, ev=100, param_offset=5
    NotifyFunc<false>(param_base[5]);
    // kernel=test_kernel, ev=101, param_offset=6
    NotifyFunc<false>(param_base[6]);
}
"""
                assert "NotifyFunc<false>" in sub_op.notify_block


if __name__ == "__main__":
    pytest.main()