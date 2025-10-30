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

THIS_FILE_NAME = __file__
FILE_PATH = os.path.dirname(os.path.realpath(THIS_FILE_NAME))
UTILS_FILE_PATH = os.path.join(FILE_PATH, "../")
sys.path.append(UTILS_FILE_PATH)
SUPER_KERNEL_PATH = os.path.join(FILE_PATH, "../../src")
sys.path.append(SUPER_KERNEL_PATH)
GLODEN_FILE_PATH = os.path.join(FILE_PATH, "./golden")

from utils import validate_codegen_output, validate_compile_options, compare_files
from superkernel.super_kernel import *

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
        "dav-c220-cube":{
            "func_name": "add_aic",
            "obj_files": "add_aic.o"
        },
        "dav-c220-vec":{
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


class TestSuperKernel:
    def setup_method(self):
        print(f"---------------SetUp---------------")

    def teardown_method(self):
        print(f"--------------TearDown-------------")

    def test_super_kernel_sub_op_add_compile(self):
        with mock.patch("builtins.open", new_callable=mock.mock_open, read_data="{}"):
            with mock.patch("json.load", return_value=sub_op_add_json):
                with mock.patch("subprocess.run"):
                    with mock.patch.object(CommonUtility, 'is_support_super_kernel', return_value = True):
                        with mock.patch("superkernel.super_kernel.super_kernel_compile"):
                            kernel_info = {
                                "op_list": [
                                    {
                                        "bin_path": "",
                                        "json_path": "",
                                        "kernel_name": "add"
                                    }],
                            }
                            super_kernel_optype = "test_add"
                            compile(kernel_info, super_kernel_optype)
                            code_gen_path = os.path.join(CommonUtility.get_kernel_meta_dir(), super_kernel_optype + "kernel.cpp")
                            gloden_code_path = os.path.join(GLODEN_FILE_PATH, super_kernel_optype + "kernel.cpp")
                            assert compare_files(code_gen_path, gloden_code_path)

    def test_gen_early_start_config_pre_op_is_aiv(self):
        info_dict = {
            "bin_path": "",
            "json_path": "",
            "kernel_name": "add"
        }
        with mock.patch("json.load", return_value=sub_op_add_json):
            pre_sub_operator = SubOperatorInfos(0, info_dict, 0, {})
            sub_operator = SubOperatorInfos(0, info_dict, 0, {})
            pre_sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_AIV_ONLY
            sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_AIV_ONLY
            assert gen_early_start_config(pre_sub_operator, sub_operator) == "g_super_kernel_early_start_config = 5;\n"

            sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIV_1_0
            assert gen_early_start_config(pre_sub_operator, sub_operator) == "g_super_kernel_early_start_config = 5;\n"

            sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_AIC_ONLY
            assert gen_early_start_config(pre_sub_operator, sub_operator) == "g_super_kernel_early_start_config = 4;\n"

            sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIC_1_0
            assert gen_early_start_config(pre_sub_operator, sub_operator) == "g_super_kernel_early_start_config = 4;\n"

            sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIC_1_1
            assert gen_early_start_config(pre_sub_operator, sub_operator) == "g_super_kernel_early_start_config = 6;\n"

            sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIC_1_2
            assert gen_early_start_config(pre_sub_operator, sub_operator) == "g_super_kernel_early_start_config = 6;\n"

    def test_gen_early_start_config_pre_op_is_aic(self):
        info_dict = {
            "bin_path": "",
            "json_path": "",
            "kernel_name": "add"
        }
        with mock.patch("json.load", return_value=sub_op_add_json):
            pre_sub_operator = SubOperatorInfos(0, info_dict, 0, {})
            sub_operator = SubOperatorInfos(0, info_dict, 0, {})
            pre_sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_AIC_ONLY
            sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_AIV_ONLY
            assert gen_early_start_config(pre_sub_operator, sub_operator) == "g_super_kernel_early_start_config = 1;\n"

            sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIV_1_0
            assert gen_early_start_config(pre_sub_operator, sub_operator) == "g_super_kernel_early_start_config = 1;\n"

            sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_AIC_ONLY
            assert gen_early_start_config(pre_sub_operator, sub_operator) == "g_super_kernel_early_start_config = 0;\n"

            sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIC_1_0
            assert gen_early_start_config(pre_sub_operator, sub_operator) == "g_super_kernel_early_start_config = 0;\n"

            sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIC_1_1
            assert gen_early_start_config(pre_sub_operator, sub_operator) == "g_super_kernel_early_start_config = 2;\n"

            sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIC_1_2
            assert gen_early_start_config(pre_sub_operator, sub_operator) == "g_super_kernel_early_start_config = 2;\n"

    def test_gen_early_start_config_pre_op_is_mix(self):
        info_dict = {
            "bin_path": "",
            "json_path": "",
            "kernel_name": "add"
        }
        with mock.patch("json.load", return_value=sub_op_add_json):
            pre_sub_operator = SubOperatorInfos(0, info_dict, 0, {})
            sub_operator = SubOperatorInfos(0, info_dict, 0, {})
            pre_sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIC_1_1
            sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_AIV_ONLY
            assert gen_early_start_config(pre_sub_operator, sub_operator) == "g_super_kernel_early_start_config = 9;\n"

            sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIV_1_0
            assert gen_early_start_config(pre_sub_operator, sub_operator) == "g_super_kernel_early_start_config = 9;\n"

            sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_AIC_ONLY
            assert gen_early_start_config(pre_sub_operator, sub_operator) == "g_super_kernel_early_start_config = 8;\n"

            sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIC_1_0
            assert gen_early_start_config(pre_sub_operator, sub_operator) == "g_super_kernel_early_start_config = 8;\n"

            sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIC_1_1
            assert gen_early_start_config(pre_sub_operator, sub_operator) == "g_super_kernel_early_start_config = 10;\n"

            sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIC_1_2
            assert gen_early_start_config(pre_sub_operator, sub_operator) == "g_super_kernel_early_start_config = 10;\n"

    def test_gen_early_start_config_pre_op_raise(self):
        info_dict = {
            "bin_path": "",
            "json_path": "",
            "kernel_name": "add"
        }
        with mock.patch("json.load", return_value=sub_op_add_json):
            pre_sub_operator = SubOperatorInfos(0, info_dict, 0, {})
            sub_operator = SubOperatorInfos(0, info_dict, 0, {})

            pre_sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_MAX
            with pytest.raises(Exception):
                gen_early_start_config(pre_sub_operator, sub_operator)

            pre_sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIC_1_1
            sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_MAX
            with pytest.raises(Exception):
                gen_early_start_config(pre_sub_operator, sub_operator)

    def test_gen_notify_wait_func(self):
        notify_func = f"""
template<bool aic_flag>
__aicore__ inline void NotifyFunc(GM_ADDR notify_lock_addr)
{{
    if constexpr (aic_flag) {{
        if (get_block_idx() == 0) {{
            __gm__ uint64_t* notifyLock = reinterpret_cast<__gm__ uint64_t*>(notify_lock_addr);
            *notifyLock = 1;
            dcci(notifyLock, 0, 2);
        }}
    }} else {{
        if (AscendC::GetBlockIdx() == 0) {{
            __gm__ uint64_t* notifyLock = reinterpret_cast<__gm__ uint64_t*>(notify_lock_addr);
            *notifyLock = 1;
            dcci(notifyLock, 0, 2);
        }}
    }}
}}\n
"""
        wait_func = f"""
template<bool aic_flag>
__aicore__ inline void WaitFunc(GM_ADDR wait_lock_addr)
{{
    if constexpr (aic_flag) {{
        __gm__ volatile uint64_t* waitLock = reinterpret_cast<__gm__ uint64_t*>(wait_lock_addr);
        if (get_block_idx() == 0) {{
            dcci(waitLock, 0, 2);
            while(*waitLock != 1) {{
                dcci(waitLock, 0, 2);
            }}
        }}
    }} else {{
        __gm__ volatile uint64_t* waitLock = reinterpret_cast<__gm__ uint64_t*>(wait_lock_addr);
        if (AscendC::GetBlockIdx() == 0) {{
            dcci(waitLock, 0, 2);
            while(*waitLock != 1) {{
                dcci(waitLock, 0, 2);
            }}
        }}
    }}
}}\n
"""
        gen_code = gen_notify_wait_func()
        assert gen_code == notify_func + wait_func

    def test_get_sync_code_by_kernel_type(self):
        code_gen = get_sync_code_by_kernel_type(KernelMetaType.KERNEL_TYPE_MIX_AIC_1_1)
        gloden_code = "AscendC::SyncAll<false>();\n\n"
        assert code_gen == gloden_code

        code_gen = get_sync_code_by_kernel_type(KernelMetaType.KERNEL_TYPE_MIX_AIC_1_2)
        assert code_gen == gloden_code

        code_gen = get_sync_code_by_kernel_type(KernelMetaType.KERNEL_TYPE_AIC_ONLY)
        gloden_code = """
ffts_cross_core_sync(PIPE_FIX, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIC_FLAG));
wait_flag_dev(AscendC::SYNC_AIC_FLAG);
"""
        assert code_gen == gloden_code

        code_gen = get_sync_code_by_kernel_type(KernelMetaType.KERNEL_TYPE_MIX_AIC_1_0)
        assert code_gen == gloden_code

        code_gen = get_sync_code_by_kernel_type(KernelMetaType.KERNEL_TYPE_AIV_ONLY)
        gloden_code = """
ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIV_ONLY_ALL));
wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL);
"""
        assert code_gen == gloden_code

        code_gen = get_sync_code_by_kernel_type(KernelMetaType.KERNEL_TYPE_MIX_AIV_1_0)
        assert code_gen == gloden_code

    def test_gen_inter_ops_barrier(self):
        kernel_info = {
            "op_list": [],
        }
        info_dict = {
            "bin_path": "",
            "json_path": "",
            "kernel_name": "add"
        }
        with mock.patch("json.load", return_value=sub_op_add_json):
            super_operator = SuperOperatorInfos(kernel_info, "test_gen_inter_ops_barrier")
            pre_sub_operator = SubOperatorInfos(0, info_dict, 0, {})
            sub_operator = SubOperatorInfos(0, info_dict, 0, {})

            super_operator.early_start_mode = SuperKernelEarlyStartMode.EarlyStartDisable
            code_gen = gen_inter_ops_barrier(super_operator, pre_sub_operator, sub_operator)
            assert "AscendC::SyncAll<false>();" in code_gen

            super_operator.early_start_mode = SuperKernelEarlyStartMode.EarlyStartEnableV1
            pre_sub_operator.early_start_complement_set_flag_block = "pre_sub_op_key"
            code_gen = gen_inter_ops_barrier(super_operator, pre_sub_operator, sub_operator)
            assert "pre_sub_op_key" in code_gen

            super_operator.early_start_mode = SuperKernelEarlyStartMode.EarlyStartEnableV2
            pre_sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_AIV_ONLY
            sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_AIV_ONLY
            code_gen = gen_inter_ops_barrier(super_operator, pre_sub_operator, sub_operator)
            assert "pre_sub_op_key" in code_gen
            assert "g_super_kernel_early_start_config = 5" in code_gen

    def test_gen_op_end_debug_dcci_all(self):
        kernel_info = {
            "op_list": [],
        }
        super_operator = SuperOperatorInfos(kernel_info, "test_gen_op_end_debug_dcci_all")
        super_operator.debug_dcci_all_mode = SuperKernelDebugDcciAllMode.DebugDcciAllEnable
        code_gen = gen_op_end_debug_dcci_all(super_operator)
        assert "dcci((__gm__ uint64_t*)0, cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);" in code_gen

        super_operator.debug_dcci_all_mode = SuperKernelDebugDcciAllMode.DebugDcciAllDisable
        code_gen = gen_op_end_debug_dcci_all(super_operator)
        assert code_gen == ""

    def test_gen_op_end_debug_sync_all(self):
        kernel_info = {
            "op_list": [],
        }
        super_operator = SuperOperatorInfos(kernel_info, "test_gen_op_end_debug_sync_all")
        super_operator.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIC_1_2
        super_operator.debug_sync_all_mode = SuperKernelDebugSyncAllMode.DebugSyncAllEnable
        code_gen = gen_op_end_debug_sync_all(super_operator)
        assert "AscendC::SyncAll<false>();" in code_gen

        super_operator.debug_sync_all_mode = SuperKernelDebugSyncAllMode.DebugSyncAllDisable
        code_gen = gen_op_end_debug_sync_all(super_operator)
        assert code_gen == ""

    def test_gen_2_real_stream_op_end_debug_sync_all_by_arch(self):
        kernel_info = {
            "op_list": [],
        }
        super_operator = SuperOperatorInfos(kernel_info, "test_gen_2_real_stream_op_end_debug_sync_all_by_arch")
        super_operator.debug_sync_all_mode = SuperKernelDebugSyncAllMode.DebugSyncAllEnable
        code_gen = gen_2_real_stream_op_end_debug_sync_all_by_arch(super_operator, "aiv")
        assert "wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL);" in code_gen

        code_gen = gen_2_real_stream_op_end_debug_sync_all_by_arch(super_operator, "aic")
        assert "wait_flag_dev(AscendC::SYNC_AIC_FLAG);" in code_gen

        super_operator.debug_sync_all_mode = SuperKernelDebugSyncAllMode.DebugSyncAllDisable
        code_gen = gen_op_end_debug_sync_all(super_operator)
        assert code_gen == ""

if __name__ == "__main__":
    pytest.main()