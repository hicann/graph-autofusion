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

    def test_super_kernel_sub_op_add_compile(self, tmp_dir):
        with mock.patch("builtins.open", new_callable=mock.mock_open, read_data="{}"):
            with mock.patch("json.load", return_value=sub_op_add_json):
                with mock.patch("subprocess.run"):
                    with mock.patch.object(CommonUtility, 'is_support_super_kernel', return_value=True):
                        with mock.patch.object(CommonUtility, 'get_kernel_meta_dir', return_value=tmp_dir):
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

    @staticmethod
    def test_print_params_addr():
        kernel_info = {
            "op_list": [],
        }
        with mock.patch.object(CommonUtility, 'is_c310', return_value=False):
            super_operator = SuperOperatorInfos(kernel_info, "test_print_params_addr")
            super_operator.super_kernel_params = ["param1", "param2"]
            code_gen = print_params_addr(super_operator.super_kernel_params)
            result = ''
            index = 0
            result += 'AscendC::printf("ffts_addr: %p\\n", ffts_addr); //para index: 0\n'
            index += 1
            for param in super_operator.super_kernel_params:
                result += f'AscendC::printf("{param}: %p\\n", {param}); //para index: {index}\n'
                index += 1
            assert code_gen == result
    
    @staticmethod
    def test_tpl_of_gen_switch_case_call():
        kernel_info = {
            "op_list": [],
        }
        info_dict = {
            "bin_path": "",
            "json_path": "",
            "kernel_name": "add"
        }
        with mock.patch("json.load", return_value=sub_op_add_json):
            sub_operator = SubOperatorInfos(0, info_dict, 0, {})
            super_operator = SuperOperatorInfos(kernel_info, "test_tpl_of_gen_switch_case_call")

            super_operator.kernel_type = KernelMetaType.KERNEL_TYPE_AIC_ONLY
            code_gen = tpl_of_gen_switch_case_call(sub_operator.start_block_idx, sub_operator, super_operator)

            super_operator.kernel_type = KernelMetaType.KERNEL_TYPE_AIV_ONLY
            code_gen = tpl_of_gen_switch_case_call(sub_operator.start_block_idx, sub_operator, super_operator)
            assert "AscendC::GetBlockIdx" in code_gen

    @staticmethod
    def test_gen_switch_case_call_block_of_dynamic_op():
        kernel_info = {
            "op_list": [],
        }
        info_dict = {
            "bin_path": "",
            "json_path": "",
            "kernel_name": "add"
        }
        with mock.patch("json.load", return_value=sub_op_add_json):
            sub_operator = SubOperatorInfos(0, info_dict, 0, {})
            super_operator = SuperOperatorInfos(kernel_info, "test_gen_switch_case_call_block_of_dynamic_op")
            pre_sub_operator = SubOperatorInfos(0, info_dict, 0, {})
            next_sub_operator = SubOperatorInfos(0, info_dict, 0, {})

            sub_operator.sub_op_task_type = SubOperatorType.DYNAMIC_OP
            sub_operator.switch_func_called_flag = False

            pre_sub_operator = None
            super_operator.enable_double_stream = False

            code_gen = gen_switch_case_call_block_of_dynamic_op(super_operator, \
                                                                next_sub_operator, \
                                                                sub_operator, \
                                                                pre_sub_operator)

            assert "pipe_barrier(PIPE_ALL);\n" in code_gen \
                and "AscendC::SyncAll<false>(); // reason3: dynamic gen_switch_case_block when no pre op\n" in code_gen
    
    @staticmethod
    def test_gen_clear_wait_sync_addr_code():
        kernel_info = {
            "op_list": [],
        }
        info_dict = {
            "bin_path": "",
            "json_path": "",
            "kernel_name": "add"
        }
        with mock.patch("json.load", return_value=sub_op_add_json):
            super_operator = SuperOperatorInfos(kernel_info, "test_gen_clear_wait_sync_addr_code")
            super_operator.info_base = [SubOperatorInfos(0, info_dict, 0, {}), \
                                        SubOperatorInfos(0, info_dict, 0, {})]
            for op in super_operator.info_base:
                op.recv_event_list = info_dict.get('recv_event_list', [100, 101])
            super_operator.info_base[0].kernel_type = KernelMetaType.KERNEL_TYPE_AIC_ONLY
            super_operator.info_base[1].kernel_type = KernelMetaType.KERNEL_TYPE_AIV_ONLY
            code_gen = gen_clear_wait_sync_addr_code(super_operator)

            assert "if ASCEND_IS_AIC {\n" in code_gen and f"    if (get_block_idx() == 0) {{\n" in code_gen
            assert "if ASCEND_IS_AIV {\n" in code_gen and f"    if (AscendC::GetBlockIdx() == 0) {{\n" in code_gen
            assert f"*(reinterpret_cast<__gm__ uint64_t*>" in code_gen

    @staticmethod
    def test_process_gen_stream_send_code():
        kernel_info = {
            "op_list": [],
        }
        info_dict = {
            "bin_path": "",
            "json_path": "",
            "kernel_name": "add"
        }
        super_operator = SuperOperatorInfos(kernel_info, "test_process_gen_stream_send_code")
        super_operator.cub_op_list = [SubOperatorInfos(1, info_dict, 0, {})]
        super_operator.vec_op_list = [SubOperatorInfos(1, info_dict, 0, {})]
        op = SubOperatorInfos(0, info_dict, 0, {})
        arch = 'aic'
        need_flag = True
        
        with mock.patch("json.load", return_value=sub_op_add_json):
            arch = 'aic'
            need_flag = True
            code = "init_code_str"
            code_gen = process_gen_stream_send_code(super_operator, op, arch, need_flag, code)
            assert "init_code_str" in code_gen

            arch = 'aic'
            need_flag = False
            code = "init_code_str"
            code = "init_code_str"
            code_gen = process_gen_stream_send_code(super_operator, op, arch, need_flag, code)
            assert f"   pipe_barrier(PIPE_ALL);\n" in code_gen

            arch = 'aiv'
            need_flag = True
            code = "init_code_str"
            code_gen = process_gen_stream_send_code(super_operator, op, arch, need_flag, code)
            assert "init_code_str" in code_gen

            arch = 'aiv'
            need_flag = False
            code = "init_code_str"
            code_gen = process_gen_stream_send_code(super_operator, op, arch, need_flag, code)
            assert f"   pipe_barrier(PIPE_ALL);\n" in code_gen

    @staticmethod
    def test_gen_2_real_stream_send_code():
        kernel_info = {
            "op_list": [],
        }
        info_dict = {
            "bin_path": "",
            "json_path": "",
            "kernel_name": "add"
        }
        with mock.patch("json.load", return_value=sub_op_add_json):
            super_operator = SuperOperatorInfos(kernel_info, "test_gen_2_real_stream_send_code")
            super_operator.info_base = [SubOperatorInfos(0, info_dict, 0, {}), \
                                        SubOperatorInfos(0, info_dict, 0, {})]
            super_operator.cub_op_list = [SubOperatorInfos(1, info_dict, 0, {})]
            op = SubOperatorInfos(0, info_dict, 0, {})
            op.send_info = {"op1": "cub:cub", "op2": "cub:vec", "op3": "vec:vec", "op4": "vec:cub"}

            arch = 'aic'
            op.index = 0
            super_operator.info_base[-1].index = 0
            code_gen = gen_2_real_stream_send_code(super_operator, op, arch)

            op.index = 1
            super_operator.info_base[-1].index = 0
            code_gen = gen_2_real_stream_send_code(super_operator, op, arch)

            assert "wait_flag_dev(AscendC::SYNC_AIC_FLAG);" in code_gen
            assert "ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIC_AIV_FLAG));" in code_gen

            arch = 'aiv'
            code_gen = gen_2_real_stream_send_code(super_operator, op, arch)

            assert "wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL);" in code_gen
            assert "ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIV_FLAG));" in code_gen

    @staticmethod
    def test_gen_2_real_stream_recv_code():

        info_dict = {
            "bin_path": "",
            "json_path": "",
            "kernel_name": "add"
        }
        with mock.patch("json.load", return_value=sub_op_add_json):
            op = SubOperatorInfos(0, info_dict, 0, {})
            op.recv_info = {"op1": "vec:cub", "op2": "cub:vec"}

            arch = 'aic'
            code_gen = gen_2_real_stream_recv_code(op, arch)
            assert "wait_flag_dev(AscendC::SYNC_AIV_FLAG);\n" in code_gen

            arch = 'aiv'
            code_gen = gen_2_real_stream_recv_code(op, arch)
            assert "wait_flag_dev(AscendC::SYNC_AIC_AIV_FLAG);\n" in code_gen

    @staticmethod
    def test_gen_2_real_stream_sync_code():
        kernel_info = {
            "op_list": [],
        }
        info_dict = {
            "bin_path": "",
            "json_path": "",
            "kernel_name": "add"
        }
        with mock.patch("json.load", return_value=sub_op_add_json):
            super_operator = SuperOperatorInfos(kernel_info, "test_gen_2_real_stream_sync_code")
            super_operator.info_base = [SubOperatorInfos(0, info_dict, 0, {}), \
                                        SubOperatorInfos(0, info_dict, 0, {})]
            pre_op = SubOperatorInfos(0, info_dict, 0, {})
            cur_op = SubOperatorInfos(0, info_dict, 0, {})

            pre_op.index = 0
            super_operator.info_base[-1].index = 1
            
            pre_op.send_info = {"op1": "cub:cub", "op2": "cub:vec", "op3": "vec:vec", "op4": "vec:cub"}
            cur_op.recv_info = {"op1": "vec:cub", "op2": "cub:vec"}
            
            code_gen = gen_2_real_stream_sync_code(super_operator, pre_op, cur_op, 'aic')
            assert "ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIC_AIV_FLAG));" in code_gen
            assert "wait_flag_dev(AscendC::SYNC_AIV_FLAG);\n" in code_gen

            code_gen = gen_2_real_stream_sync_code(super_operator, pre_op, cur_op, 'aiv')
            assert "ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIV_FLAG));" in code_gen
            assert "wait_flag_dev(AscendC::SYNC_AIC_AIV_FLAG);\n" in code_gen

    @staticmethod
    def test_gen_sync_and_event_code_for_two_stream():
        kernel_info = {
            "op_list": [],
        }
        info_dict = {
            "bin_path": "",
            "json_path": "",
            "kernel_name": "add"
        }
        with mock.patch("json.load", return_value=sub_op_add_json):
            super_operator = SuperOperatorInfos(kernel_info, "test_gen_sync_and_event_code_for_two_stream")
            super_operator.info_base = [SubOperatorInfos(0, info_dict, 0, {}), \
                                        SubOperatorInfos(0, info_dict, 0, {})]
            pre_sub_operator = SubOperatorInfos(0, info_dict, 0, {})
            sub_operator = SubOperatorInfos(0, info_dict, 0, {})
            sub_operator.wait_block = "wait_code"
            
            pre_sub_operator.send_event_list = [100]
            sub_operator.recv_event_list = [101]
            
            arch = 'aic'
            pre_sub_operator.notify_block = {"aic": "notify_code", "aiv": "notify_code"}
            sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIC_1_1
            code_gen = gen_sync_and_event_code_for_two_stream(super_operator, pre_sub_operator, sub_operator, arch)
            assert "AscendC::SyncAll<false>(); // reason3: for continues notify/wait event" in code_gen

            sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_AIC_ONLY
            code_gen = gen_sync_and_event_code_for_two_stream(super_operator, pre_sub_operator, sub_operator, arch)
            assert "wait_flag_dev(AscendC::SYNC_AIC_FLAG);" in code_gen

            sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_AIV_ONLY
            code_gen = gen_sync_and_event_code_for_two_stream(super_operator, pre_sub_operator, sub_operator, arch)
            assert "wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL);" in code_gen

            sub_operator.recv_event_list = []
            code_gen = gen_sync_and_event_code_for_two_stream(super_operator, pre_sub_operator, sub_operator, arch)
            assert "notify_code" in code_gen

    @staticmethod
    def test_gen_2_real_stream_code_by_arch():
        kernel_info = {
            "op_list": [],
        }
        info_dict = {
            "bin_path": "",
            "json_path": "",
            "kernel_name": "add"
        }
        with mock.patch("json.load", return_value=sub_op_add_json):
            with mock.patch.object(CommonUtility, 'ascendc_raise_python_err') as mock_raise:
                super_operator = SuperOperatorInfos(kernel_info, "test_gen_2_real_stream_code_by_arch")
                super_operator.info_base = [SubOperatorInfos(0, info_dict, 0, {}), \
                                            SubOperatorInfos(0, info_dict, 0, {})]
                sub_ops = [SubOperatorInfos(0, info_dict, 0, {}), \
                           SubOperatorInfos(0, info_dict, 0, {}), \
                            SubOperatorInfos(0, info_dict, 0, {})]
                
                arch = 'aic'
                super_kernel_params_str = ''
                exits_dynamic_op = True
                super_operator.split_mode = 4
                code_gen = gen_2_real_stream_code_by_arch(super_operator, \
                                                          arch, \
                                                          super_kernel_params_str, \
                                                          exits_dynamic_op, \
                                                          sub_ops)

                assert "uint64_t aic_func_addr_split3 = 0;" in code_gen

                super_operator.preload_mode = SuperKernelPreLoadMode.PreLoadByWhole

                code_gen = gen_2_real_stream_code_by_arch(super_operator, \
                                                          arch, \
                                                          super_kernel_params_str, \
                                                          exits_dynamic_op, \
                                                          sub_ops)
                
                assert "AscendC::PreLoad(8);" in code_gen

                super_operator.preload_mode = SuperKernelPreLoadMode.PreLoadStepByStep
                sub_ops[1].preload_call_block = "sub_operator.preload_call_block"

                code_gen = gen_2_real_stream_code_by_arch(super_operator, \
                                                          arch, \
                                                          super_kernel_params_str, \
                                                          exits_dynamic_op, \
                                                          sub_ops)
                assert "sub_operator.preload_call_block" in code_gen

                super_operator.preload_mode = SuperKernelPreLoadMode.PreloadByAdanvanceStep
                sub_ops[2].preload_call_block = "next_sub_operator.preload_call_block"
                code_gen = gen_2_real_stream_code_by_arch(super_operator, \
                                                          arch, \
                                                          super_kernel_params_str, \
                                                          exits_dynamic_op, \
                                                          sub_ops)
                assert "next_sub_operator.preload_call_block" in code_gen
                

                super_operator.datacache_mode = SuperKernelDataCacheMode.DataCacheLoadAdancanceStep
                sub_ops[1].data_cache_preload_call = "sub_operator.data_cache_preload_call"
                code_gen = gen_2_real_stream_code_by_arch(super_operator, \
                                                          arch, \
                                                          super_kernel_params_str, \
                                                          exits_dynamic_op, \
                                                          sub_ops)

                assert "sub_operator.data_cache_preload_call" in code_gen
                sub_ops[2].data_cache_preload_call = "next_sub_operator.data_cache_preload_call"
                code_gen = gen_2_real_stream_code_by_arch(super_operator, \
                                                          arch, \
                                                          super_kernel_params_str, \
                                                          exits_dynamic_op, \
                                                          sub_ops)
                assert "next_sub_operator.data_cache_preload_call" in code_gen

                sub_ops[2].send_event_list = [100]
                sub_ops[2].notify_block = {"aic": "notify_code", "aiv": "notify_code"}

                code_gen = gen_2_real_stream_code_by_arch(super_operator, \
                                                          arch, \
                                                          super_kernel_params_str, \
                                                          exits_dynamic_op, \
                                                          sub_ops)
                mock_raise.assert_called_once_with(ERR_CODE, \
f"last op of super kernel must not have any send event, op:{sub_ops[2].kernel_name}, \
event_list:{sub_ops[2].send_event_list}")



if __name__ == "__main__":
    pytest.main()