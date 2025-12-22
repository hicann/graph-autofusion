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


class TestSuperKernel:
    @staticmethod
    def setup_method():
        print(f"---------------SetUp---------------")

    @staticmethod
    def teardown_method():
        print(f"--------------TearDown-------------")

    @staticmethod
    def test_super_kernel_sub_op_add_compile(tmp_dir):
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
                                code_gen_path = os.path.join(CommonUtility.get_kernel_meta_dir(),
                                                             super_kernel_optype + "kernel.cpp")
                                gloden_code_path = os.path.join(GLODEN_FILE_PATH, super_kernel_optype + "kernel.cpp")
                                assert compare_files(code_gen_path, gloden_code_path)

                    with mock.patch.object(CommonUtility, 'is_support_super_kernel', return_value=False):
                        with mock.patch.object(CommonUtility, 'ascendc_raise_python_err') as mock_raise:
                            with mock.patch("superkernel.super_kernel.super_kernel_compile"):
                                kernel_info = {
                                    "op_list": ""
                                }
                                super_kernel_optype = "test_add"
                                compile(kernel_info, super_kernel_optype)
                                mock_raise.assert_called()

    @staticmethod
    def test_gen_early_start_config_pre_op_is_aiv():
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

    @staticmethod
    def test_gen_early_start_config_pre_op_is_aic():
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

    @staticmethod
    def test_gen_early_start_config_pre_op_is_mix():
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

    @staticmethod
    def test_gen_early_start_config_pre_op_raise():
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

    @staticmethod
    def test_gen_notify_wait_func():
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

    @staticmethod
    def test_get_sync_code_by_kernel_type():
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

    @staticmethod
    def test_gen_inter_ops_barrier():
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

    @staticmethod
    def test_gen_op_end_debug_dcci_all():
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

    @staticmethod
    def test_gen_op_end_debug_sync_all():
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

    @staticmethod
    def test_gen_2_real_stream_op_end_debug_sync_all_by_arch():
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
            sub_operator.call_dynamic_switch_func = "call_dynamic_switch_func_mock"
            code_gen = tpl_of_gen_switch_case_call(sub_operator.start_block_idx, sub_operator, super_operator)
            assert "\n    call_dynamic_switch_func_mock\n" == code_gen

            super_operator.kernel_type = KernelMetaType.KERNEL_TYPE_AIV_ONLY
            code_gen = tpl_of_gen_switch_case_call(sub_operator.start_block_idx, sub_operator, super_operator)
            goden_code = """
    call_dynamic_switch_func_mock
    if ASCEND_IS_AIV {
      if (AscendC::GetBlockIdx() < 0) {
        uint8_t coreid = (uint8_t)get_coreid();
        if ((coreid % 4) == 0) {
          preload((const void *)aiv_func_addr, 8);
        } else if ((coreid % 4) == 1) {
          preload((const void *)aiv_func_addr_split1, 8);
        } else if ((coreid % 4) == 2) {
          preload((const void *)aiv_func_addr_split2, 8);
        } else {
          preload((const void *)aiv_func_addr_split3, 8);
        }

      }

    }\n\n\n"""
            assert goden_code == code_gen

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
            goden_code = """\n    \n    pipe_barrier(PIPE_ALL);
    AscendC::SyncAll<false>(); // reason3: dynamic gen_switch_case_block when no pre op
"""
            assert goden_code == code_gen

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
            
            goden_code = """\
    if ASCEND_IS_AIC {
        if (get_block_idx() == 0) {
            *(reinterpret_cast<__gm__ uint64_t*>(param_base[0])) = 0;
        }
    }
    if ASCEND_IS_AIC {
        if (get_block_idx() == 0) {
            *(reinterpret_cast<__gm__ uint64_t*>(param_base[1])) = 0;
        }
    }
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() == 0) {
            *(reinterpret_cast<__gm__ uint64_t*>(param_base[0])) = 0;
        }
    }
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() == 0) {
            *(reinterpret_cast<__gm__ uint64_t*>(param_base[1])) = 0;
        }
    }
"""
            assert goden_code == code_gen
            
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
            assert "init_code_str" == code_gen

            arch = 'aic'
            need_flag = False
            code = "init_code_str"
            code = "init_code_str"
            code_gen = process_gen_stream_send_code(super_operator, op, arch, need_flag, code)
            goden_code = """\
// insert pipe all for ops
   pipe_barrier(PIPE_ALL);
"""
            assert goden_code == code_gen

            arch = 'aiv'
            need_flag = True
            code = "init_code_str"
            code_gen = process_gen_stream_send_code(super_operator, op, arch, need_flag, code)
            assert "init_code_str" == code_gen

            arch = 'aiv'
            need_flag = False
            code = "init_code_str"
            code_gen = process_gen_stream_send_code(super_operator, op, arch, need_flag, code)
            goden_code = """\
// insert pipe all for ops
   pipe_barrier(PIPE_ALL);
"""
            assert goden_code == code_gen

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
            assert "" == code_gen

            op.index = 1
            super_operator.info_base[-1].index = 0
            code_gen = gen_2_real_stream_send_code(super_operator, op, arch)
            goden_code = """\
// Rule 1 : sync all aic must be insert behind each aic sub operator, when has real send info
// sync all C->C kernel_name:, send_info:{'op1': 'cub:cub', 'op2': 'cub:vec', 'op3': 'vec:vec', 'op4': 'vec:cub'}
ffts_cross_core_sync(PIPE_FIX, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIC_FLAG));
wait_flag_dev(AscendC::SYNC_AIC_FLAG);

// Rule 3.1 : sync all c2v must be insert when sendinfo has c2v, kernel_name:, send_info:{'op1': 'cub:cub', 'op2': 'cub:vec', 'op3': 'vec:vec', 'op4': 'vec:cub'}
// send sync of C->V;
ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIC_AIV_FLAG));

"""
            assert goden_code == code_gen

            arch = 'aiv'
            code_gen = gen_2_real_stream_send_code(super_operator, op, arch)
            goden_code = """\
// Rule 1 : sync all aiv must be insert behind each aiv sub operator, when has real send info
// sync all V->V kernel_name:, send_info:{'op1': 'cub:cub', 'op2': 'cub:vec', 'op3': 'vec:vec', 'op4': 'vec:cub'}
ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIV_ONLY_ALL));
wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL);

// Rule 3.1 : sync all v2c must be insert when sendinfo has v2c, kernel_name:, send_info:{'op1': 'cub:cub', 'op2': 'cub:vec', 'op3': 'vec:vec', 'op4': 'vec:cub'}
// send sync of V->C;
ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIV_FLAG));

"""
            assert goden_code == code_gen

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
            goden_code = """\
// Rule 3.2 : sync all v2c must be insert when recvinfo has v2c, kernel_name:, send_info:{'op1': 'vec:cub', 'op2': 'cub:vec'}
// receive sync of V->C;
wait_flag_dev(AscendC::SYNC_AIV_FLAG);
"""
            assert goden_code == code_gen

            arch = 'aiv'
            code_gen = gen_2_real_stream_recv_code(op, arch)
            goden_code = """\
// Rule 3.2 : sync all c2v must be insert when recvinfo has c2v, kernel_name:, send_info:{'op1': 'vec:cub', 'op2': 'cub:vec'}
// receive sync of C->V;
wait_flag_dev(AscendC::SYNC_AIC_AIV_FLAG);
"""
            assert goden_code == code_gen

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
            goden_code = """\
// Rule 1 : sync all aic must be insert behind each aic sub operator, when has real send info
// sync all C->C kernel_name:, send_info:{'op1': 'cub:cub', 'op2': 'cub:vec', 'op3': 'vec:vec', 'op4': 'vec:cub'}
ffts_cross_core_sync(PIPE_FIX, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIC_FLAG));
wait_flag_dev(AscendC::SYNC_AIC_FLAG);

// Rule 3.1 : sync all c2v must be insert when sendinfo has c2v, kernel_name:, send_info:{'op1': 'cub:cub', 'op2': 'cub:vec', 'op3': 'vec:vec', 'op4': 'vec:cub'}
// send sync of C->V;
ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIC_AIV_FLAG));

// Rule 3.2 : sync all v2c must be insert when recvinfo has v2c, kernel_name:, send_info:{'op1': 'vec:cub', 'op2': 'cub:vec'}
// receive sync of V->C;
wait_flag_dev(AscendC::SYNC_AIV_FLAG);
"""
            assert goden_code == code_gen

            code_gen = gen_2_real_stream_sync_code(super_operator, pre_op, cur_op, 'aiv')
            goden_code = """\
// Rule 1 : sync all aiv must be insert behind each aiv sub operator, when has real send info
// sync all V->V kernel_name:, send_info:{'op1': 'cub:cub', 'op2': 'cub:vec', 'op3': 'vec:vec', 'op4': 'vec:cub'}
ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIV_ONLY_ALL));
wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL);

// Rule 3.1 : sync all v2c must be insert when sendinfo has v2c, kernel_name:, send_info:{'op1': 'cub:cub', 'op2': 'cub:vec', 'op3': 'vec:vec', 'op4': 'vec:cub'}
// send sync of V->C;
ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIV_FLAG));

// Rule 3.2 : sync all c2v must be insert when recvinfo has c2v, kernel_name:, send_info:{'op1': 'vec:cub', 'op2': 'cub:vec'}
// receive sync of C->V;
wait_flag_dev(AscendC::SYNC_AIC_AIV_FLAG);
"""
            assert goden_code == code_gen

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
            goden_code = """\
    notify_code    wait_code// two stream when has wait event, add sync by current operator kernel type
    AscendC::SyncAll<false>(); // reason3: for continues notify/wait event \n
"""
            assert goden_code == code_gen

            sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_AIC_ONLY
            code_gen = gen_sync_and_event_code_for_two_stream(super_operator, pre_sub_operator, sub_operator, arch)
            goden_code = """\
    notify_code    wait_code// two stream when has wait event, add sync by current operator kernel type
// reason3: for continues notify/wait event
ffts_cross_core_sync(PIPE_FIX, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIC_FLAG));
wait_flag_dev(AscendC::SYNC_AIC_FLAG);\n
"""
            assert goden_code == code_gen

            sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_AIV_ONLY
            code_gen = gen_sync_and_event_code_for_two_stream(super_operator, pre_sub_operator, sub_operator, arch)
            goden_code = """\
    notify_code    wait_code// two stream when has wait event, add sync by current operator kernel type
// reason3: for continues notify/wait event
ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIV_ONLY_ALL));
wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL);\n
"""
            assert goden_code == code_gen

            sub_operator.recv_event_list = []
            code_gen = gen_sync_and_event_code_for_two_stream(super_operator, pre_sub_operator, sub_operator, arch)
            goden_code = "    notify_code"
            assert goden_code == code_gen

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
                goden_code = """\
__aicore__ inline void auto_gen_test_gen_2_real_stream_code_by_arch_kernel_aic(void) {
    GM_ADDR *param_base = (GM_ADDR *)get_para_base();
    uint64_t aiv_func_addr = 0;
    uint64_t aic_func_addr = 0;
    uint64_t dy_blockDim = 0;
    uint64_t aiv_func_addr_split1 = 0;
    uint64_t aic_func_addr_split1 = 0;
    uint64_t aiv_func_addr_split2 = 0;
    uint64_t aic_func_addr_split2 = 0;
    uint64_t aiv_func_addr_split3 = 0;
    uint64_t aic_func_addr_split3 = 0;
    //begin func call of sub operator 
    //begin func call of sub operator 
    //begin func call of sub operator 
}\n
"""
                assert goden_code == code_gen

                super_operator.preload_mode = SuperKernelPreLoadMode.PreLoadByWhole

                code_gen = gen_2_real_stream_code_by_arch(super_operator, \
                                                          arch, \
                                                          super_kernel_params_str, \
                                                          exits_dynamic_op, \
                                                          sub_ops)
                goden_code = """\
__aicore__ inline void auto_gen_test_gen_2_real_stream_code_by_arch_kernel_aic(void) {
    GM_ADDR *param_base = (GM_ADDR *)get_para_base();
    uint64_t aiv_func_addr = 0;
    uint64_t aic_func_addr = 0;
    uint64_t dy_blockDim = 0;
    uint64_t aiv_func_addr_split1 = 0;
    uint64_t aic_func_addr_split1 = 0;
    uint64_t aiv_func_addr_split2 = 0;
    uint64_t aic_func_addr_split2 = 0;
    uint64_t aiv_func_addr_split3 = 0;
    uint64_t aic_func_addr_split3 = 0;
    AscendC::PreLoad(8);
    //begin func call of sub operator 
    //begin func call of sub operator 
    //begin func call of sub operator 
}\n
"""
                assert goden_code == code_gen

                super_operator.preload_mode = SuperKernelPreLoadMode.PreLoadStepByStep
                sub_ops[1].preload_call_block = "sub_operator.preload_call_block"

                code_gen = gen_2_real_stream_code_by_arch(super_operator, \
                                                          arch, \
                                                          super_kernel_params_str, \
                                                          exits_dynamic_op, \
                                                          sub_ops)
                goden_code = """\
__aicore__ inline void auto_gen_test_gen_2_real_stream_code_by_arch_kernel_aic(void) {
    GM_ADDR *param_base = (GM_ADDR *)get_para_base();
    uint64_t aiv_func_addr = 0;
    uint64_t aic_func_addr = 0;
    uint64_t dy_blockDim = 0;
    uint64_t aiv_func_addr_split1 = 0;
    uint64_t aic_func_addr_split1 = 0;
    uint64_t aiv_func_addr_split2 = 0;
    uint64_t aic_func_addr_split2 = 0;
    uint64_t aiv_func_addr_split3 = 0;
    uint64_t aic_func_addr_split3 = 0;
    //begin func call of sub operator 
    //begin func call of sub operator 
    sub_operator.preload_call_block    //begin func call of sub operator 
}\n
"""
                assert goden_code == code_gen

                super_operator.preload_mode = SuperKernelPreLoadMode.PreloadByAdanvanceStep
                sub_ops[2].preload_call_block = "next_sub_operator.preload_call_block"
                code_gen = gen_2_real_stream_code_by_arch(super_operator, \
                                                          arch, \
                                                          super_kernel_params_str, \
                                                          exits_dynamic_op, \
                                                          sub_ops)
                goden_code = """\
__aicore__ inline void auto_gen_test_gen_2_real_stream_code_by_arch_kernel_aic(void) {
    GM_ADDR *param_base = (GM_ADDR *)get_para_base();
    uint64_t aiv_func_addr = 0;
    uint64_t aic_func_addr = 0;
    uint64_t dy_blockDim = 0;
    uint64_t aiv_func_addr_split1 = 0;
    uint64_t aic_func_addr_split1 = 0;
    uint64_t aiv_func_addr_split2 = 0;
    uint64_t aic_func_addr_split2 = 0;
    uint64_t aiv_func_addr_split3 = 0;
    uint64_t aic_func_addr_split3 = 0;
    //begin func call of sub operator 
    sub_operator.preload_call_block    //begin func call of sub operator 
    next_sub_operator.preload_call_block    //begin func call of sub operator 
}\n
"""
                assert goden_code == code_gen

                super_operator.datacache_mode = SuperKernelDataCacheMode.DataCacheLoadAdancanceStep
                sub_ops[1].data_cache_preload_call = "sub_operator.data_cache_preload_call"
                code_gen = gen_2_real_stream_code_by_arch(super_operator, \
                                                          arch, \
                                                          super_kernel_params_str, \
                                                          exits_dynamic_op, \
                                                          sub_ops)
                goden_code = """\
__aicore__ inline void auto_gen_test_gen_2_real_stream_code_by_arch_kernel_aic(void) {
    GM_ADDR *param_base = (GM_ADDR *)get_para_base();
    uint64_t aiv_func_addr = 0;
    uint64_t aic_func_addr = 0;
    uint64_t dy_blockDim = 0;
    uint64_t aiv_func_addr_split1 = 0;
    uint64_t aic_func_addr_split1 = 0;
    uint64_t aiv_func_addr_split2 = 0;
    uint64_t aic_func_addr_split2 = 0;
    uint64_t aiv_func_addr_split3 = 0;
    uint64_t aic_func_addr_split3 = 0;
    //begin func call of sub operator 
    sub_operator.preload_call_block    sub_operator.data_cache_preload_call
    //begin func call of sub operator 
    next_sub_operator.preload_call_block
    //begin func call of sub operator 

}\n
"""
                assert goden_code == code_gen
                sub_ops[2].data_cache_preload_call = "next_sub_operator.data_cache_preload_call"
                code_gen = gen_2_real_stream_code_by_arch(super_operator, \
                                                          arch, \
                                                          super_kernel_params_str, \
                                                          exits_dynamic_op, \
                                                          sub_ops)
                goden_code = """\
__aicore__ inline void auto_gen_test_gen_2_real_stream_code_by_arch_kernel_aic(void) {
    GM_ADDR *param_base = (GM_ADDR *)get_para_base();
    uint64_t aiv_func_addr = 0;
    uint64_t aic_func_addr = 0;
    uint64_t dy_blockDim = 0;
    uint64_t aiv_func_addr_split1 = 0;
    uint64_t aic_func_addr_split1 = 0;
    uint64_t aiv_func_addr_split2 = 0;
    uint64_t aic_func_addr_split2 = 0;
    uint64_t aiv_func_addr_split3 = 0;
    uint64_t aic_func_addr_split3 = 0;
    //begin func call of sub operator 
    sub_operator.preload_call_block    sub_operator.data_cache_preload_call
    //begin func call of sub operator 
    next_sub_operator.preload_call_block    next_sub_operator.data_cache_preload_call
    //begin func call of sub operator 

}\n
"""
                assert goden_code == code_gen

                sub_ops[2].send_event_list = [100]
                sub_ops[2].notify_block = {"aic": "notify_code", "aiv": "notify_code"}
                code_gen = gen_2_real_stream_code_by_arch(super_operator, \
                                                          arch, \
                                                          super_kernel_params_str, \
                                                          exits_dynamic_op, \
                                                          sub_ops)
                mock_raise.assert_called()
                goden_code = """\
__aicore__ inline void auto_gen_test_gen_2_real_stream_code_by_arch_kernel_aic(void) {
    GM_ADDR *param_base = (GM_ADDR *)get_para_base();
    uint64_t aiv_func_addr = 0;
    uint64_t aic_func_addr = 0;
    uint64_t dy_blockDim = 0;
    uint64_t aiv_func_addr_split1 = 0;
    uint64_t aic_func_addr_split1 = 0;
    uint64_t aiv_func_addr_split2 = 0;
    uint64_t aic_func_addr_split2 = 0;
    uint64_t aiv_func_addr_split3 = 0;
    uint64_t aic_func_addr_split3 = 0;
    //begin func call of sub operator 
    sub_operator.preload_call_block    sub_operator.data_cache_preload_call
    //begin func call of sub operator 
    next_sub_operator.preload_call_block    next_sub_operator.data_cache_preload_call
    //begin func call of sub operator 

    notify_code}\n
"""
                assert goden_code == code_gen

    @staticmethod
    def test_gen_profling_func_code():
        kernel_info = {
            "op_list": [],
        }
        super_operator = SuperOperatorInfos(kernel_info, "test_gen_profling_func_code")

        super_operator.profiling_mode = SuperKernelProfilingMode.ProfilingEnable
        code_gen = gen_profling_func_code(super_operator)
        goden_code = """\
\n__BLOCK_LOCAL__ __inline__ uint32_t g_profiling_task_id;
__BLOCK_LOCAL__ __inline__ __gm__ uint8_t* g_profiling_base_addr;
__BLOCK_LOCAL__ __inline__ __gm__ uint8_t* g_profiling_working_addr;
__BLOCK_LOCAL__ __inline__ __gm__ uint8_t* g_profiling_max_addr;
__BLOCK_LOCAL__ __inline__ bool g_profiling_off;
__BLOCK_LOCAL__ __inline__ uint32_t g_percore_size;
constexpr uint64_t PROFILING_MAGIC_NUMBER = 0xbdca8756;
constexpr uint32_t PROFILING_WORKINF_PTR_OFFSET = 8;
constexpr uint32_t PROFILING_MAX_PTR_OFFSET = 16;
constexpr uint32_t ONE_PROFILING_HEAD_SIZE = 16;
constexpr uint32_t ONE_PROFILING_DATA_SIZE = 16;
__aicore__ inline bool ProfilingAreaIsValid()
{
    return (*((__gm__ uint64_t*)g_profiling_base_addr) == PROFILING_MAGIC_NUMBER) && 
        ((*((__gm__ uint64_t*)g_profiling_working_addr)) < (*((__gm__ uint64_t*)g_profiling_max_addr)));
}

__aicore__ inline uint8_t GetProfilingBlockIdx()
{
    if ASCEND_IS_AIV {
        return get_block_idx() * get_subblockdim() + get_subblockid();
    } else {
        return get_block_idx() + 50;
    }
}
 
__aicore__ inline void RecordProfiling()
{
    if (g_profiling_off) {
        return;
    }
    uint8_t blockIdx = GetProfilingBlockIdx();
    uint64_t workAddr = *((__gm__ uint64_t*)g_profiling_working_addr);
    *((__gm__ uint64_t*)workAddr) = ((uint64_t)g_profiling_task_id << 32) | (((uint64_t)blockIdx) << 8) | 0xff;
    *((__gm__ uint64_t*)workAddr + 1) = static_cast<uint64_t>(AscendC::GetSystemCycle());
    dcci((__gm__ uint64_t*)workAddr, 0, 2);
    *((__gm__ uint64_t*)g_profiling_working_addr) += ONE_PROFILING_DATA_SIZE;
    if (!ProfilingAreaIsValid()) {
        g_profiling_off = true;
    }
    dcci((__gm__ uint64_t*)g_profiling_working_addr, 0, 2);
}
 
__aicore__ inline void RecordProfiling(uint32_t index, uint8_t profilingType, bool startFlag)
{
    if (g_profiling_off) {
        return;
    }
    uint8_t blockIdx = GetProfilingBlockIdx();
    uint64_t workAddr = *((__gm__ uint64_t*)g_profiling_working_addr);
    if (startFlag) {
        *((__gm__ uint64_t*)workAddr) = ((uint64_t)index << 32) | (((uint64_t)profilingType & 0xf) << 8) | 0x0;
    } else {
        *((__gm__ uint64_t*)workAddr) =
            ((uint64_t)index << 32) | (1 << 12) | (((uint64_t)profilingType & 0xf) << 8) | 0x0;
    }
    *((__gm__ uint64_t*)workAddr + 1) = static_cast<uint64_t>(AscendC::GetSystemCycle());
    dcci((__gm__ uint64_t*)workAddr, 0, 2);
    *((__gm__ uint64_t*)g_profiling_working_addr) += ONE_PROFILING_DATA_SIZE;
    if (!ProfilingAreaIsValid()) {
        g_profiling_off = true;
    }
    dcci((__gm__ uint64_t*)g_profiling_working_addr, 0, 2);
}
 
__aicore__ inline void InitProfiling(uint32_t taskId, GM_ADDR profilingPtr)
{
    g_profiling_off = false;
    uint8_t blockIdx = GetProfilingBlockIdx();
    g_percore_size = *((__gm__ uint32_t*)(profilingPtr + 12));
    g_profiling_base_addr = profilingPtr + 64 + blockIdx * g_percore_size;
    g_profiling_working_addr = g_profiling_base_addr + PROFILING_WORKINF_PTR_OFFSET;
    g_profiling_max_addr = g_profiling_base_addr + PROFILING_MAX_PTR_OFFSET;
    if (!ProfilingAreaIsValid()) {
        g_profiling_off = true;
        return;
    }
    g_profiling_task_id = taskId;
    RecordProfiling();
}
"""
        assert goden_code == code_gen

    @staticmethod
    def test_gen_profiling_start_and_end_record():
        kernel_info = {
            "op_list": [],
        }
        super_operator = SuperOperatorInfos(kernel_info, "test_gen_profiling_start_and_end_record")

        super_operator.profiling_mode = SuperKernelProfilingMode.ProfilingEnable
        code_gen = gen_profiling_start_and_end_record(super_operator, True)
        assert "RecordProfiling(0, 0, true);\n" == code_gen

        code_gen = gen_profiling_start_and_end_record(super_operator, False)
        assert "RecordProfiling(0, 0, false);\n" == code_gen

    @staticmethod
    def test_gen_2_real_stream_super_kernel_file():
        kernel_info = {
            "op_list": [],
        }
        info_dict = {
            "bin_path": "",
            "json_path": "",
            "kernel_name": "add"
        }

        with mock.patch("json.load", return_value=sub_op_add_json):
            super_operator = SuperOperatorInfos(kernel_info, "test_gen_2_real_stream_super_kernel_file")
            super_operator.info_base = [SubOperatorInfos(0, info_dict, 0, {})]
            sub_operator = super_operator.info_base[0]
            sub_operator.kernel_params = ["kernel_params"]
            sub_operator.extra_kernel_params = ["extra_kernel_params"]
            sub_operator.sub_op_task_type = SubOperatorType.DYNAMIC_OP
            sub_operator.kernel_name = "sub_operator.kernel_name"

            super_operator.cub_op_list = []
            super_operator.vec_op_list = []
            super_operator.timestamp_option = True
            super_operator.feed_sync_all_mode = SuperKernelFeedSyncAllMode.FeedSyncAllEnable
            super_operator.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIC_1_1
            super_operator.profiling_mode = SuperKernelProfilingMode.ProfilingEnable
            gen_2_real_stream_super_kernel_file(super_operator)
            assert os.path.exists(super_operator.kernel_file)
            with open(super_operator.kernel_file, 'r') as f:
                code_gen = f.read()
                assert "#if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON" in code_gen

            super_operator.timestamp_option = False
            super_operator.profiling_mode = SuperKernelProfilingMode.ProfilingEnable
            gen_2_real_stream_super_kernel_file(super_operator)
            assert os.path.exists(super_operator.kernel_file)
            with open(super_operator.kernel_file, 'r') as f:
                code_gen = f.read()
                assert "#if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON" not in code_gen

    @staticmethod
    def test_judge_need_feed_sync_all():
        kernel_info = {
            "op_list": [],
        }
        info_dict = {
            "bin_path": "",
            "json_path": "",
            "kernel_name": "add"
        }
        with mock.patch("json.load", return_value=sub_op_add_json):
            super_operator = SuperOperatorInfos(kernel_info, "test_judge_need_feed_sync_all")
            sub_op = SubOperatorInfos(0, info_dict, 0, {})

            sub_op.with_sync_all = False
            assert judge_need_feed_sync_all(super_operator, sub_op) == False

            sub_op.with_sync_all = True
            super_operator.block_dim = 20
            sub_op.block_dim = 20
            super_operator.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIC_1_0
            sub_op.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIC_1_0
            assert judge_need_feed_sync_all(super_operator, sub_op) == False

            sub_op.block_dim = 10
            super_operator.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIC_1_0
            assert judge_need_feed_sync_all(super_operator, sub_op) == True

            super_operator.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIC_1_2
            sub_op.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIC_1_2
            assert judge_need_feed_sync_all(super_operator, sub_op) == True

            sub_op.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIV_1_0
            assert judge_need_feed_sync_all(super_operator, sub_op) == True

            super_operator.block_dim = 10
            sub_op.block_dim = 20
            assert judge_need_feed_sync_all(super_operator, sub_op) == False

    @staticmethod
    def test_gen_feed_syncall_var_init_code():
        kernel_info = {
            "op_list": [],
        }
        info_dict = {
            "bin_path": "",
            "json_path": "",
            "kernel_name": "add"
        }
        with mock.patch("json.load", return_value=sub_op_add_json):
            super_operator = SuperOperatorInfos(kernel_info, "test_gen_feed_syncall_var_init_code")
            sub_op = SubOperatorInfos(0, info_dict, 0, {})
            super_operator.info_base = [sub_op]

            super_operator.feed_sync_all_mode = SuperKernelFeedSyncAllMode.FeedSyncAllDisable
            code_gen, sync_flag = gen_feed_syncall_var_init_code(super_operator, sub_op)
            assert code_gen == ""
            assert sync_flag == False

            super_operator.feed_sync_all_mode = SuperKernelFeedSyncAllMode.FeedSyncAllEnable
            sub_op.with_sync_all = False
            code_gen, sync_flag = gen_feed_syncall_var_init_code(super_operator, sub_op)
            assert "AscendC::g_superKernelAutoSyncAllEnable = false;\n" == code_gen
            assert sync_flag == False

            sub_op.with_sync_all = True
            super_operator.block_dim = 36
            sub_op.block_dim = 18
            super_operator.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIC_1_0
            code_gen, sync_flag = gen_feed_syncall_var_init_code(super_operator, sub_op)
            goden_code = """\
\nAscendC::g_superKernelAutoSyncAllSyncIdx = 0;
AscendC::g_superKernelAutoSyncAllEnable = true;
if ASCEND_IS_AIC {
    AscendC::g_superKernelAutoSyncAllConfigGmAddr = AscendC::g_superKernelAutoSyncAllConfigGmBaseAddr + 0 * 64;
}
if ASCEND_IS_AIV {
    AscendC::g_superKernelAutoSyncAllConfigGmAddr = AscendC::g_superKernelAutoSyncAllConfigGmBaseAddr + 1 * 64 + 0 * 64;
}
"""
            assert goden_code == code_gen
            assert sync_flag == True

    @staticmethod
    def test_gen_clear_syncall_worskspace():
        kernel_info = {
            "op_list": [],
        }
        with mock.patch("json.load", return_value=sub_op_add_json):
            with mock.patch.object(CommonUtility, 'is_c310', return_value=True):
                super_operator = SuperOperatorInfos(kernel_info, "test_gen_clear_syncall_worskspace")

                super_operator.feed_sync_all_mode = SuperKernelFeedSyncAllMode.FeedSyncAllDisable
                code_gen = gen_clear_syncall_worskspace(super_operator)
                assert code_gen == ""

                super_operator.feed_sync_all_mode = SuperKernelFeedSyncAllMode.FeedSyncAllEnable
                super_operator.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIC_1_0
                super_operator.workspace_size = 1024
                code_gen = gen_clear_syncall_worskspace(super_operator)
                goden_code = """\
\nif ASCEND_IS_AIC {
    uint32_t sizePerCore = 1024 / get_block_num();
    const uint32_t repeatTimes = sizePerCore / 512;
    __gm__ uint8_t* startAddr  = (__gm__ uint8_t*)(workspace + sizePerCore * AscendC::GetBlockIdxImpl());
    create_cbuf_matrix((__cbuf__ uint32_t*)(0), 0x10010, 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
    for (size_t i = 0; i < repeatTimes; i++) {
        copy_cbuf_to_gm((__gm__ void*)(startAddr), (__cbuf__ void*)(0), 0, 1, 16, 1, 1);
        startAddr += 512;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    ffts_cross_core_sync(PIPE_FIX, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIC_FLAG));
    wait_flag_dev(AscendC::SYNC_AIC_FLAG);
}
"""
                assert goden_code == code_gen

                super_operator.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIV_1_0
                code_gen = gen_clear_syncall_worskspace(super_operator)
                goden_code = """\
\nif ASCEND_IS_AIV {
    uint32_t sizePerCore = 1024 / get_block_num();
    const uint32_t repeatTimes = sizePerCore / 512;
    __gm__ uint8_t* startAddr  = (__gm__ uint8_t*)(workspace + sizePerCore * AscendC::GetBlockIdxImpl());
    AscendC::DuplicateImpl((__ubuf__ uint32_t*)(0), (uint32_t)0, 128);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    for (size_t i = 0; i < repeatTimes; i++) {
        copy_ubuf_to_gm((__gm__ void*)(startAddr), (__ubuf__ void*)(0), 0, 1, 16, 1, 1);
        startAddr += 512;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIV_ONLY_ALL));
    wait_flag_dev(AscendC::SYNC_AIV_ONLY_ALL);
}
"""
                assert goden_code == code_gen

                super_operator.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIC_1_2
                code_gen = gen_clear_syncall_worskspace(super_operator)
                goden_code = """\
\nif ASCEND_IS_AIV {
    uint32_t sizePerCore = 512 / get_block_num();
    const uint32_t repeatTimes = sizePerCore / 512;
    __gm__ uint8_t* startAddr  = (__gm__ uint8_t*)(workspace + sizePerCore * AscendC::GetBlockIdxImpl());
    AscendC::DuplicateImpl((__ubuf__ uint32_t*)(0), (uint32_t)0, 128);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    for (size_t i = 0; i < repeatTimes; i++) {
        copy_ubuf_to_gm_align_v2((__gm__ void*)(startAddr), (__ubuf__ void*)(0), 0, 1, 512, 0, 512, 512);
        startAddr += 512;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x0, AscendC::SYNC_AIV_ONLY_ALL));
    wait_flag_dev(PIPE_S, AscendC::SYNC_AIV_ONLY_ALL);
    ffts_cross_core_sync(PIPE_MTE3, AscendC::GetffstMsg(0x02, AscendC::SYNC_AIV_FLAG));
}

if ASCEND_IS_AIC {
    wait_flag_dev(PIPE_S, AscendC::SYNC_AIV_FLAG);
}
"""
                assert goden_code == code_gen

    @staticmethod
    def test_gen_sync_and_event_code():
        kernel_info = {
            "op_list": [],
        }
        info_dict = {
            "bin_path": "",
            "json_path": "",
            "kernel_name": "add"
        }
        with mock.patch("json.load", return_value=sub_op_add_json):
            super_operator = SuperOperatorInfos(kernel_info, "test_gen_sync_and_event_code")
            pre_sub_operator = SubOperatorInfos(0, info_dict, 0, {})
            sub_operator = SubOperatorInfos(0, info_dict, 0, {})

            sub_operator.recv_event_list = [100]
            pre_sub_operator.send_event_list = [100]
            pre_sub_operator.notify_block = "pre_sub_operator.notify_block"
            sub_operator.wait_block = "sub_operator.wait_block"
            super_operator.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIC_1_1
            pre_sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIC_1_1
            sub_operator.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIC_1_1

            code_gen = gen_sync_and_event_code(super_operator, pre_sub_operator, sub_operator)
            goden_code = """\
    // begin inter ops barrier
    g_super_kernel_early_start_config = 10;
    pre_sub_operator.notify_block    sub_operator.wait_block// reason3: for continues notify/wait event
    AscendC::SyncAll<false>();\n
"""
            assert goden_code == code_gen

            sub_operator.recv_event_list = [100]
            pre_sub_operator.send_event_list = []
            code_gen = gen_sync_and_event_code(super_operator, pre_sub_operator, sub_operator)
            goden_code = """\
    sub_operator.wait_block    // begin inter ops barrier
    g_super_kernel_early_start_config = 10;
"""
            assert goden_code == code_gen

            sub_operator.recv_event_list = []
            pre_sub_operator.send_event_list = [100]
            code_gen = gen_sync_and_event_code(super_operator, pre_sub_operator, sub_operator)
            goden_code = """\
    // begin inter ops barrier
    g_super_kernel_early_start_config = 10;
    pre_sub_operator.notify_block\
"""
            assert goden_code == code_gen

    @staticmethod
    def test_gen_super_kernel_file():
        kernel_info = {
            "op_list": [],
        }
        info_dict = {
            "bin_path": "",
            "json_path": "",
            "kernel_name": "add"
        }
        with mock.patch("json.load", return_value=sub_op_add_json):
            with mock.patch.object(CommonUtility, 'is_c310', return_value=True):
                super_operator = SuperOperatorInfos(kernel_info, "test_gen_super_kernel_file")

                super_operator.enable_double_stream = True
                gen_super_kernel_file(super_operator)
                with open(super_operator.kernel_file, 'r') as f:
                    code_gen = f.read()
                    goden_code = """\
\n#if 1
#include "kernel_operator.h"

template<bool aic_flag>
__aicore__ inline void NotifyFunc(GM_ADDR notify_lock_addr)
{
    if constexpr (aic_flag) {
        if (get_block_idx() == 0) {
            __gm__ uint64_t* notifyLock = reinterpret_cast<__gm__ uint64_t*>(notify_lock_addr);
            *notifyLock = 1;
            dcci(notifyLock, 0, 2);
        }
    } else {
        if (AscendC::GetBlockIdx() == 0) {
            __gm__ uint64_t* notifyLock = reinterpret_cast<__gm__ uint64_t*>(notify_lock_addr);
            *notifyLock = 1;
            dcci(notifyLock, 0, 2);
        }
    }
}


template<bool aic_flag>
__aicore__ inline void WaitFunc(GM_ADDR wait_lock_addr)
{
    if constexpr (aic_flag) {
        __gm__ volatile uint64_t* waitLock = reinterpret_cast<__gm__ uint64_t*>(wait_lock_addr);
        if (get_block_idx() == 0) {
            dcci(waitLock, 0, 2);
            while(*waitLock != 1) {
                dcci(waitLock, 0, 2);
            }
        }
    } else {
        __gm__ volatile uint64_t* waitLock = reinterpret_cast<__gm__ uint64_t*>(wait_lock_addr);
        if (AscendC::GetBlockIdx() == 0) {
            dcci(waitLock, 0, 2);
            while(*waitLock != 1) {
                dcci(waitLock, 0, 2);
            }
        }
    }
}

extern "C"  __global__ __attribute__((aligned(512))) __aicore__ void auto_gen_test_gen_super_kernel_file_kernel(void) {
    GM_ADDR *param_base = (GM_ADDR *)get_para_base();
    GM_ADDR ffts_addr = param_base[0];
    if (ffts_addr != nullptr) {
        set_ffts_base_addr((uint64_t)ffts_addr);
    }

}\n
"""
                    assert goden_code == code_gen

                super_operator.enable_double_stream = False
                super_operator.info_base = [SubOperatorInfos(0, info_dict, 0, {}), \
                                            SubOperatorInfos(0, info_dict, 0, {}), \
                                            SubOperatorInfos(0, info_dict, 0, {})]
                for sub_op in super_operator.info_base:
                    sub_op.kernel_type = KernelMetaType.KERNEL_TYPE_MIX_AIC_1_1
                    sub_op.recv_event_list = [100]
                    sub_op.send_event_list = [100]
                    sub_op.wait_block = "sub_op.wait_block"
                    sub_op.notify_block = "sub_op.notify_block"
                sub_operator = super_operator.info_base[1]
                next_sub_operator = super_operator.info_base[2]
                sub_operator.dynamic_impl_func_block = "dynamic_impl_func_block"
                sub_operator.sub_op_task_type = SubOperatorType.DYNAMIC_OP
                super_operator.timestamp_option = True
                super_operator.feed_sync_all_mode = SuperKernelFeedSyncAllMode.FeedSyncAllEnable
                super_operator.profiling_mode = SuperKernelProfilingMode.ProfilingEnable
                super_operator.split_mode = 2
                super_operator.preload_mode = SuperKernelPreLoadMode.PreLoadByWhole
                sub_operator.preload_call_block = "sub_operator.preload_call_block"
                next_sub_operator.preload_call_block = "next_sub_operator.preload_call_block"
                sub_operator.data_cache_preload_call = "sub_operator.data_cache_preload_call"
                next_sub_operator.data_cache_preload_call = "next_sub_operator.data_cache_preload_call"
                super_operator.datacache_mode = SuperKernelDataCacheMode.DataCacheLoadAdancanceStep

                with mock.patch.object(CommonUtility, 'ascendc_raise_python_err') as mock_raise:
                    gen_super_kernel_file(super_operator)
                    mock_raise.assert_called()
                    with open(super_operator.kernel_file, 'r') as f:
                        code_gen = f.read()
                        assert "#if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON" in code_gen

                with mock.patch.object(CommonUtility, 'ascendc_raise_python_err') as mock_raise:
                    super_operator.timestamp_option = False
                    gen_super_kernel_file(super_operator)
                    mock_raise.assert_called()
                    with open(super_operator.kernel_file, 'r') as f:
                        code_gen = f.read()
                        assert "#if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON" not in code_gen

                with mock.patch.object(CommonUtility, 'ascendc_raise_python_err') as mock_raise:
                    super_operator.preload_mode = SuperKernelPreLoadMode.PreLoadStepByStep
                    gen_super_kernel_file(super_operator)
                    mock_raise.assert_called()
                    sub_operator.preload_call_block = "sub_operator.preload_call_block"
                    with open(super_operator.kernel_file, 'r') as f:
                        code_gen = f.read()
                        assert "sub_operator.preload_call_block" in code_gen

                with mock.patch.object(CommonUtility, 'ascendc_raise_python_err') as mock_raise:
                    super_operator.preload_mode = SuperKernelPreLoadMode.PreloadByAdanvanceStep
                    gen_super_kernel_file(super_operator)
                    mock_raise.assert_called()
                    with open(super_operator.kernel_file, 'r') as f:
                        code_gen = f.read()
                        assert "next_sub_operator.preload_call_block" in code_gen


if __name__ == "__main__":
    pytest.main()