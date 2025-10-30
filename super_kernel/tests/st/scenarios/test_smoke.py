#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------------------

"""Minimal smoke test ensuring the ST harness executes."""

import pytest
from superkernel import super_kernel
from utils import validate_codegen_output, validate_compile_options


@pytest.mark.parametrize(
    # 测试参数：
    #   1. 子内核1 fixture
    #   2. 子内核2 fixture
    #   3. 编译前配置的编译选项
    #   4. golden_codegen_path: 期望编译生成的代码结果路径，路径内部按照${dir}/expect_sk_code.cc ${dir}/expect_compiled_json.json组织
    #   5. golden_options: 期望编译结果的编译选项
    "subkernel_inf, subkernel_pows, compile_options, golden_codegen_path, golden_options",
    [
        # 默认配置
        ("subkernel_is_inf_default", "subkernel_pows_default", "compile-options=-g:", "test_sk_1_stream_2_ops_default",
         [
            "-D__ASCENDC_ENABLE_SET_NEXT_TASK_START",
            "-D__ASCENDC_ENABLE_WAIT_PRE_TASK_END",
            "-D__ASCENDC_SUPERKERNEL_EARLY_START_V2",
        ]),
        # 新配置1：early-start=0
        ("subkernel_is_inf_default", "subkernel_pows_default", "compile-options=-g:early-start=0:", "test_sk_1_stream_2_ops_early_start_0",
         []),
        # 新配置2：func-align=256, 默认是512
        ("subkernel_is_inf_default", "subkernel_pows_default", "compile-options=-g:func-align=256:", "test_sk_1_stream_2_ops_func_align_256",
         [
             "-D__ASCENDC_ENABLE_SET_NEXT_TASK_START",
             "-D__ASCENDC_ENABLE_WAIT_PRE_TASK_END",
             "-D__ASCENDC_SUPERKERNEL_EARLY_START_V2",
         ]),
        # 新配置2：preload-code=max, 支持[max, pre-func, none], 默认pre-func
        ("subkernel_is_inf_default", "subkernel_pows_default", "compile-options=-g:preload-code=max:", "test_sk_1_stream_2_ops_preload_code_max",
         [
             "-D__ASCENDC_ENABLE_SET_NEXT_TASK_START",
             "-D__ASCENDC_ENABLE_WAIT_PRE_TASK_END",
             "-D__ASCENDC_SUPERKERNEL_EARLY_START_V2",
         ]),
        # 新配置3：debug-dcci-all=1, 支持0跟1，默认0
        ("subkernel_is_inf_default", "subkernel_pows_default", "compile-options=-g:debug-dcci-all=1:", "test_sk_1_stream_2_ops_debug_dcci_all_1",
         [
             "-D__ASCENDC_ENABLE_SET_NEXT_TASK_START",
             "-D__ASCENDC_ENABLE_WAIT_PRE_TASK_END",
             "-D__ASCENDC_SUPERKERNEL_EARLY_START_V2",
         ]),
        # 新配置4：debug-sync-all=1, 支持0跟1，默认0
        ("subkernel_is_inf_default", "subkernel_pows_default", "compile-options=-g:debug-sync-all=1:", "test_sk_1_stream_2_ops_debug_sync_all_1",
         [
             "-D__ASCENDC_ENABLE_SET_NEXT_TASK_START",
             "-D__ASCENDC_ENABLE_WAIT_PRE_TASK_END",
             "-D__ASCENDC_SUPERKERNEL_EARLY_START_V2",
         ]),
        # 新配置5：feed-sync-all=1, 支持0跟1，默认0
        ("subkernel_is_inf_default", "subkernel_pows_default", "compile-options=-g:feed-sync-all=1:", "test_sk_1_stream_2_ops_feed_sync_all_1",
         [
             "-D__ASCENDC_ENABLE_SET_NEXT_TASK_START",
             "-D__ASCENDC_ENABLE_WAIT_PRE_TASK_END",
             "-D__ASCENDC_SUPERKERNEL_EARLY_START_V2",
         ]),
        # 新配置6：stream-fusion=1, 支持0跟1，默认0
        ("subkernel_is_inf_default", "subkernel_pows_default", "compile-options=-g:stream-fusion=1:", "test_sk_1_stream_2_ops_stream_fusion_1",
         [
             "-D__ASCENDC_ENABLE_SET_NEXT_TASK_START",
             "-D__ASCENDC_ENABLE_WAIT_PRE_TASK_END",
             "-D__ASCENDC_SUPERKERNEL_EARLY_START_V2",
         ]),
        # # 新配置7：split-mode=1
        # ("subkernel_is_inf_split_mode1", "subkernel_is_finite_split_mode1", "compile-options=-g:split-mode=1:", "test_sk_1_stream_2_ops_split_mode_1",
        #  [
        #     "-D__ASCENDC_ENABLE_SET_NEXT_TASK_START",
        #     "-D__ASCENDC_ENABLE_WAIT_PRE_TASK_END",
        #     "-D__ASCENDC_SUPERKERNEL_EARLY_START_V2",
        # ]),
    ],
    # 通过 indirect 实现 fixture 动态传入
    indirect=["subkernel_inf", "subkernel_pows"],
)
def test_sk_1_stream_2_ops(
    soc_version,
    tmp_dir,
    data_dir,
    subkernel_inf,
    subkernel_pows,
    compile_options,
    golden_codegen_path,
    golden_options,
):
    kernel_meta_dir = tmp_dir / golden_codegen_path

    from tbe.tikcpp.global_storage import global_var_storage
    global_var_storage.set_variable("ascendc_compile_debug_config", True)

    from tbe.common.platform.platform_info import set_current_compile_soc_info
    set_current_compile_soc_info(soc_version)

    from tbe.tvm.contrib.ccec import current_build_config
    from tbe.common.buildcfg.buildcfg_mapping import kernel_meta_parent_dir, op_debug_config, tbe_debug_level

    current_build_config()[kernel_meta_parent_dir] = str(kernel_meta_dir)
    current_build_config()[op_debug_config] = ["dump_cce"]
    current_build_config()[tbe_debug_level] = 0

    kernel_info = {
        "super_kernel_options": compile_options,
        "op_list": [
            {
                "stream_id": 1,
                "bin_path": str(subkernel_inf.o()),
                "json_path": str(subkernel_inf.json()),
            },
            {
                "stream_id": 1,
                "bin_path": str(subkernel_pows.o()),
                "json_path": str(subkernel_pows.json()),
            },
        ],
    }
    kernel_name = "te_superkernel_1"

    super_kernel.compile(kernel_info, kernel_name)

    scenario_dir = data_dir / golden_codegen_path
    validate_codegen_output(
        kernel_meta_dir,
        kernel_name,
        scenario_dir / "expect_sk_code.cc",
    )
    validate_compile_options(
        kernel_meta_dir,
        kernel_name,
        golden_options,
    )


@pytest.mark.parametrize(
    # 测试参数：
    #   1. 子内核1 fixture
    #   2. 子内核2 fixture
    #   3. 子内核3 fixture
    #   4. 编译前配置的编译选项
    #   5. golden_codegen_path: 期望编译生成的代码结果路径，路径内部按照${dir}/expect_sk_code.cc ${dir}/expect_compiled_json.json组织
    #   6. golden_options: 期望编译结果的编译选项
    "subkernel_inf, subkernel_pows, compile_options, golden_codegen_path, golden_options",
    [
        # 默认配置
        ("subkernel_is_inf_default", "subkernel_pows_default", "compile-options=-g:", "test_sk_1_stream_3_ops_default",
         [
             "-D__ASCENDC_ENABLE_SET_NEXT_TASK_START",
             "-D__ASCENDC_ENABLE_WAIT_PRE_TASK_END",
             "-D__ASCENDC_SUPERKERNEL_EARLY_START_V2",
         ]),
    ],
    # 通过 indirect 实现 fixture 动态传入
    indirect=["subkernel_inf", "subkernel_pows"],
)

def test_sk_1_stream_3_ops(
        soc_version,
        tmp_dir,
        data_dir,
        subkernel_inf,
        subkernel_pows,
        compile_options,
        golden_codegen_path,
        golden_options,
):
    kernel_meta_dir = tmp_dir / golden_codegen_path

    from tbe.tikcpp.global_storage import global_var_storage
    global_var_storage.set_variable("ascendc_compile_debug_config", True)

    from tbe.common.platform.platform_info import set_current_compile_soc_info
    set_current_compile_soc_info(soc_version)

    from tbe.tvm.contrib.ccec import current_build_config
    from tbe.common.buildcfg.buildcfg_mapping import kernel_meta_parent_dir, op_debug_config, tbe_debug_level

    current_build_config()[kernel_meta_parent_dir] = str(kernel_meta_dir)
    current_build_config()[op_debug_config] = ["dump_cce"]
    current_build_config()[tbe_debug_level] = 0

    kernel_info = {
        "super_kernel_options": "compile-options=-g:",
        "op_list": [
            {
                "stream_id": 1,
                "bin_path": str(subkernel_inf.o()),
                "json_path": str(subkernel_inf.json()),
            },
            {
                "stream_id": 1,
                "bin_path": str(subkernel_pows.o()),
                "json_path": str(subkernel_pows.json()),
            },
            {
                "stream_id": 1,
                "bin_path": str(subkernel_inf.o()),
                "json_path": str(subkernel_inf.json()),
            },
        ],
    }
    kernel_name = "te_superkernel_2"

    super_kernel.compile(kernel_info, kernel_name)

    scenario_dir = data_dir / golden_codegen_path
    validate_codegen_output(
        kernel_meta_dir,
        kernel_name,
        scenario_dir / "expect_sk_code.cc",
        )
    validate_compile_options(
        kernel_meta_dir,
        kernel_name,
        golden_options,
    )
