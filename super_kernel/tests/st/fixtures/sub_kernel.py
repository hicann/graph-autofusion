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


from pathlib import Path
import pytest

from tbe.common.context.op_context import OpContext
from tbe.common.context import op_info
from tbe.common.context import get_context
from tbe.common.buildcfg.buildcfg_mapping import kernel_meta_parent_dir, op_debug_config, tbe_debug_level
from tbe.common.buildcfg.buildcfg import build_config
from tbe.tvm.contrib.ccec import current_build_config
from tbe.common.platform.platform_info import set_current_compile_soc_info


def compile_sub_kernel(kernel_meta_dir, op_name, op_type, func, extend_op_info: dict = None):
    current_build_config()[kernel_meta_parent_dir] = kernel_meta_dir
    current_build_config()[tbe_debug_level] = 0
    set_current_compile_soc_info("Ascend910_9391")

    # compile_op 函数一开始就会对 global_var_storage 做 reset，因此直接如下配置是无法生效的：
    # global_var_storage.set_variable("ascendc_compile_debug_config", True)
    # 这样配置才能生效
    current_build_config()[op_debug_config] = ["dump_cce", ]

    # 必须配置 enable_deterministic_mode，否则在调用 C++ tiling 函数时，
    # 会将 extra_params_c 中的 deterministic 设置为 null，导致 C++ 侧 core dump
    current_build_config()['enable_deterministic_mode'] = 0

    current_build_config()[kernel_meta_parent_dir] = kernel_meta_dir

    current_build_config()['enable_super_kernel'] = 1
    sp_info = {}
    sp_info['super_kernel_sub_loc'] = 'middle'
    sp_info['super_kernel_options'] = 'early-start=0'
    sp_info['super_kernel_count'] = 0
    sp_info['super_kernel_sub_id'] = 0
    if extend_op_info:
        sp_info.update(extend_op_info)

    with OpContext('static'):
        opinfo = op_info.OpInfo(op_name, op_type)
        get_context().set_graph_op_info(opinfo)
        get_context().add_addition('super_kernel_sub_info', sp_info)

        func()

class SubkernelPath:
    def __init__(self, path, name):
        self.root = path
        self.name = name
    
    def o(self):
        return self.root / "kernel_meta" / (self.name + ".o")

    def json(self):
        return self.root / "kernel_meta" / (self.name + ".json")


@pytest.fixture(scope="session")
def subkernel_is_inf(tmp_dir):
    kernel_meta_dir = Path(tmp_dir) / "subkernel_is_inf"

    from impl.dynamic import is_inf
    x = {}
    x["shape"] = [1024]
    x["ori_shape"] = [1024]
    x["format"] = "ND"
    x["ori_format"] = "ND"
    x["dtype"] = "float16"

    y = {}
    y["shape"] = [1024]
    y["ori_shape"] = [1024]
    y["format"] = "ND"
    y["ori_format"] = "ND"
    y["dtype"] = "float16"

def make_1_in_1_out_subkernel_fixture(
        impl_module_name,  # 实现模块名（如 "is_inf"）
        func_name,         # 函数名（如 "is_inf"）
        op_name,           # 算子名（如 "IsInf_SplitMode1"）最好唯一，根据该name生成不同的编译子kernel路径名字，如果存在相同name，会覆盖之前的
        op_type,           # 算子类型（如 "IsInf"）
        extend_op_info=None  # 新的扩展配置
):
    """生成子内核 fixture 的工厂函数"""
    @pytest.fixture(scope="session")
    def fixture_func(tmp_dir):
        # 1. 定义内核元数据目录
        kernel_meta_dir = Path(tmp_dir) / f"subkernel_{op_name}"

        # 2. 动态导入实现模块和函数
        module = __import__(f"impl.dynamic.{impl_module_name}", fromlist=[func_name])
        func = getattr(module, func_name)

        # 3. 定义输入参数（可根据需要扩展）
        x = {
            "shape": [1024],
            "ori_shape": [1024],
            "format": "ND",
            "ori_format": "ND",
            "dtype": "float16"
        }
        y = {
            "shape": [1024],
            "ori_shape": [1024],
            "format": "ND",
            "ori_format": "ND",
            "dtype": "float16"
        }

        # 4. 编译子内核（传入新的 extend_op_info）
        with build_config():
            compile_sub_kernel(
                str(kernel_meta_dir),
                op_name,
                op_type,
                extend_op_info=extend_op_info,
                func=lambda: func(x, y)  # 调用实际的算子函数
            )

        # 5. 返回路径管理对象
        return SubkernelPath(kernel_meta_dir, impl_module_name)

    return fixture_func

NEW_EXTEND_OP_INFO = {
    "super_kernel_options": "split-mode=1:early-start=1",  # 新的分割模式和启动参数
}
# --------------------------
# 2. 原有 fixture（保持兼容，使用默认配置）
# --------------------------
subkernel_is_inf_default = make_1_in_1_out_subkernel_fixture(
    impl_module_name="is_inf",
    func_name="is_inf",
    op_name="IsInf_Default",
    op_type="IsInf",
    extend_op_info=None  # 原有默认配置
)
# --------------------------
subkernel_is_finite_default = make_1_in_1_out_subkernel_fixture(
    impl_module_name="is_finite",
    func_name="is_finite",
    op_name="IsFinite_Default",
    op_type="IsFinite",
    extend_op_info=None  # 原有默认配置
)

# 基于新配置生成子内核 fixture
subkernel_is_inf_split_mode1 = make_1_in_1_out_subkernel_fixture(
    impl_module_name="is_inf",
    func_name="is_inf",
    op_name="IsInf_SplitMode1",
    op_type="IsInf",
    extend_op_info=NEW_EXTEND_OP_INFO
)

subkernel_is_finite_split_mode1 = make_1_in_1_out_subkernel_fixture(
    impl_module_name="is_finite",
    func_name="is_finite",
    op_name="IsFinite_SplitMode1",
    op_type="IsFinite",
    extend_op_info=NEW_EXTEND_OP_INFO
)

@pytest.fixture
def subkernel_inf(request):
    # request.param 接收来自 parametrize 的参数，即你写的 "subkernel_is_inf" 字符串
    fixture_name = request.param
    # 使用 request.getfixturevalue 根据字符串名字动态获取对应的夹具值
    return request.getfixturevalue(fixture_name)

@pytest.fixture
def subkernel_finite(request):
    fixture_name = request.param
    return request.getfixturevalue(fixture_name)

@pytest.fixture(scope="session")
def subkernel_pows_default(tmp_dir):
    kernel_meta_dir = Path(tmp_dir) / "subkernel_pows"

    from impl.dynamic import pows
    x = {}
    x["shape"] = [1024]
    x["ori_shape"] = [1024]
    x["format"] = "ND"
    x["ori_format"] = "ND"
    x["dtype"] = "float16"

    x1 = {}
    x1["shape"] = [1024]
    x1["ori_shape"] = [1024]
    x1["format"] = "ND"
    x1["ori_format"] = "ND"
    x1["dtype"] = "float16"

    y = {}
    y["shape"] = [1024]
    y["ori_shape"] = [1024]
    y["format"] = "ND"
    y["ori_format"] = "ND"
    y["dtype"] = "float16"

    with build_config():
        compile_sub_kernel(str(kernel_meta_dir), "Pows", "Pows", lambda:pows.pows(x, x1, y))

    return SubkernelPath(kernel_meta_dir, "pows")


@pytest.fixture
def subkernel_pows(request):
    fixture_name = request.param
    return request.getfixturevalue(fixture_name)