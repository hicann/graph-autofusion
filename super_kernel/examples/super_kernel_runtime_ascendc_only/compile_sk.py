#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------------------

import os
import json
from pathlib import Path
from typing import List
import ctypes
import dataclasses

from superkernel import super_kernel
from utils import SkCompileContext

# TODO: AscendC需要重构部分，需要与tbe解耦
from tbe.common.context.op_context import OpContext
from tbe.common.context import op_info
from tbe.common.context import get_context
from tbe.common.buildcfg.buildcfg_mapping import kernel_meta_parent_dir, op_debug_config, tbe_debug_level
from tbe.common.buildcfg.buildcfg import build_config
from tbe.tvm.contrib.ccec import current_build_config
from tbe.common.platform.platform_info import set_current_compile_soc_info
from tbe.tikcpp.global_storage import global_var_storage


MAGIC_MAPPING = {
    "RT_DEV_BINARY_MAGIC_PLAIN": 0xabceed50,
    "RT_DEV_BINARY_MAGIC_PLAIN_AICPU": 0xabceed51,
    "RT_DEV_BINARY_MAGIC_PLAIN_AIVEC": 0xabceed52,
    "RT_DEV_BINARY_MAGIC_ELF": 0x43554245,
    "RT_DEV_BINARY_MAGIC_ELF_AICPU": 0x41415243,
    "RT_DEV_BINARY_MAGIC_ELF_AIVEC": 0x41415246,
    "RT_DEV_BINARY_MAGIC_ELF_AICUBE": 0x41494343
}


@dataclasses.dataclass
class BinData:
    bin_data: ctypes.c_void_p
    bin_size: int


class KernelResult:
    def __init__(self, path, name):
        self.root = path
        self.name = name
        self._json_data = self._load_json()
        self._bin = self._load_bin()



    @property
    def bin_data(self):
        return self._bin.bin_data

    @property
    def bin_size(self):
        return self._bin.bin_size

    @staticmethod
    def _get_magic_number(magic_str: str):
        return MAGIC_MAPPING.get(magic_str)

    def json_path(self):
        return self.root / "kernel_meta" / (self.name + ".json")

    def bin_path(self):
        return self.root / "kernel_meta" / (self.name + ".o")

    def block_dim(self):
        return self._json_data["blockDim"]

    def magic_number(self):
        return self._get_magic_number(self._json_data["magic"])

    def op_para_size(self):
        return self._json_data["opParaSize"]

    def kernel_name(self):
        return self._json_data["kernelName"]

    def bin_file_name(self):
        return self._json_data["binFileName"]

    def _load_json(self):
        try:
            # 检查文件是否存在
            if not os.path.exists(self.json_path()):
                raise FileNotFoundError(f"文件不存在: {self.json_path()}")
            # 读取并解析JSON文件
            with open(self.json_path(), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            return json_data
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"JSON解析错误: {e}")
        except Exception as e:
            raise Exception(f"读取文件时发生错误: {e}")

    def _load_bin(self):
        file_path = self.root / "kernel_meta" / (self.name + ".o")
        with open(file_path, 'rb') as f:
            data = f.read()

        # 获取文件大小
        file_size = len(data)

        # 创建ctypes缓冲区
        buffer = (ctypes.c_ubyte * file_size)()

        # 将数据复制到缓冲区
        ctypes.memmove(buffer, data, file_size)
        c_pointer = ctypes.cast(buffer, ctypes.c_void_p)

        return BinData(c_pointer, file_size)


class SubkernelResult(KernelResult):
    def __init__(self, path, name):
        super().__init__(path, name)
        self._input_addr: List[ctypes.c_void_p] = []
        self._output_addr: List[ctypes.c_void_p] = []
        self._workspace_addr: List[ctypes.c_void_p] = []
        self._input: List[str] = []
        self._output: List[str] = []

    @property
    def input(self) -> List[str]:
        return self._input

    @property
    def output(self) -> List[str]:
        return self._output

    @property
    def workspaces_addr(self) -> List[ctypes.c_void_p]:
        """I'm the 'workspaces_addr' property."""
        return self._workspace_addr

    @workspaces_addr.setter
    def workspaces_addr(self, value: List[ctypes.c_void_p]):
        self._workspace_addr = value

    @property
    def input_addr(self) -> List[ctypes.c_void_p]:
        """I'm the 'input_addr' property."""
        return self._input_addr

    @input_addr.setter
    def input_addr(self, value: List[ctypes.c_void_p]):
        self._input_addr = value

    @property
    def output_addr(self) -> List[ctypes.c_void_p]:
        """I'm the 'output_addr' property."""
        return self._output_addr

    @output_addr.setter
    def output_addr(self, value: List[ctypes.c_void_p]):
        self._output_addr = value

    def workspaces_size(self) -> List[int]:
            return self._json_data["workspace"]["size"]

    # 通过set_input和set_output设置输入输出名称，目的是为了在Superkernel时获取复用关系，
    # 比如A->B，A的输出就是B的输入， 此时需要将A的输出名称设置为B的输入名称
    # 对于workspace的复用而言，输入输出名称是没有意义的，不同算子的workspace可以根据相同stream时通过小的复用大的workspace来实现
    def set_input(self, input_name: List[str]):
        self._input = input_name

    def set_output(self, output_name: List[str]):
        self._output = output_name


class SuperkernelResult(KernelResult):
    def __init__(self, path, name):
        super().__init__(path, name)
        self._output = List[str]

    @property
    def output(self) -> List[str]:
        return self._output

    @output.setter
    def output(self, value: List[str]):
        self._output = value


def _compile_sub_kernel(kernel_meta_dir, op_name, op_type, func, extend_op_info: dict = None):
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


def compile_subkernel(ctx: SkCompileContext):
    def make_subkernel(
            impl_module_name,  # 实现模块名
            func_name,  # 函数名
            op_name,  # 算子名
            op_type,  # 算子类型
            input_count=1,  # 输入参数数量
            output_count=1,  # 输出参数数量
            extend_op_info=None  # 扩展配置
    ):
        with ctx.tmp_dir as tmp_dir:
            # 1. 定义内核元数据目录
            kernel_meta_dir = Path(tmp_dir) / f"subkernel_{op_name}"

            # 2. 动态导入实现模块和函数
            module = __import__(f"impl.dynamic.{impl_module_name}", fromlist=[func_name])
            func = getattr(module, func_name)

            # 3. 动态创建输入输出参数
            tensor_template = {
                "shape": [256],
                "ori_shape": [256],
                "format": "ND",
                "ori_format": "ND",
                "dtype": "float32"
            }

            # 创建输入张量列表
            inputs = [tensor_template.copy() for _ in range(input_count)]
            # 创建输出张量列表
            outputs = [tensor_template.copy() for _ in range(output_count)]

            # 4. 编译子内核
            with build_config():
                _compile_sub_kernel(
                    str(kernel_meta_dir),
                    op_name,
                    op_type,
                    extend_op_info=extend_op_info,
                    func=lambda: func(*inputs, *outputs)  # 动态调用函数
                )

            # 5. 返回路径管理对象
            return SubkernelResult(kernel_meta_dir, impl_module_name)

    # 使用统一的make_subkernel函数创建不同配置的算子
    is_inf_op = make_subkernel(
        impl_module_name="is_inf",
        func_name="is_inf",
        op_name="IsInf_Default_1",
        op_type="IsInf",
        input_count=1,
        output_count=1
    )

    pows_op = make_subkernel(
        impl_module_name="pows",
        func_name="pows",
        op_name="Pows_Default_2",
        op_type="Pows",
        input_count=2,
        output_count=1
    )

    return [is_inf_op, pows_op]


def compile_superkernel(ctx: SkCompileContext, sub_kernels: list[KernelResult]):
    with ctx.tmp_dir as tmp_dir:
        kernel_meta_dir = tmp_dir / "superkernel_1"

        global_var_storage.set_variable("ascendc_compile_debug_config", True)
        soc_version = "Ascend910_9391"
        set_current_compile_soc_info(soc_version)

        current_build_config()[kernel_meta_parent_dir] = str(kernel_meta_dir)
        current_build_config()[op_debug_config] = ["dump_cce"]
        current_build_config()[tbe_debug_level] = 0

        compile_options = "compile-options=-g:"
        kernel_info = {
            "super_kernel_options": compile_options,
            "op_list": [
                {
                    "stream_id": 1,
                    "bin_path": str(sub_kernels[0].bin_path()),
                    "json_path": str(sub_kernels[0].json_path()),
                },
                {
                    "stream_id": 1,
                    "bin_path": str(sub_kernels[1].bin_path()),
                    "json_path": str(sub_kernels[1].json_path()),
                },
            ],
        }
        kernel_name = "te_superkernel_1"

        super_kernel.compile(kernel_info, kernel_name)

        return SuperkernelResult(kernel_meta_dir, kernel_name)
