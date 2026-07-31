#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import ctypes
import os
import re
import sys
import shutil
import argparse
import subprocess
import platform
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from functools import wraps
from typing import List
from asc_op_compile_base.common.platform.platform_info import get_soc_spec

PYF_PATH = os.path.dirname(os.path.realpath(__file__))
ASCEND_PATH = os.path.join(PYF_PATH, "..", "..", "..")
machine = platform.machine()
HOST_LINK_LIBRARIES = ["tiling_api", "platform", "graph_base", "register"]
CV_HOST_LINK_LIBRARIES = HOST_LINK_LIBRARIES + ["nnopbase"]
INDUCTOR_COMPILE_TRACE_LABEL = "InductorCompile"
HOST_COMPILE_MAX_WORKERS = 32
if not os.path.exists(ASCEND_PATH):
    ASCEND_PATH = os.getenv("ASCEND_HOME_PATH", ASCEND_PATH)


class CompileError(Exception):
    """Compile failed exception."""


def parse_env_flags(env_name):
    result = {}
    flags = os.getenv(env_name)
    if not flags:
        return result
    params = flags.split(";")
    for param in params:
        if "=" in param:
            key_part, value_part = param.split("=", 1)
            result[key_part.lstrip("-")] = value_part
    return result


def record_inductor_compile_duration(stage, step, graph_name, start, duration):
    from autofuse.pyautofuse import ascir

    labels = [INDUCTOR_COMPILE_TRACE_LABEL, stage, step, graph_name]
    ascir.utils.duration_record(labels, int(start), int(duration))


class InductorCompileDuration:
    def __init__(self, args, step):
        self.stage = getattr(args, "trace_stage", getattr(args, "stage", "unknown"))
        self.step = step
        self.graph_name = getattr(args, "graph_name", "unknown")
        self.start = None

    def __enter__(self):
        self.start = time.time_ns()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        end = time.time_ns()
        record_inductor_compile_duration(
            self.stage, self.step, self.graph_name, self.start, end - self.start
        )
        return False


def inductor_compile_duration(step, args_index=0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            compile_args = (
                args[args_index] if len(args) > args_index else kwargs.get("args")
            )
            with InductorCompileDuration(compile_args, step):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def get_soc_type(args):
    """根据 soc_version 返回对应的类型"""
    if args.soc_version.startswith("Ascend910B"):
        return "dav-2201"
    elif args.soc_version.startswith("Ascend910_93"):
        return "dav-2201"
    elif args.soc_version.startswith("Ascend950"):
        return "dav-3510"
    else:
        raise ValueError(f"Unsupported soc_version: {args.soc_version}")


def run_compile_command(cmd: List[str], stage_name):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        error_msg = f"{stage_name} compile failed with code {result.returncode}"
        if result.stderr:
            error_msg += f"\nstderr: {result.stderr}"
        raise CompileError(error_msg)
    if result.stdout:
        print(f"[{stage_name}] {result.stdout}")


def link_shared(target_file, obj_files, link_libraries=None):
    link_command = [f"{ASCEND_PATH}/tools/bisheng_compiler/bin/bisheng"]
    link_command.extend(obj_files)
    link_command.extend(["-fPIC", "--shared", "-o", target_file])
    if link_libraries:
        link_command.extend(["-L", f"{ASCEND_PATH}/lib64"])
        link_command.extend(["-L", f"{ASCEND_PATH}/{machine}-linux/lib64"])
        link_command.extend([f"-l{link_library}" for link_library in link_libraries])
    run_compile_command(link_command, "LinkObj")
    return target_file


def iter_compile_source_files(args: argparse.Namespace):
    for source_files in (
        getattr(args, "host_files", None),
        getattr(args, "device_files", None),
    ):
        for source_file in normalize_to_list(source_files):
            yield source_file


def contains_cv_tiling_data(source_file):
    if not source_file or not os.path.exists(source_file):
        return False
    with open(source_file, "r") as f:
        return "CVAutofuseTilingData" in f.read()


def is_cv_fusion_compile(args: argparse.Namespace):
    return any(
        contains_cv_tiling_data(source_file)
        for source_file in iter_compile_source_files(args)
    )


def get_host_abi_compile_options(args: argparse.Namespace):
    if is_cv_fusion_compile(args):
        return [
            option
            for option in args.compile_options.split()
            if "_GLIBCXX_USE_CXX11_ABI" not in option
        ], ["-D", "_GLIBCXX_USE_CXX11_ABI=0"]
    return args.compile_options.split(), (
        []
        if "-D_GLIBCXX_USE_CXX11_ABI=" in args.compile_options
        else ["-D", "_GLIBCXX_USE_CXX11_ABI=0"]
    )


def build_host_include_options(temp_dir):
    return [
        "-I",
        f"{temp_dir}/host",
        "-I",
        f"{ASCEND_PATH}/include",
        "-I",
        f"{ASCEND_PATH}/pkg_inc/base",
        "-I",
        f"{ASCEND_PATH}/include/base",
        "-I",
        f"{ASCEND_PATH}/include/experiment",
        "-I",
        f"{ASCEND_PATH}/{machine}-linux/pkg_inc/base",
        "-I",
        f"{ASCEND_PATH}/{machine}-linux/include",
        "-I",
        f"{ASCEND_PATH}/{machine}-linux/include/aclnn",
        "-I",
        f"{ASCEND_PATH}/{machine}-linux/ascendc/include/highlevel_api/tiling/platform",
        "-I",
        f"{ASCEND_PATH}/{machine}-linux/ascendc/include/highlevel_api",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/mat_mul_v3",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/mat_mul_v3",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/conv2d_v2",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/conv2d_v2",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common",
    ]


def build_host_base_options(args: argparse.Namespace, temp_dir):
    soc_version = get_soc_type(args)
    return [
        f"{ASCEND_PATH}/tools/bisheng_compiler/bin/bisheng",
        "-D",
        "kernel_EXPORTS",
        *build_host_include_options(temp_dir),
        "-fPIC",
        f"--npu-arch={soc_version}",
        "-O2",
        "-fno-common",
        "-Wextra",
        "-Wfloat-equal",
        "-fvisibility=default",
    ]


def build_host_output_options(source_file, obj_file):
    return [
        "-o",
        obj_file,
        "-c",
        "-x",
        "asc",
        source_file,
    ]


def build_host_compile_cmd(args: argparse.Namespace, temp_dir, source_file, obj_file):
    compile_options, host_abi_option = get_host_abi_compile_options(args)
    return [
        *build_host_base_options(args, temp_dir),
        *compile_options,
        "-D",
        "LOG_CPP",
        *host_abi_option,
        *build_host_output_options(source_file, obj_file),
    ]


def get_host_obj_path(source_file, temp_dir):
    return os.path.join(temp_dir, "host", os.path.basename(source_file) + ".o")


def compile_host_obj_file(args: argparse.Namespace, temp_dir, source_file):
    obj_file = get_host_obj_path(source_file, temp_dir)
    host_compile_cmd = build_host_compile_cmd(args, temp_dir, source_file, obj_file)
    try:
        run_compile_command(host_compile_cmd, "Host")
    except CompileError as ex:
        raise CompileError(f"Host compile failed for {source_file}: {ex}") from ex
    return obj_file


def get_host_compile_worker_count(cpp_file_count):
    return min(HOST_COMPILE_MAX_WORKERS, cpp_file_count, os.cpu_count() or 1)


def normalize_to_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


@inductor_compile_duration("CompileHostObj")
def compile_host_objs(args: argparse.Namespace, temp_dir):
    host_files = normalize_to_list(args.host_files)
    if len(host_files) == 1:
        return [compile_host_obj_file(args, temp_dir, host_files[0])]

    obj_files = [None] * len(host_files)
    worker_count = get_host_compile_worker_count(len(host_files))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {
            executor.submit(compile_host_obj_file, args, temp_dir, source_file): index
            for index, source_file in enumerate(host_files)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            obj_files[index] = future.result()
    return obj_files


@inductor_compile_duration("CompileHostObj")
def compile_host_obj(args: argparse.Namespace, temp_dir):
    host_files = normalize_to_list(args.host_files)
    if len(host_files) != 1:
        raise CompileError("compile_host_obj expects exactly one host source")
    return compile_host_obj_file(args, temp_dir, host_files[0])


@inductor_compile_duration("CompileDeviceObj")
def compile_device_obj(args: argparse.Namespace, temp_dir):
    base_device_file = os.path.basename(args.device_files)
    soc_version = get_soc_type(args)
    device_compile_cmd = [
        f"{ASCEND_PATH}/tools/bisheng_compiler/bin/bisheng",
        "-I",
        f"{temp_dir}/device",
        "-I",
        f"{ASCEND_PATH}/{machine}-linux/asc/include",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/common",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/mat_mul_v3",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/batch_mat_mul_v3",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/mat_mul_v3",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/batch_mat_mul_v3",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/common",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/common",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/common/cmct",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common",
        "-I",
        f"{ASCEND_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common/cmct",
        "-fPIC",
        "-D",
        "HAVE_TILING",
        "-D",
        "AUTO_FUSE_DEVICE=1",
        f"--npu-arch={soc_version}",
        "-o",
        f"{temp_dir}/device/{base_device_file}.o",
        "-c",
        "-x",
        "asc",
        f"{temp_dir}/device/{base_device_file}",
    ]
    run_compile_command(device_compile_cmd, "Device")
    return f"{temp_dir}/device/{base_device_file}.o"


@inductor_compile_duration("BuildDeviceSo")
def build_device_so(args: argparse.Namespace, host_obj_path, temp_dir):
    device_obj_path = compile_device_obj(args, temp_dir)
    target_file = os.path.join(temp_dir, os.path.basename(args.output_file))
    obj_files = [device_obj_path]
    host_obj_paths = normalize_to_list(host_obj_path)
    if host_obj_paths:
        obj_files = host_obj_paths + obj_files
    link_libraries = (
        CV_HOST_LINK_LIBRARIES
        if host_obj_paths and is_cv_fusion_compile(args)
        else (HOST_LINK_LIBRARIES if host_obj_paths else None)
    )
    with InductorCompileDuration(args, "LinkDeviceSo"):
        return link_shared(target_file, obj_files, link_libraries=link_libraries)


def clean_before_modify(temp_dir):
    src_directory = os.getcwd()
    keep_dirs = {"host", "device"}
    for entry in os.listdir(temp_dir):
        entry_path = os.path.join(temp_dir, entry)
        if os.path.isfile(entry_path):
            os.remove(entry_path)
            print(f"delete file: {entry_path}")
        elif entry not in keep_dirs:
            shutil.rmtree(entry_path)
            print(f"delete dir: {entry_path}")
    os.chdir(src_directory)


def remove_tiling_data_from_launch(lines, start_index, launch_pattern):
    launch_lines = []
    index = start_index
    while index < len(lines):
        launch_lines.append(lines[index])
        if ");" in lines[index]:
            break
        index += 1

    launch_code = "".join(launch_lines)
    launch_code, replace_count = launch_pattern.subn("", launch_code, count=1)
    return launch_code.splitlines(keepends=True), index + 1, replace_count > 0


def remove_tiling_data_from_kernel_definition(
    lines, start_index, kernel_param_pattern, tiling_repr
):
    definition_lines = []
    index = start_index
    while index < len(lines):
        definition_lines.append(lines[index])
        if "{" in lines[index]:
            break
        index += 1

    definition_code = "".join(definition_lines)
    definition_code, replace_count = kernel_param_pattern.subn(
        "", definition_code, count=1
    )
    if replace_count == 0:
        return definition_lines, index + 1, False

    definition_lines = definition_code.splitlines(keepends=True)
    if tiling_repr is None:
        definition_lines.append("  const AutofuseTilingData t;\n")
    else:
        definition_lines.append(f"  constexpr AutofuseTilingData t = {tiling_repr};\n")
    return definition_lines, index + 1, True


def should_select_inductor_const_tiling_branch(
    args: argparse.Namespace, tiling_repr=None
):
    if tiling_repr is not None:
        return True
    return getattr(args, "stage", None) != "device" and is_cv_fusion_compile(args)


def prepare_static_shape_kernel(args: argparse.Namespace, temp_dir, tiling_repr=None):
    base_device_files = os.path.basename(args.device_files)
    kernel_file = os.path.join(temp_dir, "device", base_device_files)
    with open(kernel_file, "r") as f:
        lines = f.readlines()
    has_inductor_const_tiling = any(
        "INDUCTOR_CONST_TILING_DATA" in line for line in lines
    )
    if has_inductor_const_tiling:
        lines = expand_inductor_const_tiling_data(args, lines, tiling_repr)
    return kernel_file, lines


def write_kernel_lines(kernel_file, lines):
    with open(kernel_file, "w") as f:
        f.writelines(lines)


def build_static_shape_patterns():
    return (
        re.compile(
            r'^\s*(extern\s+"C"\s+)?__global__\s+__aicore__\s+void\s+(\w+)\s*\('
        ),
        re.compile(r",\s*AutofuseTilingData\s+t(?=\s*\)\s*\{)", re.DOTALL),
        re.compile(r"\b(\w+)\s*(?:<[^<>]*>)?\s*<<<"),
        re.compile(r",\s*\*\s*tiling_data(?=\s*\);)", re.DOTALL),
    )


def rewrite_static_shape_kernel_definitions(
    lines, kernel_start_pattern, kernel_param_pattern, tiling_repr=None
):
    result = []
    rewritten_kernels = set()
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        kernel_match = kernel_start_pattern.match(line)
        if kernel_match:
            definition_lines, line_index, is_rewritten = (
                remove_tiling_data_from_kernel_definition(
                    lines, line_index, kernel_param_pattern, tiling_repr
                )
            )
            result.extend(definition_lines)
            if is_rewritten:
                rewritten_kernels.add(kernel_match.group(2))
            continue

        result.append(line)
        line_index += 1
    return result, rewritten_kernels


def rewrite_static_shape_kernel_launches(
    lines, rewritten_kernels, launch_start_pattern, launch_pattern
):
    result = []
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        launch_match = launch_start_pattern.search(line)
        if launch_match and launch_match.group(1) in rewritten_kernels:
            launch_lines, line_index, is_rewritten = remove_tiling_data_from_launch(
                lines, line_index, launch_pattern
            )
            if not is_rewritten:
                raise CompileError(
                    f"Failed to remove tiling data from launch of {launch_match.group(1)}"
                )
            result.extend(launch_lines)
            continue
        result.append(line)
        line_index += 1
    return result


def static_shape_kernel_proc(args: argparse.Namespace, temp_dir, tiling_repr=None):
    with InductorCompileDuration(args, "RewriteDeviceKernelForStaticTiling"):
        clean_before_modify(temp_dir)
        kernel_file, lines = prepare_static_shape_kernel(args, temp_dir, tiling_repr)
        if tiling_repr is None and getattr(args, "stage", None) == "device":
            write_kernel_lines(kernel_file, lines)
            return

        patterns = build_static_shape_patterns()
        definition_result, rewritten_kernels = rewrite_static_shape_kernel_definitions(
            lines, patterns[0], patterns[1], tiling_repr
        )
        result = rewrite_static_shape_kernel_launches(
            definition_result, rewritten_kernels, patterns[2], patterns[3]
        )
        write_kernel_lines(kernel_file, result)


def expand_inductor_const_tiling_data(
    args: argparse.Namespace, lines, tiling_repr=None
):
    result = []
    index = 0
    pending_const_tiling = False
    select_const_branch = should_select_inductor_const_tiling_branch(args, tiling_repr)
    while index < len(lines):
        line = lines[index]
        if is_inductor_dynamic_tiling_param(lines, index):
            if tiling_repr is None and result:
                result[-1] = result[-1].rstrip("\n") + ", AutofuseTilingData t"
            elif tiling_repr is not None:
                pending_const_tiling = True
            index += 3
            continue
        if is_inductor_const_tiling_branch(lines, index):
            result.append(lines[index + 1] if select_const_branch else lines[index + 3])
            index += 5
            continue
        result.append(line)
        if pending_const_tiling and line.strip() == ") {":
            result.append(f"  constexpr AutofuseTilingData t = {tiling_repr};\n")
            pending_const_tiling = False
        index += 1
    return result


def is_inductor_dynamic_tiling_param(lines, index):
    if index + 2 >= len(lines):
        return False
    return (
        lines[index].strip() == "#ifndef INDUCTOR_CONST_TILING_DATA"
        and lines[index + 1].strip() == ", AutofuseTilingData t"
        and lines[index + 2].strip() == "#endif"
    )


def is_inductor_const_tiling_branch(lines, index):
    if index + 4 >= len(lines):
        return False
    return (
        lines[index].strip() == "#ifdef INDUCTOR_CONST_TILING_DATA"
        and lines[index + 2].strip() == "#else"
        and lines[index + 4].strip() == "#endif"
    )


def has_inductor_const_tiling_data(args: argparse.Namespace, temp_dir):
    base_device_files = os.path.basename(args.device_files)
    kernel_file = os.path.join(temp_dir, "device", base_device_files)
    with open(kernel_file, "r") as f:
        return "INDUCTOR_CONST_TILING_DATA" in f.read()


def init_torch_npu_for_const_tiling():
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    torch.npu.synchronize()


@inductor_compile_duration("PrepareStaticShapeRecompile")
def try_static_shape_compile(args: argparse.Namespace, temp_dir, so_path):
    if args.force_unknown:
        return False
    lib = ctypes.CDLL(so_path)
    lib.AutofuseIsStaticShape.argtypes = []
    lib.AutofuseIsStaticShape.restype = ctypes.c_bool
    if not bool(lib.AutofuseIsStaticShape()):
        return False
    print("static shape detected, recompile kernel with const tiling data")
    static_shape_kernel_proc(args, temp_dir)
    with InductorCompileDuration(args, "InitTorchNpu"):
        init_torch_npu_for_const_tiling()
    with InductorCompileDuration(args, "GenerateConstTilingData"):
        lib.GenConstTilingData.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        lib.GenConstTilingData.restype = ctypes.c_char_p
        config_file = ctypes.c_char_p(args.config_file.encode("utf-8"))
        aiv_num = int(get_soc_spec("vector_core_cnt"))
        ub_size = int(get_soc_spec("ub_size"))
        result = lib.GenConstTilingData(
            config_file, ctypes.c_int(aiv_num), ctypes.c_int(ub_size)
        )
    const_tiling_data = result.decode("utf-8")
    tiling_data = os.path.join(temp_dir, "device", "autofuse_tiling_data.h")
    tiling_data_bak = os.path.join(temp_dir, "device", "autofuse_tiling_data_bak.h")
    shutil.copy(tiling_data, tiling_data_bak)
    with open(tiling_data, "w") as file:
        file.write(const_tiling_data)
    return True


def link_host_target(args, temp_dir):
    # 处理 host 编译阶段
    host_obj_paths = compile_host_objs(args, temp_dir)
    so_file = os.path.join(temp_dir, os.path.basename(args.output_file))
    link_libraries = (
        CV_HOST_LINK_LIBRARIES if is_cv_fusion_compile(args) else HOST_LINK_LIBRARIES
    )
    with InductorCompileDuration(args, "LinkHostSo"):
        link_shared(so_file, host_obj_paths, link_libraries=link_libraries)
    return so_file


@inductor_compile_duration("BuildKernelTarget")
def link_kernel_target(args, host_obj_path, temp_dir):
    if args.stage == "device":
        if args.tiling_repr is not None or has_inductor_const_tiling_data(
            args, temp_dir
        ):
            if args.tiling_repr is not None:
                print("process static shape kernel with tiling_repr")
            static_shape_kernel_proc(args, temp_dir, args.tiling_repr)

    # 首次编译
    so_file = build_device_so(args, host_obj_path, temp_dir)

    # kernel_compile场景一次性生成so，链接device.o
    if args.stage == "device":
        return so_file

    # jit_compile场景，检测是否为静态shape
    re_compile = try_static_shape_compile(args, temp_dir, so_file)
    if not re_compile:
        return so_file

    # 重编译，最终产物链接host.o+device.o
    return build_device_so(args, host_obj_path, temp_dir)


@inductor_compile_duration("CopyOutput", args_index=1)
def copy_so_to_output(so_file, args, src_directory):
    dst_file = os.path.realpath(args.output_file)
    dst_dir_path = os.path.dirname(dst_file)
    if not os.path.exists(dst_dir_path):
        os.makedirs(dst_dir_path)

    shutil.copy(so_file, dst_file)
    print(f"copy file {so_file} to {dst_file}")
    os.chdir(src_directory)


def main(args):
    print("compile args:", args)
    src_directory = os.getcwd()
    os.chdir(args.temp_dir)
    print("change work dir:", os.getcwd())

    if args.stage == "host":
        so_file = link_host_target(args, args.temp_dir)
    elif args.stage == "device":
        so_file = link_kernel_target(args, None, args.temp_dir)
    else:  # all
        host_obj_paths = compile_host_objs(args, args.temp_dir)
        so_file = link_kernel_target(args, host_obj_paths, args.temp_dir)

    copy_so_to_output(so_file, args, src_directory)


def main_with_except(argv: List[str]):
    """Main process with except exceptions."""
    try:
        print("Enter main func")
        return main(argv)
    except argparse.ArgumentError as ex:
        print(f"error: check arguments error, {ex}")
        return False


if __name__ == "__main__":
    if not main_with_except(sys.argv[1:]):
        sys.exit(1)
