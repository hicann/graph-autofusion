# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import types

import pytest

from compile_test_utils import PYTHON_DIR, load_compile_module

MODULE_NAME = "autofuse.compiler.python.ascendc_compile"
MODULE_PATH = os.path.join(PYTHON_DIR, "ascendc_compile.py")

_LAUNCH_FUNC = (
    'extern "C" int64_t AutofuseLaunch(uint32_t blockDim, void* stream, void* input0, void* output0, '
    "void* workspace, AutofuseTilingData* tiling_data)\n"
    "{\n"
    "  static_kernel<<<blockDim, nullptr, stream>>>(\n"
    "      (uint8_t*)input0, (uint8_t*)output0,\n"
    "      (uint8_t*)workspace, *tiling_data);\n"
    "  return 0;\n"
    "}\n"
)


def fake_get_soc_spec(key):
    return {
        "vector_core_cnt": 20,
        "ub_size": 262144,
    }.get(key, "")


@pytest.fixture()
def ascendc_compile_module():
    platform_info_module = types.ModuleType(
        "asc_op_compile_base.common.platform.platform_info"
    )
    platform_info_module.get_soc_spec = fake_get_soc_spec
    extra_modules = {
        "asc_op_compile_base": types.ModuleType("asc_op_compile_base"),
        "asc_op_compile_base.common": types.ModuleType("asc_op_compile_base.common"),
        "asc_op_compile_base.common.platform": types.ModuleType(
            "asc_op_compile_base.common.platform"
        ),
        "asc_op_compile_base.common.platform.platform_info": platform_info_module,
    }
    with load_compile_module(
        MODULE_NAME, MODULE_PATH, extra_modules=extra_modules
    ) as loaded_module:
        yield loaded_module


def _noop_run_compile_command(cmd, stage_name):
    return None


def test_link_shared_adds_requested_libraries(ascendc_compile_module):
    captured = {}

    def fake_run_compile_command(cmd, stage_name):
        captured["cmd"] = cmd
        captured["stage_name"] = stage_name

    ascendc_compile_module.module.run_compile_command = fake_run_compile_command
    ascendc_compile_module.module.ASCEND_PATH = "/usr/local/Ascend/cann"
    ascendc_compile_module.module.machine = "x86_64"

    result = ascendc_compile_module.link_shared(
        "kernel.so", ["host.o"], link_libraries=["graph_base", "register"]
    )

    assert result == "kernel.so"
    assert captured["stage_name"] == "LinkObj"
    assert "-L" in captured["cmd"]
    assert "/usr/local/Ascend/cann/x86_64-linux/lib64" in captured["cmd"]
    assert "-lgraph_base" in captured["cmd"]
    assert "-lregister" in captured["cmd"]
    assert captured["cmd"].index("-lgraph_base") < captured["cmd"].index("-lregister")


def test_link_shared_skips_libraries_by_default(ascendc_compile_module):
    captured = {}

    def fake_run_compile_command(cmd, stage_name):
        captured["cmd"] = cmd

    ascendc_compile_module.module.run_compile_command = fake_run_compile_command

    ascendc_compile_module.link_shared("kernel.so", ["device.o"])

    assert "-lgraph_base" not in captured["cmd"]


def test_host_target_records_compile_and_link_stage(
    ascendc_compile_module, tmpdir, capsys
):
    host_dir = tmpdir.mkdir("host")
    host_file = host_dir.join("graph_tiling_func.cpp")
    host_file.write("host")
    args = type(
        "Args",
        (),
        {
            "host_files": str(host_file),
            "compile_options": "",
            "soc_version": "Ascend910B",
            "stage": "host",
            "graph_name": "graph",
            "output_file": str(tmpdir.join("host.so")),
        },
    )()

    ascendc_compile_module.module.run_compile_command = _noop_run_compile_command
    ascendc_compile_module.link_host_target(args, str(tmpdir))

    labels = [item[0] for item in ascendc_compile_module.duration_records]
    assert ["InductorCompile", "host", "CompileHostObj", "graph"] in labels
    assert ["InductorCompile", "host", "LinkHostSo", "graph"] in labels
    assert capsys.readouterr().out == ""


def test_kernel_target_records_device_compile_and_link_stage(
    ascendc_compile_module, tmpdir
):
    device_dir = tmpdir.mkdir("device")
    device_file = device_dir.join("graph_op_kernel.cpp")
    device_file.write(
        'extern "C" __global__ __aicore__ void graph_kernel(GM_ADDR input0) {}\n'
    )
    output_file = tmpdir.join("kernel.so")
    args = type(
        "Args",
        (),
        {
            "device_files": str(device_file),
            "output_file": str(output_file),
            "soc_version": "Ascend910B",
            "stage": "device",
            "tiling_repr": None,
            "force_unknown": True,
            "graph_name": "graph",
        },
    )()

    ascendc_compile_module.module.run_compile_command = _noop_run_compile_command
    ascendc_compile_module.link_kernel_target(args, None, str(tmpdir))

    labels = [item[0] for item in ascendc_compile_module.duration_records]
    assert ["InductorCompile", "device", "CompileDeviceObj", "graph"] in labels
    assert ["InductorCompile", "device", "LinkDeviceSo", "graph"] in labels


def test_compile_device_obj_includes_machine_asc_headers(
    ascendc_compile_module, tmpdir
):
    calls = []
    device_file = tmpdir.mkdir("device").join("graph_op_kernel.cpp")
    args = type(
        "Args",
        (),
        {
            "device_files": str(device_file),
            "soc_version": "Ascend910B",
            "stage": "device",
            "graph_name": "graph",
        },
    )()

    def fake_run_compile_command(cmd, stage_name):
        calls.append((cmd, stage_name))

    ascendc_compile_module.module.ASCEND_PATH = "/usr/local/Ascend/cann"
    ascendc_compile_module.module.machine = "aarch64"
    ascendc_compile_module.module.run_compile_command = fake_run_compile_command

    ascendc_compile_module.compile_device_obj(args, str(tmpdir))

    assert len(calls) == 1
    cmd, stage_name = calls[0]
    assert stage_name == "Device"
    assert "/usr/local/Ascend/cann/aarch64-linux/asc/include" in cmd


def test_try_static_shape_compile_records_stage_when_force_unknown(
    ascendc_compile_module, tmpdir
):
    args = type(
        "Args", (), {"force_unknown": True, "stage": "all", "graph_name": "graph"}
    )()

    assert (
        ascendc_compile_module.try_static_shape_compile(args, str(tmpdir), "kernel.so")
        is False
    )

    labels = [item[0] for item in ascendc_compile_module.duration_records]
    assert ["InductorCompile", "all", "PrepareStaticShapeRecompile", "graph"] in labels


def test_copy_so_to_output_records_stage(ascendc_compile_module, tmpdir):
    src_file = tmpdir.join("source.so")
    dst_file = tmpdir.mkdir("out").join("target.so")
    src_file.write("binary")
    src_directory = os.getcwd()
    args = type(
        "Args",
        (),
        {
            "output_file": str(dst_file),
            "stage": "host",
            "graph_name": "graph",
        },
    )()

    ascendc_compile_module.copy_so_to_output(str(src_file), args, src_directory)

    labels = [item[0] for item in ascendc_compile_module.duration_records]
    assert ["InductorCompile", "host", "CopyOutput", "graph"] in labels
    assert dst_file.read() == "binary"


def test_static_shape_kernel_proc_removes_tiling_data_from_launch(
    ascendc_compile_module, tmpdir
):
    device_dir = tmpdir.mkdir("device")
    kernel_file = device_dir.join("static_kernel.cpp")
    kernel_file.write(
        'extern "C" __global__ __aicore__ void static_kernel(GM_ADDR input0, GM_ADDR output0, '
        "GM_ADDR workspace, AutofuseTilingData t) {\n"
        "  use(t);\n"
        "}\n"
        "void init_static_kernel(void) {}\n"
        'extern "C" int64_t AutofuseLaunch(uint32_t blockDim, void* stream, void* input0, void* output0, '
        "void* workspace, AutofuseTilingData* tiling_data)\n"
        "{\n"
        "  static_kernel<<<blockDim, nullptr, stream>>>((uint8_t*)input0, (uint8_t*)output0, "
        "(uint8_t*)workspace, *tiling_data);\n"
        "  return 0;\n"
        "}\n"
    )
    args = type("Args", (), {"device_files": str(kernel_file)})()

    ascendc_compile_module.static_shape_kernel_proc(
        args, str(tmpdir), "AutofuseTilingData{.block_dim = 8}"
    )

    content = kernel_file.read()
    assert (
        'extern "C" __global__ __aicore__ void static_kernel(GM_ADDR input0, GM_ADDR output0, GM_ADDR workspace) {'
        in content
    )
    assert (
        "constexpr AutofuseTilingData t = AutofuseTilingData{.block_dim = 8};"
        in content
    )
    assert (
        "static_kernel<<<blockDim, nullptr, stream>>>("
        "(uint8_t*)input0, (uint8_t*)output0, (uint8_t*)workspace);" in content
    )
    assert "*tiling_data" not in content


def test_static_shape_kernel_proc_removes_multiline_tiling_data_from_launch(
    ascendc_compile_module, tmpdir
):
    device_dir = tmpdir.mkdir("device")
    kernel_file = device_dir.join("static_kernel.cpp")
    kernel_file.write(
        'extern "C" __global__ __aicore__ void static_kernel(GM_ADDR input0, GM_ADDR output0, '
        "GM_ADDR workspace, AutofuseTilingData t) {\n"
        "  use(t);\n"
        "}\n" + _LAUNCH_FUNC
    )
    args = type("Args", (), {"device_files": str(kernel_file)})()

    ascendc_compile_module.static_shape_kernel_proc(
        args, str(tmpdir), "AutofuseTilingData{.block_dim = 8}"
    )

    content = kernel_file.read()
    assert "*tiling_data" not in content
    assert "(uint8_t*)workspace);" in content


def test_static_shape_kernel_proc_removes_multiline_kernel_param(
    ascendc_compile_module, tmpdir
):
    device_dir = tmpdir.mkdir("device")
    kernel_file = device_dir.join("static_kernel.cpp")
    kernel_file.write(
        'extern "C" __global__ __aicore__ void static_kernel(\n'
        "    GM_ADDR input0,\n"
        "    GM_ADDR output0,\n"
        "    GM_ADDR workspace,\n"
        "    AutofuseTilingData t) {\n"
        "  use(t);\n"
        "}\n" + _LAUNCH_FUNC
    )
    args = type("Args", (), {"device_files": str(kernel_file)})()

    ascendc_compile_module.static_shape_kernel_proc(
        args, str(tmpdir), "AutofuseTilingData{.block_dim = 8}"
    )

    content = kernel_file.read()
    assert "GM_ADDR workspace) {" in content
    assert "AutofuseTilingData t) {" not in content
    assert (
        "constexpr AutofuseTilingData t = AutofuseTilingData{.block_dim = 8};"
        in content
    )
    assert "(uint8_t*)workspace);" in content


def test_static_shape_kernel_proc_removes_indented_kernel_param(
    ascendc_compile_module, tmpdir
):
    device_dir = tmpdir.mkdir("device")
    kernel_file = device_dir.join("static_kernel.cpp")
    kernel_file.write(
        '  extern "C" __global__ __aicore__ void static_kernel(GM_ADDR input0, GM_ADDR output0, '
        "GM_ADDR workspace, AutofuseTilingData t) {\n"
        "  use(t);\n"
        "}\n" + _LAUNCH_FUNC
    )
    args = type("Args", (), {"device_files": str(kernel_file)})()

    ascendc_compile_module.static_shape_kernel_proc(
        args, str(tmpdir), "AutofuseTilingData{.block_dim = 8}"
    )

    content = kernel_file.read()
    assert "AutofuseTilingData t) {" not in content
    assert (
        "constexpr AutofuseTilingData t = AutofuseTilingData{.block_dim = 8};"
        in content
    )
    assert "(uint8_t*)workspace);" in content


def test_static_shape_kernel_proc_keeps_launch_when_kernel_definition_not_rewritten(
    ascendc_compile_module, tmpdir
):
    device_dir = tmpdir.mkdir("device")
    kernel_file = device_dir.join("static_kernel.cpp")
    kernel_file.write(
        "__global__ __aicore__ __launch_bounds__(1) void static_kernel("
        "GM_ADDR input0, GM_ADDR workspace, AutofuseTilingData t) {\n"
        "  use(t);\n"
        "}\n"
        'extern "C" int64_t AutofuseLaunch(uint32_t blockDim, void* stream, void* input0, '
        "void* workspace, AutofuseTilingData* tiling_data)\n"
        "{\n"
        "  static_kernel<<<blockDim, nullptr, stream>>>((uint8_t*)input0, "
        "(uint8_t*)workspace, *tiling_data);\n"
        "  return 0;\n"
        "}\n"
    )
    args = type("Args", (), {"device_files": str(kernel_file)})()

    ascendc_compile_module.static_shape_kernel_proc(
        args, str(tmpdir), "AutofuseTilingData{.block_dim = 8}"
    )

    content = kernel_file.read()
    assert "AutofuseTilingData t) {" in content
    assert "constexpr AutofuseTilingData t =" not in content
    assert (
        "static_kernel<<<blockDim, nullptr, stream>>>((uint8_t*)input0, (uint8_t*)workspace, *tiling_data);"
        in content
    )


def test_static_shape_kernel_proc_only_rewrites_launch_for_rewritten_kernel(
    ascendc_compile_module, tmpdir
):
    device_dir = tmpdir.mkdir("device")
    kernel_file = device_dir.join("static_kernel.cpp")
    kernel_file.write(
        'extern "C" __global__ __aicore__ void static_kernel(GM_ADDR input0, GM_ADDR workspace, '
        "AutofuseTilingData t) {\n"
        "  use(t);\n"
        "}\n"
        'extern "C" int64_t AutofuseLaunch(uint32_t blockDim, void* stream, void* input0, '
        "void* workspace, AutofuseTilingData* tiling_data)\n"
        "{\n"
        "  static_kernel<<<blockDim, nullptr, stream>>>((uint8_t*)input0, (uint8_t*)workspace, *tiling_data);\n"
        "  other_kernel<<<blockDim, nullptr, stream>>>((uint8_t*)input0, (uint8_t*)workspace, *tiling_data);\n"
        "  return 0;\n"
        "}\n"
    )
    args = type("Args", (), {"device_files": str(kernel_file)})()

    ascendc_compile_module.static_shape_kernel_proc(
        args, str(tmpdir), "AutofuseTilingData{.block_dim = 8}"
    )

    content = kernel_file.read()
    assert (
        "static_kernel<<<blockDim, nullptr, stream>>>((uint8_t*)input0, (uint8_t*)workspace);"
        in content
    )
    assert (
        "other_kernel<<<blockDim, nullptr, stream>>>((uint8_t*)input0, (uint8_t*)workspace, *tiling_data);"
        in content
    )


def test_static_shape_kernel_proc_keeps_dynamic_inductor_tiling_when_tiling_repr_is_none(
    ascendc_compile_module, tmpdir
):
    device_dir = tmpdir.mkdir("device")
    kernel_file = device_dir.join("static_kernel.cpp")
    kernel_file.write(
        'extern "C" __global__ __aicore__ void static_kernel(GM_ADDR input0, GM_ADDR workspace, '
        "AutofuseTilingData t) {\n"
        "#ifdef INDUCTOR_CONST_TILING_DATA\n"
        "  const AutofuseTilingData t;\n"
        "#else\n"
        "  use(t);\n"
        "#endif\n"
        "}\n"
        'extern "C" int64_t AutofuseLaunch(uint32_t blockDim, void* stream, void* input0, '
        "void* workspace, AutofuseTilingData* tiling_data)\n"
        "{\n"
        "  static_kernel<<<blockDim, nullptr, stream>>>((uint8_t*)input0, "
        "(uint8_t*)workspace, *tiling_data);\n"
        "  return 0;\n"
        "}\n"
    )
    args = type("Args", (), {"device_files": str(kernel_file), "stage": "device"})()

    ascendc_compile_module.static_shape_kernel_proc(args, str(tmpdir), None)

    content = kernel_file.read()
    assert "INDUCTOR_CONST_TILING_DATA" not in content
    assert "AutofuseTilingData t) {" in content
    assert "const AutofuseTilingData t;" not in content
    assert (
        "static_kernel<<<blockDim, nullptr, stream>>>((uint8_t*)input0, "
        "(uint8_t*)workspace, *tiling_data);" in content
    )


def _make_device_stage_args():
    return type("Args", (), {"stage": "device"})()


def _make_inductor_const_tiling_branch_lines():
    return [
        "#ifdef INDUCTOR_CONST_TILING_DATA\n",
        "const CVAutofuseTilingData& gm_tiling_data = kConstTilingData;\n",
        "#else\n",
        "const TILING_DATA_T& gm_tiling_data = t;\n",
        "#endif\n",
    ]


def test_expand_inductor_const_tiling_data_selects_dynamic_branch_without_tiling_repr(
    ascendc_compile_module,
):
    args = _make_device_stage_args()
    lines = _make_inductor_const_tiling_branch_lines()

    result = ascendc_compile_module.expand_inductor_const_tiling_data(args, lines, None)

    assert result == ["const TILING_DATA_T& gm_tiling_data = t;\n"]


def test_expand_inductor_const_tiling_data_selects_const_branch_with_tiling_repr(
    ascendc_compile_module,
):
    args = _make_device_stage_args()
    lines = _make_inductor_const_tiling_branch_lines()

    result = ascendc_compile_module.expand_inductor_const_tiling_data(
        args, lines, "AutofuseTilingData{}"
    )

    assert result == [
        "const CVAutofuseTilingData& gm_tiling_data = kConstTilingData;\n"
    ]


def _make_compile_args(host_files=None):
    return type(
        "Args",
        (),
        {
            "host_files": host_files,
            "device_files": "/tmp/build/device/kernel.cpp",
            "soc_version": "Ascend910B",
            "compile_options": "-Werror",
            "output_file": "/tmp/build/kernel.so",
            "force_unknown": True,
            "stage": "all",
            "tiling_repr": None,
        },
    )()


def test_build_host_compile_cmd_uses_bisheng_without_cmake(ascendc_compile_module):
    ascendc_compile_module.module.ASCEND_PATH = "/usr/local/Ascend/cann"
    ascendc_compile_module.module.machine = "x86_64"
    args = _make_compile_args("/tmp/build/host/graph_tiling_func.cpp")

    cmd = ascendc_compile_module.build_host_compile_cmd(
        args,
        "/tmp/build",
        "/tmp/build/host/graph_tiling_func.cpp",
        "/tmp/build/host/graph_tiling_func.cpp.o",
    )

    assert cmd[0] == "/usr/local/Ascend/cann/tools/bisheng_compiler/bin/bisheng"
    assert "-c" in cmd
    assert "-x" in cmd
    assert "asc" in cmd
    assert "/tmp/build/host/graph_tiling_func.cpp" in cmd
    assert "/tmp/build/host/graph_tiling_func.cpp.o" in cmd
    assert "cmake" not in cmd
    assert "make" not in cmd


def test_compile_host_objs_keeps_single_file_compatible(ascendc_compile_module):
    calls = []
    args = _make_compile_args("/tmp/build/host/graph_tiling_func.cpp")

    def fake_run_compile_command(cmd, stage_name):
        calls.append((cmd, stage_name))

    ascendc_compile_module.module.run_compile_command = fake_run_compile_command

    result = ascendc_compile_module.compile_host_objs(args, "/tmp/build")

    assert result == ["/tmp/build/host/graph_tiling_func.cpp.o"]
    assert len(calls) == 1
    assert calls[0][1] == "Host"


def test_compile_host_objs_compiles_multiple_files(ascendc_compile_module, monkeypatch):
    calls = []
    args = _make_compile_args(
        [
            "/tmp/build/host/graph_tiling_func_a.cpp",
            "/tmp/build/host/graph_tiling_func_b.cpp",
        ]
    )

    def fake_run_compile_command(cmd, stage_name):
        calls.append((cmd, stage_name))

    ascendc_compile_module.module.run_compile_command = fake_run_compile_command
    monkeypatch.setattr(ascendc_compile_module.os, "cpu_count", lambda: 2)

    result = ascendc_compile_module.compile_host_objs(args, "/tmp/build")

    assert result == [
        "/tmp/build/host/graph_tiling_func_a.cpp.o",
        "/tmp/build/host/graph_tiling_func_b.cpp.o",
    ]
    assert [call[1] for call in calls] == ["Host", "Host"]
    compile_cmds = [call[0] for call in calls]
    assert any("/tmp/build/host/graph_tiling_func_a.cpp" in cmd for cmd in compile_cmds)
    assert any("/tmp/build/host/graph_tiling_func_b.cpp" in cmd for cmd in compile_cmds)
    assert all("cmake" not in cmd and "make" not in cmd for cmd in compile_cmds)


def test_get_host_compile_worker_count_uses_32_worker_limit(
    ascendc_compile_module, monkeypatch
):
    monkeypatch.setattr(ascendc_compile_module.os, "cpu_count", lambda: 64)
    assert ascendc_compile_module.get_host_compile_worker_count(32) == 32
    assert ascendc_compile_module.get_host_compile_worker_count(12) == 12
    monkeypatch.setattr(ascendc_compile_module.os, "cpu_count", lambda: 16)
    assert ascendc_compile_module.get_host_compile_worker_count(32) == 16
    monkeypatch.setattr(ascendc_compile_module.os, "cpu_count", lambda: None)
    assert ascendc_compile_module.get_host_compile_worker_count(3) == 1


def test_compile_host_objs_reports_failed_source(ascendc_compile_module, monkeypatch):
    args = _make_compile_args(
        [
            "/tmp/build/host/graph_tiling_func_a.cpp",
            "/tmp/build/host/graph_tiling_func_b.cpp",
        ]
    )

    def fake_run_compile_command(cmd, stage_name):
        if "/tmp/build/host/graph_tiling_func_b.cpp" in cmd:
            raise ascendc_compile_module.CompileError("stderr: fail")

    ascendc_compile_module.module.run_compile_command = fake_run_compile_command
    monkeypatch.setattr(ascendc_compile_module.os, "cpu_count", lambda: 1)

    with pytest.raises(ascendc_compile_module.CompileError) as exc_info:
        ascendc_compile_module.compile_host_objs(args, "/tmp/build")

    assert "/tmp/build/host/graph_tiling_func_b.cpp" in str(exc_info.value)
    assert "stderr: fail" in str(exc_info.value)


def test_compile_host_obj_rejects_multiple_sources_without_compile(
    ascendc_compile_module,
):
    args = _make_compile_args(
        [
            "/tmp/build/host/graph_tiling_func_a.cpp",
            "/tmp/build/host/graph_tiling_func_b.cpp",
        ]
    )

    def fake_run_compile_command(cmd, stage_name):
        pytest.fail("should not compile")

    ascendc_compile_module.module.run_compile_command = fake_run_compile_command

    with pytest.raises(ascendc_compile_module.CompileError) as exc_info:
        ascendc_compile_module.compile_host_obj(args, "/tmp/build")

    assert "expects exactly one host source" in str(exc_info.value)


def test_build_device_so_links_all_host_objects(ascendc_compile_module):
    captured = {}
    args = _make_compile_args()

    def fake_compile_device_obj(compile_args, temp_dir):
        return "/tmp/build/device/kernel.cpp.o"

    ascendc_compile_module.module.compile_device_obj = fake_compile_device_obj

    def fake_link_shared(target_file, obj_files, link_libraries=None):
        captured["target_file"] = target_file
        captured["obj_files"] = obj_files
        captured["link_libraries"] = link_libraries
        return target_file

    ascendc_compile_module.module.link_shared = fake_link_shared

    result = ascendc_compile_module.build_device_so(args, ["a.o", "b.o"], "/tmp/build")

    assert result == "/tmp/build/kernel.so"
    assert captured["obj_files"] == ["a.o", "b.o", "/tmp/build/device/kernel.cpp.o"]
    assert captured["link_libraries"] == ascendc_compile_module.HOST_LINK_LIBRARIES


def test_link_host_target_links_multiple_host_objects(ascendc_compile_module):
    captured = {}
    args = _make_compile_args(
        [
            "/tmp/build/host/graph_tiling_func_a.cpp",
            "/tmp/build/host/graph_tiling_func_b.cpp",
        ]
    )

    def fake_compile_host_objs(compile_args, temp_dir):
        return ["a.o", "b.o"]

    ascendc_compile_module.module.compile_host_objs = fake_compile_host_objs

    def fake_link_shared(target_file, obj_files, link_libraries=None):
        captured["target_file"] = target_file
        captured["obj_files"] = obj_files
        captured["link_libraries"] = link_libraries
        return target_file

    ascendc_compile_module.module.link_shared = fake_link_shared

    result = ascendc_compile_module.link_host_target(args, "/tmp/build")

    assert result == "/tmp/build/kernel.so"
    assert captured["target_file"] == "/tmp/build/kernel.so"
    assert captured["obj_files"] == ["a.o", "b.o"]
    assert captured["link_libraries"] == ascendc_compile_module.HOST_LINK_LIBRARIES


def test_link_kernel_target_reuses_host_objects_for_static_recompile(
    ascendc_compile_module,
):
    calls = []
    args = _make_compile_args()
    args.force_unknown = False

    def fake_try_static_shape_compile(compile_args, temp_dir, so_path):
        return True

    ascendc_compile_module.module.try_static_shape_compile = (
        fake_try_static_shape_compile
    )

    def fake_build_device_so(compile_args, host_obj_paths, temp_dir):
        calls.append(list(host_obj_paths))
        return f"/tmp/build/kernel_{len(calls)}.so"

    ascendc_compile_module.module.build_device_so = fake_build_device_so

    result = ascendc_compile_module.link_kernel_target(
        args, ["a.o", "b.o"], "/tmp/build"
    )

    assert result == "/tmp/build/kernel_2.so"
    assert calls == [["a.o", "b.o"], ["a.o", "b.o"]]
