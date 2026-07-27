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

import inspect
import os
import types

import pytest

from compile_test_utils import PYTHON_DIR, load_compile_module

MODULE_NAME = "autofuse.compile_adapter"
MODULE_PATH = os.path.join(PYTHON_DIR, "compile_adapter.py")


def _host_compile_args(tmpdir):
    return type(
        "Args",
        (),
        {
            "stage": "host",
            "temp_dir": str(tmpdir),
            "graph_name": "graph",
            "trace_stage": "host_compile",
        },
    )()


@pytest.fixture()
def compile_adapter_module():
    ascendc_compile_module = types.ModuleType("autofuse.ascendc_compile")

    class CompileError(Exception):
        pass

    def _noop_main(args):
        return None

    ascendc_compile_module.CompileError = CompileError
    ascendc_compile_module.main = _noop_main
    with load_compile_module(
        MODULE_NAME,
        MODULE_PATH,
        extra_autofuse_attrs={"ascendc_compile": ascendc_compile_module},
        extra_modules={"autofuse.ascendc_compile": ascendc_compile_module},
    ) as loaded_module:
        yield loaded_module


def test_host_compile_defaults_to_abi_1(compile_adapter_module):
    argv = ["--output_file=host.so", "--compile_options=-Werror"]

    args, temp_dir_ctx, auto_cleanup = compile_adapter_module.prepare_compile_context(
        argv, "host", None
    )

    assert auto_cleanup is True
    assert temp_dir_ctx is not None
    assert "-Werror" in args.compile_options
    assert "-D_GLIBCXX_USE_CXX11_ABI=1" in args.compile_options
    temp_dir_ctx.cleanup()


@pytest.mark.parametrize("abi", ["0", "1"])
def test_host_compile_keeps_explicit_abi(compile_adapter_module, abi):
    option = f"-D_GLIBCXX_USE_CXX11_ABI={abi}"
    argv = ["--output_file=host.so", f"--compile_options=-Werror {option}"]

    args, temp_dir_ctx, _ = compile_adapter_module.prepare_compile_context(
        argv, "host", None
    )

    assert args.compile_options == f"-Werror {option}"
    temp_dir_ctx.cleanup()


def test_device_compile_does_not_add_host_default_abi(compile_adapter_module):
    argv = ["--output_file=device.so", "--compile_options=-Werror"]

    args, temp_dir_ctx, _ = compile_adapter_module.prepare_compile_context(
        argv, "device", None
    )

    assert args.compile_options == "-Werror"
    temp_dir_ctx.cleanup()


def test_jit_compile_records_atrace_and_reports(compile_adapter_module, tmpdir, capsys):
    output_file = tmpdir.join("jit.so")
    argv = [f"--output_file={output_file}", f"--output_path={tmpdir}"]

    compile_adapter_module.jit_compile("tiling", "host", "kernel", argv)

    labels = [item[0] for item in compile_adapter_module.duration_records]
    assert [
        "InductorCompile",
        "jit_compile",
        "GenerateHostSource",
        "autofuse",
    ] in labels
    assert [
        "InductorCompile",
        "jit_compile",
        "GenerateDeviceSource",
        "autofuse",
    ] in labels
    assert ["InductorCompile", "jit_compile", "AscendCCompile", "autofuse"] in labels
    assert ["InductorCompile", "jit_compile", "Total", "autofuse"] in labels
    assert compile_adapter_module.duration_reports == [True]
    assert capsys.readouterr().out == ""


def test_host_compile_records_duration_without_stdout(
    compile_adapter_module, tmpdir, capsys
):
    output_file = tmpdir.join("host.so")
    argv = [f"--output_file={output_file}", f"--output_path={tmpdir}"]

    compile_adapter_module.host_compile("tiling", "host", argv)

    labels = [item[0] for item in compile_adapter_module.duration_records]
    assert [
        "InductorCompile",
        "host_compile",
        "GenerateHostSource",
        "autofuse",
    ] in labels
    assert ["InductorCompile", "host_compile", "AscendCCompile", "autofuse"] in labels
    assert ["InductorCompile", "host_compile", "Total", "autofuse"] in labels
    assert capsys.readouterr().out == ""


def test_kernel_compile_records_device_stage(compile_adapter_module, tmpdir):
    output_file = tmpdir.join("kernel.so")
    argv = [f"--output_file={output_file}", f"--output_path={tmpdir}"]

    compile_adapter_module.kernel_compile(
        "tiling", "kernel", argv, tiling_repr="AutofuseTilingData{}"
    )

    labels = [item[0] for item in compile_adapter_module.duration_records]
    assert [
        "InductorCompile",
        "kernel_compile",
        "GenerateDeviceSource",
        "autofuse",
    ] in labels
    assert ["InductorCompile", "kernel_compile", "AscendCCompile", "autofuse"] in labels
    assert ["InductorCompile", "kernel_compile", "Total", "autofuse"] in labels


def test_execute_compile_keeps_single_host_file_without_marker(
    compile_adapter_module, tmpdir
):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    compile_adapter_module.ascendc_compile.main = fake_main
    args = _host_compile_args(tmpdir)

    compile_adapter_module.execute_compile(
        {
            "tiling_struct_code": "struct AutofuseTilingData {};",
            "host_impl_code": 'extern "C" int TilingFunc() { return 0; }',
            "kernel_impl_code": None,
        },
        args,
    )

    host_file = os.path.join(str(tmpdir), "host", "graph_tiling_func.cpp")
    assert captured["args"].host_files == host_file
    assert os.path.exists(host_file)


def test_execute_compile_splits_host_files_with_marker(compile_adapter_module, tmpdir):
    captured = {}

    def fake_main(args):
        captured["args"] = args

    compile_adapter_module.ascendc_compile.main = fake_main
    host_impl = "\n".join(
        [
            "// AUTOFUSE_SPLIT_FILE_BEGIN: TilingHead",
            "struct CommonType {};",
            "// AUTOFUSE_SPLIT_FILE_END: TilingHead",
            "// AUTOFUSE_SPLIT_FILE_BEGIN: solver_func",
            'extern "C" int Solver() { return 0; }',
            "// AUTOFUSE_SPLIT_FILE_END: solver_func",
            "// AUTOFUSE_SPLIT_FILE_BEGIN: asc_graph0_schedule_result0_g0",
            '#include "autofuse_tiling_func_common.h"',
            'extern "C" int TilingFunc() { return 0; }',
            "// AUTOFUSE_SPLIT_FILE_END: asc_graph0_schedule_result0_g0",
        ]
    )
    args = _host_compile_args(tmpdir)

    compile_adapter_module.execute_compile(
        {
            "tiling_struct_code": "struct AutofuseTilingData {};",
            "host_impl_code": host_impl,
            "kernel_impl_code": None,
        },
        args,
    )

    host_dir = os.path.join(str(tmpdir), "host")
    assert os.path.exists(os.path.join(host_dir, "autofuse_tiling_func_common.h"))
    assert captured["args"].host_files == [
        os.path.join(host_dir, "graph_tiling_func_solver_func.cpp"),
        os.path.join(host_dir, "graph_tiling_func_asc_graph0_schedule_result0_g0.cpp"),
    ]
    for cpp_file in captured["args"].host_files:
        with open(cpp_file) as f:
            assert f.read().count('#include "autofuse_tiling_func_common.h"') == 1
    assert not os.path.exists(
        os.path.join(host_dir, "graph_tiling_func_TilingHead.cpp")
    )


def test_write_split_host_sources_writes_split_headers_without_injecting_common(
    tmpdir, compile_adapter_module
):
    host_code = """// AUTOFUSE_SPLIT_FILE_BEGIN: TilingStateHeader
state header
// AUTOFUSE_SPLIT_FILE_END: TilingStateHeader
// AUTOFUSE_SPLIT_FILE_BEGIN: TilingSolverHeader
solver header
// AUTOFUSE_SPLIT_FILE_END: TilingSolverHeader
// AUTOFUSE_SPLIT_FILE_BEGIN: TilingApiHeader
api header
// AUTOFUSE_SPLIT_FILE_END: TilingApiHeader
// AUTOFUSE_SPLIT_FILE_BEGIN: ACubeKernelTilingWrapperHpp
wrapper header
// AUTOFUSE_SPLIT_FILE_END: ACubeKernelTilingWrapperHpp
// AUTOFUSE_SPLIT_FILE_BEGIN: solver_func
#include "autofuse_tiling_func_state.h"
#include "autofuse_tiling_func_solver.h"
int Solver() { return 0; }
// AUTOFUSE_SPLIT_FILE_END: solver_func
"""
    host_files = compile_adapter_module.write_split_host_sources(
        str(tmpdir), "demo_graph", host_code
    )

    assert not tmpdir.join("autofuse_tiling_func_common.h").check()
    assert tmpdir.join("autofuse_tiling_func_state.h").read() == "state header\n"
    assert tmpdir.join("autofuse_tiling_func_solver.h").read() == "solver header\n"
    assert tmpdir.join("autofuse_tiling_func_api.h").read() == "api header\n"
    assert tmpdir.join("cube_kernel_tiling_wrapper.h").read() == "wrapper header\n"
    cpp = tmpdir.join("demo_graph_tiling_func_solver_func.cpp").read()
    assert cpp.count('#include "autofuse_tiling_func_common.h"') == 0
    assert '#include "autofuse_tiling_func_state.h"' in cpp
    assert len(host_files) == 1


def test_write_split_host_sources_legacy_marker_injects_common(
    tmpdir, compile_adapter_module
):
    host_code = """// AUTOFUSE_SPLIT_FILE_BEGIN: TilingHead
common header
// AUTOFUSE_SPLIT_FILE_END: TilingHead
// AUTOFUSE_SPLIT_FILE_BEGIN: solver_func
int Solver() { return 0; }
// AUTOFUSE_SPLIT_FILE_END: solver_func
"""
    compile_adapter_module.write_split_host_sources(
        str(tmpdir), "demo_graph", host_code
    )

    cpp = tmpdir.join("demo_graph_tiling_func_solver_func.cpp").read()
    assert cpp.startswith('#include "autofuse_tiling_func_common.h"')


def test_write_split_host_sources_historical_split_keeps_cpp(
    tmpdir, compile_adapter_module
):
    host_code = """// AUTOFUSE_SPLIT_FILE_BEGIN: TilingBaseHeader
base
// AUTOFUSE_SPLIT_FILE_END: TilingBaseHeader
// AUTOFUSE_SPLIT_FILE_BEGIN: solver_func
int Solver() { return 0; }
// AUTOFUSE_SPLIT_FILE_END: solver_func
"""
    compile_adapter_module.write_split_host_sources(
        str(tmpdir), "demo_graph", host_code
    )

    cpp = tmpdir.join("demo_graph_tiling_func_solver_func.cpp").read()
    assert not cpp.startswith('#include "autofuse_tiling_func_common.h"')


@pytest.mark.parametrize(
    "header_key", ["TilingSolverHeader", "UnknownHeader", "UnknownHpp"]
)
def test_write_split_host_sources_rejects_undetermined_header_format(
    tmpdir, compile_adapter_module, header_key
):
    host_code = f"""// AUTOFUSE_SPLIT_FILE_BEGIN: {header_key}
header
// AUTOFUSE_SPLIT_FILE_END: {header_key}
// AUTOFUSE_SPLIT_FILE_BEGIN: solver_func
int Solver() {{ return 0; }}
// AUTOFUSE_SPLIT_FILE_END: solver_func
"""
    with pytest.raises(Exception):
        compile_adapter_module.write_split_host_sources(
            str(tmpdir), "demo_graph", host_code
        )


@pytest.mark.parametrize(
    "host_impl",
    [
        "\n".join(
            [
                "// AUTOFUSE_SPLIT_FILE_BEGIN: TilingHead",
                "struct CommonType {};",
                "// AUTOFUSE_SPLIT_FILE_END: solver_func",
            ]
        ),
        "\n".join(
            [
                "// AUTOFUSE_SPLIT_FILE_BEGIN: TilingHead",
                "// AUTOFUSE_SPLIT_FILE_BEGIN: solver_func",
                "// AUTOFUSE_SPLIT_FILE_END: solver_func",
                "// AUTOFUSE_SPLIT_FILE_END: TilingHead",
            ]
        ),
        "\n".join(
            [
                "// AUTOFUSE_SPLIT_FILE_BEGIN: TilingHead",
                "struct CommonType {};",
            ]
        ),
        "\n".join(
            [
                "// AUTOFUSE_SPLIT_FILE_BEGIN: TilingHead",
                "struct CommonType {};",
                "// AUTOFUSE_SPLIT_FILE_END: TilingHead",
                "// AUTOFUSE_SPLIT_FILE_BEGIN: solver_func",
                "int Solver() { return 0; }",
                "// AUTOFUSE_SPLIT_FILE_END: solver_func",
                "// AUTOFUSE_SPLIT_FILE_BEGIN: solver_func",
                "int Solver2() { return 0; }",
                "// AUTOFUSE_SPLIT_FILE_END: solver_func",
            ]
        ),
        "\n".join(
            [
                "// AUTOFUSE_SPLIT_FILE_BEGIN: TilingHead",
                "struct CommonType {};",
                "// AUTOFUSE_SPLIT_FILE_END: TilingHead",
                "// AUTOFUSE_SPLIT_FILE_BEGIN: ../solver",
                "int Solver() { return 0; }",
                "// AUTOFUSE_SPLIT_FILE_END: ../solver",
            ]
        ),
        "\n".join(
            [
                "// AUTOFUSE_SPLIT_FILE_BEGIN: TilingHead",
                "struct CommonType {};",
                "// AUTOFUSE_SPLIT_FILE_END: TilingHead",
                "// AUTOFUSE_SPLIT_FILE_BEGIN: a/b",
                "int Solver() { return 0; }",
                "// AUTOFUSE_SPLIT_FILE_END: a/b",
            ]
        ),
        "\n".join(
            [
                "// AUTOFUSE_SPLIT_FILE_BEGIN: TilingHead",
                "struct CommonType {};",
                "// AUTOFUSE_SPLIT_FILE_END: TilingHead",
                "// AUTOFUSE_SPLIT_FILE_BEGIN: a\\b",
                "int Solver() { return 0; }",
                "// AUTOFUSE_SPLIT_FILE_END: a\\b",
            ]
        ),
        "\n".join(
            [
                "// AUTOFUSE_SPLIT_FILE_BEGIN: TilingHead",
                "struct CommonType {};",
                "// AUTOFUSE_SPLIT_FILE_END: TilingHead",
                "// AUTOFUSE_SPLIT_FILE_BEGIN: ",
                "int Solver() { return 0; }",
                "// AUTOFUSE_SPLIT_FILE_END: ",
            ]
        ),
        "\n".join(
            [
                "// AUTOFUSE_SPLIT_FILE_BEGIN: solver_func",
                "int Solver() { return 0; }",
                "// AUTOFUSE_SPLIT_FILE_END: solver_func",
            ]
        ),
        "\n".join(
            [
                "// AUTOFUSE_SPLIT_FILE_BEGIN: TilingHead",
                "struct CommonType {};",
                "// AUTOFUSE_SPLIT_FILE_END: TilingHead",
            ]
        ),
    ],
)
def test_execute_compile_rejects_invalid_split_marker(
    compile_adapter_module, tmpdir, host_impl
):
    def fake_main(args):
        return None

    compile_adapter_module.ascendc_compile.main = fake_main
    args = _host_compile_args(tmpdir)

    with pytest.raises(compile_adapter_module.ascendc_compile.CompileError) as exc_info:
        compile_adapter_module.execute_compile(
            {
                "tiling_struct_code": "struct AutofuseTilingData {};",
                "host_impl_code": host_impl,
                "kernel_impl_code": None,
            },
            args,
        )

    assert "split host source" in str(
        exc_info.value
    ) or "invalid split host source key" in str(exc_info.value)


def test_host_compile_and_jit_compile_signatures_keep_stable(compile_adapter_module):
    assert list(inspect.signature(compile_adapter_module.jit_compile).parameters) == [
        "tiling_def",
        "host_tiling",
        "op_kernel",
        "argv",
    ]
    assert list(inspect.signature(compile_adapter_module.host_compile).parameters) == [
        "tiling_def_code",
        "tiling_impl_code",
        "argv",
    ]
