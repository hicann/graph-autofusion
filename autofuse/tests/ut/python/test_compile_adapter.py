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
import autofuse

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


def _scheme_a_sources():
    return {
        "tiling_struct_code": "struct AutofuseTilingData {};",
        "host_impl_code": _make_scheme_a_split_host_impl(),
        "kernel_impl_code": None,
    }


def _execute_scheme_a_host_compile(compile_adapter_module, tmpdir):
    args = _host_compile_args(tmpdir)
    compile_adapter_module.execute_compile(_scheme_a_sources(), args)
    return args


def _make_mspti_install(tmpdir):
    cann_root = tmpdir.mkdir("cann")
    mspti_dir = cann_root.mkdir("tools").mkdir("mspti")
    mspti_dir.mkdir("include").join("mspti.h").write("header")
    lib_dir = mspti_dir.mkdir("lib64")
    mspti_so = lib_dir.join("libmspti.so")
    mspti_so.write("library")
    return cann_root, mspti_dir, lib_dir, mspti_so


def _clear_cann_root_envs(monkeypatch):
    for env_name in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH", "ASCEND_HOME"):
        monkeypatch.delenv(env_name, raising=False)


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


def test_pgo_get_top_result_missing_file_returns_failure(
    compile_adapter_module, tmpdir
):
    missing_search = str(tmpdir.join("missing_search.txt"))

    assert compile_adapter_module.pgo_get_top_result(missing_search) == (
        None,
        None,
        None,
    )


def test_jit_compile_records_atrace_and_reports(compile_adapter_module, tmpdir, capsys):
    output_file = tmpdir.join("jit.so")
    argv = [f"--output_file={output_file}", f"--output_path={tmpdir}"]

    compile_adapter_module.jit_compile("tiling", "host", "kernel", argv)

    labels = [item[0] for item in compile_adapter_module.duration_records]
    assert [
        "InductorCompile",
        "jit_compile",
        "WriteHostSource",
        "autofuse",
    ] in labels
    assert [
        "InductorCompile",
        "jit_compile",
        "WriteDeviceSource",
        "autofuse",
    ] in labels
    assert [
        "InductorCompile",
        "jit_compile",
        "BuildCompiledArtifacts",
        "autofuse",
    ] in labels
    assert [
        "InductorCompile",
        "jit_compile",
        "AutoFuseCompileTotal",
        "autofuse",
    ] in labels
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
        "WriteHostSource",
        "autofuse",
    ] in labels
    assert [
        "InductorCompile",
        "host_compile",
        "BuildCompiledArtifacts",
        "autofuse",
    ] in labels
    assert [
        "InductorCompile",
        "host_compile",
        "AutoFuseCompileTotal",
        "autofuse",
    ] in labels
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
        "WriteDeviceSource",
        "autofuse",
    ] in labels
    assert [
        "InductorCompile",
        "kernel_compile",
        "BuildCompiledArtifacts",
        "autofuse",
    ] in labels
    assert [
        "InductorCompile",
        "kernel_compile",
        "AutoFuseCompileTotal",
        "autofuse",
    ] in labels


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


def test_execute_compile_merges_host_files_with_marker(compile_adapter_module, tmpdir):
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
    # host 编译为单个 cpp 源文件（cpp 段合并），header 段拆出独立 .h 供 include 引用。
    host_file = os.path.join(host_dir, "graph_tiling_func.cpp")
    assert captured["args"].host_files == host_file
    assert os.path.exists(os.path.join(host_dir, "autofuse_tiling_func_common.h"))
    with open(host_file) as f:
        merged = f.read()
    assert merged.count('#include "autofuse_tiling_func_common.h"') == 1
    assert 'extern "C" int Solver()' in merged
    assert 'extern "C" int TilingFunc()' in merged
    assert not os.path.exists(
        os.path.join(host_dir, "graph_tiling_func_solver_func.cpp")
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


def _make_pgo_split_host_impl(runner_key="PgoRunner"):
    return "\n".join(
        [
            "// AUTOFUSE_SPLIT_FILE_BEGIN: TilingHead",
            "struct CommonType {};",
            "// AUTOFUSE_SPLIT_FILE_END: TilingHead",
            "// AUTOFUSE_SPLIT_FILE_BEGIN: solver_func",
            'extern "C" int Solver() { return 0; }',
            "// AUTOFUSE_SPLIT_FILE_END: solver_func",
            f"// AUTOFUSE_SPLIT_FILE_BEGIN: {runner_key}",
            "int main() { return Solver(); }",
            f"// AUTOFUSE_SPLIT_FILE_END: {runner_key}",
        ]
    )


def _make_scheme_a_split_host_impl(
    runner_key="PgoRunner", device_key="PgoDeviceSource"
):
    return "\n".join(
        [
            "// AUTOFUSE_SPLIT_FILE_BEGIN: TilingHead",
            "struct CommonType {};",
            "// AUTOFUSE_SPLIT_FILE_END: TilingHead",
            "// AUTOFUSE_SPLIT_FILE_BEGIN: solver_func",
            'extern "C" int Solver() { return 0; }',
            "// AUTOFUSE_SPLIT_FILE_END: solver_func",
            f"// AUTOFUSE_SPLIT_FILE_BEGIN: {runner_key}",
            "int main() { return Solver(); }",
            f"// AUTOFUSE_SPLIT_FILE_END: {runner_key}",
            f"// AUTOFUSE_SPLIT_FILE_BEGIN: {device_key}",
            'extern "C" __global__ __aicore__ void graph() {}',
            f"// AUTOFUSE_SPLIT_FILE_END: {device_key}",
        ]
    )


def test_write_split_host_sources_excludes_exact_pgo_runner(
    compile_adapter_module, tmpdir
):
    host_dir = str(tmpdir.mkdir("host"))

    host_files = compile_adapter_module.write_split_host_sources(
        host_dir, "graph", _make_pgo_split_host_impl()
    )

    assert host_files == [os.path.join(host_dir, "graph_tiling_func_solver_func.cpp")]
    assert os.path.exists(os.path.join(host_dir, "graph_tiling_func_PgoRunner.cpp"))


def test_write_split_host_sources_keeps_similar_runner_key(
    compile_adapter_module, tmpdir
):
    host_dir = str(tmpdir.mkdir("host"))

    host_files = compile_adapter_module.write_split_host_sources(
        host_dir, "graph", _make_pgo_split_host_impl("PgoRunnerExtra")
    )

    assert host_files == [
        os.path.join(host_dir, "graph_tiling_func_solver_func.cpp"),
        os.path.join(host_dir, "graph_tiling_func_PgoRunnerExtra.cpp"),
    ]


def test_write_inductor_pgo_sources_classifies_exact_versioned_splits(
    compile_adapter_module, tmpdir
):
    host_dir = str(tmpdir.mkdir("host"))
    device_dir = str(tmpdir.mkdir("device"))

    normal_files, runner_file, device_file = (
        compile_adapter_module.write_inductor_pgo_sources(
            host_dir, device_dir, "graph", _make_scheme_a_split_host_impl()
        )
    )

    assert normal_files == [os.path.join(host_dir, "graph_tiling_func_solver_func.cpp")]
    assert runner_file == os.path.join(host_dir, "graph_tiling_func_PgoRunner.cpp")
    assert device_file == os.path.join(device_dir, "graph_pgo_device.cpp")
    assert all(
        "PgoRunner" not in path and "PgoDeviceSource" not in path
        for path in normal_files
    )
    assert "int main()" in open(runner_file).read()
    assert "__aicore__ void graph()" in open(device_file).read()


def test_write_inductor_pgo_sources_supports_final_split_headers(
    compile_adapter_module, tmpdir
):
    host_impl = """// AUTOFUSE_SPLIT_FILE_BEGIN: TilingStateHeader
state header
// AUTOFUSE_SPLIT_FILE_END: TilingStateHeader
// AUTOFUSE_SPLIT_FILE_BEGIN: solver_func
#include "autofuse_tiling_func_state.h"
int Solver() { return 0; }
// AUTOFUSE_SPLIT_FILE_END: solver_func
// AUTOFUSE_SPLIT_FILE_BEGIN: PgoRunner
int main() { return 0; }
// AUTOFUSE_SPLIT_FILE_END: PgoRunner
// AUTOFUSE_SPLIT_FILE_BEGIN: PgoDeviceSource
extern "C" __global__ __aicore__ void graph() {}
// AUTOFUSE_SPLIT_FILE_END: PgoDeviceSource
"""
    host_dir = str(tmpdir.mkdir("host"))
    device_dir = str(tmpdir.mkdir("device"))

    normal_files, runner_file, device_file = (
        compile_adapter_module.write_inductor_pgo_sources(
            host_dir, device_dir, "graph", host_impl
        )
    )

    assert not tmpdir.join("host", "autofuse_tiling_func_common.h").check()
    assert (
        tmpdir.join("host", "autofuse_tiling_func_state.h").read() == "state header\n"
    )
    assert (
        '#include "autofuse_tiling_func_common.h"' not in open(normal_files[0]).read()
    )
    assert '#include "autofuse_tiling_func_common.h"' not in open(runner_file).read()
    assert device_file == os.path.join(device_dir, "graph_pgo_device.cpp")


@pytest.mark.parametrize(
    "host_impl",
    [
        _make_pgo_split_host_impl(),
    ],
)
def test_write_inductor_pgo_sources_rejects_missing_pair(
    compile_adapter_module, tmpdir, host_impl
):
    host_dir = str(tmpdir.mkdir("host"))
    device_dir = str(tmpdir.mkdir("device"))

    with pytest.raises(
        compile_adapter_module.ascendc_compile.CompileError, match="PGO|Pgo"
    ):
        compile_adapter_module.write_inductor_pgo_sources(
            host_dir, device_dir, "graph", host_impl
        )


def test_write_split_host_sources_keeps_similar_pgo_keys(
    compile_adapter_module, tmpdir
):
    host_dir = str(tmpdir.mkdir("host"))
    host_impl = _make_scheme_a_split_host_impl("PgoRunnerExtra", "PgoDeviceSourceExtra")

    host_files = compile_adapter_module.write_split_host_sources(
        host_dir, "graph", host_impl
    )

    assert any("PgoRunnerExtra" in path for path in host_files)
    assert any("PgoDeviceSourceExtra" in path for path in host_files)


def test_execute_compile_prepares_scheme_a_sidecars_without_changing_host_api(
    compile_adapter_module, tmpdir, monkeypatch
):
    captured = {}

    def capture_args(args):
        captured["args"] = args

    compile_adapter_module.ascendc_compile.main = capture_args
    monkeypatch.setattr(
        compile_adapter_module.module,
        "get_inductor_pgo_mspti_config",
        lambda: ("/mspti", ["/mspti/libmspti.so"], ["-lmspti"]),
    )
    _execute_scheme_a_host_compile(compile_adapter_module, tmpdir)

    compiled_args = captured["args"]
    assert compiled_args.host_files == [
        os.path.join(str(tmpdir), "host", "graph_tiling_func_solver_func.cpp")
    ]
    assert compiled_args.pgo_runner_file.endswith("graph_tiling_func_PgoRunner.cpp")
    assert compiled_args.pgo_device_file.endswith("graph_pgo_device.cpp")
    assert compiled_args.pgo_mspti_config == (
        "/mspti",
        ["/mspti/libmspti.so"],
        ["-lmspti"],
    )


def test_execute_compile_scheme_a_without_mspti_keeps_pgo_proxy_runtime_linkage(
    compile_adapter_module, tmpdir, monkeypatch
):
    captured = {}

    def capture_args(args):
        captured["args"] = args

    compile_adapter_module.ascendc_compile.main = capture_args
    monkeypatch.setattr(
        compile_adapter_module.module, "get_inductor_pgo_mspti_config", lambda: None
    )
    _execute_scheme_a_host_compile(compile_adapter_module, tmpdir)

    compiled_args = captured["args"]
    assert compiled_args.pgo_runner_file.endswith("graph_tiling_func_PgoRunner.cpp")
    assert compiled_args.pgo_device_file.endswith("graph_pgo_device.cpp")
    assert compiled_args.pgo_mspti_config is None


def test_execute_compile_scheme_a_rejects_stage_all(
    compile_adapter_module, tmpdir, monkeypatch
):
    monkeypatch.setattr(
        compile_adapter_module.module,
        "get_inductor_pgo_mspti_config",
        lambda: ("/mspti", ["/mspti/libmspti.so"], ["-lmspti"]),
    )
    args = type(
        "Args",
        (),
        {
            "stage": "all",
            "temp_dir": str(tmpdir),
            "graph_name": "graph",
            "trace_stage": "jit_compile",
        },
    )()

    with pytest.raises(
        compile_adapter_module.ascendc_compile.CompileError, match="host_compile"
    ):
        compile_adapter_module.execute_compile(
            {
                "tiling_struct_code": "struct AutofuseTilingData {};",
                "host_impl_code": _make_scheme_a_split_host_impl(),
                "kernel_impl_code": 'extern "C" void kernel() {}',
            },
            args,
        )


def test_get_inductor_pgo_mspti_config_uses_ascend_toolkit_home(
    compile_adapter_module, tmpdir, monkeypatch
):
    _clear_cann_root_envs(monkeypatch)
    cann_root, mspti_dir, lib_dir, mspti_so = _make_mspti_install(tmpdir)
    monkeypatch.setenv("ASCEND_TOOLKIT_HOME", str(cann_root))

    config = compile_adapter_module.get_inductor_pgo_mspti_config()

    assert config == (
        os.path.realpath(str(mspti_dir)),
        [os.path.realpath(str(mspti_so))],
        [f"-L{os.path.realpath(str(lib_dir))}", "-lmspti"],
    )


def test_get_inductor_pgo_mspti_config_includes_optional_prof_common(
    compile_adapter_module, tmpdir, monkeypatch
):
    _clear_cann_root_envs(monkeypatch)
    cann_root = tmpdir.mkdir("cann")
    mspti_dir = cann_root.mkdir("tools").mkdir("mspti")
    mspti_dir.mkdir("include").join("mspti.h").write("header")
    lib_dir = mspti_dir.mkdir("lib64")
    mspti_so = lib_dir.join("libmspti.so")
    prof_common_so = lib_dir.join("libprof_common.so")
    mspti_so.write("library")
    prof_common_so.write("library")
    monkeypatch.setenv("ASCEND_TOOLKIT_HOME", str(cann_root))

    config = compile_adapter_module.get_inductor_pgo_mspti_config()

    assert config == (
        os.path.realpath(str(mspti_dir)),
        [os.path.realpath(str(prof_common_so)), os.path.realpath(str(mspti_so))],
        [f"-L{os.path.realpath(str(lib_dir))}", "-lmspti", "-lprof_common"],
    )


def test_get_inductor_pgo_mspti_config_rejects_incomplete_cann_root(
    compile_adapter_module, tmpdir, monkeypatch
):
    _clear_cann_root_envs(monkeypatch)
    cann_root = tmpdir.mkdir("cann")
    mspti_dir = cann_root.mkdir("tools").mkdir("mspti")
    mspti_dir.mkdir("include").join("mspti.h").write("header")
    monkeypatch.setenv("ASCEND_TOOLKIT_HOME", str(cann_root))

    assert compile_adapter_module.get_inductor_pgo_mspti_config() is None


def test_get_inductor_pgo_mspti_config_uses_current_cann_root(
    compile_adapter_module, tmpdir, monkeypatch
):
    _clear_cann_root_envs(monkeypatch)
    cann_root, mspti_dir, lib_dir, mspti_so = _make_mspti_install(tmpdir)
    package_dir = cann_root.mkdir("python").mkdir("site-packages").mkdir("autofuse")
    monkeypatch.setattr(
        compile_adapter_module.ascendc_compile,
        "__file__",
        str(package_dir.join("ascendc_compile.py")),
        raising=False,
    )

    assert compile_adapter_module.get_inductor_pgo_mspti_config() == (
        os.path.realpath(str(mspti_dir)),
        [os.path.realpath(str(mspti_so))],
        [f"-L{os.path.realpath(str(lib_dir))}", "-lmspti"],
    )


def test_get_inductor_pgo_mspti_config_uses_cann_root_without_importing_codegen(
    compile_adapter_module, tmpdir, monkeypatch
):
    _clear_cann_root_envs(monkeypatch)
    cann_root, mspti_dir, lib_dir, mspti_so = _make_mspti_install(tmpdir)
    monkeypatch.setenv("ASCEND_TOOLKIT_HOME", str(cann_root))

    asc_codegen_compile = types.SimpleNamespace(
        pgo_get_mspti_config=lambda: (_ for _ in ()).throw(
            AssertionError("asc_codegen_compile should not be used")
        )
    )
    monkeypatch.setattr(
        autofuse, "asc_codegen_compile", asc_codegen_compile, raising=False
    )

    assert compile_adapter_module.get_inductor_pgo_mspti_config() == (
        os.path.realpath(str(mspti_dir)),
        [os.path.realpath(str(mspti_so))],
        [f"-L{os.path.realpath(str(lib_dir))}", "-lmspti"],
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
