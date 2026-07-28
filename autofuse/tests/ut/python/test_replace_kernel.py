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

import importlib.util
import os
import sys
import types

import pytest


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    )
)
PYTHON_DIR = os.path.join(BASE_DIR, "autofuse/compiler/python")
MODULE_NAME = "autofuse.compiler.python.asc_codegen_compile"
MODULE_PATH = os.path.join(PYTHON_DIR, "asc_codegen_compile.py")


class _SimpleNamespace(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _DummyLogger(object):
    @staticmethod
    def info(*args, **kwargs):
        return None

    @staticmethod
    def error(*args, **kwargs):
        return None

    @staticmethod
    def warning(*args, **kwargs):
        return None


def _stub_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


@pytest.fixture(scope="module")
def asc_codegen_compile_module():
    _stub_module("tbe")
    _stub_module("tbe.common")
    _stub_module("tbe.common.buildcfg", get_current_build_config=lambda: {})
    _stub_module("tbe.common.utils")
    _stub_module(
        "tbe.common.utils.log",
        info=_DummyLogger.info,
        error=_DummyLogger.error,
        warning=_DummyLogger.warning,
    )
    _stub_module(
        "tbe.common.utils.op_tiling", do_op_tiling=lambda *args, **kwargs: None
    )
    _stub_module(
        "tbe.common.context",
        get_context=lambda: _SimpleNamespace(get_compile_info=lambda: {}),
    )
    _stub_module("tbe.tikcpp")
    _stub_module(
        "tbe.tikcpp.compile_op",
        CommonUtility=_SimpleNamespace(print_compile_log=lambda *args, **kwargs: None),
        AscendCLogLevel=_SimpleNamespace(
            LOG_ERROR="error", LOG_DEBUG="debug", LOG_WARNING="warning"
        ),
    )
    _stub_module(
        "tbe.tikcpp.get_op_tiling",
        TilingInfo=object,
        _change_param_name_to_name=lambda *args, **kwargs: None,
        gen_static_shape_v2=lambda *args, **kwargs: None,
    )
    sys.modules["tbe.tikcpp"].OpInfo = object
    _stub_module("asc_op_compile_base")
    _stub_module("asc_op_compile_base.common")
    _stub_module("asc_op_compile_base.common.platform")
    _stub_module(
        "asc_op_compile_base.common.platform.platform_info",
        get_soc_spec=lambda *args, **kwargs: None,
    )

    package = _stub_module("autofuse")
    compiler_pkg = _stub_module("autofuse.compiler")
    python_pkg = _stub_module("autofuse.compiler.python")
    package.compiler = compiler_pkg
    compiler_pkg.python = python_pkg

    package_prefix = MODULE_NAME.rsplit(".", 1)[0]
    _stub_module(
        package_prefix + ".pyautofuse",
        Schedule=object,
        CodeGen=object,
        ascir=_SimpleNamespace(),
    )
    _stub_module(
        package_prefix + ".ascbc_kernel_compile",
        ascbc_kernel_compile=lambda *args, **kwargs: ("kernel.o", "kernel.json"),
        camel_to_snake=lambda value: value,
    )
    _stub_module(
        package_prefix + ".compile_adapter",
        get_pgo_env_flag=lambda: False,
        get_pgo_topn=lambda: 5,
    )

    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules[MODULE_NAME] = module
    return module


def test_get_replace_kernel_root_returns_none_when_env_is_missing(
    monkeypatch, asc_codegen_compile_module
):
    monkeypatch.delenv("AUTOFUSE_DFX_FLAGS", raising=False)
    assert asc_codegen_compile_module.get_replace_kernel_root() is None


def test_replace_device_kernel_replaces_exact_kernel_file(
    tmpdir, asc_codegen_compile_module
):
    replace_root = tmpdir.mkdir("replace_root")
    device_build_dir = tmpdir.ensure("build", "device", dir=True)

    replace_root.join("demo_graph_op_kernel.cpp").write("new device kernel")
    dst = device_build_dir.join("demo_graph_op_kernel.cpp")
    dst.write("old device kernel")

    asc_codegen_compile_module.replace_device_kernel(
        str(replace_root), str(device_build_dir), "demo_graph"
    )

    assert dst.read() == "new device kernel"


def test_replace_host_files_replaces_host_tiling_group_and_removes_stale_files(
    tmpdir, asc_codegen_compile_module
):
    replace_root = tmpdir.mkdir("replace_root")
    source_dir = replace_root.ensure("a", "b", "whatever", dir=True)
    host_build_dir = tmpdir.ensure("build", "host", dir=True)

    source_dir.join("demo_graph_tiling_func_0.cpp").write("new0")
    source_dir.join("demo_graph_tiling_func_1.cpp").write("new1")
    host_build_dir.join("demo_graph_tiling_func_old.cpp").write("stale")

    asc_codegen_compile_module.replace_host_files(
        str(replace_root), str(host_build_dir), "demo_graph"
    )

    host_files = sorted(
        [
            path.basename
            for path in host_build_dir.visit(fil="demo_graph*tiling_func*.cpp")
        ]
    )
    assert host_files == [
        "demo_graph_tiling_func_0.cpp",
        "demo_graph_tiling_func_1.cpp",
    ]


def test_find_host_replace_source_dir_raises_when_multiple_dirs_match(
    tmpdir, asc_codegen_compile_module
):
    replace_root = tmpdir.mkdir("replace_root")
    dir1 = replace_root.mkdir("x")
    dir2 = replace_root.mkdir("y")
    dir1.join("demo_graph_tiling_func_a.cpp").write("a")
    dir2.join("demo_graph_tiling_func_b.cpp").write("b")

    with pytest.raises(RuntimeError):
        asc_codegen_compile_module.find_host_replace_source_dir(
            str(replace_root), "demo_graph"
        )


def test_replace_host_files_allows_missing_optional_headers(
    tmpdir, asc_codegen_compile_module
):
    replace_root = tmpdir.mkdir("replace_root")
    source_dir = replace_root.mkdir("nested")
    host_build_dir = tmpdir.ensure("build", "host", dir=True)

    source_dir.join("demo_graph_tiling_func_0.cpp").write("new0")

    asc_codegen_compile_module.replace_host_files(
        str(replace_root), str(host_build_dir), "demo_graph"
    )

    assert host_build_dir.join("demo_graph_tiling_func_0.cpp").read() == "new0"
    assert not host_build_dir.join("autofuse_tiling_data.h").check()
    assert not host_build_dir.join("autofuse_tiling_func_common.h").check()


def test_replace_host_files_skips_same_file_copy_when_source_is_host_dir(
    tmpdir, asc_codegen_compile_module
):
    replace_root = tmpdir.mkdir("replace_root")
    host_build_dir = replace_root.ensure("host", dir=True)
    host_build_dir.join("demo_graph_tiling_func_0.cpp").write("same cpp")
    host_build_dir.join("autofuse_tiling_data.h").write("same header")

    asc_codegen_compile_module.replace_host_files(
        str(replace_root), str(host_build_dir), "demo_graph"
    )

    assert host_build_dir.join("demo_graph_tiling_func_0.cpp").read() == "same cpp"
    assert host_build_dir.join("autofuse_tiling_data.h").read() == "same header"


def test_replace_host_files_copies_autofuse_cube_tiling_data_when_present(
    tmpdir, asc_codegen_compile_module
):
    replace_root = tmpdir.mkdir("replace_root")
    source_dir = replace_root.mkdir("nested")
    host_build_dir = tmpdir.ensure("build", "host", dir=True)

    source_dir.join("demo_graph_tiling_func_0.cpp").write("new0")
    source_dir.join("autofuse_cube_tiling_data.h").write("cube")

    asc_codegen_compile_module.replace_host_files(
        str(replace_root), str(host_build_dir), "demo_graph"
    )

    assert host_build_dir.join("autofuse_cube_tiling_data.h").read() == "cube"


def test_host_miss_device_hit_keeps_device_replacement_independent(
    tmpdir, asc_codegen_compile_module
):
    replace_root = tmpdir.mkdir("replace_root")
    device_build_dir = tmpdir.ensure("build", "device", dir=True)
    host_build_dir = tmpdir.ensure("build", "host", dir=True)

    replace_root.join("demo_graph_op_kernel.cpp").write("new device kernel")
    device_build_dir.join("demo_graph_op_kernel.cpp").write("old device kernel")

    asc_codegen_compile_module.replace_device_kernel(
        str(replace_root), str(device_build_dir), "demo_graph"
    )
    asc_codegen_compile_module.replace_host_files(
        str(replace_root), str(host_build_dir), "demo_graph"
    )

    assert (
        device_build_dir.join("demo_graph_op_kernel.cpp").read() == "new device kernel"
    )
    assert host_build_dir.listdir() == []


def test_device_miss_host_hit_keeps_host_replacement_independent(
    tmpdir, asc_codegen_compile_module
):
    replace_root = tmpdir.mkdir("replace_root")
    source_dir = replace_root.mkdir("nested")
    device_build_dir = tmpdir.ensure("build", "device", dir=True)
    host_build_dir = tmpdir.ensure("build", "host", dir=True)

    source_dir.join("demo_graph_tiling_func_0.cpp").write("new0")
    device_build_dir.join("demo_graph_op_kernel.cpp").write("old device kernel")

    asc_codegen_compile_module.replace_device_kernel(
        str(replace_root), str(device_build_dir), "demo_graph"
    )
    asc_codegen_compile_module.replace_host_files(
        str(replace_root), str(host_build_dir), "demo_graph"
    )

    assert (
        device_build_dir.join("demo_graph_op_kernel.cpp").read() == "old device kernel"
    )
    assert host_build_dir.join("demo_graph_tiling_func_0.cpp").read() == "new0"


def _mock_host_replacement_compile_flow(
    asc_codegen_compile_module, monkeypatch, tmpdir, calls
):
    def fake_static_shape_compile(**kwargs):
        if "use_cv_common" in kwargs:
            kwargs["use_cv_common"][0] = True
        calls.append(("static_shape_compile", kwargs["graph_name"], kwargs))

    monkeypatch.setattr(
        asc_codegen_compile_module,
        "get_graph_basic_info",
        lambda params, args: ("demo_graph", 1, 1, True, False, {}),
    )
    monkeypatch.setattr(
        asc_codegen_compile_module,
        "create_compile_dirs",
        lambda temp_dir: (
            str(tmpdir.ensure("device", dir=True)),
            str(tmpdir.ensure("host", dir=True)),
        ),
    )
    monkeypatch.setattr(
        asc_codegen_compile_module,
        "generate_device_and_host_code",
        lambda **kwargs: ("kernel", {"k": "v"}),
    )
    monkeypatch.setattr(
        asc_codegen_compile_module,
        "is_static_compile",
        lambda params, tiling_func_srcs: True,
    )
    monkeypatch.setattr(
        asc_codegen_compile_module,
        "ascbc_matmul_kernel_tiling_pro",
        lambda *args, **kwargs: kwargs["use_cv_common"].__setitem__(0, True),
    )
    monkeypatch.setattr(
        asc_codegen_compile_module, "static_shape_compile", fake_static_shape_compile
    )
    monkeypatch.setattr(
        asc_codegen_compile_module,
        "get_replace_kernel_root",
        lambda: tmpdir.ensure("replace_root", dir=True),
    )
    monkeypatch.setattr(
        asc_codegen_compile_module,
        "replace_host_files",
        lambda root, host_dir, graph_name: calls.append(
            ("replace_host", host_dir, graph_name)
        ),
    )
    monkeypatch.setattr(
        asc_codegen_compile_module,
        "replace_kernel",
        lambda *args, **kwargs: calls.append(("replace_device",)),
    )
    monkeypatch.setattr(
        asc_codegen_compile_module,
        "ascbc_kernel_compile",
        lambda *args, **kwargs: ("kernel.o", "kernel.json"),
    )
    monkeypatch.setattr(
        asc_codegen_compile_module,
        "asc_graph_compile_post",
        lambda *args, **kwargs: calls.append(("post", args[0])),
    )
    monkeypatch.setattr(
        asc_codegen_compile_module, "timestamp_set", lambda *args, **kwargs: None
    )


def test_host_replacement_happens_once_before_static_shape_compile_and_uses_final_host_dir(
    monkeypatch, tmpdir, asc_codegen_compile_module
):
    calls = []
    host_cv_common_dir = tmpdir.ensure("host", "cv_common", dir=True)
    _mock_host_replacement_compile_flow(
        asc_codegen_compile_module, monkeypatch, tmpdir, calls
    )

    asc_codegen_compile_module.asc_graph_compile(
        "arg0",
        "kernel_name",
        temp_dir=str(tmpdir),
        params={
            "schedule_results": object(),
            "vector_core_num": 1,
            "impl_mode": None,
        },
    )

    replace_calls = [item for item in calls if item[0] == "replace_host"]
    assert len(replace_calls) == 1
    assert replace_calls[0][1] == str(host_cv_common_dir)
    order = [item[0] for item in calls]
    assert order.index("replace_host") < order.index("static_shape_compile")
    static_shape_calls = [item for item in calls if item[0] == "static_shape_compile"]
    assert static_shape_calls[0][2]["vector_core_num"] == 1


class _LogCapture(object):
    def __init__(self):
        self.messages = []

    def info(self, message, *args):
        if args:
            message = message % args
        self.messages.append(message)


def test_replace_host_files_logs_source_target_removed_and_copied(
    monkeypatch, tmpdir, asc_codegen_compile_module
):
    replace_root = tmpdir.mkdir("replace_root")
    source_dir = replace_root.ensure("nested", dir=True)
    host_build_dir = tmpdir.ensure("build", "host", dir=True)
    log_capture = _LogCapture()

    source_dir.join("demo_graph_tiling_func_0.cpp").write("new0")
    source_dir.join("autofuse_tiling_data.h").write("tiling")
    host_build_dir.join("demo_graph_tiling_func_old.cpp").write("stale")

    monkeypatch.setattr(asc_codegen_compile_module, "logger", log_capture)

    asc_codegen_compile_module.replace_host_files(
        str(replace_root), str(host_build_dir), "demo_graph"
    )

    assert any(
        "replace host source dir:" in message for message in log_capture.messages
    )
    assert any(
        "replace host target dir:" in message for message in log_capture.messages
    )
    assert any(
        "cleanup stale host tiling files:" in message
        for message in log_capture.messages
    )
    assert any("copied host files:" in message for message in log_capture.messages)


def test_find_host_replace_source_dir_error_message_contains_conflict_dirs(
    tmpdir, asc_codegen_compile_module
):
    replace_root = tmpdir.mkdir("replace_root")
    dir1 = replace_root.mkdir("x")
    dir2 = replace_root.mkdir("y")
    dir1.join("demo_graph_tiling_func_a.cpp").write("a")
    dir2.join("demo_graph_tiling_func_b.cpp").write("b")

    with pytest.raises(RuntimeError) as exc_info:
        asc_codegen_compile_module.find_host_replace_source_dir(
            str(replace_root), "demo_graph"
        )

    error_message = str(exc_info.value)
    assert str(dir1) in error_message
    assert str(dir2) in error_message


def test_replace_host_files_does_not_touch_infershape_cpp(
    tmpdir, asc_codegen_compile_module
):
    replace_root = tmpdir.mkdir("replace_root")
    source_dir = replace_root.mkdir("nested")
    host_build_dir = tmpdir.ensure("build", "host", dir=True)

    source_dir.join("demo_graph_tiling_func_0.cpp").write("new0")
    source_dir.join("demo_graph_infershape.cpp").write("new infershape")
    host_build_dir.join("demo_graph_infershape.cpp").write("old infershape")

    asc_codegen_compile_module.replace_host_files(
        str(replace_root), str(host_build_dir), "demo_graph"
    )

    assert host_build_dir.join("demo_graph_tiling_func_0.cpp").read() == "new0"
    assert host_build_dir.join("demo_graph_infershape.cpp").read() == "old infershape"


def test_ascbc_host_compile_uses_32_jobs_for_tiling_func_split_compile(
    monkeypatch, tmpdir, asc_codegen_compile_module
):
    host_build_dir = tmpdir.mkdir("host")
    commands = []

    class FakeCompletedProcess(object):
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, capture_output, text):
        commands.append(cmd)
        return FakeCompletedProcess()

    monkeypatch.setattr(asc_codegen_compile_module.subprocess, "run", fake_run)

    asc_codegen_compile_module.ascbc_host_compile(
        asc_codegen_compile_module.HostCompileContext(
            "demo_graph", "demo_kernel", str(host_build_dir), True, True, False
        )
    )

    assert commands[1] == ["make", "-C", "./", "-j", "32"]
