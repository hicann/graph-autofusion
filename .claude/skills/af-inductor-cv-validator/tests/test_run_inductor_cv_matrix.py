# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_inductor_cv_matrix.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("run_inductor_cv_matrix", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_fake_cann(tmp_path: Path) -> Path:
    cann = tmp_path / "cann"
    (cann / "lib64").mkdir(parents=True)
    (cann / "x86_64-linux" / "include").mkdir(parents=True)
    opp = cann / "opp"
    opp.mkdir()
    (cann / "set_env.sh").write_text(
        'export ASCEND_HOME_PATH="$PWD"\n'
        'export ASCEND_AICPU_PATH="$PWD"\n'
        'export ASCEND_OPP_PATH="$PWD/opp"\n'
        'export TOOLCHAIN_HOME="$PWD/toolkit"\n'
        'export LD_LIBRARY_PATH="$PWD/lib64:${LD_LIBRARY_PATH}"\n',
        encoding="utf-8",
    )
    return cann


def write_fake_matmul_header(cann: Path, text: str) -> Path:
    header = (
        cann
        / "opp"
        / "built-in"
        / "op_impl"
        / "ai_core"
        / "tbe"
        / "impl"
        / "ops_nn"
        / "ascendc"
        / "mat_mul_v3"
        / "arch35"
        / "mat_mul_v3_tiling_key_public.h"
    )
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text(text, encoding="utf-8")
    return header


def make_fake_torchair(tmp_path: Path) -> Path:
    experimental = tmp_path / "torchair" / "experimental"
    package = experimental / "_inductor_npu_ext" / "python" / "inductor_npu_ext"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    return experimental


def make_fake_builtin_ascendc(tmp_path: Path) -> Path:
    site = tmp_path / "site"
    torch_pkg = site / "torch"
    ascendc_pkg = site / "torch_npu" / "_inductor" / "ascendc"
    torch_pkg.mkdir(parents=True)
    ascendc_pkg.mkdir(parents=True)
    (torch_pkg / "__init__.py").write_text(
        "__version__ = 'fake'\n"
        "class _Npu:\n"
        "    @staticmethod\n"
        "    def is_available():\n"
        "        return True\n"
        "npu = _Npu()\n",
        encoding="utf-8",
    )
    (site / "torch_npu" / "__init__.py").write_text("", encoding="utf-8")
    (site / "torch_npu" / "_inductor" / "__init__.py").write_text("", encoding="utf-8")
    (ascendc_pkg / "__init__.py").write_text("", encoding="utf-8")
    (ascendc_pkg / "config.py").write_text(
        "disable_canfuse = False\n"
        "enable_matmul_fuse = True\n"
        "def get_cached_soc_version():\n"
        "    return 'fake-soc'\n",
        encoding="utf-8",
    )
    return site


def test_import_probe_prefers_builtin_ascendc_backend(tmp_path: Path):
    runner = load_runner()
    site = make_fake_builtin_ascendc(tmp_path)
    env = {
        "PYTHONPATH": str(site),
        "TORCHINDUCTOR_NPU_BACKEND": "ascendc",
    }

    probe = runner.import_probe(env)

    assert probe.get("probe_exit_code") == 0
    backend = probe.get("npu_backend_package") or {}
    assert backend.get("ok") is True
    assert backend.get("kind") == "torch_npu._inductor.ascendc"
    assert backend.get("enable_matmul_fuse") is True


def test_fragment08_uses_2d_mm_inputs_to_avoid_view_fallback():
    runner = load_runner()
    case = next(item for item in runner.MATRIX if item.name == "fragment08_mm_add_exp")

    assert case.shape == {"x": [16384, 26], "w": [26, 256], "bias": [1, 256]}
    assert case.variants == ({"m": 16384, "k": 26, "n": 256},)


def test_fragment08_expects_cv_fused_non_ub_mode():
    runner = load_runner()
    case = next(item for item in runner.MATRIX if item.name == "fragment08_mm_add_exp")

    assert case.fusion_mode == 0
    assert case.ub_mode == 0


def test_core_mode_expectations_match_current_tiling_results():
    runner = load_runner()
    by_name = {case.name: case for case in runner.MATRIX}

    cv_static_mm_small_ub = by_name.get("cv_static_mm_small_ub")
    cv_dynamic_mm_small_ub = by_name.get("cv_dynamic_mm_small_ub")
    cv_static_bmm_to_mul_k1_mix2 = by_name.get("cv_static_bmm_to_mul_k1_mix2")
    cv_dynamic_bmm_to_mul_k1_mix2 = by_name.get("cv_dynamic_bmm_to_mul_k1_mix2")

    assert cv_static_mm_small_ub is not None
    assert cv_dynamic_mm_small_ub is not None
    assert cv_static_bmm_to_mul_k1_mix2 is not None
    assert cv_dynamic_bmm_to_mul_k1_mix2 is not None
    assert cv_static_mm_small_ub.ub_mode == 0
    assert cv_dynamic_mm_small_ub.ub_mode == 0
    assert cv_static_bmm_to_mul_k1_mix2.mix_mode == 1
    assert cv_dynamic_bmm_to_mul_k1_mix2.mix_mode == 1


def test_matrix_includes_historical_template_mixed_dtype_and_extended_cases():
    runner = load_runner()
    names = {case.name for case in runner.MATRIX}

    assert len(names) > 40
    for name in (
        "cv_static_mm_fallback_stream_k",
        "cv_dynamic_bmm_high_level_iter_batch",
        "cv_static_mixed_fp32_mm_cast_fp16_mul_add",
        "cv_static_mm_compare_fp32_bf16_brc",
        "cv_dynamic_vv_large_cast_chain",
        "cv_static_mm_add_row_large_n1001",
    ):
        assert name in names


def test_cv_cases_have_vv_only_variants_for_ab_comparison():
    runner = load_runner()
    by_name = {case.name: case for case in runner.MATRIX}

    cv_case = by_name["cv_static_mm_512_non_ub"]
    vv_case = by_name["vv_only_cv_static_mm_512_non_ub"]

    assert cv_case.mode_variant == "cv"
    assert vv_case.mode_variant == "vv_only"
    assert vv_case.kind == cv_case.kind
    assert vv_case.fusion_mode is None
    assert vv_case.ub_mode is None
    assert vv_case.expect_cv is False


def test_selected_cases_supports_family_and_mode_variant_filters():
    runner = load_runner()

    by_family = runner.selected_cases("template_coverage")
    by_mode = runner.selected_cases("vv_only")

    assert by_family
    assert all(case.family == "template_coverage" for case in by_family)
    assert by_mode
    assert all(case.mode_variant == "vv_only" for case in by_mode)


def test_vv_only_case_does_not_require_or_allow_cv_artifacts():
    runner = load_runner()
    case = next(
        item for item in runner.MATRIX if item.name == "vv_only_cv_static_mm_512_non_ub"
    )
    result = {
        "name": case.name,
        "ok": True,
        "stage": "summary",
        "artifact_paths": {
            "asc_kernel": [],
            "wrapper": [],
            "kernel_so": [],
            "dump_dirs": [],
        },
        "cv_artifacts": {"wrapper": []},
        "fusion_mode": None,
        "ub_mode": None,
        "mix_mode": None,
    }

    runner.validate_case_result(case, result)

    assert result.get("ok") is True

    result_with_cv = {
        **result,
        "ok": True,
        "cv_artifacts": {"wrapper": ["w"]},
        "fusion_mode": 0,
    }
    runner.validate_case_result(case, result_with_cv)

    assert result_with_cv.get("ok") is False
    assert "unexpected CV artifact" in result_with_cv.get("error", "")


def test_prepare_case_env_removes_matmul_debug_for_vv_only(tmp_path: Path):
    runner = load_runner()
    cv_case = next(
        item for item in runner.MATRIX if item.name == "cv_static_mm_small_ub"
    )
    vv_case = next(
        item for item in runner.MATRIX if item.name == "vv_only_cv_static_mm_small_ub"
    )
    run_dir = tmp_path / "run"
    base_env = {"TORCHINDUCTOR_NPU_EXT_DEBUG": "lowering+matmul+nocat"}

    cv_env = runner.prepare_case_env(base_env, cv_case, run_dir)
    vv_env = runner.prepare_case_env(base_env, vv_case, run_dir)

    assert "matmul" in cv_env["TORCHINDUCTOR_NPU_EXT_DEBUG"].split("+")
    assert "matmul" not in vv_env.get("TORCHINDUCTOR_NPU_EXT_DEBUG", "").split("+")
    assert "lowering" in vv_env["TORCHINDUCTOR_NPU_EXT_DEBUG"].split("+")


def test_tensor_metrics_supports_bool_outputs(monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    torch = pytest.importorskip("torch")
    runner = load_runner()
    actual = torch.tensor([[True, False], [False, True]])
    expected = torch.tensor([[True, True], [False, True]])

    metrics = runner.tensor_metrics(torch, actual, expected, 0.0, 0.0)

    assert metrics["fail_count"] == 1
    assert metrics["ok"] is False


def test_timeout_stream_to_text_accepts_bytes_and_none():
    runner = load_runner()

    assert runner.timeout_stream_to_text(b"abc\xff") == "abc�"
    assert runner.timeout_stream_to_text("abc") == "abc"
    assert runner.timeout_stream_to_text(None) == ""


def test_mandatory_mixed_dtype_case_has_short_timeout():
    runner = load_runner()
    case = next(
        item
        for item in runner.MATRIX
        if item.name == "cv_static_mixed_fp32_mm_cast_fp16_mul_add"
    )

    assert case.timeout <= 180


def test_case_progress_line_includes_index_name_and_timeout():
    runner = load_runner()
    case = next(
        item
        for item in runner.MATRIX
        if item.name == "cv_static_mixed_fp32_mm_cast_fp16_mul_add"
    )

    line = runner.case_progress_line("START", 3, 15, case, 180)

    assert line == "[3/15] START cv_static_mixed_fp32_mm_cast_fp16_mul_add timeout=180s"


def test_matrix_plan_records_family_mode_variant_and_expected_template(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    cann = make_fake_cann(tmp_path)
    torchair = make_fake_torchair(tmp_path)
    run_dir = repo / "temp" / "inductor_cv_validation" / "dry_run_fields"

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(repo),
            "--cann-toolkit",
            str(cann),
            "--torchair-experimental",
            str(torchair),
            "--run-dir",
            str(run_dir),
            "--mode",
            "dry-run",
            "--case-filter",
            "cv_static_mm_fallback_stream_k",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    plan = json.loads(
        (run_dir / "logs" / "matrix_plan.json").read_text(encoding="utf-8")
    )
    first = plan[0]
    assert first.get("name") == "cv_static_mm_fallback_stream_k"
    assert first.get("family") == "template_coverage"
    assert first.get("mode_variant") == "cv"
    assert first.get("expected_template") == "stream_k"


def test_env_probe_records_portable_runner_inputs(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    cann = make_fake_cann(tmp_path)
    torchair = make_fake_torchair(tmp_path)
    run_dir = repo / "temp" / "inductor_cv_validation" / "fresh"

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(repo),
            "--cann-toolkit",
            str(cann),
            "--torchair-experimental",
            str(torchair),
            "--run-dir",
            str(run_dir),
            "--mode",
            "env-probe",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (run_dir / "script" / "run_inductor_cv_matrix.py").is_file()
    assert (run_dir / "logs" / "matrix_plan.json").is_file()
    env_report = json.loads((run_dir / "logs" / "env.log").read_text(encoding="utf-8"))
    assert env_report.get("repo_root") == str(repo.resolve())
    assert env_report.get("cann_selected") == str(cann.resolve())
    assert env_report.get("torchair_experimental") == str(torchair.resolve())
    assert (env_report.get("argv") or [""])[0].endswith("run_inductor_cv_matrix.py")
    assert (env_report.get("args") or {}).get("mode") == "env-probe"
    assert (env_report.get("probe_env_keys") or {}).get(
        "TORCHINDUCTOR_NPU_BACKEND"
    ) == "ascendc"
    assert (env_report.get("matmul_v3_header_probe") or {}).get("ok") is False


def test_matmul_v3_header_probe_reports_missing_required_macros(tmp_path: Path):
    runner = load_runner()
    cann = make_fake_cann(tmp_path)
    header = write_fake_matmul_header(
        cann, "#define MAT_MUL_BASIC 0\n#define MAT_MUL_SLICE 5\n"
    )
    env = {"ASCEND_OPP_PATH": str(cann / "opp")}

    probe = runner.matmul_v3_header_probe(env, cann)

    assert probe.get("path") == str(header)
    assert probe.get("exists") is True
    assert probe.get("ok") is False
    assert probe.get("missing_macros") == [
        "MAT_MUL_BASIC_SPLIT_K",
        "MAT_MUL_SK_SPLIT_K",
    ]


def test_matmul_v3_header_probe_passes_when_required_macros_exist(tmp_path: Path):
    runner = load_runner()
    cann = make_fake_cann(tmp_path)
    write_fake_matmul_header(
        cann,
        "#define MAT_MUL_SLICE 5\n#define MAT_MUL_BASIC_SPLIT_K 6\n#define MAT_MUL_SK_SPLIT_K 7\n",
    )
    env = {"ASCEND_OPP_PATH": str(cann / "opp")}

    probe = runner.matmul_v3_header_probe(env, cann)

    assert probe.get("ok") is True
    assert probe.get("missing_macros") == []


def test_runtime_env_snapshot_includes_cann_paths(monkeypatch):
    runner = load_runner()
    monkeypatch.setenv("ASCEND_HOME_PATH", "/cann")
    monkeypatch.setenv("ASCEND_AICPU_PATH", "/cann")
    monkeypatch.setenv("ASCEND_OPP_PATH", "/cann/opp")
    monkeypatch.setenv("TOOLCHAIN_HOME", "/cann/toolkit")

    snapshot = runner.runtime_env_snapshot()

    assert snapshot.get("ASCEND_HOME_PATH") == "/cann"
    assert snapshot.get("ASCEND_AICPU_PATH") == "/cann"
    assert snapshot.get("ASCEND_OPP_PATH") == "/cann/opp"
    assert snapshot.get("TOOLCHAIN_HOME") == "/cann/toolkit"


def test_run_mode_blocks_cv_cases_when_matmul_header_lacks_required_macros(
    tmp_path: Path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    cann = make_fake_cann(tmp_path)
    torchair = make_fake_torchair(tmp_path)
    run_dir = repo / "temp" / "inductor_cv_validation" / "missing_header_macros"

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(repo),
            "--cann-toolkit",
            str(cann),
            "--torchair-experimental",
            str(torchair),
            "--run-dir",
            str(run_dir),
            "--mode",
            "run",
            "--case-filter",
            "cv_static_mm_small_ub",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert proc.returncode == 2
    env_report = json.loads((run_dir / "logs" / "env.log").read_text(encoding="utf-8"))
    assert (env_report.get("matmul_v3_header_probe") or {}).get("ok") is False
    assert any(
        "MatMulV3 header missing required macros" in item
        for item in env_report["env_errors"]
    )


def test_torchair_experimental_path_normalization(tmp_path: Path):
    runner = load_runner()
    experimental = make_fake_torchair(tmp_path)
    base, package, source = runner.normalize_torchair_experimental(
        str(experimental), tmp_path / "repo"
    )

    assert base == experimental.resolve()
    assert package == (experimental / "_inductor_npu_ext" / "python").resolve()
    assert source == "arg"


def test_cv_case_fails_when_artifacts_or_modes_are_missing():
    runner = load_runner()
    case = next(item for item in runner.MATRIX if item.name == "cv_static_mm_small_ub")
    result = {
        "name": case.name,
        "ok": True,
        "stage": "summary",
        "artifact_paths": {
            "asc_kernel": [],
            "wrapper": [],
            "kernel_so": [],
            "dump_dirs": [],
        },
        "fusion_mode": None,
        "ub_mode": None,
        "mix_mode": None,
    }

    runner.validate_case_result(case, result)

    assert result.get("ok") is False
    assert result.get("stage") == "summary"
    assert "missing artifacts" in result.get("error", "")


def test_cv_case_fails_when_only_non_cv_artifacts_exist():
    runner = load_runner()
    case = next(item for item in runner.MATRIX if item.name == "cv_static_bmm_ub_basic")
    result = {
        "name": case.name,
        "ok": True,
        "stage": "summary",
        "artifact_paths": {
            "asc_kernel": ["a"],
            "wrapper": ["w"],
            "kernel_so": ["k"],
            "dump_dirs": ["d"],
        },
        "cv_artifacts": {"wrapper": []},
        "fusion_mode": None,
        "ub_mode": None,
        "mix_mode": None,
    }

    runner.validate_case_result(case, result)

    assert result.get("ok") is False
    assert "missing CV artifact" in result.get("error", "")


def test_run_continuation_defaults_to_collecting_all_selected_cases():
    runner = load_runner()

    assert runner.should_stop_after_case({"ok": False}, fail_fast=False) is False
    assert runner.should_stop_after_case({"ok": False}, fail_fast=True) is True
    assert runner.should_stop_after_case({"ok": True}, fail_fast=True) is False


def test_cv_case_fails_when_expected_mode_is_missing():
    runner = load_runner()
    case = next(item for item in runner.MATRIX if item.name == "cv_dynamic_bmm_key1")
    result = {
        "name": case.name,
        "ok": True,
        "stage": "summary",
        "artifact_paths": {
            "asc_kernel": ["a"],
            "wrapper": ["w"],
            "kernel_so": ["k"],
            "dump_dirs": ["d"],
        },
        "fusion_mode": None,
        "ub_mode": 0,
        "mix_mode": None,
    }

    runner.validate_case_result(case, result)

    assert result.get("ok") is False
    assert result.get("stage") == "summary"
    assert "missing fusion_mode" in result.get("error", "")


def test_extract_modes_ignores_struct_comments_and_uses_assignments(tmp_path: Path):
    runner = load_runner()
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "inductor_wrapper.cpp").write_text(
        "uint8_t fusion_mode; // 0:ub; 1:safety\n"
        "uint8_t ub_mode; // 0:no db; 1:db\n"
        "uint8_t mix_mode;\n",
        encoding="utf-8",
    )
    (case_dir / "asc_kernel.py").write_text(
        "tiling->cv_tiling_data.fusion_mode = 0;\n"
        "tiling->cv_tiling_data.ub_mode = 1;\n"
        "tiling->cv_tiling_data.mix_mode = 2;\n",
        encoding="utf-8",
    )

    modes = runner.extract_modes(case_dir)

    assert modes == {"fusion_mode": 0, "ub_mode": 1, "mix_mode": 2}


def test_filter_path_list_removes_foreign_cann_paths(tmp_path: Path):
    runner = load_runner()
    repo = tmp_path / "repo"
    cann = tmp_path / "selected" / "cann-9.1.0"
    overlay = tmp_path / "run" / "overlay"
    driver = tmp_path / "usr" / "local" / "Ascend" / "driver" / "lib64"
    foreign = (
        tmp_path
        / "home"
        / "developer"
        / "Ascend"
        / "cann"
        / "tools"
        / "msdebug"
        / "lib"
    )
    simulator = tmp_path / "tools" / "simulator" / "Ascend910B1" / "lib"

    paths = (
        overlay / "lib64",
        repo / "build" / "autofuse",
        cann / "lib64",
        driver,
        foreign,
        simulator,
    )
    value = ":".join(str(path) for path in paths)

    filtered = runner.filter_path_list(value, repo, cann, overlay, allow_driver=True)

    assert str(foreign) not in filtered
    assert str(simulator) not in filtered
    assert str(overlay / "lib64") in filtered
    assert str(repo / "build" / "autofuse") in filtered
    assert str(cann / "lib64") in filtered
    assert str(driver) in filtered


def test_with_debug_option_preserves_existing_options():
    runner = load_runner()

    assert (
        runner.with_debug_option("lowering+nocat", "matmul") == "lowering+matmul+nocat"
    )
    assert runner.with_debug_option("matmul+nocat", "matmul") == "matmul+nocat"
    assert runner.with_debug_option("", "matmul") == "matmul"


def test_cann_host_include_env_contains_matmul_arch_parent(tmp_path: Path):
    runner = load_runner()
    cann = tmp_path / "cann"
    matmul_parent = (
        cann
        / "opp"
        / "built-in"
        / "op_impl"
        / "ai_core"
        / "tbe"
        / "impl"
        / "ops_nn"
        / "ascendc"
        / "mat_mul_v3"
    )
    ops_nn_common = (
        cann
        / "opp"
        / "built-in"
        / "op_impl"
        / "ai_core"
        / "tbe"
        / "impl"
        / "ops_nn"
        / "ascendc"
        / "common"
    )
    (matmul_parent / "arch35").mkdir(parents=True)
    (ops_nn_common / "cmct" / "block").mkdir(parents=True)
    env = {}

    runner.add_cann_host_include_env(env, cann)

    include_paths = env.get("CPLUS_INCLUDE_PATH", "").split(":")
    assert str(matmul_parent) in include_paths
    assert str(ops_nn_common) in include_paths
