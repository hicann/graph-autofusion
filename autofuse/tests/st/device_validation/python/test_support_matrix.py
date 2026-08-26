# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import json
from pathlib import Path

import pytest

from device_validation.python.support_matrix import (
    _profile_preflight,
    load_profile,
    resolve_case_profile,
)
from device_validation.python.case import load_typed_case
from device_validation.tools.run_device_validation import (
    _resolve_case_selection,
    build_parser,
    load_case,
    select_support,
)


CASE_DIR = Path(__file__).parents[1] / "cases" / "isinf_maskedfill_fusion"
PROFILE_DIR = CASE_DIR.parents[1] / "profiles"


def _raw_case_with(tmp_path, **overrides):
    raw_case = json.loads((CASE_DIR / "case.json").read_text(encoding="utf-8"))
    raw_case["support_matrix"].append({**raw_case["support_matrix"][0], **overrides})
    case_path = tmp_path / "case"
    case_path.mkdir()
    (case_path / "case.json").write_text(json.dumps(raw_case), encoding="utf-8")
    return case_path


def _case_and_profiles(tmp_path):
    case_path = _raw_case_with(tmp_path, soc="ascend910")
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    source = json.loads(
        (CASE_DIR.parents[1] / "profiles" / "ascend950.json").read_text(
            encoding="utf-8"
        )
    )
    return case_path, profiles, source


def test_ascend910_9362_profile_loads_and_declares_real_device_backend():
    profile = load_profile(PROFILE_DIR / "ascend910_9362.json")
    assert profile["profile"] == "ascend910_9362"
    assert profile["real_device_backend"] == "ascendc_real_device"
    assert profile["tools"]["toolkit"] == "ASCEND_HOME_PATH"


def test_ascend910_9362_profile_declares_confirmed_platform_facts():
    profile = load_profile(PROFILE_DIR / "ascend910_9362.json")
    assert profile["soc_version"] == "Ascend910_9362"
    assert profile["ascir"] == {"platform": "2201", "core_type": 40, "ub_size": 196608}
    assert profile["resources"]["max_block_dimension"] == 40
    assert "max_tiling_bytes" not in profile["resources"]
    assert "max_workspace_bytes" not in profile["resources"]
    assert profile["soc_version"] != "Ascend950"


def test_isinf_case_declares_ascend910_9362_for_all_supported_shapes():
    case = load_typed_case(CASE_DIR)
    entries = []
    for entry in case.support_matrix:
        if (
            entry["backend"] == "ascendc_real_device"
            and entry["soc"] == "ascend910_9362"
        ):
            entries.append(entry)
    assert len(entries) == 1
    assert entries[0]["shapes"] == [[128, 128], [128, 130], [127, 129], [512, 512]]
    assert entries[0]["input_dtypes"] == ["float16", "uint8", "float16"]
    assert entries[0]["output_dtypes"] == ["float16", "uint8"]
    assert entries[0]["compile"] == "required"
    assert entries[0]["functional"] == "required"
    assert entries[0]["precision"] == "required"
    assert entries[0]["performance"] == "optional"


def test_explicit_ascend910_9362_profile_selects_matching_case_entry():
    case = load_case(CASE_DIR)
    soc, selected, profile = resolve_case_profile(
        CASE_DIR,
        "ascendc_real_device",
        "ascend910_9362",
        PROFILE_DIR / "ascend910_9362.json",
    )
    entry = select_support(
        case,
        "ascendc_real_device",
        soc,
        shape=(127, 129),
    )
    assert soc == profile["profile"] == entry["soc"] == "ascend910_9362"
    assert selected == PROFILE_DIR / "ascend910_9362.json"
    assert profile["real_device_backend"] == entry["backend"]
    assert entry["backend"] == "ascendc_real_device"
    assert entry["dtypes"] == ["float16", "uint8"]
    assert set(entry["dtypes"]).issubset(profile["dtypes"])
    assert entry["input_dtypes"] == ["float16", "uint8", "float16"]
    assert entry["output_dtypes"] == ["float16", "uint8"]
    assert entry["shapes"] == [[128, 128], [128, 130], [127, 129], [512, 512]]


def test_explicit_ascend910_9362_cli_selection_uses_matching_profile():
    profile = PROFILE_DIR / "ascend910_9362.json"
    args = build_parser().parse_args(
        [
            "--case",
            str(CASE_DIR),
            "--variant",
            "all",
            "--mode",
            "prepare",
            "--soc-profile",
            "ascend910_9362",
            "--profile",
            str(profile),
        ]
    )
    _resolve_case_selection(args)
    assert args.soc_profile == "ascend910_9362"
    assert Path(args.profile) == profile


def test_explicit_soc_requires_matching_case_entry():
    case = load_case(CASE_DIR)
    decision = select_support(case, "ascendc_real_device", "unknown_soc", (128, 128))
    assert decision.status == "unsupported"
    assert decision.result == "not_applicable"


def test_profile_soc_mismatch_is_not_supported():
    profile = {
        "profile": "other_soc",
        "allowed_abi": "AutofuseLaunchV2",
        "dtypes": ["float16", "uint8"],
        "real_device_backend": "ascendc_real_device",
        "ascir": {"platform": "other", "core_type": 1, "ub_size": 1},
    }
    case = load_case(CASE_DIR)
    decision = _profile_preflight(
        profile, case, "ascendc_real_device", "ascend950", (128, 128)
    )
    assert decision.result == "not_applicable"


def test_profile_only_derives_soc_from_profile_file():
    profile = CASE_DIR.parents[1] / "profiles" / "ascend950.json"
    soc, selected, loaded = resolve_case_profile(
        CASE_DIR, "ascendc_real_device", None, profile
    )
    assert soc == "ascend950"
    assert selected == profile
    assert loaded["profile"] == soc


def test_soc_only_resolves_profile_from_profile_directory():
    profile_dir = CASE_DIR.parents[1] / "profiles"
    soc, selected, loaded = resolve_case_profile(
        CASE_DIR, "ascendc_real_device", "ascend950", profile_dir
    )
    assert soc == "ascend950"
    assert selected == profile_dir / "ascend950.json"
    assert loaded["profile"] == soc


def test_unique_case_candidate_resolves_without_explicit_selection(tmp_path):
    raw_case = json.loads((CASE_DIR / "case.json").read_text(encoding="utf-8"))
    raw_case["support_matrix"] = [raw_case["support_matrix"][0]]
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "case.json").write_text(json.dumps(raw_case), encoding="utf-8")
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    source = json.loads(
        (CASE_DIR.parents[1] / "profiles" / "ascend950.json").read_text(
            encoding="utf-8"
        )
    )
    (profile_dir / "ascend950.json").write_text(json.dumps(source), encoding="utf-8")
    soc, selected, _ = resolve_case_profile(
        case_dir, "ascendc_real_device", None, profile_dir
    )
    assert (soc, selected.name) == ("ascend950", "ascend950.json")


def test_matrix_all_flow_resolves_missing_cli_selection():
    args = build_parser().parse_args(
        ["--case", str(CASE_DIR), "--variant", "all", "--mode", "prepare"]
    )
    with pytest.raises(ValueError, match="multiple candidate SoCs"):
        _resolve_case_selection(args)


def test_multiple_candidate_profiles_require_explicit_soc(tmp_path):
    case_path, profiles, source = _case_and_profiles(tmp_path)
    for soc in ("ascend950", "ascend910"):
        profile = dict(source)
        profile["profile"] = soc
        (profiles / f"{soc}.json").write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ValueError, match="multiple"):
        resolve_case_profile(case_path, "ascendc_real_device", None, profiles)


def test_single_profile_file_does_not_hide_multiple_case_candidates(tmp_path):
    case_path, profiles, source = _case_and_profiles(tmp_path)
    (profiles / "ascend950.json").write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="multiple"):
        resolve_case_profile(case_path, "ascendc_real_device", None, profiles)


def test_soc_from_other_backend_is_rejected(tmp_path):
    case_path = _raw_case_with(tmp_path, backend="other_backend", soc="other_soc")
    with pytest.raises(ValueError, match="unsupported SoC"):
        resolve_case_profile(case_path, "ascendc_real_device", "other_soc", None)


def test_explicit_soc_and_profile_require_case_backend_soc_entry(tmp_path):
    case_path = _raw_case_with(tmp_path, backend="other_backend", soc="other_soc")
    profile_path = tmp_path / "other_soc.json"
    profile = json.loads(
        (CASE_DIR.parents[1] / "profiles" / "ascend950.json").read_text(
            encoding="utf-8"
        )
    )
    profile["profile"] = "other_soc"
    (profile_path).write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported SoC"):
        resolve_case_profile(
            case_path, "ascendc_real_device", "other_soc", profile_path
        )


def test_explicit_soc_and_profile_require_existing_profile(tmp_path):
    missing_profile = tmp_path / "missing.json"

    with pytest.raises(ValueError, match="profile is required"):
        resolve_case_profile(
            CASE_DIR, "ascendc_real_device", "ascend950", missing_profile
        )


def test_explicit_profile_backend_mismatch_is_rejected(tmp_path):
    profile_path = tmp_path / "ascend950.json"
    profile = json.loads(
        (CASE_DIR.parents[1] / "profiles" / "ascend950.json").read_text(
            encoding="utf-8"
        )
    )
    profile["real_device_backend"] = "other_backend"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(ValueError, match="profile backend mismatch"):
        resolve_case_profile(CASE_DIR, "ascendc_real_device", "ascend950", profile_path)


def test_profile_only_backend_mismatch_is_rejected(tmp_path):
    profile_path = tmp_path / "ascend950.json"
    profile = json.loads(
        (CASE_DIR.parents[1] / "profiles" / "ascend950.json").read_text(
            encoding="utf-8"
        )
    )
    profile["real_device_backend"] = "other_backend"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(ValueError, match="profile backend mismatch"):
        resolve_case_profile(CASE_DIR, "ascendc_real_device", None, profile_path)


def test_undeclared_backend_resolves_profile_for_support_preflight():
    profile = CASE_DIR.parents[1] / "profiles" / "ascend950.json"

    soc, selected, loaded = resolve_case_profile(
        CASE_DIR, "unknown_backend", None, profile
    )

    assert soc == "ascend950"
    assert selected == profile
    assert loaded["real_device_backend"] == "ascendc_real_device"
