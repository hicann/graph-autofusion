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
"""Case/backend/SoC/shape/variant support decisions."""

from dataclasses import dataclass
import json
from pathlib import Path

try:
    from .case import SUPPORTED_DTYPES
except ImportError:
    from case import SUPPORTED_DTYPES


@dataclass(frozen=True)
class SupportDecision:
    status: str
    result: str
    reason: str
    capabilities: dict
    entry: dict | None = None

    @property
    def matched(self):
        return self.entry is not None and self.status == "supported"


@dataclass(frozen=True)
class SupportConfig:
    variant: str = "fused"
    profile: dict | None = None


def _validate_basic(profile):
    required = {"profile", "allowed_abi", "dtypes", "real_device_backend"}
    if not isinstance(profile, dict) or not required.issubset(profile):
        raise ValueError("invalid device profile contract")
    bad_profile_name = not isinstance(profile["profile"], str) or not profile["profile"]
    bad_dtypes = (
        not isinstance(profile["dtypes"], list)
        or not profile["dtypes"]
        or any(dtype not in SUPPORTED_DTYPES for dtype in profile["dtypes"])
    )
    if bad_profile_name or bad_dtypes:
        raise ValueError("invalid device profile dtypes")
    if not isinstance(profile["allowed_abi"], str) or not all(
        item.strip() for item in profile["allowed_abi"].split(",")
    ):
        raise ValueError("invalid device profile ABI contract")
    ascir = profile.get("ascir")
    bad_backend = (
        not isinstance(profile["real_device_backend"], str)
        or not profile["real_device_backend"]
    )
    bad_platform = (
        not isinstance(ascir, dict)
        or not isinstance(ascir.get("platform"), str)
        or not ascir["platform"]
    )
    bad_ascir_values = any(
        not isinstance(ascir.get(key), int)
        or isinstance(ascir.get(key), bool)
        or ascir[key] < 0
        for key in ("core_type", "ub_size")
    )
    if bad_backend or bad_platform or bad_ascir_values:
        raise ValueError("invalid device profile backend")


def _validate_resources(profile):
    resources = profile.get("resources", {})
    if not isinstance(resources, dict):
        raise ValueError("invalid device profile resources")
    resource_keys = {
        "max_tiling_bytes",
        "max_workspace_bytes",
        "max_block_dimension",
        "max_shape_elements",
    }
    if any(
        key not in resource_keys
        or isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for key, value in resources.items()
    ):
        raise ValueError("invalid device profile resources")
    for key in ("max_shape_elements",):
        if key in profile:
            bad_limit = (
                isinstance(profile[key], bool)
                or not isinstance(profile[key], int)
                or profile[key] < 0
            )
            if bad_limit:
                raise ValueError("invalid device profile resources")


def load_profile(path):
    """Load the profile contract used by host policy and device execution."""
    path = Path(path)
    if not path.is_file():
        raise ValueError("device profile is required and must be a regular file")
    profile = json.loads(path.read_text(encoding="utf-8"))
    _validate_basic(profile)
    _validate_resources(profile)
    return profile


def _candidate_socs(case, backend):
    return sorted(
        {
            entry.get("soc")
            for entry in case.support_matrix
            if entry.get("backend") == backend and isinstance(entry.get("soc"), str)
        }
    )


def _profile_dir(profile_path):
    if profile_path is None:
        return Path(__file__).resolve().parents[1] / "profiles"
    profile_path = Path(profile_path)
    if profile_path.is_dir():
        return (
            profile_path / "profiles"
            if (profile_path / "profiles").is_dir()
            else profile_path
        )
    return profile_path.parent


def _reject_soc_without_explicit_profile(soc_profile, profile_path, case_socs):
    explicit_profile = profile_path is not None and not Path(profile_path).is_dir()
    if soc_profile and soc_profile not in case_socs and not explicit_profile:
        raise ValueError(f"unsupported SoC for backend: {soc_profile}")


def _selection_with_explicit_profile(soc_profile, profile_path):
    selected_path = Path(profile_path)
    if selected_path.is_dir():
        selected_path = selected_path / f"{soc_profile}.json"
    profile = load_profile(selected_path)
    if profile.get("profile") != soc_profile:
        raise ValueError("profile SoC does not match --soc-profile")
    return soc_profile, selected_path, profile


def _resolve_profile_selection(soc_profile, profile_path, profile_dir, case_socs):
    _reject_soc_without_explicit_profile(soc_profile, profile_path, case_socs)
    if soc_profile and profile_path:
        return _selection_with_explicit_profile(soc_profile, profile_path)
    if profile_path and not Path(profile_path).is_dir():
        selected_path = Path(profile_path)
        profile = load_profile(selected_path)
        return profile["profile"], selected_path, profile
    if soc_profile:
        selected_path = profile_dir / f"{soc_profile}.json"
        profile = load_profile(selected_path)
        return soc_profile, selected_path, profile
    candidates = sorted(case_socs)
    if len(candidates) != 1:
        if len(candidates) > 1:
            raise ValueError("multiple candidate SoCs require explicit --soc-profile")
        raise ValueError("no candidate SoC declared by case support matrix")
    selected_path = profile_dir / f"{candidates[0]}.json"
    return candidates[0], selected_path, load_profile(selected_path)


def resolve_case_profile(case_dir, backend, soc_profile, profile_path):
    """Resolve the selected case SoC and its profile without framework defaults."""
    case_dir = Path(case_dir)
    case = case_dir if hasattr(case_dir, "support_matrix") else None
    if case is None:
        try:
            from .case import load_typed_case
        except ImportError:
            from case import load_typed_case
        case = load_typed_case(case_dir)
    case_socs = set(_candidate_socs(case, backend))
    soc_profile, selected_path, profile = _resolve_profile_selection(
        soc_profile, profile_path, _profile_dir(profile_path), case_socs
    )
    if profile.get("profile") != soc_profile:
        raise ValueError("profile SoC does not match --soc-profile")
    if case_socs and backend != profile.get("real_device_backend"):
        raise ValueError("profile backend mismatch")
    if case_socs and soc_profile not in case_socs:
        raise ValueError(f"unsupported SoC for backend: {soc_profile}")
    return soc_profile, selected_path, profile


def _profile_preflight(profile, case, backend, soc, shape):
    if profile is None:
        return None
    if soc != profile["profile"]:
        return SupportDecision(
            "unsupported", "not_applicable", "profile SoC mismatch", {}, None
        )
    if backend != profile["real_device_backend"]:
        return SupportDecision(
            "unsupported", "not_applicable", "profile backend mismatch", {}, None
        )
    if any(
        dtype not in profile["dtypes"]
        for dtype in (*case.input_dtypes, *case.output_dtypes)
    ):
        return SupportDecision(
            "unsupported", "not_applicable", "profile dtype mismatch", {}, None
        )
    if shape is None:
        return None
    elements = 1
    for dimension in shape:
        elements *= int(dimension)
    if elements > profile.get("max_shape_elements", float("inf")):
        return SupportDecision(
            "unsupported", "not_applicable", "profile shape resource limit", {}, None
        )
    resources = profile.get("resources", {})
    if not isinstance(resources, dict):
        return SupportDecision(
            "unsupported", "not_applicable", "invalid profile resources", {}, None
        )
    if elements > resources.get("max_shape_elements", float("inf")):
        return SupportDecision(
            "unsupported", "not_applicable", "profile resource limit", {}, None
        )
    return None


def _match_entry(case, backend, soc, shape):
    entries = [
        entry
        for entry in case.support_matrix
        if entry.get("backend") == backend and entry.get("soc") == soc
    ]
    if not entries:
        return SupportDecision(
            "unsupported", "not_applicable", "undeclared backend/SoC", {}, None
        )
    matching = [
        entry
        for entry in entries
        if shape is None or list(shape) in entry.get("shapes", [])
    ]
    if shape is not None and not matching:
        return SupportDecision(
            "unsupported",
            "not_applicable",
            f"unsupported shape: {list(shape)}",
            {},
            None,
        )
    if shape is None:
        return SupportDecision(
            "unsupported", "not_applicable", "shape is required", {}, None
        )
    return matching[0]


def _dtype_signature_error(entry, case):
    if entry.get("input_dtypes", list(case.input_dtypes)) != list(case.input_dtypes):
        return SupportDecision(
            "unsupported",
            "not_applicable",
            "unsupported input dtype signature",
            {},
            None,
        )
    declared_outputs = entry.get("output_dtypes", list(case.output_dtypes))
    if not set(case.output_dtypes).issubset(declared_outputs):
        return SupportDecision(
            "unsupported",
            "not_applicable",
            "unsupported output dtype signature",
            {},
            None,
        )
    return None


def resolve_support(case, backend, soc, shape=None, config=None):
    variant = config.variant if config is not None else "fused"
    profile = config.profile if config is not None else None
    profile_error = _profile_preflight(profile, case, backend, soc, shape)
    if profile_error is not None:
        return profile_error
    entry = _match_entry(case, backend, soc, shape)
    if isinstance(entry, SupportDecision):
        return entry
    if variant not in case.variants:
        return SupportDecision(
            "unsupported", "not_applicable", f"unsupported variant: {variant}", {}, None
        )
    signature_error = _dtype_signature_error(entry, case)
    if signature_error is not None:
        return signature_error
    variants = case.raw.get("variants", {})
    variant_config = variants.get(variant, {}) if isinstance(variants, dict) else {}
    allowed_abi = (
        tuple(item.strip() for item in profile.get("allowed_abi", "").split(","))
        if profile
        else ()
    )
    case_abi = variant_config.get("abi") or entry.get("abi") or "AutofuseLaunchV2"
    if allowed_abi and case_abi not in allowed_abi:
        return SupportDecision(
            "unsupported", "not_applicable", "profile ABI mismatch", {}, None
        )
    return SupportDecision(
        "supported",
        "passed",
        "supported",
        {
            name: entry.get(name, "unsupported")
            for name in ("compile", "functional", "precision", "performance")
        },
        {
            **entry,
            "variant": variant,
            "codegen_entry": variant_config.get("codegen_entry", "input_ascir.py"),
            "allowed_abi": allowed_abi,
        },
    )
