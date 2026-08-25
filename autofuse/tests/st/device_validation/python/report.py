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
"""Stable report schema v2 builder."""


def build_report(
    case_id,
    backend,
    soc,
    status,
    reason="",
    *,
    stage="execution",
    variant="fused",
    support_decisions=None,
    precision=None,
    performance=None,
    artifact_paths=None,
    run_parameters=None,
    error_code="",
    performance_comparison=None,
    step=-1,
    actual_abi="",
    requested_abi="",
    abi_metadata=None,
    steps=None,
):
    return {
        "schema_version": 2,
        "case_id": case_id,
        "case": case_id,
        "backend": backend,
        "soc_profile": soc,
        "soc": soc,
        "variant": variant,
        "step": step,
        "requested_abi": requested_abi,
        "actual_abi": actual_abi,
        "abi_metadata": abi_metadata if isinstance(abi_metadata, dict) else {},
        "stage_status": status,
        "stage": stage,
        "reason": reason,
        "error_code": error_code,
        "support_decisions": support_decisions or {},
        "precision": _default_precision(precision, status),
        "performance": _default_performance(performance),
        "performance_comparison": _default_comparison(performance_comparison),
        "run_parameters": run_parameters or {},
        "artifact_paths": artifact_paths or [],
        "artifact": artifact_paths or [],
        "steps": steps or [],
    }


def _default_precision(precision, status):
    if precision is not None:
        return precision
    return {
        "passed": status == "passed",
        "tensor_count": 0,
        "element_count": 0,
        "mismatch_count": 0,
        "nan_mismatch_count": 0,
        "inf_mismatch_count": 0,
        "first_mismatch": None,
    }


def _default_performance(performance):
    if performance is not None:
        return performance
    return {"declared": {}, "actual": {}, "status": "not_applicable"}


def _default_comparison(performance_comparison):
    if performance_comparison is not None:
        return performance_comparison
    return {"status": "not_applicable"}
