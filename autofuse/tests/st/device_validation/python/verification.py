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
"""Structured output decoding, precision verification and failure classification."""

import numpy as np

from .shape import DTYPE_NUMPY, DTYPE_SIZES, checked_shape_size

EXIT_CODES = {
    "passed": 0,
    "skipped": 0,
    "not_applicable": 0,
    "failed": 4,
    "precision_failed": 5,
}
EXIT_CODES["required_capability_failed"] = 3


def decode_output_payload(payload):
    if not isinstance(payload, dict) or payload.get("dtype") not in DTYPE_SIZES:
        raise ValueError("invalid output payload")
    raw_shape = payload.get("shape", ())
    if not isinstance(raw_shape, list) or not raw_shape:
        raise ValueError("output payload shape must be a non-empty list")
    shape, element_count, _ = checked_shape_size(
        raw_shape, DTYPE_SIZES[payload["dtype"]]
    )
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != element_count:
        raise ValueError("output payload has invalid shape or values")
    return np.asarray(data, dtype=DTYPE_NUMPY[payload["dtype"]]).reshape(shape)


def _json_value(value):
    value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return "nan" if np.isnan(value) else ("inf" if value > 0 else "-inf")
    return value


def _verify_tensor(lhs, rhs, tensor_index, atol, rtol):
    if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
        raise ValueError(f"output {tensor_index} metadata mismatch")
    if np.issubdtype(lhs.dtype, np.integer) or np.issubdtype(lhs.dtype, np.bool_):
        mismatch = lhs != rhs
    else:
        mismatch = ~np.isclose(lhs, rhs, atol=atol, rtol=rtol, equal_nan=True)
    indices = np.flatnonzero(mismatch)
    finite = np.isfinite(lhs) & np.isfinite(rhs)
    abs_error = np.abs(lhs.astype(np.float64) - rhs.astype(np.float64))
    rel_error = np.divide(
        abs_error,
        np.abs(rhs.astype(np.float64)),
        out=np.zeros_like(abs_error),
        where=rhs != 0,
    )
    item = {
        "passed": not indices.size,
        "tensor_index": tensor_index,
        "element_count": int(lhs.size),
        "mismatch_count": int(indices.size),
        "nan_mismatch_count": int(np.count_nonzero(np.isnan(lhs) != np.isnan(rhs))),
        "inf_mismatch_count": int(
            np.count_nonzero((np.isinf(lhs) | np.isinf(rhs)) & (lhs != rhs))
        ),
        "max_abs_error": float(np.max(abs_error[finite])) if np.any(finite) else 0.0,
        "max_rel_error": float(np.max(rel_error[finite])) if np.any(finite) else 0.0,
        "first_mismatch": None,
    }
    if indices.size:
        index = int(indices[0])
        item["first_mismatch"] = {
            "linear_index": index,
            "actual": _json_value(lhs.flat[index]),
            "expected": _json_value(rhs.flat[index]),
        }
    return item


def verify_outputs(actual, expected, atol, rtol):
    if len(actual) != len(expected):
        raise ValueError("output tensor count mismatch")
    outputs = []
    for tensor_index, (lhs, rhs) in enumerate(zip(actual, expected)):
        outputs.append(_verify_tensor(lhs, rhs, tensor_index, atol, rtol))
    return {
        "passed": all(item["passed"] for item in outputs),
        "tensor_count": len(outputs),
        "element_count": sum(item["element_count"] for item in outputs),
        "mismatch_count": sum(item["mismatch_count"] for item in outputs),
        "nan_mismatch_count": sum(item["nan_mismatch_count"] for item in outputs),
        "inf_mismatch_count": sum(item["inf_mismatch_count"] for item in outputs),
        "max_abs_error": max((item["max_abs_error"] for item in outputs), default=0.0),
        "max_rel_error": max((item["max_rel_error"] for item in outputs), default=0.0),
        "first_mismatch": next(
            (item["first_mismatch"] for item in outputs if item["first_mismatch"]), None
        ),
        "outputs": outputs,
    }


def report_precision(precision, artifact):
    report = dict(precision)
    report["artifact"] = str(artifact)
    return report


def classify_failure(reason, capability_status):
    text = reason.lower()
    if capability_status == "unsupported":
        return "not_applicable", 0
    if capability_status == "optional":
        return "skipped", 0
    if "precision" in text or "output" in text:
        return "precision_failed", 5
    return "failed", 4


def classify_backend_payload(payload, returncode=0):
    if returncode != 0:
        if not (
            payload.get("stage") == "preflight"
            and payload.get("stage_status") in ("skipped", "not_applicable")
        ):
            return "failed", 4
    decisions = payload.get("support_decisions", {})
    stage = payload.get("stage", "")
    is_preflight = stage in ("preflight", "capability") or not stage
    required_failure = any(
        item.get("result") == "failed" for item in decisions.values()
    )
    if is_preflight and required_failure:
        return "required_capability_failed", 3
    if is_preflight and payload.get("stage_status") == "skipped":
        return "skipped", 0
    if is_preflight and payload.get("stage_status") == "not_applicable":
        return "not_applicable", 0
    if (
        stage in ("verification", "post_run")
        and payload.get("precision", {}).get("passed") is False
    ):
        return "precision_failed", 5
    if payload.get("stage_status") == "failed":
        return "failed", 4
    status = payload.get("stage_status", "failed")
    return (status, EXIT_CODES[status]) if status in EXIT_CODES else ("failed", 4)
