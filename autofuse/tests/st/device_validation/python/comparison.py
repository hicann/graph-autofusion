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
"""Comparable fused/unfused performance report helpers."""

import math

METRIC_UNITS = {"runner_wall_clock": "ms", "device_kernel_duration": "us"}
SPEEDUP_KEYS = ("p50_speedup", "mean_speedup", "p90_speedup", "p99_speedup")
THRESHOLD_KEYS = ("max_regression_ratio", "min_speedup", "kernel_reduction_required")


def _shared_inputs(fused, unfused):
    shared = ("shape", "dtype", "soc", "warmup", "repeat", "reference")
    if any(fused.get(key) != unfused.get(key) for key in shared):
        return None, "comparison inputs are not shared"
    metric = fused.get("metric")
    if metric != unfused.get("metric") or metric not in METRIC_UNITS:
        return None, "metric mismatch"
    unit = fused.get("unit") or METRIC_UNITS[metric]
    if (
        unit != (unfused.get("unit") or METRIC_UNITS[metric])
        or unit != METRIC_UNITS[metric]
    ):
        return None, "metric mismatch"
    return metric, unit


def _invalid_item(item, metrics=()):
    if not isinstance(item.get("samples"), list):
        return "sample count does not match repeat"
    sample_count = item.get("sample_count")
    if not _valid_sample_count(item, sample_count):
        return "sample count does not match repeat"
    kernel_count = item.get("kernel_count")
    if not _valid_kernel_count(kernel_count):
        return "kernel count is missing or invalid"
    for value in item["samples"]:
        if not _valid_sample_value(value):
            return "benchmark samples are invalid"
    if item.get("precision_passed") is not True:
        return "precision gate failed"
    for key in metrics:
        value = item.get(key)
        if not _valid_sample_value(value):
            return "benchmark summary is incomplete"
    return None


def _valid_sample_count(item, sample_count):
    repeat = item.get("repeat")
    return not (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or not isinstance(repeat, int)
        or isinstance(repeat, bool)
        or sample_count != len(item["samples"])
        or sample_count != repeat
        or sample_count <= 0
    )


def _valid_kernel_count(kernel_count):
    return not (
        isinstance(kernel_count, bool)
        or not isinstance(kernel_count, (int, float))
        or kernel_count <= 0
        or not math.isfinite(kernel_count)
    )


def _valid_sample_value(value):
    return not (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    )


def _validate_shared(fused, unfused):
    metric, unit = _shared_inputs(fused, unfused)
    if metric is None:
        return None, unit
    return (metric, unit), None


def _apply_thresholds(result, thresholds):
    configured = any(key in thresholds for key in THRESHOLD_KEYS)
    result["threshold_status"] = "not_configured" if not configured else "passed"
    if not configured:
        return result
    max_regression_ratio = thresholds.get("max_regression_ratio")
    if max_regression_ratio is not None:
        min_ratio = min(result.get(key, 0.0) for key in SPEEDUP_KEYS)
        if min_ratio < 1.0 / max_regression_ratio:
            result["threshold_status"] = "failed"
    min_speedup = thresholds.get("min_speedup")
    if min_speedup is not None:
        min_ratio = min(result.get(key, 0.0) for key in SPEEDUP_KEYS)
        if min_ratio < min_speedup:
            result["threshold_status"] = "failed"
    kernel_reduction_required = thresholds.get("kernel_reduction_required")
    if (
        kernel_reduction_required is not None
        and result["kernel_reduction"] < kernel_reduction_required
    ):
        result["threshold_status"] = "failed"
    if result["threshold_status"] == "failed":
        result["status"] = "failed"
        result["reason"] = "performance threshold failed"
    return result


def _validate_thresholds(thresholds):
    if thresholds is None:
        return {}
    if not isinstance(thresholds, dict):
        raise ValueError("invalid threshold configuration")
    rules = {
        "max_regression_ratio": 0,
        "min_speedup": 0,
        "kernel_reduction_required": 0,
    }
    for key, value in thresholds.items():
        if key not in rules:
            raise ValueError("invalid threshold configuration")
        if not _valid_sample_value(value):
            raise ValueError("invalid threshold configuration")
    return thresholds


def compare_variants(fused, unfused, thresholds=None):
    inputs, reason = _validate_shared(fused, unfused)
    if inputs is None:
        return {"status": "not_applicable", "reason": reason}
    metric, unit = inputs
    for item in (fused, unfused):
        reason = _invalid_item(item, ("p50", "mean", "p90", "p99"))
        if reason is not None:
            return {"status": "failed", "reason": reason}
    result = {
        "status": "passed",
        "metric": metric,
        "unit": unit,
        "p50_speedup": unfused["p50"] / fused["p50"],
        "mean_speedup": unfused["mean"] / fused["mean"],
        "p90_speedup": unfused["p90"] / fused["p90"],
        "p99_speedup": unfused["p99"] / fused["p99"],
        "kernel_reduction": unfused["kernel_count"] - fused["kernel_count"],
    }
    try:
        threshold_config = _validate_thresholds(thresholds)
    except ValueError:
        return {
            "status": "failed",
            "threshold_status": "failed",
            "reason": "invalid threshold configuration",
        }
    result = _apply_thresholds(result, threshold_config)
    if any(not math.isfinite(result.get(key, 0.0)) for key in SPEEDUP_KEYS):
        return {
            "status": "failed",
            "threshold_status": result.get("threshold_status", "not_configured"),
            "reason": "speedup ratio is not finite",
        }
    return result
