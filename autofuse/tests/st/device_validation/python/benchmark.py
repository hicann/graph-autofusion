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
"""Deterministic benchmark sample aggregation for device validation."""

from pathlib import Path
import json
import statistics


def save_samples(artifact, samples, *, variant="fused", step=None):
    path = Path(artifact) / (
        f"samples_{variant}_{step}.json" if step else f"samples_{variant}.json"
    )
    path.write_text(json.dumps(list(samples)), encoding="utf-8")
    return path


def sum_step_samples(step_samples):
    rows = [list(values) for values in step_samples]
    if not rows:
        return []
    if any(len(values) != len(rows[0]) for values in rows):
        raise ValueError("benchmark sample counts differ between steps")
    return [sum(values[index] for values in rows) for index in range(len(rows[0]))]


def summarize_samples(
    samples,
    *,
    variant="fused",
    step_count=1,
    runtime_state=None,
    metric="runner_wall_clock",
    unit=None,
):
    values = [float(value) for value in samples]
    if not values:
        raise ValueError("benchmark samples must not be empty")
    ordered = sorted(values)

    def percentile(percent):
        position = (len(ordered) - 1) * percent / 100
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    result = {
        "variant": variant,
        "samples": values,
        "sample_count": len(values),
        "p50": percentile(50),
        "mean": statistics.fmean(values),
        "p90": percentile(90),
        "p99": percentile(99),
        "kernel_count": step_count if variant == "unfused" else 1,
        "metric": metric,
        "unit": unit or ("us" if metric == "device_kernel_duration" else "ms"),
    }
    result.update(runtime_state or {})
    return result


def chain_step_outputs(steps, artifact):
    """Return explicit file inputs/outputs for a sequential unfused iteration."""
    root = Path(artifact)
    chained = []
    previous = None
    for index, step in enumerate(steps):
        inputs = list(step.get("inputs", []))
        if previous is not None:
            inputs = [previous if value == "$previous" else value for value in inputs]
        output = root / f"step_{index}_output_0.bin"
        chained.append(
            {
                "name": step["name"],
                "inputs": inputs,
                "output": str(output),
                "input_count": len(inputs),
                "output_count": len(step.get("outputs", [{"dtype": "float16"}])),
            }
        )
        previous = str(output)
    return chained
