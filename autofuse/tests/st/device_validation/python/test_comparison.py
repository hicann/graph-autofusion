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

from device_validation.python.case import load_typed_case
from device_validation.python.benchmark import summarize_samples
from device_validation.python.comparison import compare_variants
from device_validation.python.matrix import MatrixSpec, execute_matrix
from device_validation.python.test_helpers import passed_payload


def _case(tmp_path):
    case_dir = tmp_path / "matrix_case"
    case_dir.mkdir()
    case = {
        "schema_version": 1,
        "case_id": "matrix_case",
        "inputs": [{"dtype": "float16", "dynamic": True}],
        "outputs": [{"dtype": "float16", "dynamic": True}],
        "variants": {
            "fused": {},
            "unfused": {"steps": [{"name": "step0"}, {"name": "step1"}]},
        },
        "support_matrix": [
            {
                "backend": "fake",
                "soc": "soc_a",
                "shapes": [[2, 2]],
                "input_dtypes": ["float16"],
                "output_dtypes": ["float16"],
                "compile": "required",
                "functional": "required",
                "precision": "required",
                "performance": "required",
            },
            {
                "backend": "fake",
                "soc": "soc_b",
                "shapes": [[2, 2]],
                "input_dtypes": ["float16"],
                "output_dtypes": ["float16"],
                "compile": "required",
                "functional": "required",
                "precision": "required",
                "performance": "optional",
            },
            {
                "backend": "fake",
                "soc": "soc_a",
                "shapes": [[3, 3]],
                "input_dtypes": ["float16"],
                "output_dtypes": ["float16"],
                "compile": "required",
                "functional": "required",
                "precision": "unsupported",
                "performance": "unsupported",
            },
        ],
    }
    (case_dir / "case.json").write_text(json.dumps(case), encoding="utf-8")
    return load_typed_case(case_dir)


def _performance(
    soc, shape, variant, *, precision=True, samples=None, metric="runner_wall_clock"
):
    samples = (
        samples
        if samples is not None
        else ([1.0, 2.0, 3.0, 4.0] if variant == "fused" else [2.0, 4.0, 6.0, 8.0])
    )
    summary = summarize_samples(
        samples, variant=variant, step_count=2 if variant == "unfused" else 1
    )
    summary.update(
        {
            "shape": list(shape),
            "dtype": "float16",
            "soc": soc,
            "warmup": 3,
            "repeat": len(samples),
            "reference": "reference.py",
            "metric": metric,
            "unit": "ms",
            "precision_passed": precision,
        }
    )
    return summary


def _fake_executor(calls):
    def fake_executor(request, artifact):
        calls.append(request)
        return {
            "stage_status": "passed",
            "performance": {
                "actual": _performance(
                    request["soc"], request["shape"], request["variant"]
                )
            },
        }

    return fake_executor


def _run_matrix(case, tmp_path, executor, socs, shapes):
    return execute_matrix(
        case,
        "fake",
        MatrixSpec(socs=socs, shapes=shapes, variants=("fused", "unfused")),
        tmp_path / "artifacts",
        executor,
    )


def test_matrix_preflight_skips_unsupported_and_isolates_every_artifact(tmp_path):
    case = _case(tmp_path)
    calls = []
    result = _run_matrix(
        case, tmp_path, _fake_executor(calls), ("soc_a", "soc_b"), ((2, 2), (3, 3))
    )

    assert len(calls) == 4
    assert all(call["soc"] != "soc_a" or call["shape"] != [3, 3] for call in calls)
    artifacts = [Path(item["artifact"]) for item in result["runs"]]
    assert len(artifacts) == len(set(artifacts))
    assert all((artifact / "report.json").exists() for artifact in artifacts)
    unsupported = [
        item
        for item in result["runs"]
        if item["soc"] == "soc_a" and item["shape"] == [3, 3]
    ]
    assert {item["variant"] for item in unsupported} == {"fused", "unfused"}
    assert all(item["stage_status"] == "not_applicable" for item in unsupported)


def test_matrix_comparison_gate_never_emits_speedup_for_invalid_pair(tmp_path):
    case = _case(tmp_path)

    def fake_executor(request, artifact):
        valid = request["variant"] == "fused"
        performance = (
            {
                "shape": [2, 2],
                "dtype": "float16",
                "soc": request["soc"],
                "warmup": 3,
                "repeat": 4,
                "reference": "reference.py",
                "metric": "runner_wall_clock",
                "unit": "ms",
                "samples": [],
                "sample_count": 0,
                "precision_passed": False,
            }
            if not valid
            else _performance(request["soc"], request["shape"], request["variant"])
        )
        return {"stage_status": "passed", "performance": {"actual": performance}}

    result = _run_matrix(case, tmp_path, fake_executor, ("soc_a",), ((2, 2),))
    comparison = result["comparisons"][0]["performance_comparison"]
    assert comparison["status"] in {"not_applicable", "failed"}
    assert "speedup" not in json.dumps(comparison)


def test_matrix_skips_capability_unsupported_entry_without_executor(tmp_path):
    case = _case(tmp_path)
    case.raw["support_matrix"].append(
        {
            "backend": "fake",
            "soc": "soc_b",
            "shapes": [[3, 3]],
            "input_dtypes": ["float16"],
            "output_dtypes": ["float16"],
            "compile": "required",
            "functional": "unsupported",
            "precision": "unsupported",
            "performance": "unsupported",
        }
    )
    calls = []
    result = _run_matrix(case, tmp_path, _fake_executor(calls), ("soc_b",), ((3, 3),))
    assert calls == []
    assert all(item["stage_status"] == "not_applicable" for item in result["runs"])


def test_successful_comparison_reports_all_speedups_and_kernel_reduction(tmp_path):
    case = _case(tmp_path)

    def fake_executor(request, artifact):
        payload = passed_payload(list(request["shape"]))
        payload.update(
            case="matrix_case",
            backend="fake",
            soc_profile=request["soc"],
            variant=request["variant"],
            performance={
                "declared": {},
                "actual": _performance(
                    request["soc"], request["shape"], request["variant"]
                ),
                "status": "passed",
            },
        )
        return payload

    result = _run_matrix(case, tmp_path, fake_executor, ("soc_a",), ((2, 2),))
    comparison = result["comparisons"][0]["performance_comparison"]
    assert comparison["status"] == "passed"
    assert comparison["p50_speedup"] == 2.0
    assert comparison["mean_speedup"] == 2.0
    assert comparison["p90_speedup"] == 2.0
    assert comparison["p99_speedup"] == 2.0
    assert comparison["kernel_reduction"] == 1


def test_comparison_rejects_missing_samples_and_unit_mismatch_without_speedup():
    base = _performance("soc_a", (2, 2), "fused")
    assert (
        compare_variants(base, {**base, "samples": [], "sample_count": 0})["status"]
        == "failed"
    )
    mismatch = {**base, "unit": "us"}
    result = compare_variants(base, mismatch)
    assert result["status"] == "not_applicable"
    assert "speedup" not in json.dumps(result)
