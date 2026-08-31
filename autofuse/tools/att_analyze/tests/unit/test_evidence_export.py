#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import hashlib
import json
import os
import subprocess
import sys


SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
ENTRYPOINT = os.path.join(SRC, "att.py")
DATA = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../data/test_complete.log")
)


def test_evidence_jsonl_contains_records_and_source_locations(tmp_path):
    # Existing CSV/SVG artifacts are not rewritten by the exporter, but must
    # remain discoverable in the manifest for archive consumers.
    raw_csv = tmp_path / "summary.csv"
    raw_svg = tmp_path / "perf.svg"
    raw_csv.write_text("operator,cycle\nAdd,1\n", encoding="utf-8")
    raw_svg.write_text("<svg/>\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, ENTRYPOINT, "evidence", DATA, "-o", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    evidence_path = tmp_path / "att-evidence.jsonl"
    manifest_path = tmp_path / "tool-manifest.json"
    records = [
        json.loads(line)
        for line in evidence_path.read_text(encoding="utf-8").splitlines()
    ]
    assert {
        (r["operator"], r["graph_id"], r["result_id"], r["group_id"], r["case_id"])
        for r in records
    } == {
        ("Add", 0, 0, 0, 0),
        ("Mul", 1, 1, 0, 0),
    }
    for record in records:
        assert record["tiling_values"]
        assert record["objective"] is not None
        assert record["source_path"] == DATA
        assert isinstance(record["source_line"], int) and record["source_line"] > 0
        assert record["parse_status"] == "ok"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["record_count"] == 2
    artifact_paths = {entry["path"] for entry in manifest["artifacts"]}
    assert {"att-evidence.jsonl", "summary.csv", "perf.svg"} <= artifact_paths
    csv_entry = next(
        entry for entry in manifest["artifacts"] if entry["path"] == "summary.csv"
    )
    assert csv_entry["sha256"] == hashlib.sha256(raw_csv.read_bytes()).hexdigest()


def test_missing_template_selection_is_explicit(tmp_path):
    log = tmp_path / "incomplete.log"
    log.write_text(
        "[Add] [PROF]The value of block_dim is 2 in graph0_result0_g0_0.\n"
        "[Add] [PROF]The objective value of the tiling data is 10 in graph0_result0_g0_0.\n",
        encoding="utf-8",
    )
    output = tmp_path / "evidence"
    result = subprocess.run(
        [sys.executable, ENTRYPOINT, "evidence", str(log), "-o", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(
        (output / "att-evidence.jsonl").read_text(encoding="utf-8").strip()
    )
    assert record["parse_status"] == "missing_group_case"


def test_missing_graph_selection_is_marked_inferred(tmp_path):
    log = tmp_path / "inferred.log"
    log.write_text(
        "[Add] [PROF]The value of block_dim is 2 in graph0_result0_g0_0.\n"
        "[Add] [PROF]The objective value of the tiling data is 10 in graph0_result0_g0_0.\n"
        "[Add] [PROF]Among the templates, tiling case 0 of graph0_result0_g0 is the best choice.\n",
        encoding="utf-8",
    )
    output = tmp_path / "evidence"
    result = subprocess.run(
        [sys.executable, ENTRYPOINT, "evidence", str(log), "-o", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(
        (output / "att-evidence.jsonl").read_text(encoding="utf-8").strip()
    )
    assert "inferred_graph_result" in record["parse_status"]


def test_evidence_accepts_info_prefix_and_r_subcase_labels(tmp_path):
    """Severity prefixes must not be mistaken for the operator name."""
    log = tmp_path / "prefixed.log"
    log.write_text(
        "[INFO] [Add] [PROF]The value of block_dim is 2 in graph0_result0_g0_R0.\n"
        "[INFO] [Add] [PROF]The objective value of the tiling data is 10 in graph0_result0_g0_R0.\n"
        "[INFO] [Add] [PROF]Among the templates, tiling case 0 of graph0_result0_g0_R0 is the best choice.\n"
        "[INFO] [Add] [PROF]Among all schedule results, graph0_result0 is the best choice.\n"
        "[INFO] [Add] [PROF]The value of graph0_result0 is 10.\n",
        encoding="utf-8",
    )
    output = tmp_path / "evidence"
    result = subprocess.run(
        [sys.executable, ENTRYPOINT, "evidence", str(log), "-o", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(
        (output / "att-evidence.jsonl").read_text(encoding="utf-8").strip()
    )
    assert record["operator"] == "Add"
    assert record["graph_id"] == 0
    assert record["result_id"] == 0
    assert record["group_id"] == 0
    assert record["case_id"] == 0
    assert record["tiling_values"] == {"block_dim": 2.0}
    assert record["objective"] == 10.0
    assert record["result_performance"] == 10.0
    assert record["parse_status"] == "ok"


def test_prefixed_prof_result_performance_satisfies_evidence_status(tmp_path):
    log = tmp_path / "prefixed-performance.log"
    log.write_text(
        "[INFO] [Add] [PROF]The value of block_dim is 2 in graph0_result0_g0_0.\n"
        "[INFO] [Add] [PROF]The objective value of the tiling data is 10 in graph0_result0_g0_0.\n"
        "[INFO] [Add] [PROF]Among the templates, tiling case 0 of graph0_result0_g0 is the best choice.\n"
        "[INFO] [Add] [PROF]Among all schedule results, graph0_result0 is the best choice.\n"
        "[INFO] [Add] [PROF]The value of graph0_result0 is 10.\n",
        encoding="utf-8",
    )
    output = tmp_path / "evidence"
    result = subprocess.run(
        [sys.executable, ENTRYPOINT, "evidence", str(log), "-o", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(
        (output / "att-evidence.jsonl").read_text(encoding="utf-8").strip()
    )
    assert record["result_performance"] == 10.0
    assert record["parse_status"] == "ok"


def test_evidence_accepts_tilingsummary_prefix(tmp_path):
    """CANN GE logs may put a non-bracketed component tag before the operator."""
    log = tmp_path / "tiling-summary.log"
    log.write_text(
        "[INFO] GE(1,python): TilingSummary:[Add][PROF]The value of block_dim is 2 in graph0_result0_g0_0.\n"
        "[INFO] GE(1,python): TilingSummary:[Add][PROF]The objective value of the tiling data is 10 in graph0_result0_g0_0.\n"
        "[INFO] GE(1,python): TilingSummary:[Add][PROF]Among the templates, tiling case 0 of graph0_result0_g0 is the best choice.\n"
        "[INFO] GE(1,python): TilingSummary:[Add][PROF]Among all schedule results, graph0_result0 is the best choice.\n"
        "[INFO] GE(1,python): TilingSummary:[Add]The value of graph0_result0 is 10.\n",
        encoding="utf-8",
    )
    output = tmp_path / "evidence"
    result = subprocess.run(
        [sys.executable, ENTRYPOINT, "evidence", str(log), "-o", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(
        (output / "att-evidence.jsonl").read_text(encoding="utf-8").strip()
    )
    assert record["operator"] == "Add"
    assert record["tiling_values"] == {"block_dim": 2.0}
    assert record["objective"] == 10.0
    assert record["result_performance"] == 10.0
