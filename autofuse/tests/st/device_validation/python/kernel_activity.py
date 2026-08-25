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
"""AI Core kernel activity parsing and measurement extraction from msprof output.

Durations in msprof kernel activity tables (op_summary/kernel_details/task_time
CSV files) are treated as microseconds. Host fixture CSVs never require a real
device; the same layout rules apply to real msprof output directories.
"""

from dataclasses import dataclass
import csv
import math
from pathlib import Path


class KernelActivityError(ValueError):
    def __init__(self, error_code, reason):
        super().__init__(reason)
        self.error_code = error_code
        self.reason = reason


@dataclass(frozen=True)
class KernelRecord:
    kernel_name: str
    duration_us: float
    start_us: float | None = None
    end_us: float | None = None


@dataclass(frozen=True)
class _TableColumns:
    columns: dict
    normalized: list
    name_column: str
    duration_column: str
    start_column: str | None
    end_column: str | None


_NAME_COLUMNS = frozenset(("op_name", "kernel_name", "name"))
_DURATION_COLUMNS = frozenset(
    (
        "task_duration",
        "task_duration_us",
        "duration",
        "duration_us",
        "dur",
        "kernel_duration",
        "kernel_duration_us",
        "task_time_us",
    )
)
_START_COLUMNS = frozenset(
    (
        "task_start_time",
        "task_start_time_us",
        "start_time",
        "start_time_us",
        "start_us",
        "begin_us",
    )
)
_END_COLUMNS = frozenset(
    (
        "task_end_time",
        "task_end_time_us",
        "end_time",
        "end_time_us",
        "end_us",
        "finish_us",
    )
)
_FAMILY_PRIORITY = ("kernel_details", "op_summary", "task_time")
_FILENAME_PATTERNS = (
    "**/mindstudio_profiler_output/op_summary*.csv",
    "**/mindstudio_profiler_output/kernel_details*.csv",
    "**/op_summary*.csv",
    "**/kernel_details*.csv",
    "**/task_time*.csv",
)


def _normalize_header(header):
    name = header.strip().lower().replace("(", " ").replace(")", "").replace("-", " ")
    return "_".join(part for part in name.split() if part)


def _discover_files(output_dir):
    files = []
    for pattern in _FILENAME_PATTERNS:
        files.extend(sorted(Path(output_dir).glob(pattern)))
    seen = set()
    unique = []
    for path in files:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    for family in _FAMILY_PRIORITY:
        matched = [path for path in unique if family in path.name]
        if matched:
            return matched
    return []


def _parse_time(row, columns, column):
    if column is None:
        return None
    index = columns[column]
    if index >= len(row):
        return None
    raw = (row[index] or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_row(table, row):
    name = (row[table.columns[table.name_column]] or "").strip()
    if not name:
        type_column = next(
            (
                i
                for i, n in enumerate(table.normalized)
                if n in ("kernel_type", "task_type")
            ),
            None,
        )
        name = (
            row[type_column]
            if type_column is not None and type_column < len(row)
            else ""
        ) or "unknown"
    raw_duration = (row[table.columns[table.duration_column]] or "").strip()
    if not raw_duration:
        return None
    try:
        duration = float(raw_duration)
    except ValueError as error:
        raise KernelActivityError(
            "invalid_activity_row",
            f"kernel activity row for '{name}' has an invalid duration",
        ) from error
    return KernelRecord(
        name,
        duration,
        _parse_time(row, table.columns, table.start_column),
        _parse_time(row, table.columns, table.end_column),
    )


def _parse_table(path, records):
    with path.open("r", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header_row = next(reader, None)
    if not header_row:
        return
    normalized = [_normalize_header(item) for item in header_row]
    columns = {name: index for index, name in enumerate(normalized)}
    name_column = next((name for name in _NAME_COLUMNS if name in columns), None)
    duration_column = next(
        (name for name in _DURATION_COLUMNS if name in columns), None
    )
    if name_column is None or duration_column is None:
        return
    start_column = next((name for name in _START_COLUMNS if name in columns), None)
    end_column = next((name for name in _END_COLUMNS if name in columns), None)
    table = _TableColumns(
        columns, normalized, name_column, duration_column, start_column, end_column
    )
    with path.open("r", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        next(reader, None)
        for row in reader:
            record = _parse_row(table, row)
            if record is not None:
                records.append(record)


def parse_profiler_output(output_dir):
    """Parse msprof kernel activity CSV tables into ordered kernel records."""
    files = _discover_files(output_dir)
    if not files:
        raise KernelActivityError(
            "no_profiler_output", f"no profiling output found under {output_dir}"
        )
    records = []
    for path in files:
        _parse_table(path, records)
    return records


def filter_kernel_records(records, kernel_names):
    """Keep records whose kernel name matches any of the declared names.

    Matching is containment-based in either direction: generated kernel names
    carry per-graph suffixes (for example ``isinf_3_1``), so a declared graph
    name matches the generated kernel name without needing an exact equality.
    """
    if kernel_names is None:
        return list(records)
    if not isinstance(kernel_names, (list, tuple)):
        raise ValueError("kernel_names must be a sequence of strings")
    names = [name for name in kernel_names if isinstance(name, str) and name]
    if not names:
        return list(records)
    filtered = []
    for record in records:
        for name in names:
            if name in record.kernel_name or record.kernel_name in name:
                filtered.append(record)
                break
    return filtered


def extract_measured_records(records, repeat, warmup):
    """Exclude warmup records and return exactly ``repeat`` durations.

    With serial execution the warmup runs happen first: when more records than
    ``warmup + repeat`` exist the last ``repeat`` records are the measured
    iterations; otherwise (partial warmup) the first ``repeat`` records are
    used. A record count below ``repeat`` is a structured error.
    """
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
        raise ValueError("repeat must be a positive integer")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if len(records) < repeat:
        raise KernelActivityError(
            "kernel_activity_count_mismatch",
            f"kernel activity record count {len(records)} is below repeat {repeat}",
        )
    if len(records) >= warmup + repeat:
        measured = records[len(records) - repeat:]  # fmt: skip
    else:
        measured = records[:repeat]
    durations = [record.duration_us for record in measured]
    if any(
        not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or not math.isfinite(duration)
        or duration <= 0
        for duration in durations
    ):
        raise KernelActivityError(
            "kernel_activity_invalid_duration",
            "kernel activity durations must be finite and positive",
        )
    return durations


def align_unfused_iterations(step_records):
    """Sum per-step measured durations by iteration index."""
    rows = [list(values) for values in step_records]
    if not rows:
        raise ValueError("step records must not be empty")
    if len({len(values) for values in rows}) != 1:
        raise KernelActivityError(
            "kernel_activity_step_mismatch",
            "unfused step kernel activity record counts differ",
        )
    return [sum(values[index] for values in rows) for index in range(len(rows[0]))]
