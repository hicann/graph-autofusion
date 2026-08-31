#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class OperatorSummary:
    operator_name: str
    graph: int = 0
    result: int = 0
    group: int = -1
    case: int = 0
    aiv_mte2: Optional[float] = None
    aiv_mte3: Optional[float] = None
    objective_value: Optional[float] = None
    result_performance: Optional[float] = None
    tiling_values: Dict[str, float] = field(default_factory=dict)
    # 状态用于区分缺失证据与真实的 0 值；不改变既有 CSV 字段。
    parse_status: str = "ok"


class LogParser:
    def __init__(self):
        self.patterns = {
            "operator_name": re.compile(r"\[([^\]]+)\]\s*\[PROF\]"),
            "graph_result": re.compile(
                r"\[PROF\]Among all schedule results,\s*graph(\d+)_result(\d+)\s+is the best choice"
            ),
            "group_case": re.compile(
                r"\[PROF\]Among the templates,\s*tiling case\s+(\d+)\s+of\s+graph(\d+)_result(\d+)_g(\d+)\s+is the best choice"
            ),
            "tiling_value": re.compile(
                r"\[PROF\]The value of\s+(\w+)\s+is\s+([\d.]+)\s+in\s+graph(\d+)_result(\d+)_g(\d+)_(\d+)"
            ),
            "objective_value": re.compile(
                r"\[PROF\]The objective value of the tiling data is\s+([\d.]+)\s+in\s+graph(\d+)_result(\d+)_g(\d+)_(\d+)"
            ),
            "result_performance": re.compile(
                r"\[([^\]]+)\]\s+The value of\s+graph(\d+)_result(\d+)\s+is\s+([\d.]+)"
            ),
        }

    def extract_operator_names(self, log_content: str) -> List[str]:
        operators = []
        seen = set()
        for match in self.patterns["operator_name"].finditer(log_content):
            name = match.group(1)
            if name not in seen:
                seen.add(name)
                operators.append(name)
        return operators

    def _extract_graph_result_status(self, log_content: str, operator_name: str):
        pattern = re.compile(
            rf"\[{re.escape(operator_name)}\]\s*\[PROF\]Among all schedule results,\s*graph(\d+)_result(\d+)\s+is the best choice"
        )
        match = pattern.search(log_content)
        if match:
            return int(match.group(1)), int(match.group(2)), "ok"
        # 模板行也带有 graph/result，可用于兼容旧日志，但要明确标记推断来源。
        pattern = re.compile(
            rf"\[{re.escape(operator_name)}\]\s*\[PROF\]Among the templates,\s*tiling case\s+\d+\s+of\s+graph(\d+)_result(\d+)_g\d+\s+is the best choice"
        )
        match = pattern.search(log_content)
        if match:
            return int(match.group(1)), int(match.group(2)), "inferred_graph_result"
        return None, None, "missing_graph_result"

    def extract_graph_result(self, log_content: str, operator_name: str) -> tuple:
        pattern = re.compile(
            rf"\[{re.escape(operator_name)}\]\s*\[PROF\]Among all schedule results,\s*graph(\d+)_result(\d+)\s+is the best choice"
        )
        match = pattern.search(log_content)
        if match:
            return int(match.group(1)), int(match.group(2))
        graph, result, _ = self._extract_graph_result_status(log_content, operator_name)
        # 保留历史 API 的 0 默认值；parse_log_file 使用带状态接口，避免丢失缺失信息。
        return (graph if graph is not None else 0), (
            result if result is not None else 0
        )

    def extract_group_case(
        self, log_content: str, operator_name: str, graph_id: int, result_id: int
    ) -> tuple:
        """单 group 接口（兼容旧调用），返回第一个匹配的 (group_id, case_id)"""
        pattern = re.compile(
            rf"\[{re.escape(operator_name)}\]\s*\[PROF\]Among the templates,\s*tiling case\s+(\d+)\s+of\s+graph{graph_id}_result{result_id}_g(\d+)\s+is the best choice"
        )
        match = pattern.search(log_content)
        if match:
            return int(match.group(2)), int(match.group(1))
        return -1, 0

    def extract_all_group_cases(
        self, log_content: str, operator_name: str, graph_id: int, result_id: int
    ) -> Dict[int, int]:
        """多 group 接口，返回 {group_id: case_id}，同一 group 重复出现时取最后一次"""
        pattern = re.compile(
            rf"\[{re.escape(operator_name)}\]\s*\[PROF\]Among the templates,\s*tiling case\s+(\d+)\s+of\s+graph{graph_id}_result{result_id}_g(\d+)\s+is the best choice"
        )
        result: Dict[int, int] = {}
        for match in pattern.finditer(log_content):
            case_id = int(match.group(1))
            group_id = int(match.group(2))
            result[group_id] = case_id  # 后出现的覆盖前面的
        return result

    def extract_all_graph_results(
        self, log_content: str, operator_name: str
    ) -> List[Tuple[int, int]]:
        results = set()

        result_perf_pattern = re.compile(
            rf"\[{re.escape(operator_name)}\]\s+The value of\s+graph(\d+)_result(\d+)\s+is\b"
        )
        for match in result_perf_pattern.finditer(log_content):
            results.add((int(match.group(1)), int(match.group(2))))

        group_case_pattern = re.compile(
            rf"\[{re.escape(operator_name)}\]\s*\[PROF\]Among the templates,\s*tiling case\s+\d+\s+of\s+graph(\d+)_result(\d+)_g\d+\s+is the best choice"
        )
        for match in group_case_pattern.finditer(log_content):
            results.add((int(match.group(1)), int(match.group(2))))

        if not results:
            results.add(self.extract_graph_result(log_content, operator_name))
        return sorted(results)

    def extract_tiling_values(
        self,
        log_content: str,
        operator_name: str,
        graph_id: int,
        result_id: int,
        group_id: int,
        case_id: int,
    ) -> Dict[str, float]:
        tiling_values = {}
        pattern = re.compile(
            rf"\[{re.escape(operator_name)}\]\s*\[PROF\]The value of\s+(\w+)\s+is\s+([\d.]+)\s+in\s+graph{graph_id}_result{result_id}_g{group_id}_{case_id}"
        )
        for match in pattern.finditer(log_content):
            tiling_values[match.group(1)] = float(match.group(2))
        return tiling_values

    def extract_performance_metrics(
        self,
        log_content: str,
        operator_name: str,
        graph_id: int,
        result_id: int,
        group_id: int,
        case_id: int,
    ) -> Dict[str, float]:
        metrics = {}
        objective_pattern = re.compile(
            rf"\[{re.escape(operator_name)}\]\s*\[PROF\]The objective value of the tiling data is\s+([\d.]+)\s+in\s+graph{graph_id}_result{result_id}_g{group_id}_{case_id}"
        )
        match = objective_pattern.search(log_content)
        if match:
            metrics["objective_value"] = float(match.group(1))
        tiling_values = self.extract_tiling_values(
            log_content, operator_name, graph_id, result_id, group_id, case_id
        )
        if "AIV_MTE2" in tiling_values:
            metrics["aiv_mte2"] = tiling_values["AIV_MTE2"]
        if "AIV_MTE3" in tiling_values:
            metrics["aiv_mte3"] = tiling_values["AIV_MTE3"]
        return metrics

    def extract_result_performance(
        self, log_content: str, operator_name: str, graph_id: int, result_id: int
    ) -> Optional[float]:
        pattern = re.compile(
            rf"\[{re.escape(operator_name)}\]\s+The value of\s+graph{graph_id}_result{result_id}\s+is\s+([\d.]+)"
        )
        match = pattern.search(log_content)
        if match:
            return float(match.group(1).rstrip("."))
        return None

    def _build_summary(
        self,
        log_content: str,
        operator_name: str,
        graph_id: int,
        result_id: int,
        group_id: int,
        case_id: int,
        parse_status: str = "ok",
    ) -> OperatorSummary:
        metrics = self.extract_performance_metrics(
            log_content, operator_name, graph_id, result_id, group_id, case_id
        )
        tiling_values = self.extract_tiling_values(
            log_content, operator_name, graph_id, result_id, group_id, case_id
        )
        result_performance = self.extract_result_performance(
            log_content, operator_name, graph_id, result_id
        )
        result_perf_missing = result_performance is None
        if result_perf_missing and parse_status == "ok":
            parse_status = "missing_result_performance"

        return OperatorSummary(
            operator_name=operator_name,
            graph=graph_id,
            result=result_id,
            group=group_id,
            case=case_id,
            aiv_mte2=metrics.get("aiv_mte2"),
            aiv_mte3=metrics.get("aiv_mte3"),
            objective_value=metrics.get("objective_value"),
            result_performance=result_performance,
            tiling_values=tiling_values,
            parse_status=parse_status,
        )

    def _collect_result_summaries(
        self,
        log_content: str,
        operator_name: str,
        graph_id: int,
        result_id: int,
        parse_status: str = "ok",
    ) -> List[OperatorSummary]:
        summaries: List[OperatorSummary] = []
        group_cases = self.extract_all_group_cases(
            log_content, operator_name, graph_id, result_id
        )
        if group_cases:
            for group_id, case_id in sorted(group_cases.items()):
                status = parse_status
                if status == "ok":
                    status = "ok"
                summaries.append(
                    self._build_summary(
                        log_content,
                        operator_name,
                        graph_id,
                        result_id,
                        group_id,
                        case_id,
                        status,
                    )
                )
            return summaries

        group_id, case_id = self.extract_group_case(
            log_content, operator_name, graph_id, result_id
        )
        status = parse_status if parse_status != "ok" else "missing_group_case"
        summaries.append(
            self._build_summary(
                log_content,
                operator_name,
                graph_id,
                result_id,
                group_id,
                case_id,
                status,
            )
        )
        return summaries

    def parse_log_file(
        self, file_path: str, summary_mode: str = "best_result_all_groups"
    ) -> List[OperatorSummary]:
        with open(file_path, "r", encoding="utf-8") as f:
            log_content = f.read()

        operator_names = self.extract_operator_names(log_content)
        summaries = []

        for operator_name in operator_names:
            graph_status = "ok"
            if summary_mode == "all_results_all_groups":
                graph_results = self.extract_all_graph_results(
                    log_content, operator_name
                )
            else:
                graph, result, graph_status = self._extract_graph_result_status(
                    log_content, operator_name
                )
                graph_results = [] if graph is None else [(graph, result)]

            if not graph_results:
                continue

            for graph_id, result_id in graph_results:
                summaries.extend(
                    self._collect_result_summaries(
                        log_content, operator_name, graph_id, result_id, graph_status
                    )
                )

        return summaries
