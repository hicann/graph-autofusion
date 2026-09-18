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
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
import unittest
from core.log_parser import LogParser

MULTI_GROUP_LOG = """
[FlashAttentionScore] [PROF]Among the templates, tiling case 0 of graph0_result1_g0 is the best choice
[FlashAttentionScore] [PROF]Among the templates, tiling case 1 of graph0_result1_g1 is the best choice
[FlashAttentionScore] [PROF]Among all schedule results, graph0_result1 is the best choice
"""

RERUN_LOG = """
[FlashAttentionScore] [PROF]Among the templates, tiling case 0 of graph0_result1_g0 is the best choice
[FlashAttentionScore] [PROF]Among the templates, tiling case 2 of graph0_result1_g0 is the best choice
[FlashAttentionScore] [PROF]Among all schedule results, graph0_result1 is the best choice
"""

MULTI_GROUP_DETAIL_LOG = """
[FlashAttentionScore] [PROF]Among the templates, tiling case 0 of graph0_result1_g0 is the best choice
[FlashAttentionScore] [PROF]Among the templates, tiling case 1 of graph0_result1_g1 is the best choice
[FlashAttentionScore] [PROF]Among all schedule results, graph0_result1 is the best choice
[FlashAttentionScore] [PROF]The objective value of the tiling data is 100.0 in graph0_result1_g0_0
[FlashAttentionScore] [PROF]The objective value of the tiling data is 200.0 in graph0_result1_g1_1
[FlashAttentionScore] [PROF]The value of s0t_size is 16 in graph0_result1_g0_0
[FlashAttentionScore] [PROF]The value of s0t_size is 32 in graph0_result1_g1_1
[FlashAttentionScore] The value of graph0_result1 is 300.0
"""

MULTI_RESULT_DETAIL_LOG = """
[FlashAttentionScore] [PROF]Among the templates, tiling case 0 of graph0_result0_g0 is the best choice
[FlashAttentionScore] [PROF]Among the templates, tiling case 1 of graph0_result0_g1 is the best choice
[FlashAttentionScore] [PROF]Among the templates, tiling case 2 of graph0_result1_g0 is the best choice
[FlashAttentionScore] [PROF]Among the templates, tiling case 3 of graph0_result1_g1 is the best choice
[FlashAttentionScore] [PROF]Among all schedule results, graph0_result1 is the best choice
[FlashAttentionScore] [PROF]The objective value of the tiling data is 10.0 in graph0_result0_g0_0
[FlashAttentionScore] [PROF]The objective value of the tiling data is 11.0 in graph0_result0_g1_1
[FlashAttentionScore] [PROF]The objective value of the tiling data is 20.0 in graph0_result1_g0_2
[FlashAttentionScore] [PROF]The objective value of the tiling data is 21.0 in graph0_result1_g1_3
[FlashAttentionScore] The value of graph0_result0 is 100.0
[FlashAttentionScore] The value of graph0_result1 is 200.0
"""

MISSING_GROUP_DETAIL_LOG = """
[FlashAttentionScore] [PROF]Among all schedule results, graph0_result1 is the best choice
[FlashAttentionScore] [PROF]The objective value of the tiling data is 123.0 in graph0_result1_g-1_0
"""


class TestExtractAllGroupCases(unittest.TestCase):
    def setUp(self):
        self.parser = LogParser()

    def test_multi_group(self):
        result = self.parser.extract_all_group_cases(
            MULTI_GROUP_LOG, "FlashAttentionScore", 0, 1
        )
        self.assertEqual(result, {0: 0, 1: 1})

    def test_rerun_takes_last(self):
        result = self.parser.extract_all_group_cases(
            RERUN_LOG, "FlashAttentionScore", 0, 1
        )
        self.assertEqual(result, {0: 2})

    def test_not_found(self):
        result = self.parser.extract_all_group_cases("", "FlashAttentionScore", 0, 1)
        self.assertEqual(result, {})


class TestExtractAllGraphResults(unittest.TestCase):
    def setUp(self):
        self.parser = LogParser()

    def test_prefers_result_value_and_deduplicates(self):
        result = self.parser.extract_all_graph_results(
            MULTI_RESULT_DETAIL_LOG, "FlashAttentionScore"
        )
        self.assertEqual(result, [(0, 0), (0, 1)])

    def test_falls_back_to_template_lines(self):
        log = """
[FlashAttentionScore] [PROF]Among the templates, tiling case 0 of graph1_result2_g0 is the best choice
[FlashAttentionScore] [PROF]Among the templates, tiling case 1 of graph1_result3_g0 is the best choice
"""
        result = self.parser.extract_all_graph_results(log, "FlashAttentionScore")
        self.assertEqual(result, [(1, 2), (1, 3)])


class TestParseLogFileSummaryModes(unittest.TestCase):
    def setUp(self):
        self.parser = LogParser()

    def _write_temp_log(self, content: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".log")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        return path

    def test_default_returns_best_result_all_groups(self):
        path = self._write_temp_log(MULTI_GROUP_DETAIL_LOG)
        summaries = self.parser.parse_log_file(path)

        self.assertEqual(len(summaries), 2)
        summaries_by_group = {s.group: s for s in summaries}
        self.assertEqual(sorted(summaries_by_group.keys()), [0, 1])
        self.assertEqual(summaries_by_group[0].case, 0)
        self.assertEqual(summaries_by_group[1].case, 1)

    def test_all_results_all_groups_returns_multiple_results(self):
        path = self._write_temp_log(MULTI_RESULT_DETAIL_LOG)
        summaries = self.parser.parse_log_file(
            path, summary_mode="all_results_all_groups"
        )

        self.assertEqual(len(summaries), 4)
        pairs = sorted((s.result, s.group, s.case) for s in summaries)
        self.assertEqual(pairs, [(0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)])

    def test_default_only_returns_best_result_groups(self):
        path = self._write_temp_log(MULTI_RESULT_DETAIL_LOG)
        summaries = self.parser.parse_log_file(path)

        self.assertEqual(len(summaries), 2)
        self.assertTrue(all(s.result == 1 for s in summaries))

    def test_rerun_takes_last_case_per_group(self):
        path = self._write_temp_log("""
[FlashAttentionScore] [PROF]Among the templates, tiling case 0 of graph0_result1_g0 is the best choice
[FlashAttentionScore] [PROF]Among the templates, tiling case 2 of graph0_result1_g0 is the best choice
[FlashAttentionScore] [PROF]Among all schedule results, graph0_result1 is the best choice
[FlashAttentionScore] [PROF]The objective value of the tiling data is 222.0 in graph0_result1_g0_2
""")
        summaries = self.parser.parse_log_file(path)

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].group, 0)
        self.assertEqual(summaries[0].case, 2)
        self.assertEqual(summaries[0].objective_value, 222.0)

    def test_falls_back_when_group_missing(self):
        path = self._write_temp_log(MISSING_GROUP_DETAIL_LOG)
        summaries = self.parser.parse_log_file(path)

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].group, -1)
        self.assertEqual(summaries[0].case, 0)
        self.assertEqual(summaries[0].objective_value, 123.0)


class TestFinalTilingRecords(unittest.TestCase):
    def setUp(self):
        self.parser = LogParser()

    def test_single_line_decodes_descriptive_fields(self):
        content = (
            "[ATT][FINAL_TILING] schema=1 source=runtime selection_mode=default operator=Add_0 graph=0 result=0 group=0 case_id=2 tiling_key=2 "
            "sub_case_tag=R template=graph0_result0_g0_R2 repr_kind=full_json score=3 "
            'pipe_estimates="{\\"M2\\":128.0,\\"V\\":null}" '
            'tiling_repr="{\\"tile_m\\":64}"\n'
        )
        record = self.parser.extract_final_tiling_records(content, "single.log")[0]
        self.assertEqual(record.op, "Add_0")
        self.assertEqual(record.pipe_est["M2"], 128.0)
        self.assertEqual(record.tiling_repr["tile_m"], 64)
        self.assertEqual(record.selection_mode, "default")
        self.assertEqual(record.sub_case_tag, "R")
        self.assertEqual(record.template_name, "graph0_result0_g0_R2")
        self.assertEqual(record.repr_kind, "full_json")
        self.assertEqual(record.score, 3)
        self.assertEqual(record.parse_status, "ok")

    def test_descriptive_field_names_are_supported(self):
        content = (
            "[ATT][FINAL_TILING] schema=1 source=runtime selection_mode=default "
            "operator=Add_0 graph=0 result=0 group=0 case_id=2 tiling_key=2 score=3 "
            "sub_case_tag=R template=graph0_result0_g0_R2 repr_kind=full_json "
            'pipe_estimates="{\\"M2\\":128.0}" tiling_repr="{\\"tile_m\\":64}"\n'
        )
        record = self.parser.extract_final_tiling_records(content, "descriptive.log")[0]
        self.assertEqual(record.op, "Add_0")
        self.assertEqual(record.graph, 0)
        self.assertEqual(record.case, 2)
        self.assertEqual(record.score, 3)
        self.assertEqual(record.tiling_repr["tile_m"], 64)

    def test_short_field_names_are_not_accepted(self):
        content = (
            "[ATT][FINAL_TILING] s=1 src=runtime mode=default op=Add_0 "
            'g=0 r=0 gr=0 c=2 k=2 p="{}" tr="{}"\n'
        )
        record = self.parser.extract_final_tiling_records(content, "short.log")[0]
        self.assertEqual(record.parse_status, "invalid_final_tiling")

    def test_full_json_preserves_inductor_cpp_repr(self):
        content = (
            "[ATT][FINAL_TILING] schema=1 source=runtime selection_mode=default operator=Add_0 graph=0 result=0 group=0 case_id=0 tiling_key=0 "
            'repr_kind=full_json pipe_estimates="{}" tiling_repr="AutofuseTilingData{\\n  .block_dim = 1\\n}"\n'
        )
        record = self.parser.extract_final_tiling_records(content)[0]
        self.assertEqual(record.parse_status, "ok")
        self.assertIn("AutofuseTilingData", record.tiling_repr)

    def test_unavailable_repr_is_valid_without_payload(self):
        content = (
            '[ATT][FINAL_TILING] schema=1 source="runtime" selection_mode="default" '
            'operator="Add" graph=0 result=0 group=0 case_id=0 tiling_key=0 score=0 sub_case_tag="" template="AddCase0" '
            'repr_kind="unavailable" pipe_estimates="{}" tiling_repr=""\n'
        )
        record = self.parser.extract_final_tiling_records(content)[0]
        self.assertEqual(record.parse_status, "ok")
        self.assertIsNone(record.tiling_repr)

    def test_multiple_groups_and_results_are_retained(self):
        content = "".join(
            '[ATT][FINAL_TILING] schema=1 source=r operator=Fusion graph=0 result=%d group=%d case_id=%d tiling_key=%d pipe_estimates="{}" tiling_repr="{}"\n'
            % (result, group, group, group)
            for result in (0, 1)
            for group in (0, 1)
        )
        records = self.parser.extract_final_tiling_records(content)
        self.assertEqual(len(records), 4)
        self.assertEqual(
            {(r.result, r.group) for r in records}, {(0, 0), (0, 1), (1, 0), (1, 1)}
        )

    def test_framing_validates_sequence_length_and_hash(self):
        import hashlib

        payload = '{"x":1}'
        digest = hashlib.sha256(payload.encode()).hexdigest()
        content = (
            '[ATT][FINAL_TILING_BEGIN] schema=1 source=r id=x operator=A graph=0 result=0 group=0 case_id=1 tiling_key=1 pipe_estimates="{}" chunks=2 len=7 hash=%s\n'
            '[ATT][FINAL_TILING_CHUNK] id=x seq=0 data="{\\"x\\":"\n'
            '[ATT][FINAL_TILING_CHUNK] id=x seq=1 data="1}"\n'
            "[ATT][FINAL_TILING_END] id=x chunks=2 len=7 hash=%s\n" % (digest, digest)
        )
        record = self.parser.extract_final_tiling_records(content)[0]
        self.assertEqual(record.tiling_repr, {"x": 1})

    def test_framing_preserves_multiline_inductor_repr(self):
        import hashlib

        payload = "AutofuseTilingData{\n  .block_dim = 1\n}"
        digest = hashlib.sha256(payload.encode()).hexdigest()
        wire_payload = payload.replace("\n", "\\n")
        content = (
            "[ATT][FINAL_TILING_BEGIN] schema=1 source=runtime selection_mode=default operator=Add graph=0 result=0 "
            "group=0 case_id=0 tiling_key=0 repr_kind=full_json chunks=1 len=%d hash=%s\n"
            '[ATT][FINAL_TILING_CHUNK] id="unused" seq=0 data="%s"\n'
            "[ATT][FINAL_TILING_END] id=unused chunks=1 len=%d hash=%s\n"
            % (
                len(payload.encode()),
                digest,
                wire_payload,
                len(payload.encode()),
                digest,
            )
        )
        # The BEGIN id is intentionally added below to keep the payload fixture
        # readable while exercising the exact escaped newline transport.
        content = content.replace(
            "operator=Add graph=0 result=0 group=0 case_id=0 tiling_key=0 repr_kind=full_json",
            "id=unused operator=Add graph=0 result=0 group=0 case_id=0 tiling_key=0 repr_kind=full_json",
        )
        record = self.parser.extract_final_tiling_records(content)[0]
        self.assertEqual(record.parse_status, "ok")
        self.assertEqual(record.tiling_repr, payload)

    def test_att_mix64_hash_algorithm_is_valid(self):
        payload = "{}"
        value = 0x6D2B79F5
        for byte in payload.encode("utf-8"):
            value ^= (
                byte + 0x9E3779B9 + ((value << 6) & 0xFFFFFFFFFFFFFFFF) + (value >> 2)
            )
            value &= 0xFFFFFFFFFFFFFFFF
            value = ((value << 13) | (value >> 51)) & 0xFFFFFFFFFFFFFFFF
        digest = f"{value:016x}"
        content = (
            f'[ATT][FINAL_TILING_BEGIN] schema=1 source=r id=x operator=A graph=0 result=0 group=0 case_id=1 tiling_key=1 pipe_estimates="{{}}" chunks=1 len=2 hash_alg=att_mix64_v1 hash={digest}\n'
            '[ATT][FINAL_TILING_CHUNK] id=x seq=0 data="{}"\n'
            f"[ATT][FINAL_TILING_END] id=x chunks=1 len=2 hash_alg=att_mix64_v1 hash={digest}\n"
        )
        record = self.parser.extract_final_tiling_records(content)[0]
        self.assertEqual(record.parse_status, "ok")

    def test_unsupported_hash_algorithm_is_incomplete(self):
        content = (
            '[ATT][FINAL_TILING_BEGIN] schema=1 source=r id=x operator=A graph=0 result=0 group=0 case_id=1 tiling_key=1 pipe_estimates="{}" chunks=1 len=2 hash_alg=md5 hash=x\n'
            '[ATT][FINAL_TILING_CHUNK] id=x seq=0 data="{}"\n'
            "[ATT][FINAL_TILING_END] id=x chunks=1 len=2 hash_alg=md5 hash=x\n"
        )
        record = self.parser.extract_final_tiling_records(content)[0]
        self.assertEqual(record.parse_status, "incomplete_final_tiling")

    def test_incomplete_and_duplicate_records_are_marked(self):
        content = (
            '[ATT][FINAL_TILING_BEGIN] schema=1 source=r id=x operator=A graph=0 result=0 group=0 case_id=1 tiling_key=1 pipe_estimates="{}" chunks=1 len=1 hash=x\n'
            '[ATT][FINAL_TILING] schema=1 source=r operator=A graph=0 result=0 group=0 case_id=1 tiling_key=2 pipe_estimates="{}" tiling_repr="{}"\n'
            '[ATT][FINAL_TILING] schema=1 source=r operator=A graph=0 result=0 group=0 case_id=1 tiling_key=2 pipe_estimates="{}" tiling_repr="{}"\n'
        )
        records = self.parser.extract_final_tiling_records(content)
        self.assertEqual(records[0].parse_status, "incomplete_final_tiling")
        self.assertEqual(
            [r.parse_status for r in records[1:]], ["ok", "duplicate_final_tiling"]
        )

    def test_out_of_order_chunks_are_incomplete(self):
        content = (
            '[ATT][FINAL_TILING_BEGIN] schema=1 source=r id=x operator=A graph=0 result=0 group=0 case_id=1 tiling_key=1 pipe_estimates="{}" chunks=2 len=2 hash=x\n'
            '[ATT][FINAL_TILING_CHUNK] id=x seq=1 data="b"\n'
            '[ATT][FINAL_TILING_CHUNK] id=x seq=0 data="a"\n'
            "[ATT][FINAL_TILING_END] id=x chunks=2 len=2 hash=x\n"
        )
        record = self.parser.extract_final_tiling_records(content)[0]
        self.assertEqual(record.parse_status, "incomplete_final_tiling")
        self.assertIsNone(record.tiling_repr)

    def test_framing_requires_end_metadata_and_json_payload(self):
        content = (
            '[ATT][FINAL_TILING_BEGIN] schema=1 source=r id=x operator=A graph=0 result=0 group=0 case_id=1 tiling_key=1 pipe_estimates="{}" chunks=1 len=3 hash=x\n'
            '[ATT][FINAL_TILING_CHUNK] id=x seq=0 data="raw"\n'
            "[ATT][FINAL_TILING_END] id=x\n"
        )
        record = self.parser.extract_final_tiling_records(content)[0]
        self.assertEqual(record.parse_status, "incomplete_final_tiling")
        self.assertIsNone(record.tiling_repr)

    def test_schema_version_must_be_one(self):
        content = '[ATT][FINAL_TILING] schema=2 source=r operator=A graph=0 result=0 group=0 case_id=1 tiling_key=1 pipe_estimates="{}" tiling_repr="{}"\n'
        record = self.parser.extract_final_tiling_records(content)[0]
        self.assertEqual(record.parse_status, "invalid_final_tiling")

    def test_summary_keeps_group_mapping(self):
        content = (
            "[ATT][FINAL_TILING_SUMMARY] schema=1 source=runtime selection_mode=default operator=Fusion "
            'graph=0 result=1 groups="{\\"0\\":{\\"case_id\\":2,\\"tiling_key\\":7,\\"sub_case_tag\\":\\"R\\"}}"\n'
        )
        summaries = self.parser.extract_final_tiling_summaries(content, "summary.log")
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].groups["0"]["tiling_key"], 7)
        self.assertEqual(summaries[0].selection_mode, "default")

    def test_descriptive_summary_group_fields_are_preserved(self):
        content = (
            "[ATT][FINAL_TILING_SUMMARY] schema=1 source=runtime selection_mode=default "
            "operator=Fusion graph=0 result=1 "
            'groups="{\\"0\\":{\\"case_id\\":2,\\"tiling_key\\":7,\\"score\\":3,'
            '\\"sub_case_tag\\":\\"R\\",\\"template\\":\\"ConcatCase2\\"}}"\n'
        )
        summaries = self.parser.extract_final_tiling_summaries(content)
        self.assertEqual(summaries[0].groups["0"]["tiling_key"], 7)
        self.assertEqual(summaries[0].groups["0"]["template"], "ConcatCase2")

    def test_summary_framing_reassembles_groups(self):
        import hashlib

        payload = '{"0":{"case_id":2,"tiling_key":7}}'
        digest = hashlib.sha256(payload.encode()).hexdigest()
        content = (
            "[ATT][FINAL_TILING_SUMMARY_BEGIN] schema=1 source=runtime selection_mode=default operator=F graph=0 result=1 id=x chunks=2 len=%d hash=%s\n"
            '[ATT][FINAL_TILING_SUMMARY_CHUNK] id=x seq=0 data="{\\"0\\":{\\"case_id\\":2,"\n'
            '[ATT][FINAL_TILING_SUMMARY_CHUNK] id=x seq=1 data="\\"tiling_key\\":7}}"\n'
            "[ATT][FINAL_TILING_SUMMARY_END] id=x chunks=2 len=%d hash=%s\n"
            % (len(payload), digest, len(payload), digest)
        )
        summaries = self.parser.extract_final_tiling_summaries(content)
        self.assertEqual(summaries[0].groups["0"]["tiling_key"], 7)


if __name__ == "__main__":
    unittest.main()
