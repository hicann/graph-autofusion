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
from pathlib import Path
import importlib.util
import json

import numpy as np
import pytest

from device_validation.tools.run_device_validation import load_case, select_support
from device_validation.cases.brc_load_nddma_score.gen_input import generate_inputs
from device_validation.cases.brc_load_nddma_score.reference import compute_reference


CASE_DIR = Path(__file__).parent.parent / "cases" / "brc_load_nddma_score"


def test_brc_load_case_declares_score_shapes_and_float16_input():
    case = load_case(CASE_DIR)
    assert case["case_id"] == "brc_load_nddma_score"
    assert [item["dtype"] for item in case["inputs"]] == ["float16"]
    entry = select_support(case, "ascendc_real_device", "ascend950", shape=(64, 2048))
    assert entry["shapes"] == [[64, 1023], [64, 2048], [64, 2050]]


def test_brc_load_reference_broadcasts_singleton_load_axis():
    value = generate_inputs((2, 3))[0]
    result = compute_reference(value)
    broadcast = np.broadcast_to(value[:1, :3], (2, 3)).astype(np.float16)
    expected = (
        np.exp(broadcast).astype(np.float16)
        * np.abs(np.exp(broadcast).astype(np.float16))
    ).astype(np.float16)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.real_codegen
@pytest.mark.parametrize("platform", ["3510", "5102"])
def test_platform_v2_codegen_emits_nddma_score(tmp_path, platform):
    source = CASE_DIR / "input_ascir.py"
    spec = importlib.util.spec_from_file_location("brc_load_nddma_score_input", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    profile = tmp_path / f"{platform}.json"
    profile.write_text(
        json.dumps(
            {"ascir": {"platform": platform, "core_type": 1, "ub_size": 245760}}
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / platform
    module.generate_codegen((64, 2048), output_dir, profile)
    host_impl = (output_dir / "host_impl.cpp").read_text(encoding="utf-8")
    score_functions = host_impl.split(
        "int32_t CalcScore(const AutofuseTilingData &tiling_data)"
    )[1:]
    assert any("return -1;" in function for function in score_functions)
