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
import pytest


@pytest.fixture(autouse=True)
def explicit_device_profile(monkeypatch):
    """Keep legacy host contract tests explicit without adding CLI fallbacks."""
    from device_validation.tools import run_device_validation

    original = run_device_validation.build_parser
    profile = str(__file__.replace("conftest.py", "profiles/ascend950.json"))

    def parser_with_test_contract():
        parser = original()
        parser.set_defaults(soc_profile="ascend950", profile=profile)
        return parser

    monkeypatch.setattr(
        run_device_validation, "build_parser", parser_with_test_contract
    )

    original_metadata = run_device_validation.read_abi_metadata

    def metadata_for_contract_tests(artifact, input_count, output_count, **kwargs):
        if (Path(artifact) / "abi_metadata.json").is_file():
            return original_metadata(artifact, input_count, output_count)
        return {
            "launch_abi": "AutofuseLaunchV2",
            "input_count": input_count,
            "output_count": output_count,
            "input_dtypes": ["float16"] * input_count,
            "output_dtypes": kwargs.get("output_dtypes", ["float16"] * output_count),
        } | {"input_dtypes": kwargs.get("input_dtypes", ["float16"] * input_count)}

    monkeypatch.setattr(
        run_device_validation, "read_abi_metadata", metadata_for_contract_tests
    )
