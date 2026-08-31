# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from compare_csv import CSVComparator


def test_compare_operators_preserves_same_operator_across_groups():
    rows = [
        {"Operator": "Add", "Graph": "0", "Result": "0", "Group": "0", "Case": "1"},
        {"Operator": "Add", "Graph": "0", "Result": "0", "Group": "1", "Case": "2"},
    ]
    common, only_one, only_two = CSVComparator().compare_operators(rows, rows)
    assert len(common) == 2
    assert not only_one
    assert not only_two


def test_case_is_compared_as_a_field_not_identity():
    first = [
        {"Operator": "Add", "Graph": "0", "Result": "0", "Group": "0", "Case": "0"}
    ]
    second = [
        {"Operator": "Add", "Graph": "0", "Result": "0", "Group": "0", "Case": "2"}
    ]
    common, only_one, only_two = CSVComparator().compare_operators(first, second)
    assert common == ["Add|0|0|0"]
    assert not only_one
    assert not only_two
