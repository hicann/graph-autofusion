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
from dataclasses import dataclass
from typing import List, Optional

_DIM_ALIASES = {
    "r": "results",
    "result": "results",
    "g": "groups",
    "group": "groups",
    "c": "cases",
    "case": "cases",
}


@dataclass
class CaseFilter:
    results: Optional[List[int]] = None
    groups: Optional[List[int]] = None
    cases: Optional[List[int]] = None

    def match(self, result_id: int, group_id: int, case_id: int) -> bool:
        if self.results is not None and result_id not in self.results:
            return False
        if self.groups is not None and group_id not in self.groups:
            return False
        if self.cases is not None and case_id not in self.cases:
            return False
        return True


def parse_case_arg(value: Optional[str]) -> Optional[CaseFilter]:
    """将 --case 字符串解析为 CaseFilter；value 为 None 时返回 None"""
    if value is None:
        return None
    dims = {"results": None, "groups": None, "cases": None}
    current_dim = None
    for token in value.split(","):
        token = token.strip()
        if "=" in token:
            raw_dim, id_str = token.split("=", 1)
            raw_dim = raw_dim.strip()
            dim = _DIM_ALIASES.get(raw_dim)
            if dim is None:
                raise ValueError(f"未知维度: {raw_dim!r}，支持 r/result/g/group/c/case")
            current_dim = dim
            if dims[current_dim] is None:
                dims[current_dim] = []
            dims[current_dim].append(int(id_str.strip()))
        else:
            if current_dim is None:
                raise ValueError(
                    f"--case 解析错误：{token!r} 前缺少维度标识（r=/g=/c=）"
                )
            dims[current_dim].append(int(token))
    return CaseFilter(**dims)
