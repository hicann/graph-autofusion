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
import os
from typing import List


def find_log_files(path: str) -> List[str]:
    """递归查找 .log 文件；path 可以是文件或目录"""
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        result = []
        for root, _, files in os.walk(path):
            for fname in files:
                if fname.endswith(".log"):
                    result.append(os.path.join(root, fname))
        return sorted(result)
    return []


def ensure_output_dir(path: str) -> str:
    """确保目录存在，返回 path"""
    os.makedirs(path, exist_ok=True)
    return path


def build_case_output_path(
    base: str, op: str, result_id: int, group_id: int, case_num: int
) -> str:
    """构造拆分输出路径：base/op/graph0_result{r}/g{g}/case{n}.log"""
    return os.path.join(
        base, op, f"graph0_result{result_id}", f"g{group_id}", f"case{case_num}.log"
    )
