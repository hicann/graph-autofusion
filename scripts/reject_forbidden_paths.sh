#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

forbidden_prefix="docs/superpowers/"

while IFS= read -r -d '' path; do
    case "${path}" in
        "${forbidden_prefix}"*)
            printf '拒绝提交 %s：docs/superpowers 目录由仓库策略保护。\n' "${path}" >&2
            exit 1
            ;;
    esac
done < <(git diff --cached --name-only --diff-filter=ACMRD -z)
