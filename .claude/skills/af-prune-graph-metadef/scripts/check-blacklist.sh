#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
skill_dir="$(cd "${script_dir}/.." && pwd -P)"
repo_root="$(git -C "${skill_dir}" rev-parse --show-toplevel)"
blacklist="${skill_dir}/black-list-files.txt"
declare -A protected=()
while IFS= read -r entry; do
    [[ -z "${entry}" || "${entry}" == \#* ]] && continue
    protected["${entry}"]=1
done < "${blacklist}"

usage() {
    echo "Usage: $0 check-target <header> | check-diff" >&2
    exit 2
}

is_blacklisted() {
    local name
    name="${1##*/}"
    [[ -n "${protected[${name}]+present}" ]]
}

[[ $# -ge 1 ]] || usage
case "$1" in
    check-target)
        [[ $# -eq 2 ]] || usage
        [[ -f "$2" ]] || { echo "Target is not a file: $2" >&2; exit 2; }
        target="$(realpath -e -- "$2")"
        case "${target}" in
            "${repo_root}"/*) ;;
            *) echo "Target is outside the repository: $2" >&2; exit 2 ;;
        esac
        if is_blacklisted "${target}"; then
            echo "Blocked by graph_metadef blacklist: $2" >&2
            exit 1
        fi
        ;;
    check-diff)
        [[ $# -eq 1 ]] || usage
        blocked=0
        while IFS= read -r -d '' path; do
            if is_blacklisted "${path}"; then
                echo "Blacklisted path changed: ${path}" >&2
                blocked=1
            fi
        done < <(
            git -C "${repo_root}" diff --no-renames --name-only -z HEAD --
            git -C "${repo_root}" ls-files --others --exclude-standard -z
        )
        [[ ${blocked} -eq 0 ]] || exit 1
        ;;
    *)
        usage
        ;;
esac
