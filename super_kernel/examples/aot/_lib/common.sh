#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

sk_validate_npu_arch() {
    case "$1" in
        dav-2201|dav-3510)
            return 0
            ;;
        *)
            echo "ERROR: unsupported NPU architecture '$1'; valid values: dav-2201, dav-3510" >&2
            return 2
            ;;
    esac
}

sk_parse_npu_arch() {
    local parsed_args
    NPU_ARCH=""
    parsed_args=$(getopt -a -o h -l help,npu-arch: -- "$@") || return 2
    eval set -- "${parsed_args}"
    while true; do
        case "$1" in
            -h | --help)
                echo "Usage: bash $0 --npu-arch=<dav-2201|dav-3510>"
                exit 0
                ;;
            --npu-arch)
                NPU_ARCH="$2"
                shift 2
                ;;
            --)
                shift
                break
                ;;
        esac
    done
    if [ "$#" -ne 0 ]; then
        echo "ERROR: unexpected arguments: $*" >&2
        return 2
    fi
    if [ -z "${NPU_ARCH}" ]; then
        echo "ERROR: --npu-arch is required." >&2
        return 2
    fi
    sk_validate_npu_arch "${NPU_ARCH}"
}

sk_cleanup_local() {
    local output
    for output in log tmp sk_meta kernel_meta profiling static_kernel_compile_outputs \
        aclnn_static_shape_kernel_outputs .static_kernel_records.json; do
        rm -rf -- "${PWD}/${output}" || return 1
    done
    mkdir -p "${PWD}/tmp" "${PWD}/log"
}

sk_setup_isolated_python_userbase() {
    local userbase="$1"
    local python_cmd="${PYTHON_CMD:-python3}"
    local python_site_dir

    python_site_dir=$("${python_cmd}" - <<'PY'
import sys
print(f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
PY
    ) || return 1
    export PYTHONUSERBASE="${userbase}"
    export PYTHONPATH="${PYTHONUSERBASE}/${python_site_dir}${PYTHONPATH:+:${PYTHONPATH}}"
}

sk_run_python_with_log() {
    local script="$1"
    local run_log="$2"
    local python_cmd="${PYTHON_CMD:-python3}"

    mkdir -p "$(dirname "${run_log}")"
    "${python_cmd}" "${script}" 2>&1 | tee "${run_log}"
    return "${PIPESTATUS[0]}"
}

sk_check_static_kernel_outputs() {
    local output_dir="$1"
    local package_requirement="${2:-optional}"
    local error_log
    local run_package

    error_log=$(find "${output_dir}" -type f -name '*_compile_error.log' -print -quit 2>/dev/null)
    run_package=$(find "${output_dir}" -type f -name '*.run' -print -quit 2>/dev/null)
    if [ -n "${error_log}" ] && [ -z "${run_package}" ]; then
        echo "ERROR: static kernel compilation failed; see ${error_log}" >&2
        return 1
    fi
    if [ "${package_requirement}" = "required" ] && [ -z "${run_package}" ]; then
        echo "ERROR: static kernel run package was not generated under ${output_dir}" >&2
        return 1
    fi
}

sk_uninstall_static_kernel_from_log() {
    local run_log="$1"
    local uninstall_script

    [ -f "${run_log}" ] || return 0
    while IFS= read -r uninstall_script; do
        [ -n "${uninstall_script}" ] && [ -f "${uninstall_script}" ] && bash "${uninstall_script}" >/dev/null 2>&1
    done < <(grep -Eo '/[^[:space:]]+/uninstall\.sh' "${run_log}" | sort -u)
}
