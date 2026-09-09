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

set -e

BASEPATH=$(cd "$(dirname "$0")"; pwd)
REPO_ROOT=$(cd "${BASEPATH}/../.."; pwd)

# 选择运行样例使用的 Python 解释器。
if [ -n "${VIRTUAL_ENV:-}" ]; then
    PYTHON_CMD="python3"
elif [ -f "${REPO_ROOT}/venv/bin/python" ]; then
    PYTHON_CMD="${REPO_ROOT}/venv/bin/python"
else
    PYTHON_CMD="python3"
fi

source "${BASEPATH}/aot/_lib/common.sh"

usage() {
    echo "Usage: bash super_kernel/examples/run_example.sh --npu-arch=<dav-2201|dav-3510>"
}

NPU_ARCH=""
parsed_args=$(getopt -a -o h -l help,npu-arch: -- "$@") || {
    usage
    exit 2
}
eval set -- "${parsed_args}"

while true; do
    case "$1" in
        -h | --help)
            usage
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
        *)
            echo "ERROR: undefined option: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [ -z "${NPU_ARCH}" ]; then
    echo "ERROR: --npu-arch is required." >&2
    usage
    exit 2
fi
sk_validate_npu_arch "${NPU_ARCH}"

echo "---------------- Start running examples ----------------"
if [[ "${NPU_ARCH}" == "dav-2201" ]]; then
    "${PYTHON_CMD}" "${BASEPATH}/jit/example01_super_kernel_base/superkernel_scope.py"
    "${PYTHON_CMD}" "${BASEPATH}/jit/example02_super_kernel_profiling/superkernel_compare.py"
    "${PYTHON_CMD}" \
        "${BASEPATH}/jit/example03_super_kernel_runtime_ascendc_only/superkernel_runtime_ascendc_basic.py"
else
    echo "[INFO] Skipping SuperKernel JIT examples on ${NPU_ARCH}."
fi

export SK_NPU_ARCH="${NPU_ARCH}"
export PYTHON_CMD
bash "${BASEPATH}/aot/example01_dual_stream/run.sh"
bash "${BASEPATH}/aot/example02_sk_options/run.sh"
bash "${BASEPATH}/aot/example03_kernel_pybind/run.sh"

echo "Run all examples success"
