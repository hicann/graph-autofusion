#!/bin/bash
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------------------

set -e

BASEPATH=$(cd "$(dirname $0)"; pwd)
BUILD_PATH="${BASEPATH}/build"
OUTPUT_PATH="${BASEPATH}/build_out"

# Detect Python command to use
if [ -n "$VIRTUAL_ENV" ]; then
  PYTHON_CMD="python3"
elif [ -f "${BASEPATH}/venv/bin/python" ]; then
  PYTHON_CMD="${BASEPATH}/venv/bin/python"
else
  PYTHON_CMD="python3"
fi

# print usage message
usage() {
  echo "Usage:"
  echo "  sh build.sh [-h|--help] [--pkg] [-u|--ut] [-s|--st] [-c|--coverage]"
  echo "              [--output_path=<PATH>]"
  echo ""
  echo "Options:"
  echo "    -h, --help            Print usage"
  echo "    --pkg                 Build run package"
  echo "    -u, --ut              Run all unit test"
  echo "        =superkernel      Run superkernel unit test"
  echo "    -s, --st              Run all system test"
  echo "        =superkernel      Run superkernel system test"
  echo "    -c, --coverage        Run tests with coverage report generation"
  echo "    --output_path=<PATH>"
  echo "                          Set output path, where the run package will be generated, default ./build_out"
  echo "    --run_example         Run all examples"
  echo "        =superkernel      Run superkernel examples"
  echo ""
}

# parse and set options
checkopts() {
  if [ $# -eq 0 ]; then
    echo "ERROR: 'build.sh' has no options available, please select at least one option!"
    usage
    exit 1
  fi

  ENABLE_BUILD_PACKAGE="off"
  ENABLE_UT="off"
  ENABLE_ST="off"
  ENABLE_COVERAGE="off"
  ENABLE_SUPERKERNEL_UT="off"
  ENABLE_SUPERKERNEL_ST="off"
  ENABLE_RUN_EXAMPLE="off"
  ENABLE_SUPERKERNEL_RUN_EXAMPLE="off"

  # Process the options
  parsed_args=$(getopt -a -o j:hu::s::c -l help,pkg,run_example::,ut::,st::,coverage,output_path: -- "$@") || {
    usage
    exit 1
  }

  eval set -- "$parsed_args"

  while true; do
    case "$1" in
      -h | --help)
        usage
        exit 0
        ;;
      --pkg)
        ENABLE_BUILD_PACKAGE="on"
        shift
        ;;
      -u | --ut)
        ENABLE_UT="on"
        case "$2" in
          "")
            ENABLE_SUPERKERNEL_UT="on"
            shift 2
            ;;
          "superkernel")
            ENABLE_SUPERKERNEL_UT="on"
            shift 2
            ;;
          *)
            usage
            exit 1
        esac
        ;;
      -s | --st)
        ENABLE_ST="on"
        case "$2" in
          "")
            ENABLE_SUPERKERNEL_ST="on"
            shift 2
            ;;
          "superkernel")
            ENABLE_SUPERKERNEL_ST="on"
            shift 2
            ;;
          *)
            usage
            exit 1
        esac
        ;;
      -c | --coverage)
        ENABLE_COVERAGE="on"
        shift
        ;;
      --run_example)
        ENABLE_RUN_EXAMPLE="on"
        case "$2" in
          "")
            ENABLE_SUPERKERNEL_RUN_EXAMPLE="on"
            shift 2
            ;;
          "superkernel")
            ENABLE_SUPERKERNEL_RUN_EXAMPLE="on"
            shift 2
            ;;
          *)
            usage
            exit 1
        esac
        ;;
      --output_path)
        OUTPUT_PATH="$(realpath $2)"
        shift 2
        ;;
      --)
        shift
        break
        ;;
      *)
        echo "Undefined option: $1"
        usage
        exit 1
        ;;
    esac
  done
}

build_package() {
  echo "---------------- Start build run package ----------------"
  mkdir -pv ${BUILD_PATH} &&
  cd ${BUILD_PATH} &&
  cmake .. && make package &&
  mkdir -pv ${OUTPUT_PATH} &&
  cp _CPack_Packages/makeself_staging/cann-graph-autofusion*.run ${OUTPUT_PATH} &&
  output_run_path=`ls -1 ${OUTPUT_PATH}/cann-graph-autofusion*.run 2>/dev/null` &&
  echo "Buid run package success!" &&
  echo "package: ${output_run_path}"
}

superkernel_run_example() {
  echo "---------------- Start running examples ----------------"
  ${PYTHON_CMD} ${BASEPATH}/super_kernel/examples/super_kernel_base/superkernel_scope.py
  ${PYTHON_CMD} ${BASEPATH}/super_kernel/examples/super_kernel_profiling/superkernel_compare.py
}

superkernel_ut() {
  echo "---------------- Start UT ----------------"
  cd ${BASEPATH}/super_kernel &&
  if [ "X$ENABLE_COVERAGE" == "Xon" ]; then
    ${PYTHON_CMD} -m pytest tests/ut -m ut \
                                     --cov-config=scripts/sk_ut_cfg.toml \
                                     --cov=src/superkernel \
                                     --cov-report=term-missing \
                                     --cov-report=html \
                                     --cov-report=xml
  else
    ${PYTHON_CMD} -m pytest tests/ut -m ut
  fi
}

superkernel_st() {
  echo "---------------- Start ST ----------------"
  cd ${BASEPATH}/super_kernel &&
  if [ "X$ENABLE_COVERAGE" == "Xon" ]; then
    ${PYTHON_CMD} -m pytest tests/st -m st --cov-config=scripts/sk_st_cfg.toml \
                                     --cov=src/superkernel \
                                     --cov-report=term-missing \
                                     --cov-report=html \
                                     --cov-report=xml
  else
    ${PYTHON_CMD} -m pytest tests/st -m st
  fi
}

main() {
  checkopts "$@"

  # Clean coverage data if coverage is enabled
  if [ "X$ENABLE_COVERAGE" == "Xon" ]; then
    # All UT and ST is enabled when only '-c' without '-u' or '-s'
    if [ "X$ENABLE_UT" != "Xon" ] && [ "X$ENABLE_ST" != "Xon" ]; then
      ENABLE_UT="on"
      ENABLE_ST="on"
      ENABLE_SUPERKERNEL_UT="on"
      ENABLE_SUPERKERNEL_ST="on"
    fi
    echo "---------------- Clean Coverage Data ----------------"
    cd ${BASEPATH}/super_kernel && ${PYTHON_CMD} -m coverage erase
  fi

  if [ "X$ENABLE_BUILD_PACKAGE" == "Xon" ]; then
    build_package || { echo "Build run package failed."; exit 1; }
  fi

  if [ "X$ENABLE_UT" == "Xon" ]; then
    if [ "X$ENABLE_SUPERKERNEL_UT" == "Xon" ]; then
      superkernel_ut || { echo "Run superkernel UT failed."; exit 1; }
    fi
  fi

  if [ "X$ENABLE_ST" == "Xon" ]; then
    if [ "X$ENABLE_SUPERKERNEL_ST" == "Xon" ]; then
      superkernel_st || { echo "Run superkernel ST failed."; exit 1; }
    fi
  fi

  if [ "X$ENABLE_RUN_EXAMPLE" == "Xon" ]; then
    if [ "X$ENABLE_SUPERKERNEL_RUN_EXAMPLE" == "Xon" ]; then
      superkernel_run_example || { echo "Run examples failed."; exit 1; }
    fi
  fi
}

main "$@"
