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

echo $(grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2)
if [[ "${task_name}" == *ubuntu24* ]]; then
    export PATH=/opt/buildtools/python-3.10.2/bin:$PATH
else
    if [[ -f "/opt/rh/devtoolset-7/enable" ]]; then
        echo "source devtoolset"
        source /opt/rh/devtoolset-7/enable
    fi
    rm -rf /home/jenkins/opensource/lib_cache
    ln -s /home/jenkins/opensource/ubuntu20/lib_cache /home/jenkins/opensource/lib_cache
fi

select_compilers() {
    local selected_cc="${CC:-}"
    local selected_cxx="${CXX:-}"
    local gcc_version="${GCC_VERSION:-}"

    if [[ -n "${selected_cc}" && -z "${selected_cxx}" ]]; then
        if [[ "${selected_cc}" =~ ^gcc-([0-9]+)$ ]]; then
            selected_cxx="g++-${BASH_REMATCH[1]}"
        fi
    fi

    if [[ -n "${selected_cxx}" && -z "${selected_cc}" ]]; then
        if [[ "${selected_cxx}" =~ ^g\+\+-([0-9]+)$ ]]; then
            selected_cc="gcc-${BASH_REMATCH[1]}"
        fi
    fi

    if [[ -z "${selected_cc}" || -z "${selected_cxx}" ]]; then
        if [[ -n "${gcc_version}" ]]; then
            selected_cc="${selected_cc:-gcc-${gcc_version}}"
            selected_cxx="${selected_cxx:-g++-${gcc_version}}"
        elif [[ "${task_name}" == *ubuntu24* ]]; then
            selected_cc="${selected_cc:-gcc-14}"
            selected_cxx="${selected_cxx:-g++-14}"
        else
            selected_cc="${selected_cc:-gcc}"
            selected_cxx="${selected_cxx:-g++}"
        fi
    fi

    export CC="${selected_cc}"
    export CXX="${selected_cxx}"
    echo "Using CC=${CC}"
    echo "Using CXX=${CXX}"
}

select_compilers

if [[ "${task_name}" =~ Compile_Ascend_X86_ubuntu24 ]]; then
    sed -i "1i set(CMAKE_EXPORT_COMPILE_COMMANDS ON)" "CMakeLists.txt"
    echo "api-check=compile" >> "${ATOMGIT_OUTPUT}"
else
    echo "api-check=continue" >> "${ATOMGIT_OUTPUT}"
fi

if [[ "${target_branch}" == "master" ]] || [[ "${target_branch}" == "develop" ]]; then
    sudo update-alternatives --set lcov /opt/lcov-2.3.2/bin/lcov
    lcov --version
fi

"${CC}" --version
"${CXX}" --version
source /home/jenkins/Ascend/cann/bin/setenv.bash
set +e

pip3 install -r super_kernel/requirements-dev.txt

echo "exec cmd: [bash build.sh --pkg --cann_3rd_lib_path="/home/jenkins/opensource"]"
bash build.sh --pkg --cann_3rd_lib_path="/home/jenkins/opensource"
ret=$?
exit $ret
