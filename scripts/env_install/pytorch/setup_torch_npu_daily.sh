#!/bin/bash
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------------------

# 使用前置条件：
# 1. 已经安装 NPU 驱动、CANN Toolkit 和对应设备的 OPS 包。
# 2. 当前用户具有安装 Python 编译依赖的 sudo 权限。
# 3. torch_npu 版本要求为 2.9.0 及以上，默认安装 2.10.0，可自行修改。

# 任意一条命令执行失败时，立即终止脚本。
set -e

# 第一个参数：Python 版本。
# 没有指定时，默认使用 3.11.4。
PYTHON_VERSION="${1:-3.11.4}"

# 检查 Python 版本格式，避免通过非法参数构造越界路径。
if [[ ! "$PYTHON_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Python 版本格式错误，应为 x.y.z，例如 3.11.4。"
    exit 1
fi

# 第二个参数：torch_npu 版本。
# 没有指定时，默认使用 2.10.0。
TORCH_NPU_VERSION="${2:-2.10.0}"

# 第三个参数：环境安装目录。
# 没有指定时，默认使用 /mnt/workspace/env。
ENV_DIR="${3:-/mnt/workspace/env}"

PYENV_ROOT="$ENV_DIR/pyenv"
PYTHON_DIR="$PYENV_ROOT/versions/$PYTHON_VERSION"
BASE_PYTHON="$PYTHON_DIR/bin/python"
VENV_DIR="$ENV_DIR/venv/torch210_daily"
PKG_DIR="$ENV_DIR/torch_npu_pkg"

# 一、创建虚拟环境。

# 查看当前操作系统。
cat /etc/os-release

# 安装 Python 编译依赖。
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update

    sudo apt-get install -y \
        build-essential \
        git \
        wget \
        curl \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libncurses-dev \
        xz-utils \
        tk-dev \
        libffi-dev \
        liblzma-dev
elif command -v yum >/dev/null 2>&1; then
    sudo yum install -y \
        gcc \
        gcc-c++ \
        make \
        git \
        wget \
        curl \
        openssl-devel \
        zlib-devel \
        bzip2-devel \
        readline-devel \
        sqlite-devel \
        ncurses-devel \
        xz-devel \
        tk-devel \
        libffi-devel
else
    echo "未找到 apt-get 或 yum，请手动安装 Python 编译依赖。"
    exit 1
fi

# 创建环境目录。
mkdir -p "$ENV_DIR"

cd "$ENV_DIR"

# 安装 pyenv，已经安装时直接复用。
if [ ! -x "$PYENV_ROOT/bin/pyenv" ]; then
    rm -rf "$PYENV_ROOT"
    git clone https://github.com/pyenv/pyenv.git "$PYENV_ROOT"
fi

# 配置 pyenv 环境变量。
export PYENV_ROOT
export PATH="$PYENV_ROOT/bin:$PATH"

# 检查 Python 是否存在，以及关键扩展模块是否正常。
if [ -x "$BASE_PYTHON" ] &&
    "$BASE_PYTHON" - <<'EOF' >/dev/null 2>&1
import ssl
import sqlite3
import readline
import curses
import bz2
import lzma
import ctypes
EOF
then
    echo "Python $PYTHON_VERSION 已安装且关键扩展模块正常，直接复用。"
else
    echo "Python $PYTHON_VERSION 不存在或编译不完整，重新安装。"

    # 删除不完整的 Python 和基于该 Python 创建的虚拟环境。
    rm -rf "$PYTHON_DIR"
    rm -rf "$VENV_DIR"

    # 安装指定 Python 版本。
    PYTHON_BUILD_MIRROR_URL="https://mirrors.huaweicloud.com/python" \
    PYTHON_BUILD_MIRROR_URL_SKIP_CHECKSUM=1 \
    pyenv install "$PYTHON_VERSION"
fi

# 创建虚拟环境，已经存在时直接复用。
if [ ! -x "$VENV_DIR/bin/python" ]; then
    rm -rf "$VENV_DIR"
    mkdir -p "$ENV_DIR/venv"

    "$BASE_PYTHON" \
        -m venv "$VENV_DIR"
fi

# 激活虚拟环境。
source "$VENV_DIR/bin/activate"

# 二、安装 PyTorch 环境依赖。

# 安装 NumPy。
python -m pip install numpy

# 进入环境目录。
cd "$ENV_DIR"

# 获取当天日期。
DATE="$(date +%Y%m%d)"

# 获取 Python 版本标识，例如 Python 3.11 对应 py311。
PY_TAG="$(
    python -c 'import sys; print(f"py{sys.version_info.major}{sys.version_info.minor}")'
)"

# 获取当前机器架构，例如 aarch64 或 x86_64。
ARCH="$(uname -m)"

# 根据版本、日期和 Python 版本生成 Daily 包下载地址。
URL="https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/Daily/v${TORCH_NPU_VERSION}/${DATE}.1/pytorch_v${TORCH_NPU_VERSION}_${PY_TAG}.tar.gz"

# 判断当天的 Daily 包是否已经发布。
if ! wget --spider -q "$URL"; then
    echo "当天的 torch_npu Daily 包尚未发布，或者下载链接不可访问。"
    exit 1
fi

# 清理并创建下载目录。
rm -rf "$PKG_DIR"
mkdir -p "$PKG_DIR"

# 下载并解压 Daily 包。
wget -O "$PKG_DIR/pytorch.tar.gz" "$URL"

tar -xzf "$PKG_DIR/pytorch.tar.gz" \
    -C "$PKG_DIR"

# 查找与当前机器架构匹配的 torch_npu Wheel。
WHEEL="$(
    find "$PKG_DIR" \
        -type f \
        -name "torch_npu*${ARCH}.whl" |
        head -n 1
)"

# 安装 Daily torch_npu Wheel。
python -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --force-reinstall "$WHEEL"

# 所有步骤执行成功后输出提示。
echo "SUCCESS: PyTorch 和 torch_npu 环境安装成功。"
