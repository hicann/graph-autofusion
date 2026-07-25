#!/bin/bash
# ==============================================================================
# TF 环境一键配置脚本（仅 x86_64，支持 TF 1.15.0 和 TF 2.6.5）

# 通过 PyPI 和 GitCode 在线安装。
#
# 用法：
#   bash setup_tf_env.sh                       # 默认安装在脚本同级 env/ 目录
#   bash setup_tf_env.sh /custom/install/path  # 指定安装根目录
#
# 前置条件：
#   - 已安装 CANN toolkit 和 driver
#   - 有 sudo 权限
#   - 有网络
#
# 完成后产出：
#   $INSTALL_ROOT/venv/test_tf1/ 或 test_tf2/  # TF 虚拟环境
#   $INSTALL_ROOT/activate_tf1.sh 或 activate_tf2.sh  # 激活脚本
#
# 注意：aarch64 架构不支持本脚本，请参考 docs/env_install/tensorflow/build_tf_aarch64.md 手动编译。
# ==============================================================================
set -e

# ==================== 路径与变量定义 ====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_ROOT="${1:-$SCRIPT_DIR/env}"
ENV_ROOT="$INSTALL_ROOT"

# 镜像与版本
PY_MIRROR="https://mirrors.huaweicloud.com/python"
PY37_VERSION="3.7.11"
PY39_VERSION="3.9.25"

# 日志
log() { echo -e "\033[32m[SETUP]\033[0m $1"; }
warn() { echo -e "\033[33m[WARN]\033[0m $1"; }
err() { echo -e "\033[31m[ERROR]\033[0m $1" >&2; }

# ==================== 环境探测 ====================
log "=== 环境探测 ==="

# 架构检查（仅支持 x86_64）
ARCH=$(uname -m)
if [ "$ARCH" != "x86_64" ]; then
    err "本脚本仅支持 x86_64 架构，当前: $ARCH"
    err "aarch64 架构请参考 docs/env_install/tensorflow/build_tf_aarch64.md 手动编译"
    exit 1
fi
log "架构: $ARCH"

# 探测 CANN 路径
CANN_PATH=""
for candidate in /usr/local/Ascend/ascend-toolkit /home/developer/Ascend/cann-9.0.0 /home/developer/Ascend/cann-9.1.0; do
    if [ -f "$candidate/set_env.sh" ]; then
        CANN_PATH="$candidate"
        break
    fi
done
if [ -z "$CANN_PATH" ]; then
    for candidate in /usr/local/Ascend/cann-* /home/*/Ascend/cann-*; do
        if [ -f "$candidate/set_env.sh" ]; then
            CANN_PATH="$candidate"
            break
        fi
    done
fi
if [ -z "$CANN_PATH" ]; then
    err "未找到 CANN toolkit，请先安装 CANN（需要 set_env.sh）"
    exit 1
fi
log "CANN 路径: $CANN_PATH"

# 探测 driver setenv
DRIVER_SETENV=""
for candidate in /usr/local/Ascend/driver/bin/setenv.bash /usr/local/Ascend/driver/bin/setenv.sh; do
    if [ -f "$candidate" ]; then
        DRIVER_SETENV="$candidate"
        break
    fi
done
if [ -n "$DRIVER_SETENV" ]; then
    log "Driver setenv: $DRIVER_SETENV"
else
    warn "未找到 driver setenv（可能已在 LD_LIBRARY_PATH 中）"
fi

# ==================== 检测 Python 版本，选择 TF 路径 ====================
# 支持通过环境变量 TF_MODE（tf1/tf2）强制指定路径，绕过系统 Python 检测
if [ "${TF_MODE:-}" = "tf1" ] || [ "${TF_MODE:-}" = "tf2" ]; then
    if [ "$TF_MODE" = "tf1" ]; then
        SYSTEM_PY_MAJOR=3; SYSTEM_PY_MINOR=7
    else
        SYSTEM_PY_MAJOR=3; SYSTEM_PY_MINOR=9
    fi
    log "TF_MODE=$TF_MODE，强制选择路径（Python ${SYSTEM_PY_MAJOR}.${SYSTEM_PY_MINOR}）"
else
    SYSTEM_PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
    SYSTEM_PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
    log "系统 Python 版本: ${SYSTEM_PY_MAJOR}.${SYSTEM_PY_MINOR}"
fi

if [ "$SYSTEM_PY_MAJOR" -eq 3 ] && [ "$SYSTEM_PY_MINOR" -eq 7 ]; then
    log "=== 选择 TF 1.15.0 路径（Python 3.7）==="
    TF_MODE="tf1"
    PY_VERSION="$PY37_VERSION"
    VENV_NAME="test_tf1"
    NPU_PACKAGE="npu_bridge"
    ACTIVATE_NAME="activate_tf1.sh"
    NUMPY_VERSION="1.18.5"
else
    log "=== 选择 TF 2.6.5 路径（Python ${SYSTEM_PY_MAJOR}.${SYSTEM_PY_MINOR}）==="
    TF_MODE="tf2"
    PY_VERSION="$PY39_VERSION"
    VENV_NAME="test_tf2"
    NPU_PACKAGE="npu_device"
    ACTIVATE_NAME="activate_tf2.sh"
    NUMPY_VERSION="1.23.5"
fi

# venv 内 Python 的 major.minor（用于生成激活脚本中的库路径）
VENV_PY_VER="${PY_VERSION%.*}"

PYENV_ROOT="$ENV_ROOT/pyenv"
VENV="$ENV_ROOT/venv/$VENV_NAME"

log "安装根目录: $INSTALL_ROOT"
mkdir -p "$ENV_ROOT"

# ==================== 阶段 1：系统依赖 ====================
log "=== 阶段 1/5：安装系统依赖 ==="
sudo apt-get update -qq
sudo apt-get install -y -qq \
    libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
    libgdbm-dev liblzma-dev tk-dev \
    build-essential cmake unzip wget git > /dev/null 2>&1
log "系统依赖安装完成"

# ==================== 阶段 2：Python 环境 ====================
log "=== 阶段 2/5：创建 Python ${PY_VERSION} 环境 ==="

# 2.1 pyenv（优先 gitee 镜像，GitHub 网络差时兜底）
if [ ! -d "$PYENV_ROOT" ]; then
    log "下载 pyenv..."
    if ! git clone -q --depth 1 https://gitee.com/mirrors/pyenv.git "$PYENV_ROOT" 2>/dev/null; then
        warn "gitee 镜像失败，回退 GitHub..."
        git clone -q https://github.com/pyenv/pyenv.git "$PYENV_ROOT"
    fi
else
    log "pyenv 已存在，跳过"
fi
export PYENV_ROOT
export PATH="$PYENV_ROOT/bin:$PATH"

# 2.2 安装指定 Python 版本
if [ ! -d "$PYENV_ROOT/versions/$PY_VERSION" ]; then
    log "安装 python${PY_VERSION}（华为云镜像，约 15 分钟）..."
    PYTHON_BUILD_MIRROR_URL="$PY_MIRROR" PYTHON_BUILD_MIRROR_URL_SKIP_CHECKSUM=1 \
        pyenv install "$PY_VERSION"
else
    log "python${PY_VERSION} 已存在，跳过"
fi
PY_BIN="$PYENV_ROOT/versions/$PY_VERSION/bin/python3"

# 2.3 虚拟环境
if [ ! -f "$VENV/bin/activate" ]; then
    log "创建虚拟环境 ${VENV_NAME}..."
    "$PY_BIN" -m venv "$VENV"
fi

# 升级 pip
log "升级 pip..."
"$VENV/bin/pip3" install --upgrade pip -q 2>/dev/null || true
log "Python 环境就绪: $VENV"

# ==================== 阶段 3：安装 TensorFlow 及 NPU Adapter ====================
log "=== 阶段 3/5：安装 TensorFlow 及 NPU Adapter ==="

log "x86_64 架构，在线安装..."
if [ "$TF_MODE" = "tf1" ]; then
    "$VENV/bin/pip3" install tensorflow==1.15.0 -q 2>/dev/null
    "$VENV/bin/pip3" install h5py -q 2>/dev/null
else
    "$VENV/bin/pip3" install tensorflow==2.6.5 -q 2>/dev/null
    "$VENV/bin/pip3" install h5py -q 2>/dev/null
fi
log "TensorFlow 安装完成"

log "安装 NPU Adapter（从 GitCode 下载）..."
if [ "$TF_MODE" = "tf1" ]; then
    NPU_WHL_URL="https://gitcode.com/cann/tensorflow/releases/download/tfa_v0.0.48_9.0.0/npu_bridge-1.15.0-py3-none-manylinux2014_x86_64.whl"
else
    NPU_WHL_URL="https://gitcode.com/cann/tensorflow/releases/download/tfa_v0.0.48_9.0.0/npu_device-2.6.5-py3-none-manylinux2014_x86_64.whl"
fi
wget -q "$NPU_WHL_URL" -O /tmp/npu_adapter.whl 2>/dev/null && \
    "$VENV/bin/pip3" install /tmp/npu_adapter.whl --upgrade --no-deps -q 2>/dev/null && \
    rm -f /tmp/npu_adapter.whl || warn "NPU Adapter 下载失败，请手动安装"

# ==================== 阶段 4：Python 依赖 ====================
log "=== 阶段 4/5：安装 Python 依赖 ==="
"$VENV/bin/pip3" install Cython==0.29.14 wheel -q 2>/dev/null
if [ "$TF_MODE" = "tf1" ]; then
    "$VENV/bin/pip3" install numpy==${NUMPY_VERSION} --no-build-isolation -q
else
    "$VENV/bin/pip3" install "numpy==${NUMPY_VERSION}" -q
fi
"$VENV/bin/pip3" install protobuf==3.19.0 -q
"$VENV/bin/pip3" install -U keras_preprocessing==1.1.2 --no-deps -q
# 通过 constraints 锁定 numpy 版本，避免 pandas/scipy 把 numpy 升级到 2.x（破坏 h5py 与 TF 的 ABI）
CONSTRAINTS_FILE=$(mktemp)
echo "numpy==${NUMPY_VERSION}" > "$CONSTRAINTS_FILE"
"$VENV/bin/pip3" install -c "$CONSTRAINTS_FILE" pandas decorator sympy scipy attrs psutil -q
rm -f "$CONSTRAINTS_FILE"
log "Python 依赖安装完成"

# ==================== 阶段 5：验证与生成激活脚本 ====================
log "=== 阶段 5/5：验证与生成激活脚本 ==="

# 验证 import
log "验证 import..."
if ! "$VENV/bin/python3" -c "import tensorflow; print('TF', tensorflow.__version__)" 2>&1; then
    warn "TensorFlow import 失败（可能需要重启终端或检查依赖）"
fi
if ! "$VENV/bin/python3" -c "import $NPU_PACKAGE; print('${NPU_PACKAGE} OK')" 2>&1; then
    warn "${NPU_PACKAGE} import 失败"
fi
"$VENV/bin/python3" -c "import h5py; print('h5py', h5py.__version__)" 2>&1 || warn "h5py import 失败"
log "import 验证完成"

# 生成激活脚本
log "生成激活脚本..."
ACTIVATE_SCRIPT="$ENV_ROOT/$ACTIVATE_NAME"
cat > "$ACTIVATE_SCRIPT" << EOF
#!/bin/bash
# TF 环境（${TF_MODE}）激活脚本
# 用法: source $ACTIVATE_SCRIPT

# 激活虚拟环境
source "$VENV/bin/activate"

# CANN & driver 环境
$([ -n "$DRIVER_SETENV" ] && echo "source $DRIVER_SETENV")
source "$CANN_PATH/set_env.sh"

# NPU 设备
export ASCEND_DEVICE_ID=\${ASCEND_DEVICE_ID:-0}

# Autofuse 开关
export AUTOFUSE_FLAGS="--enable_autofuse=true"

# TF 动态库路径
export LD_LIBRARY_PATH="$VENV/lib/python${VENV_PY_VER}/site-packages/tensorflow_core:\${LD_LIBRARY_PATH:-}"

echo "TF 环境（${TF_MODE}）已激活"
echo "CANN: $CANN_PATH"
echo "VENV: $VENV"
EOF
chmod +x "$ACTIVATE_SCRIPT"

# ==================== 完成 ====================
log ""
log "========================================"
log "  环境配置完成！"
log "========================================"
log ""
log "激活环境："
log "  source $ACTIVATE_SCRIPT"
log ""
if [ "$TF_MODE" = "tf1" ]; then
    log "运行样例："
    log "  cd <autofuse>/examples/tensorflow/tf1/af_eleandele"
    log "  python3 test_abs_relu_exp.py"
fi
log ""
