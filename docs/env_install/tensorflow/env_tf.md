# TensorFlow 环境编译部署

## 前置准备

TensorFlow 及 TF Adapter 的直接安装（非源码编译路线）不依赖 CANN 包，仅在用例运行时依赖 CANN 包。请根据以下步骤完成前置准备：

1. 通过 [快速安装](../../zh/quick_install.md) 或 [安装 CANN - 昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/softwareinst/instg/instg_0000.html) 正确安装 toolkit 和 ops 包，并配置环境变量（用例运行依赖）。
2. 通过下方步骤搭建 TensorFlow 环境。

> **注意**：下文中的 `/mnt/workspace` 为华为云开发者环境挂载目录，可根据实际环境替换。

---

## 一、创建虚拟环境

x86_64 与 aarch64 两种架构均使用 pyenv 安装指定 Python 版本并创建虚拟环境，步骤一致。

### 1. 安装 pyenv

```bash
cd /mnt/workspace
git clone https://github.com/pyenv/pyenv.git

# 临时执行如下环境变量（建议将其添加到系统环境变量 PATH 中）
export PYENV_ROOT="/mnt/workspace/pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
```

### 2. 安装指定 Python 版本

根据所需 TF 版本选择 Python 版本：
- TF 1.15.0 → Python 3.7
- TF 2.6.5 → Python 3.9

```bash
# 安装 python3.7（指定国内源下载）
PYTHON_BUILD_MIRROR_URL="https://mirrors.huaweicloud.com/python" \
PYTHON_BUILD_MIRROR_URL_SKIP_CHECKSUM=1 \
pyenv install 3.7.11

# 或安装 python3.9
PYTHON_BUILD_MIRROR_URL="https://mirrors.huaweicloud.com/python" \
PYTHON_BUILD_MIRROR_URL_SKIP_CHECKSUM=1 \
pyenv install 3.9.25
```

### 3. 创建并激活虚拟环境

```bash
# 创建虚拟环境（TF 1.15）
/mnt/workspace/pyenv/versions/3.7.11/bin/python3.7 -m venv /mnt/workspace/venv/test_tf1

# 或创建虚拟环境（TF 2.6.5）
/mnt/workspace/pyenv/versions/3.9.25/bin/python3.9 -m venv /mnt/workspace/venv/test_tf2

# 激活虚拟环境
source /mnt/workspace/venv/test_tf1/bin/activate   # TF 1.15
# 或
source /mnt/workspace/venv/test_tf2/bin/activate   # TF 2.6.5

# 升级 pip
pip install --upgrade pip
```

> **注意**：后续所有命令默认都在已激活的虚拟环境中执行，请勿退出该环境。

---

## 二、安装 TensorFlow

### x86_64 架构

x86_64 架构下 TF 1.15 和 TF 2.6.5 均有官方预编译 wheel，可直接 pip 安装。

```bash
# TF 1.15
pip3 install tensorflow==1.15.0
export CUDA_VISIBLE_DEVICES=-1

# 或 TF 2.6.5
pip3 install tensorflow==2.6.5
export CUDA_VISIBLE_DEVICES=-1
```

### aarch64 架构

aarch64 架构下 TF 无官方预编译 wheel，需要源码编译。TF 1.15.0 和 TF 2.6.5 的完整编译流程（含 nsync 修改、bazel 编译、TF Adapter 编译）请参考 [aarch64 架构 TF 源码编译](build_tf_aarch64.md)，文档内对两个版本的差异（Python、numpy、h5py、bazel、TF Adapter 路径等）有分别说明。

---

## 三、安装框架插件包

### x86_64 架构

```bash
# TF 1.15（npu_bridge）
wget https://gitcode.com/cann/tensorflow/releases/download/tfa_v0.0.48_9.0.0/npu_bridge-1.15.0-py3-none-manylinux2014_x86_64.whl
pip3 install npu_bridge-1.15.0-py3-none-manylinux2014_x86_64.whl --force-reinstall

# 或 TF 2.6.5（npu_device）
wget https://gitcode.com/cann/tensorflow/releases/download/tfa_v0.0.48_9.0.0/npu_device-2.6.5-py3-none-manylinux2014_x86_64.whl
pip3 install npu_device-2.6.5-py3-none-manylinux2014_x86_64.whl --force-reinstall
```

### aarch64 架构

TF 1.15（npu_bridge）：需修改 tfa 源码（`tf_adapter/`）三处后编译，请参考 [aarch64 架构 TF 源码编译](build_tf_aarch64.md) 中的「编译安装 TF Adapter」章节。

TF 2.6.5（npu_device）：使用 tfa 源码中独立的 `tf_adapter_2.x/` 构建系统，无需修改源码。

```bash
git clone https://gitcode.com/cann/tensorflow.git
cd tensorflow/tf_adapter_2.x && bash build.sh -c -j8
pip3 install build/dist/python/dist/npu_device-2.6.5-py3-none-manylinux2014_aarch64.whl --upgrade --no-deps
```


---

## 四、安装其他依赖

```bash
# TF 1.15
pip3 install "numpy==1.18.5" pandas decorator sympy scipy attrs psutil protobuf==3.19.0
# TF 2.6.5
pip3 install "numpy==1.23.5" pandas decorator sympy scipy attrs psutil protobuf==3.19.0
```

> numpy 必须与其它包同条命令安装以锁定版本，否则 pandas/scipy 会将 numpy 升级到 2.x，破坏 h5py 的 ABI 兼容性。

---

## 五、一键配置脚本（仅 x86_64）

x86_64 架构可使用一键配置脚本自动完成上述步骤（在线安装 TF、NPU Adapter 及依赖）：

```bash
bash scripts/env_install/setup_tf_env.sh
```

脚本完成后激活环境：

```bash
source scripts/env_install/env/activate_tf1.sh    # TF 1.15
# 或
source scripts/env_install/env/activate_tf2.sh    # TF 2.6.5
```

> **aarch64 架构不适用此脚本**：aarch64 请按 [aarch64 架构 TF 源码编译](build_tf_aarch64.md) 手动编译。
