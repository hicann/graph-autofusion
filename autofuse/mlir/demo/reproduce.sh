#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# =============================================================================
# reproduce.sh — 三维 LoadAddStore 从 AFIR 到执行的一键复现
#
# 用例：out = in0 + in1，float16，具体 shape <100x16x8xf16>（12800 elems）。
# 生成的 kernel 完全 shape-agnostic：tile_len/total 是运行期 TilingData 字段，
# 核数由 launch 的 blockDim 决定，同一份 kernel.bin 覆盖对齐/非对齐尾块 + 单/多核。
#
# 子命令（默认 all = build+pipeline+sim）：
#   build      构建 af-opt + ascir-translate（容器内）
#   pipeline   ③lowering + ④翻译（AFIR -> ascendc IR -> Ascend C 源码）
#   sim        CPU 功能仿真（ccec --run-mode=cpu + libpem_davinci），全量数值校验
#   camodel    camodel 周期精确仿真（同一 kernel.bin，无硬件）
#   realnpu    真机 910C（需 real-npu env，见下）；跑 total×blockDim 9 组合
#   all        build + pipeline + sim
#
# 前置：docker + 镜像 autofuse-mlir-dev:arm64 + CANN 卷 af-cann-910。
# 真机还需 real-npu 环境文件（见 REALNPU_ENV），且只碰 /data/nyh、device 5。
#
# 所有路径都可用环境变量覆盖，默认值不写死任何机器专属绝对路径：
#   SRC          autofuse 仓库根（默认从脚本位置推导，无需设）
#   BUILD        持久 build 目录（默认 $HOME/afmlir-build，或设 AFMLIR_BUILD）
#   IMAGE        docker 镜像名
#   CANN_VOL     CANN docker 卷名
#   REALNPU_ENV  真机 SSH 环境文件（仅 realnpu 子命令需要）
# 例：BUILD=/data/me/afbuild ./reproduce.sh sim
# =============================================================================
set -euo pipefail

# 脚本自身位置：<SRC>/autofuse/mlir/demo/reproduce.sh
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# --- 路径（均可用环境变量覆盖；默认值不含机器专属绝对路径）------------------
# 仓库根从脚本位置推导（demo -> mlir -> autofuse -> 仓库根，上溯 3 层）。
SRC=${SRC:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
# 持久 build 目录：默认放项目根目录下（方便查找，已在 .gitignore）。
BUILD=${BUILD:-${AFMLIR_BUILD:-$SRC/build-afmlir-demo}}

# --- 根据系统架构自动选择镜像 -----------------------------------------------
# 支持通过环境变量 IMAGE 覆盖，否则根据架构自动选择
detect_arch() {
  local arch=$(uname -m)
  case "$arch" in
    x86_64|i686|i386)
      echo "x86_64"
      ;;
    aarch64|arm64)
      echo "arm64"
      ;;
    *)
      echo "unknown"
      ;;
  esac
}

ARCH=${ARCH:-$(detect_arch)}
case "$ARCH" in
  x86_64)
    DEFAULT_IMAGE="swr.cn-east-2.myhuaweicloud.com/ascendmlir/autofuse-mlir-dev:x86_64"
    CANN_ARCH="x86_64-linux"
    ;;
  *)
    DEFAULT_IMAGE="swr.cn-east-2.myhuaweicloud.com/ascendmlir/autofuse-mlir-dev:arm64"
    CANN_ARCH="aarch64-linux"
    [ "$ARCH" != "arm64" ] && echo "警告：未知的系统架构 '$ARCH'，请手动设置 IMAGE 环境变量" >&2
    ;;
esac
IMAGE=${IMAGE:-$DEFAULT_IMAGE}
echo "检测到系统架构: $ARCH，使用镜像: $IMAGE"

# CANN_VOL 支持：docker卷名 或 本地CANN路径
CANN_VOL=${CANN_VOL:-af-cann-910}
# 检测 CANN_VOL 是本地目录还是 docker volume
if [ -d "$CANN_VOL" ]; then
  echo "使用本地 CANN: $CANN_VOL"
else
  echo "使用 CANN docker 卷: $CANN_VOL"
fi
# 真机 SSH 环境文件：无通用默认（连接信息因人而异），仅 realnpu 子命令用到。
REALNPU_ENV=${REALNPU_ENV:-}

# 版本化的 demo 源（本目录），会被 stage 到 build 工作目录
DEMO=$SCRIPT_DIR/loadaddstore_3d
GEN=$BUILD/gen             # 共享生成物（AFIR/lowered/kbody/af_kfunc.h/kernel.bin）
SIM=$BUILD/sim             # CPU 仿真工作目录（harness + 副本）
CAMODEL=$BUILD/camodel     # camodel 仿真工作目录（launcher + 副本）
REALNPU=$BUILD/realnpu     # 真机准备区（scp 源）
# 容器内路径（$BUILD 挂到 /build）
CGEN=/build/gen
CSIM=/build/sim
CCAMODEL=/build/camodel
CREALNPU=/build/realnpu

dk()  { docker run --rm -v "$SRC":/workspace -v "$BUILD":/build -w /build "$IMAGE" bash -lc "$1"; }
dkc() { docker run --rm -v "$BUILD":/build -v "$CANN_VOL":/opt/cann:ro -w "$2" "$IMAGE" bash -lc "$1"; }

# --- schedule: 跑真实 Python UT，生成调度并 dump AFIR -------------------------
do_schedule() {
  echo "== [schedule] Python UT → 真实调度 AFIR =="
  # 容器内跑 pytest，dump AFIR 到 /build/afir_dump（挂载到 $BUILD/afir_dump）
  docker run --rm -v "$SRC":/workspace -v "$BUILD":/build \
    -v "$CANN_VOL":/opt/cann:ro -w /workspace/autofuse "$IMAGE" bash -lc '
    set -e
    DB=/workspace/build/autofuse
    PKG=/tmp/pypkg
    CANN=/opt/cann/'"$CANN_ARCH"'
    mkdir -p $PKG/autofuse /build/afir_dump
    cp -f $DB/compiler/py_module/pyautofuse.so $PKG/autofuse/pyautofuse.so
    cp -f /workspace/autofuse/compiler/python/*.py $PKG/autofuse/ 2>/dev/null || true
    export PYTHONPATH=$PKG:$PYTHONPATH
    export LD_LIBRARY_PATH=$DB:$DB/compiler/py_module:$DB/ascir/generator:$DB/ascir/meta:$DB/graph_metadef/graph:$DB/graph_metadef/graph/ascendc_ir:$DB/graph_metadef/graph/ascendc_ir/generator:$DB/graph_metadef/graph/expression:$CANN/lib64:$CANN/devlib/linux/$CANN_ARCH:$CANN/devlib/device:$LD_LIBRARY_PATH
    cd /workspace/autofuse/tests/ut/python
    AF_MLIR_AFIR_DUMP_DIR=/build/afir_dump \
      python3 -m pytest -s -q "test_python_ascir.py::TestAutofuseLoadAddStore::test_codegen" 2>&1 | tail -3
    ls -lh /build/afir_dump/afir_0_after_codegen_LoadAddStore.mlir | awk "{print \"  dump:\",\$9,\$5}"
  '
  # 拷贝到工作目录并重命名（保持后续流程兼容）
  cp "$BUILD/afir_dump/afir_0_after_codegen_LoadAddStore.mlir" "$GEN/afir_LoadAddStore_3D.mlir"
  echo "  → $GEN/afir_LoadAddStore_3D.mlir (真实调度, 动态 shape)"
}

# --- stage: 把版本化的 demo 源拷进 build 工作目录 ----------------------------
do_stage() {
  echo "== [stage] $DEMO -> $GEN / $SIM / $CAMODEL / $REALNPU =="
  mkdir -p "$GEN" "$SIM" "$CAMODEL" "$REALNPU"
  # 不再复制手写 afir.mlir，改用 do_schedule() 生成真实调度 AFIR
  do_schedule
  cp "$DEMO/kernel_wrapper.cpp"  "$GEN/afir_kernel_source.cpp"
  cp "$DEMO/harness_cpusim.cpp"  "$SIM/harness.cpp"
  cp "$DEMO/launcher.cpp"        "$CAMODEL/launcher.cpp"
  cp "$DEMO/launcher.cpp"        "$REALNPU/launcher.cpp"
}

# --- build: af-opt + ascir-translate ----------------------------------------
do_build() {
  echo "== [build] af-opt + ascir-translate =="
  mkdir -p "$BUILD/wrap"
  cat > "$BUILD/wrap/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.16)
project(af_mlir_standalone CXX C)
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0 -Wno-error)
set(ENABLE_AUTOFUSE_MLIR ON CACHE BOOL "" FORCE)
add_subdirectory(/workspace/autofuse/mlir ${CMAKE_BINARY_DIR}/mlir)
EOF
  docker run --rm -v "$SRC":/workspace \
    -v "$BUILD":/build -w /workspace "$IMAGE" bash -lc '
    cmake -G Ninja -S /build/wrap -B /build/b \
      -DMLIR_DIR=/opt/llvm/lib/cmake/mlir -DLLVM_DIR=/opt/llvm/lib/cmake/llvm
    cmake --build /build/b --target af-opt af-afir-gen ascir-translate -j 8'
}

# --- pipeline: AFIR -> ascendc IR -> Ascend C 源码 ---------------------------
do_pipeline() {
  do_stage
  echo "== [pipeline] ③lowering + ④翻译 =="
  dk "
  AFOPT=/build/b/mlir/tools/af-opt/af-opt; TR=/build/b/bin/ascir-translate
  \$AFOPT --convert-afir-to-ascendc-queue $CGEN/afir_LoadAddStore_3D.mlir > $CGEN/lowered.mlir
  \$TR -mlir-to-ascendc $CGEN/lowered.mlir > $CGEN/kbody.inc
  echo '  ③ lowered.mlir:'; grep -c 'scf.for\|data_copy_pad\|ceildivsi' $CGEN/lowered.mlir | xargs echo '    tiling/pad ops:'
  echo '  ④ kbody.inc:'; grep -c 'InitBuffer\|FreeTensor\|DataCopyPad\|GetBlockIdx' $CGEN/kbody.inc | xargs echo '    kernel ops:'
  # 从 kbody.inc 抽出 kernel 入口函数名，生成 af_kfunc.h（launcher/harness 的唯一真源）
  KF=\$(grep -oE '__global__ __aicore__ void [A-Za-z0-9_]+' $CGEN/kbody.inc | head -1 | awk '{print \$NF}')
  [ -n \"\$KF\" ] || { echo '    ERROR: 无法从 kbody.inc 解析 kernel 入口函数名' >&2; exit 1; }
  printf '#pragma once\n#define AF_KERNEL_FUNC %s\n' \"\$KF\" > $CGEN/af_kfunc.h
  echo \"    kernel entry: \$KF (-> af_kfunc.h)\"
  # 分发副本到各消费端
  cp $CGEN/kbody.inc $CSIM/kbody.inc
  cp $CGEN/af_kfunc.h $CSIM/af_kfunc.h
  cp $CGEN/af_kfunc.h $CCAMODEL/af_kfunc.h
  cp $CGEN/af_kfunc.h $CREALNPU/af_kfunc.h
  "
}

# --- sim: CPU 功能仿真 -------------------------------------------------------
do_sim() {
  echo "== [sim] CPU 功能仿真（全量数值校验）=="
  dkc '
  H=/opt/cann/'"$CANN_ARCH"'; CPUD=/opt/cann/tools/cpudebug; SOCL=$CPUD/lib64/Ascend910B1
  INC="-isystem $CPUD/include -isystem $H/asc -isystem $H/asc/include -isystem $H/asc/include/basic_api -isystem $H/asc/include/interface -isystem $H/asc/include/utils -isystem $H/include"
  # 关键：-D_GLIBCXX_USE_CXX11_ABI=0 与 cpudebug 库的旧 ABI 匹配
  $H/bin/ccec --run-mode=cpu -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 \
    -DASCENDC_CPU_DEBUG=1 -DASCENDC_DEBUG=1 -DASCENDC_DUMP=0 -D__NPU_ARCH__=2201 $INC harness.cpp \
    -Wl,-Bdynamic,--no-as-needed -L$CPUD/lib64 -ltikicpulib_stubreg -ltikicpulib_npuchk \
    -ltikicpulib_cceprint -L$SOCL -lcpudebug -L$H/lib64 -lc_sec -lstdc++ -lm -lpthread -o test_kernel
  export LD_LIBRARY_PATH=$H/simulator/dav_2201/lib:$H/devlib/device:$H/lib64:$CPUD/lib64:$SOCL
  for T in 12800 12808 12840; do
    echo -n "  total=$T (tiles=$((($T+255)/256))): "
    LD_PRELOAD=$H/simulator/dav_2201/lib/libpem_davinci.so ./test_kernel $T 2>&1 | grep -o "\[RESULT\].*"
  done
  ' "$CSIM"
}

# --- device 编译：kernel.bin（camodel 与真机共用）---------------------------
do_kernelbin() {
  echo "== [kernel.bin] bisheng -> aicore ELF =="
  dkc '
  C=/opt/cann/'"$CANN_ARCH"'; T=$C/tikcpp/tikcfw
  $C/bin/bisheng -c -x cce --cce-aicore-arch=dav-c220-vec --cce-aicore-only -std=c++17 \
    --cce-disable-kernel-global-attr-check \
    -mllvm -cce-aicore-stack-size=0x8000 -mllvm -cce-aicore-function-stack-size=0x8000 \
    -I $T -I $T/impl -I $T/interface -I $C/include -I $C/pkg_inc/base \
    -DASCENDC_DUMP=0 -D__NPU_TILING__ afir_kernel_source.cpp -o kernel.o
  $C/bin/ld.lld -m aicorelinux -Ttext=0 kernel.o -static -o kernel.bin
  ls -lh kernel.bin
  ' "$CGEN"
}

# --- camodel: 周期精确仿真（同一 kernel.bin）--------------------------------
do_camodel() {
  do_kernelbin
  echo "== [camodel] 周期精确仿真（无硬件，同一 kernel.bin）=="
  # 拷贝 kernel.bin 到 camodel 目录
  cp "$GEN/kernel.bin" "$CAMODEL/"
  dkc '
  H=/opt/cann/'"$CANN_ARCH"'
  g++ -std=c++17 -O0 launcher.cpp -ldl -o launcher_cam
  export LD_LIBRARY_PATH=$H/simulator/dav_2201/lib:$H/simulator/Ascend910B1/lib:$H/lib64:$H/devlib/device
  ./launcher_cam $H/simulator/dav_2201/lib/libruntime_camodel.so 2>&1 | grep -iE "PASS|FAIL|total=|ALL|runtime lib"
  ' "$CCAMODEL"
}

# --- realnpu: 真机 910C（9 组合）--------------------------------------------
do_realnpu() {
  do_kernelbin
  echo "== [realnpu] 910C 9 组合（total×blockDim）=="
  if [ -z "$REALNPU_ENV" ]; then
    echo "错误：realnpu 需要 SSH 环境文件，请设 REALNPU_ENV 指向它。" >&2
    echo "  该文件须导出：ASCEND_MLIR_CI_REMOTE / _REMOTE_PORT / _SSH_USERNAME / _SSH_PASSWORD" >&2
    echo "  例：REALNPU_ENV=/path/to/real-npu.local.env ./reproduce.sh realnpu" >&2
    exit 1
  fi
  [ -f "$REALNPU_ENV" ] || { echo "错误：REALNPU_ENV 文件不存在: $REALNPU_ENV" >&2; exit 1; }
  source /dev/stdin < "$REALNPU_ENV"
  local HOST="${ASCEND_MLIR_CI_REMOTE}" PORT="${ASCEND_MLIR_CI_REMOTE_PORT}" USER="${ASCEND_MLIR_CI_SSH_USERNAME}"
  export SSHPASS="${ASCEND_MLIR_CI_SSH_PASSWORD}"
  local DST=/data/nyh/afir-3d-demo
  local SSHOPT="-o StrictHostKeyChecking=no -o ServerAliveInterval=5 -o ConnectTimeout=15 -p $PORT"
  # 拷贝 kernel.bin 到 realnpu 目录，然后 scp 三个文件到远端
  cp "$GEN/kernel.bin" "$REALNPU/"
  sshpass -e ssh $SSHOPT "$USER@$HOST" "mkdir -p $DST"
  sshpass -e scp -o StrictHostKeyChecking=no -P "$PORT" "$REALNPU/kernel.bin" "$REALNPU/launcher.cpp" "$REALNPU/af_kfunc.h" "$USER@$HOST:$DST/"
  sshpass -e ssh $SSHOPT "$USER@$HOST" "bash -lc '
    cd $DST; source /data/nyh/Ascend/latest/set_env.sh 2>/dev/null; export ASCEND_DEVICE_ID=5
    H=\${ASCEND_HOME_PATH:-/data/nyh/Ascend/latest}
    g++ -std=c++17 -O0 -DAF_REAL_NPU launcher.cpp -ldl -I \$H/include -L \$H/lib64 -lascendcl -Wl,-rpath,\$H/lib64 -o launcher
    export ASCEND_GLOBAL_LOG_LEVEL=3
    ./launcher \$H/lib64/libruntime.so 2>&1 | grep -iE \"PASS|FAIL|total=|ALL|rc=\"
  '"
}

case "${1:-all}" in
  build)    do_build ;;
  pipeline) do_pipeline ;;
  sim)      do_pipeline; do_sim ;;
  camodel)  do_pipeline; do_camodel ;;
  realnpu)  do_pipeline; do_realnpu ;;
  all)      do_build; do_pipeline; do_sim ;;
  *) echo "usage: $0 {build|pipeline|sim|camodel|realnpu|all}"; exit 1 ;;
esac
echo "== done: ${1:-all} =="
