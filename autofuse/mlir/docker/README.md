# Autofuse MLIR Docker 环境使用指南

本文只说明如何使用 Phase 1 的开发/编译验证环境。镜像如何构建、LLVM 具体编译参数、
CANN 安装细节和校验项都由脚本维护；需要时执行对应脚本的 `--help`。

命令默认从 Autofuse 组件目录执行：

```bash
cd /Volumes/GM9/code/autofuse/autofuse
```

脚本会自动定位上一层 Git 仓库根目录 `/Volumes/GM9/code/autofuse`，并在容器内挂载为
`/workspace`。容器内顶层 `build.sh` 位于 `/workspace/build.sh`。

## 选择方案

### Linux 用户

Linux 用户优先直接使用 Linux 本机或远端 Linux 环境：

```text
LLVM/MLIR:  /opt/llvm
CANN:       /usr/local/Ascend/ascend-toolkit
CLion:      Local Toolchain 或 Remote Toolchain
```

这种方式不需要 CANN Docker volume。CLion 的 CMake Profile 中按实际路径配置：

```text
-DASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux
-DLLVM_BUILD_DIR=/opt/llvm
-DLLVM_DIR=/opt/llvm/lib/cmake/llvm
-DMLIR_DIR=/opt/llvm/lib/cmake/mlir
```

x86_64 Linux 环境把 `aarch64-linux` 改为 `x86_64-linux`。

### macOS/Windows 用户

非 Linux 本地开发机推荐：

```text
LLVM/MLIR:  团队 dev+LLVM Docker 镜像
CANN:       本机 Docker named volume
CLion:      Docker Toolchain
```

这条路线避免在 macOS/Windows 宿主机安装 CANN。真实 NPU 设备验证仍放到 Linux/NPU/CI
环境执行。

## 拉取团队 LLVM 镜像

如果本机已经有可用镜像，例如我们当前实测过的本地镜像：

```bash
case "$(uname -m)" in
  aarch64 | arm64)
    export AF_MLIR_DEV_IMAGE=autofuse-mlir-dev:arm64
    ;;
  x86_64 | amd64)
    export AF_MLIR_DEV_IMAGE=autofuse-mlir-dev:x86_64
    ;;
  *)
    echo "Unsupported architecture: $(uname -m)" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac
```

这时不要执行 `pull_dev_image.sh`。`pull_dev_image.sh` 只用于拉取镜像仓中的远端镜像。

团队镜像根据宿主机架构选择。远端仓库使用 `arm64` 作为 AArch64 镜像 tag，使用
`x86_64` 作为 x86-64 镜像 tag：

```bash
case "$(uname -m)" in
  aarch64 | arm64)
    export AF_MLIR_REMOTE_IMAGE=swr.cn-east-2.myhuaweicloud.com/ascendmlir/autofuse-mlir-dev:arm64
    export AF_MLIR_DEV_IMAGE=autofuse-mlir-dev:arm64
    ;;
  x86_64 | amd64)
    export AF_MLIR_REMOTE_IMAGE=swr.cn-east-2.myhuaweicloud.com/ascendmlir/autofuse-mlir-dev:x86_64
    export AF_MLIR_DEV_IMAGE=autofuse-mlir-dev:x86_64
    ;;
  *)
    echo "Unsupported architecture: $(uname -m)" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac
bash mlir/docker/pull_dev_image.sh
```

如果镜像仓需要登录：

```bash
export AF_MLIR_REGISTRY=swr.cn-east-2.myhuaweicloud.com
export AF_MLIR_REGISTRY_USER=<registry-user>
export AF_MLIR_REGISTRY_TOKEN=<registry-token>
bash mlir/docker/pull_dev_image.sh "${AF_MLIR_REMOTE_IMAGE}"
```

快速确认镜像可用：

```bash
bash mlir/docker/validate_llvm_image.sh "${AF_MLIR_DEV_IMAGE}"
```

## 准备 CANN Volume

非 Linux 用户用官方 CANN Toolkit `.run` 包初始化本机 Docker volume：

```bash
export AF_CANN_VOLUME=af-cann-910
bash mlir/docker/prepare_cann_volume.sh \
  --volume "${AF_CANN_VOLUME}" \
  --toolkit-url https://example.com/path/to/Ascend-cann-toolkit_9.1.0_linux-aarch64.run
```

如果还需要安装 ops 包：

```bash
export AF_CANN_OPS_PACKAGE_URL=https://example.com/path/to/Ascend-cann-A3-ops_9.1.0_linux-aarch64.run
bash mlir/docker/prepare_cann_volume.sh \
  --volume "${AF_CANN_VOLUME}" \
  --toolkit-url https://example.com/path/to/Ascend-cann-toolkit_9.1.0_linux-aarch64.run \
  --force
```

确认 volume 可用：

```bash
docker volume inspect af-cann-910
AF_CANN_VOLUME=af-cann-910 bash mlir/scripts/enter_dev_container.sh -- ls /opt/cann/latest
```

`enter_dev_container.sh` 会把同一个 CANN root/volume 同时挂载到 `/opt/cann` 和 CANN
`set_env.sh` 记录的安装前缀。CANN `.run` 包的默认前缀是
`/usr/local/Ascend/cann-9.1.0`；如果你的包使用其他前缀，设置
`AF_CANN_INSTALL_PREFIX=<container-path>` 覆盖。

脚本会在执行用户命令前 source `${ASCEND_HOME_PATH}/set_env.sh`，并自动设置：

```text
CANN_ROOT=/opt/cann
ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
ASCEND_CUSTOM_PATH=/usr/local/Ascend/cann-9.1.0
AF_LLVM_ROOT=/opt/llvm
LLVM_BUILD_DIR=/opt/llvm
LLVM_DIR=/opt/llvm/lib/cmake/llvm
MLIR_DIR=/opt/llvm/lib/cmake/mlir
```

一般构建不需要手工 `source ${ASCEND_HOME_PATH}/set_env.sh`。

## 准备 PyAsc

默认构建和 `--no-autofuse` 不需要 PyAsc。只有打开 MLIR 工程接入时才需要初始化 PyAsc：

```bash
bash mlir/scripts/sync_pyasc_upstream.sh
```

该脚本默认使用 Autofuse 父仓 gitlink 固定的 PyAsc commit，不会拉取官方仓最新 HEAD。
当前固定点可用下面命令查看：

```bash
git ls-files --stage mlir/externals/pyasc
```

如果只想恢复到固定 upstream 基线、不应用本仓 patchset：

```bash
bash mlir/scripts/sync_pyasc_upstream.sh --no-apply-patches
```

更新 PyAsc 基线时才显式指定官方仓 commit，并单独评审 submodule gitlink 变化：

```bash
bash mlir/scripts/sync_pyasc_upstream.sh --ref <new-official-commit> --no-apply-patches
git add mlir/externals/pyasc
```

Autofuse 需要维护的 PyAsc 改动放在 `mlir/patches/pyasc/*.patch`，不要把 `.gitmodules`
指向个人 fork。导出 patch 时必须显式选择 commit 或 range：

```bash
bash mlir/scripts/export_pyasc_patch.sh --commit <sha>
bash mlir/scripts/export_pyasc_patch.sh --range <start>..<end>
```

## 编译验证

首次在 Docker `/workspace` 路径编译前，确认顶层 `../build` 没有保留旧路径下生成的
`CMakeCache.txt`。如果之前在宿主机或其他容器路径编译过，先备份旧构建目录：

```bash
mv ../build ../build.host-backup.$(date +%Y%m%d%H%M%S) 2>/dev/null || true
mv ../build_out ../build_out.host-backup.$(date +%Y%m%d%H%M%S) 2>/dev/null || true
```

记录环境 manifest：

```bash
bash mlir/scripts/enter_dev_container.sh -- \
  bash -lc 'cd autofuse && bash mlir/scripts/collect_env_manifest.sh'
```

验证 `--no-autofuse` 语义不变：

```bash
bash mlir/scripts/enter_dev_container.sh -- \
  bash build.sh --pkg --no-autofuse -j 8
```

完整编译 Autofuse：

```bash
bash mlir/scripts/enter_dev_container.sh -- \
  bash build.sh --pkg -j 8
```

离线环境需要提前准备第三方目录，并显式传入：

```bash
bash mlir/scripts/enter_dev_container.sh -- \
  bash build.sh --pkg --cann_3rd_lib_path=<path> -j 8
```

所有 `build.sh` 构建命令必须带 `-j 8`。

## Coverage 验证

dev+LLVM 镜像内置 `gcov`、`lcov` 和 `genhtml`。`validate_llvm_image.sh` 会检查这些工具是否存在。

Autofuse MLIR 相关覆盖率需要同时打开 coverage 和 MLIR 构建开关：

```bash
bash mlir/scripts/enter_dev_container.sh -- \
  bash build.sh -u --module=autofuse_framework --enable-mlir -c -j 8
```

`build.sh` 会把 `--enable-mlir` 继续传给 `scripts/test/run_autofuse_test.sh`，最终 CMake 使用
`-DENABLE_AUTOFUSE_MLIR=ON`。coverage 报告输出到仓库顶层 `cov/coverage_report/`，中间
`.gcno/.gcda` 和 `cov/` 都是临时产物，不入库。

## 三方件缓存

Autofuse/CANN 三方件不内置到 dev+LLVM 镜像。第一次完整构建会在仓库顶层
`output/third_party` 下下载、解压和编译三方件；从当前组件目录看是
`../output/third_party`。后续构建会复用该目录。

如果要预制给其他环境使用，先在一台机器上完成一次构建，然后打包顶层缓存：

```bash
mkdir -p mlir/.artifacts
tar -C .. -czf mlir/.artifacts/autofuse-third-party-aarch64-cann910.tar.gz output/third_party
```

在新环境恢复：

```bash
mkdir -p ../output
tar -C .. -xzf /path/to/autofuse-third-party-aarch64-cann910.tar.gz
bash mlir/scripts/enter_dev_container.sh -- \
  bash build.sh --pkg --cann_3rd_lib_path=/workspace/output/third_party -j 8
```

该缓存与 CPU 架构、CANN 版本和编译环境相关；切换这些条件后需要重新预制。

## CLion Docker Toolchain

本节面向 macOS + OrbStack 等非 Linux 开发机。Linux 用户直接配置 Linux 本机或远端
Linux 的 LLVM/MLIR 路径即可。

推荐 CLion 打开 Git 仓库根目录，这样 CLion 的 Git/VCS 功能才能正常识别：

```text
/Volumes/GM9/code/autofuse
```

日常开发主要关注其中的组件目录：

```text
/Volumes/GM9/code/autofuse/autofuse
```

推荐路径映射：

```text
CLion project:   /Volumes/GM9/code/autofuse
Repo mount:      /Volumes/GM9/code/autofuse -> /workspace
Component path:  /workspace/autofuse
```

先在 `Settings > Build, Execution, Deployment > Docker` 中确认 Docker server
`Connection successful`。源码在 `/Volumes` 下时，Docker server 的 path mappings 需要包含：

```text
/Volumes -> /Volumes
```

Docker Toolchain 配置：

```text
Settings:  Build, Execution, Deployment > Toolchains
Name:      Docker
Server:    <Docker: Auto detected server>
Image:     autofuse-mlir-dev:arm64（AArch64）或 autofuse-mlir-dev:x86_64（x86-64）
```

Docker Toolchain 的 `Container settings`：

```text
--entrypoint=
--rm
-v /Volumes/GM9/code/autofuse:/workspace
-v af-cann-910:/opt/cann:ro
-v af-cann-910:/usr/local/Ascend/cann-9.1.0:ro
-e CANN_ROOT=/opt/cann
-e ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
-e ASCEND_CUSTOM_PATH=/usr/local/Ascend/cann-9.1.0
-e AF_LLVM_ROOT=/opt/llvm
-e LLVM_BUILD_DIR=/opt/llvm
-e LLVM_DIR=/opt/llvm/lib/cmake/llvm
-e MLIR_DIR=/opt/llvm/lib/cmake/mlir
```

当前顶层 Autofuse 主工程先使用 `Unix Makefiles`，不要使用 Ninja。CANN CMake 的三方件
规则中存在 `$(MAKE)`，Ninja 会把单个 `$` 当作转义符并报
`bad $-escape`。该限制只影响 Autofuse 主工程的 CLion Profile；LLVM/MLIR 镜像内部预编译
仍可以使用 Ninja。

CMake Profile 配置：

```text
Settings:         Build, Execution, Deployment > CMake
Name:             Debug-Docker
Toolchain:        Docker
Build type:       Debug
Generator:        Unix Makefiles
Build directory:  /workspace/docker-build-make
Build options:    -j 8
```

默认 Profile 保持 `ENABLE_AUTOFUSE_MLIR=OFF` 语义不变，只传入当前主工程实际使用的路径：

```text
-DASCEND_INSTALL_PATH=/usr/local/Ascend/cann-9.1.0
-DCANN_3RD_LIB_PATH=/workspace/output/third_party
```

不要复用已经用 Ninja 配置过的 `/workspace/docker-build`；如果已经创建过，可以直接换成
`/workspace/docker-build-make`。需要清理时，只删除宿主机上的
`/Volumes/GM9/code/autofuse/docker-build` 或
`/Volumes/GM9/code/autofuse/docker-build-make`，不要删除
`/Volumes/GM9/code/autofuse/output/third_party`。

后续打开 MLIR 专项构建时，再显式增加 LLVM/MLIR 路径：

```text
-DENABLE_AUTOFUSE_MLIR=ON
-DLLVM_BUILD_DIR=/opt/llvm
-DLLVM_DIR=/opt/llvm/lib/cmake/llvm
-DMLIR_DIR=/opt/llvm/lib/cmake/mlir
```

打开 MLIR 专项构建前，先按“准备 PyAsc”小节初始化固定 PyAsc 基线和本仓 patchset。

CLion 的 CMake build directory 不建议复用顶层 `/workspace/build`；该目录留给
`build.sh --pkg -j 8` 和 `build.sh --pkg --no-autofuse -j 8` baseline。组件级 CMake
使用独立目录，例如 `/workspace/docker-build-make`。

## 镜像维护

普通开发者不需要关心镜像构建细节。需要维护镜像时直接使用脚本：

```bash
bash mlir/docker/pull_dev_image.sh --help
bash mlir/docker/build_llvm_image.sh --help
bash mlir/docker/validate_llvm_image.sh --help
bash mlir/docker/build_cann_image.sh --help
bash mlir/docker/prepare_cann_volume.sh --help
```

默认 dev+LLVM 镜像不内置 CANN，也不内置 Autofuse/CANN 第三方源码包。CANN 通过
Linux 环境预装、CI 注入或本机 Docker volume 提供。

LLVM/MLIR 输入、ABI、manifest 和镜像发布校验规则见
`autofuse/mlir/docs/llvm_dependency_policy.md`。
