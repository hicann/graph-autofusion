# aarch64 架构 TF 源码编译

本文档介绍 aarch64 架构下从源码编译 TensorFlow 及 TF Adapter 的完整流程，覆盖 TF 1.15.0 和 TF 2.6.5 两个版本。

> **注意**：下文中的 `/mnt/workspace` 为华为云开发者环境挂载目录，可根据实际环境替换。

## 版本差异对照

编译前请先确认目标 TF 版本，两者的关键差异如下：

| 项目 | TF 1.15.0 | TF 2.6.5 |
|------|-----------|----------|
| Python | 3.7.11 | 3.9.25 |
| numpy | 1.18.5 | 1.23.5 |
| h5py | 2.8.0 | 3.1.0 |
| TF 源码 | v1.15.0 | v2.6.5 |
| bazel | 0.26.1 | 3.7.2 |
| NPU 插件 | npu_bridge | npu_device |
| TF Adapter 源码路径 | `tfadapter/` | `tf_adapter_2.x/` |
| TF Adapter 源码修改 | 需修改三处 | 无需修改 |

下文中各步骤如无特殊说明，两个版本流程相同；有差异时分别标注 **TF 1.15** 和 **TF 2.6.5**。

---

## 一、创建编译目录

```bash
cd /mnt/workspace
mkdir -p tf_build
cd tf_build
```

---

## 二、下载解压 hdf5 系统库

```bash
wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src/hdf5-1.10.5.tar.gz
tar -zxvf hdf5-1.10.5.tar.gz
```

---

## 三、编译 hdf5

```bash
cd hdf5-1.10.5/
./configure --prefix=/usr/local/hdf5
sudo make -j16 && sudo make install
```

---

## 四、配置环境变量

```bash
export HDF5_DIR=/usr/local/hdf5
export CPATH=/usr/local/hdf5/include:$CPATH
export LIBRARY_PATH=/usr/local/hdf5/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/hdf5/lib:$LD_LIBRARY_PATH
```

---

## 五、安装 hdf5 python 库及其依赖包

> 会触发 h5py 的本地编译，其依赖于 hdf5 系统库。

```bash
pip3 install Cython==0.29.14
pip3 install wheel
# TF 1.15
pip3 install numpy==1.18.5 --no-build-isolation
pip3 install h5py==2.8.0 --no-deps
# TF 2.6.5
pip3 install numpy==1.23.5 --no-build-isolation
pip3 install h5py==3.1.0 --no-deps
```

> 按目标 TF 版本选择对应的 numpy 和 h5py 版本，不要同时安装。

---

## 六、下载解压 TF 源码

```bash
cd /mnt/workspace/tf_build
# TF 1.15
wget https://github.com/tensorflow/tensorflow/archive/refs/tags/v1.15.0.tar.gz
tar -zxvf v1.15.0.tar.gz
# TF 2.6.5
wget https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.6.5.tar.gz
tar -zxvf v2.6.5.tar.gz
```

> 后续步骤（步骤 7~11）均为参考 [安装开源框架 TensorFlow 1.15 - TensorFlow 社区版 9.0.0 - 昇腾社区](https://www.hiascend.com/document/detail/zh/TensorFlowCommunity/900/migration/tfmigr1/tfmigr1_000001.html) 中「安装 TensorFlow」章节对 tf 源码进行修改的具体操作。
>
> **两个版本都需要步骤 7~10 的 nsync 修改**：TF 1.15 和 TF 2.6.5 均依赖 nsync-1.22.0，该版本在 aarch64 + 新 glibc 上存在 `gettid` 声明冲突，必须修改。差异在于步骤十修改的文件路径不同：
> - TF 1.15：`tensorflow-1.15.0/tensorflow/workspace.bzl`
> - TF 2.6.5：`tensorflow-2.6.5/tensorflow/workspace2.bzl`

---

## 七、下载 nsync 并解压

```bash
wget https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/nsync/archive/1.22.0.tar.gz
tar -xf 1.22.0.tar.gz -C /mnt/workspace/tf_build/
```

---

## 八、修改 nsync 代码

文件位置：`/mnt/workspace/tf_build/nsync-1.22.0/platform/c++11/atomic.h`

修改前：

```cpp
#include "nsync_cpp.h"
#include "nsync_atomic.h"

NSYNC_CPP_START_

static INLINE int atm_cas_nomb_u32_ (nsync_atomic_uint32_ *p, uint32_t o, uint32_t n) {
    return (std::atomic_compare_exchange_strong_explicit (NSYNC_ATOMIC_UINT32_PTR_ (p), &o, n,
                         std::memory_order_relaxed, std::memory_order_relaxed));
}
static INLINE int atm_cas_acq_u32_ (nsync_atomic_uint32_ *p, uint32_t o, uint32_t n) {
    return (std::atomic_compare_exchange_strong_explicit (NSYNC_ATOMIC_UINT32_PTR_ (p), &o, n,
                         std::memory_order_acquire, std::memory_order_relaxed));
}
static INLINE int atm_cas_rel_u32_ (nsync_atomic_uint32_ *p, uint32_t o, uint32_t n) {
    return (std::atomic_compare_exchange_strong_explicit (NSYNC_ATOMIC_UINT32_PTR_ (p), &o, n,
                         std::memory_order_release, std::memory_order_relaxed));
}
static INLINE int atm_cas_relacq_u32_ (nsync_atomic_uint32_ *p, uint32_t o, uint32_t n) {
    return (std::atomic_compare_exchange_strong_explicit (NSYNC_ATOMIC_UINT32_PTR_ (p), &o, n,
                         std::memory_order_acq_rel, std::memory_order_relaxed));
}
```

修改后（在每个 `atm_cas_*` 函数中增加 `ATM_CB_()` 内存屏障调用）：

```cpp
#include "nsync_cpp.h"
#include "nsync_atomic.h"

NSYNC_CPP_START_

#define ATM_CB_() __sync_synchronize()

static INLINE int atm_cas_nomb_u32_ (nsync_atomic_uint32_ *p, uint32_t o, uint32_t n) {
    int result = (std::atomic_compare_exchange_strong_explicit (NSYNC_ATOMIC_UINT32_PTR_ (p), &o, n, std::memory_order_relaxed, std::memory_order_relaxed));
    ATM_CB_();
    return result;
}
static INLINE int atm_cas_acq_u32_ (nsync_atomic_uint32_ *p, uint32_t o, uint32_t n) {
    int result = (std::atomic_compare_exchange_strong_explicit (NSYNC_ATOMIC_UINT32_PTR_ (p), &o, n, std::memory_order_acquire, std::memory_order_relaxed));
    ATM_CB_();
    return result;
}
static INLINE int atm_cas_rel_u32_ (nsync_atomic_uint32_ *p, uint32_t o, uint32_t n) {
    int result = (std::atomic_compare_exchange_strong_explicit (NSYNC_ATOMIC_UINT32_PTR_ (p), &o, n, std::memory_order_release, std::memory_order_relaxed));
    ATM_CB_();
    return result;
}
static INLINE int atm_cas_relacq_u32_ (nsync_atomic_uint32_ *p, uint32_t o, uint32_t n) {
    int result = (std::atomic_compare_exchange_strong_explicit (NSYNC_ATOMIC_UINT32_PTR_ (p), &o, n, std::memory_order_acq_rel, std::memory_order_relaxed));
    ATM_CB_();
    return result;
}
```

---

## 九、重新压缩并获取 sha256

```bash
tar -czf /mnt/workspace/tf_build/1.22.0.tar.gz -C /mnt/workspace/tf_build/ nsync-1.22.0
sha256sum /mnt/workspace/tf_build/1.22.0.tar.gz
```

得到一串数字和字母的组合，例如：

```
c0423d005fb9bd21e73df809e2a2a4d6a1da0beaf036305d3285fcaff5a4dcf3
```

---

## 十、修改 workspace.bzl（TF 1.15）/workspace2.bzl（TF 2.6.5） 参数

文件位置：
- TF 1.15：`/mnt/workspace/tf_build/tensorflow-1.15.0/tensorflow/workspace.bzl`
- TF 2.6.5：`/mnt/workspace/tf_build/tensorflow-2.6.5/tensorflow/workspace2.bzl`

修改前：

```python
tf_http_archive(
    name = "nsync",
    sha256 = "caf32e6b3d478b78cff6c2ba009c3400f8251f646804bcb65465666a9cea93c4",
    strip_prefix = "nsync-1.22.0",
    system_build_file = clean_dep("//third_party/systemlibs:nsync.BUILD"),
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/nsync/archive/1.22.0.tar.gz",
        "https://github.com/google/nsync/archive/1.22.0.tar.gz",
    ],
)
```

修改后（`sha256` 的值改为上一步得到的值，新增 `file:` 行指向本地压缩文件）：

```python
tf_http_archive(
    name = "nsync",
    sha256 = "c0423d005fb9bd21e73df809e2a2a4d6a1da0beaf036305d3285fcaff5a4dcf3",
    strip_prefix = "nsync-1.22.0",
    system_build_file = clean_dep("//third_party/systemlibs:nsync.BUILD"),
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/nsync/archive/1.22.0.tar.gz",
        "file:/mnt/workspace/tf_build/1.22.0.tar.gz",
        "https://github.com/google/nsync/archive/1.22.0.tar.gz",
    ],
)
```

---

## 十一、安装 TF 编译依赖包

```bash
pip3 install protobuf==3.19.0
pip3 install -U keras_preprocessing==1.1.2 --no-deps
```

---

## 十二、安装并编译 bazel

### 12.1 安装 bazel 依赖的 openjdk

```bash
sudo apt install -y openjdk-11-jdk
cd /mnt/workspace/tf_build
# TF 1.15 需要 bazel 0.26.1
wget https://temp-a7fd.obs.cn-north-4.myhuaweicloud.com/tmp/bazel-0.26.1-dist.zip
unzip bazel-0.26.1-dist.zip -d bazel-0.26.1
# TF 2.6.5 需要 bazel 3.7.2
wget https://github.com/bazelbuild/bazel/releases/download/3.7.2/bazel-3.7.2-dist.zip
unzip bazel-3.7.2-dist.zip -d bazel-3.7.2
```

### 12.2 修改 bazel 源码（TF 1.15 的 bazel 0.26.1）

文件：`/mnt/workspace/tf_build/bazel-0.26.1/third_party/grpc/src/core/lib/gpr/log_linux.cc`

将 `gettid` 重命名为 `grpc_gettid`，避免与 glibc 中 `gettid` 声明冲突：

```cpp
// 修改前
static long gettid(void) { return syscall(__NR_gettid); }
// 修改后
static long grpc_gettid(void) { return syscall(__NR_gettid); }
```

并且：

```cpp
// 修改前
if (tid == 0) tid = gettid();
// 修改后
if (tid == 0) tid = grpc_gettid();
```

### 12.3 编译 bazel

```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-arm64/
# TF 1.15
cd /mnt/workspace/tf_build/bazel-0.26.1
env EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk" ./compile.sh
# TF 2.6.5
cd /mnt/workspace/tf_build/bazel-3.7.2
env EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk" ./compile.sh
```

构建成功后会在控制台日志中打印 Bazel 产物位置，将其路径添加到系统环境变量中：

```bash
# TF 1.15
export PATH=/mnt/workspace/tf_build/bazel-0.26.1/output/bazel:$PATH
# TF 2.6.5
export PATH=/mnt/workspace/tf_build/bazel-3.7.2/output/bazel:$PATH
```

> **TF 2.6.5 无需步骤 12.2**：bazel 3.7.2 源码树中不包含 `third_party/grpc/src/core/lib/gpr/log_linux.cc`，编译 bazel 本身时不会触发 `gettid` 冲突。

---

## 十三、配置并构建 TF

### 13.1 进入源码目录

```bash
# TF 1.15
cd /mnt/workspace/tf_build/tensorflow-1.15.0
# TF 2.6.5
cd /mnt/workspace/tf_build/tensorflow-2.6.5
```

### 13.2 配置编译选项

> 注意：所有额外特性选项都选 `n`，其它确认项选 `y`。

```bash
./configure
```

配置交互参考如下：

```text

Found possible Python library paths:
/mnt/workspace/venv/test_tf1/lib/python3.7/site-packages
/home/developer/Ascend/cann-9.1.0/python/site-packages
/home/developer/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe
Please input the desired Python library path to use.  Default is [/mnt/workspace/venv/test_tf1/lib/python3.7/site-packages]

Do you wish to build TensorFlow with XLA JIT support? [Y/n]: n
No XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: n
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with ROCm support? [y/N]: n
No ROCm support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: n
No CUDA support will be enabled for TensorFlow.

Do you wish to download a fresh release of clang? (Experimental) [y/N]: n
Clang will not be downloaded.

Do you wish to build TensorFlow with MPI support? [y/N]: n
No MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native -Wno-sign-compare]:

Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n
Not configuring the WORKSPACE for Android builds.

Configuration finished
```

### 13.3 执行构建

> 构建过程中会下载大量的三方包，可能存在下载失败的情况，可以反复重试构建命令直到构建成功。有条件的话可在环境上配置代理。

```bash
# TF 1.15
bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/tools/pip_package:build_pip_package
# TF 2.6.5
bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/tools/pip_package:build_pip_package
```

### 13.4 处理构建报错（仅 TF 1.15）

构建 TF 1.15 过程中，bazel 会下载 grpc 依赖，其中 `log_linux.cc` 的 `gettid` 函数与新 glibc 声明冲突，修改方法与 [步骤 12.2](#122-修改-bazel-源码tf-115-的-bazel-0261) 一致。报错文件路径为 `bazel-tensorflow-1.15.0/external/grpc/src/core/lib/gpr/log_linux.cc`（构建后会出现这个目录）。

> **TF 2.6.5 不会遇到此报错**：其依赖的 grpc 已将 `gettid` 重命名为 `sys_gettid`，不再与新 glibc 冲突。

报错示例：

```text
ERROR: external/grpc/BUILD:507:1: C++ compilation of rule '@grpc//:gpr_base' failed (Exit 1)

external/grpc/src/core/lib/gpr/log_linux.cc:43:13: error: ambiguating new declaration of 'long int gettid()'
   43 | static long gettid(void) { return syscall(__NR_gettid); }
      |             ^~~~~~
/usr/include/aarch64-linux-gnu/bits/unistd_ext.h:34:16: note: old declaration '__pid_t gettid()'
   34 | extern __pid_t gettid (void) __THROW;
      |                ^~~~~~

Target //tensorflow/tools/pip_package:build_pip_package failed to build
Use --verbose_failures to see the command lines of failed build steps.
```

### 13.5 生成并安装 pip 包

```bash
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
# TF 1.15
pip3 install /tmp/tensorflow_pkg/tensorflow-1.15.0-cp37-cp37m-linux_aarch64.whl
# TF 2.6.5
pip3 install /tmp/tensorflow_pkg/tensorflow-2.6.5-cp39-cp39-linux_aarch64.whl
```

---

## 十四、编译安装 TF Adapter

> 不要使用 tfa 仓中的预编译包，存在 `_ZN10tensorflow11GraphCyclesC1Ev` 符号不兼容问题。需要相应修改 tfa 源码确保编译通过。
>
> **TF 1.15（npu_bridge）**：需修改 tfa 源码（`tfadapter/`）三处后编译，见 14.1~14.4。
> **TF 2.6.5（npu_device）**：使用 tfa 源码中独立的 `tf_adapter_2.x/` 构建系统，无需修改源码，见 14.6。

### 14.1 下载 tfa 源码

```bash
cd /mnt/workspace/tf_build
git clone https://gitcode.com/cann/tensorflow.git
```

### 14.2 安装 tfa 编译依赖

```bash
sudo apt install swig
```

### 14.3 修改 tfa 源码（仅 TF 1.15）

共有三处修改。

**第一处**：在 tfa 源码仓中 `tfadapter/build.sh` 的第 85 行。

修改前：

```bash
CMAKE_ARGS="-DENABLE_OPENSRC=True -DCMAKE_INSTALL_PREFIX=${RELEASE_PATH}"
```

修改后（`FETCHCONTENT_SOURCE_DIR_TENSORFLOW` 指向前面编译的 tf1.15 源码仓路径）：

```bash
CMAKE_ARGS="-DENABLE_OPENSRC=True -DCMAKE_INSTALL_PREFIX=${RELEASE_PATH} -DFETCHCONTENT_SOURCE_DIR_TENSORFLOW=/mnt/workspace/tf_build/tensorflow-1.15.0"
```

**第二处**：`CMakeLists.txt` 的第 81 行开始添加三段定义：

```cmake
set(TF_GRAPHCYCLES_SOURCE
    ${tensorflow_SOURCE_DIR}/tensorflow/compiler/jit/graphcycles/graphcycles.cc
)
list(APPEND SOURCES
    ${TF_GRAPHCYCLES_SOURCE}
)
set_source_files_properties(
    ${TF_GRAPHCYCLES_SOURCE}
    PROPERTIES
    COMPILE_FLAGS "-Wno-error=sign-compare -Wno-sign-compare"
)
```

**第三处**：`tfadapter/kernels/geop_npu.cc` 中的第 75 行和第 1830 行，将 `FunctionalizeControlFlow` 函数的声明和调用注释掉（这个函数实际引用的是 tf 库中的定义，但是 tf 库在 1.15 版本中没有这个函数，会导致链接错误）。

声明处注释：

```cpp
namespace tensorflow {
#ifdef TF_VERSION_TF2
Status FunctionalizeControlFlow(Graph *graph, FunctionLibraryDefinition *library, const NodeFilter &node_filter = {},
                                bool include_functions = false);
#else
//Status FunctionalizeControlFlow(Graph *graph, FunctionLibraryDefinition *library);  这里注释
#endif
namespace {
const std::string ATTR_NAME_CONST_INPUT_NAME = "_const_input";
const std::string kAutoRecompute = "auto";
const std::string kTotalStep = "TOTAL_STEP";
```

调用处注释：

```cpp
  // 动态场景下需要将动态轴更新成-1，避免频繁触发编译
  if ((jit_compile_ != "1") || (compile_dynamic_mode_ == "1") ||
      (jit_compile_ == "1" && shape_generalization_mode_ != "STRICT")) {
    ADP_LOG(INFO) << "[GEOP] UpdateInputsShapeDesc start.";
    UpdateInputsShapeDesc(graph);
  }

  graph.ToGraphDef(&graph_def);
  std::string enable_force_v2_control;
  (void)ReadStringFromEnvVar("ENABLE_FORCE_V2_CONTROL", "", &enable_force_v2_control);
  if (enable_force_v2_control == "1") {
    // Status status = FunctionalizeControlFlow(&graph, &flib_def);  这一块注释
    // if (status != Status::OK()) {
    //   LOG(WARNING) << "[GEOP] Failed functionalize control flow: " << status.error_message();
    //   return Status::OK();
    // }
    // graph.ToGraphDef(&graph_def);
  }
  return Status::OK();
}
```

### 14.4 执行编译并安装（TF 1.15）

```bash
cd /mnt/workspace/tf_build/tensorflow
bash tfadapter/build.sh -c
pip3 install build/tfadapter/dist/python/dist/npu_bridge-1.15.0-py3-none-manylinux2014_aarch64.whl --upgrade
```

### 14.5 安装其它依赖包

```bash
# TF 1.15
pip3 install "numpy==1.18.5" pandas decorator sympy scipy attrs psutil protobuf==3.19.0
# TF 2.6.5
pip3 install "numpy==1.23.5" pandas decorator sympy scipy attrs psutil protobuf==3.19.0
```

> 此处必须锁定 numpy 版本（通过 `numpy==<版本>` 与其它包同条命令安装），否则 pandas/scipy 会将 numpy 升级到 2.x，破坏前面步骤五安装的 h5py 的 ABI 兼容性。protobuf 版本统一使用 3.19.0，与步骤十一保持一致。

### 14.6 编译安装 npu_device（TF 2.6.5）

TF 2.6.5 使用 tfa 源码中独立的 `tf_adapter_2.x/` 构建系统，无需修改源码，直接编译安装：

```bash
cd /mnt/workspace/tf_build/tensorflow
bash tf_adapter_2.x/build.sh -c -j8
pip3 install build/dist/python/dist/npu_device-2.6.5-py3-none-manylinux2014_aarch64.whl --upgrade --no-deps
```
