# Autofuse
## 简介
AutoFuse是基于Ascend C的自动融合框架，支持自动融合范围识别、自动算子代码生成、Auto Tiling优化、动态shape及混合精度等特性；在算法网络中，由于存在大量的Vector计算，各个Vector计算之间会产生大量的内存搬运，导致Memory Bound问题。而AutoFuse通过自动将多个算子融合为一个算子，减少网络中的算子数量和内存搬运，从而缓解了Memory Bound问题，释放昇腾算力，提升模型的执行性能。

详细介绍，请参考《[Autofuse自动融合](https://www.hiascend.com/document/detail/zh/canncommercial/850/graph/autofuse)》

## Autofuse 目录结构

```text
autofuse/
├── ascendc                # ascendc api 定义
├── ascir                  # 算子注册 ascir
├── att                    # 自动 tiling 生成 模块
├── cmake                  # cmake 脚本文件
├── codegen                # kernel 代码生成 模块
├── common                 # 通用工具方法
├── compiler               # 对外API 接口
├── examples               # 示例脚本，演示典型用法
├── graph_metadef          # 基本图接口
├── inc                    # 供 GE 调用接口
├── optimize               # 调度切分 模块
├── scripts                # 脚本路径
├── tests                  # 测试用例与测试框架
├── tools                  # 调试与分析工具
├── v35                    # 昇腾950 芯片相关优化
├── CMakeLists.txt         # CMake 配置文件
├── blacklist.txt          # 工程配置文件
├── README.md              # 中文说明文档
└── README_en.md           # 英文说明文档
```

## 构建与安装

参考[执行构建](../docs/zh/build.md)。

## 上板验证指导
用户如果想在昇腾设备上体验 Autofuse 的功能与性能，可以先参考[快速安装](../docs/zh/quick_install.md)准备环境。无论是没有昇腾设备的开发者，还是已有昇腾设备的开发者，都可以快速搭建好环境。在此基础上，按照上一步[构建与安装](../docs/zh/build.md)，增量安装了graph-autofusion仓编译生成的cann包。

AutoFuse 当前提供 PyTorch 和 TensorFlow 两种框架下的 Sample 用例，未来我们可能会支持更多框架。可根据实际使用场景参考对应文档完成环境安装和用例执行：

- [PyTorch 场景用例](./examples/pytorch/README.md)
- [TensorFlow 场景用例](./examples/tensorflow/README.md)

以下以 Pytorch 场景为例，指导如何搭建 Pytorch 环境，跑通 Pytorch场景下用例，并通过profiling数据观察最后的kernel性能。

### 安装依赖

#### 安装 torch_npu
```bash
pip3 install numpy
pip3 install pyyaml
pip3 install setuptools
pip3 install torch_npu==2.10.0  # torch_npu版本应为 2.9.0 及以上。通过pip 安装 torch_npu 时，会自动安装依赖的torch 版本。
```

#### 其他环境依赖
```bash
CMake >= 3.16.0
GCC >= 7.3.0
```
在 openEuler 系统上，您可以通过以下命令安装：
```bash
sudo yum install cmake gcc
```
在 Ubuntu 系统上，您可以通过以下命令安装：
```bash
sudo apt-get install cmake gcc
```

### 设置环境变量

   执行用例前，需要设置如下环境变量，设置运行NPU设备。
   ```bash
    # 用户自己的 driver 包安装路径
 	source /usr/local/Ascend/driver/bin/setenv.sh
 	# 用户自己的 CANN 包安装路径
 	source /usr/local/Ascend/ascend-toolkit/set_env.sh
    # 假设跑在 0卡，和脚本保持一致
 	export ASCEND_DEVICE_ID=0

   ```
### 执行用例
假设用例名为 test.py，直接执行：
   python3 test.py

### 更多调测相关环境变量
#### TORCH_COMPILE_DEBUG
作用： torch原生环境变量，启用详细调试日志，以及编译中间产物保存等。

使用方法：
```bash
export TORCH_COMPILE_DEBUG=1
```
注意： 多次执行相同脚本，会因为缓存存在而跳过编译，可以配合 TORCHINDUCTOR_FORCE_DISABLE_CACHES 使用，强制每次执行都重新编译。

#### TORCHINDUCTOR_FORCE_DISABLE_CACHES
作用： torch原生环境变量，禁用 Inductor 缓存，每次执行都会重新编译。

使用方法：
```bash
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
```
注意： 会显著增加图启动耗时，实际部署时请勿使用该环境变量。

#### 可选：ASCEND_LAUNCH_BLOCKING
作用： torch_npu原生环境变量，启用 Ascend 内核同步执行，每次kernel下发都会等待完成，便于确定首个报错的 kernel。

使用方法：
```bash
export ASCEND_LAUNCH_BLOCKING=1
```
注意： 会显著降低下发性能，实际部署时请勿使用该环境变量。

#### 可选：AUTOFUSE_DFX_FLAGS
作用： autofuse DFX环境变量，落盘每个自动融合算子，对应的内部融合图结构。pbtxt文件可以使用netron.app 打开观察。
使用方法：
```bash
export AUTOFUSE_DFX_FLAGS="--codegen_compile_debug=true;--debug_dir=/path-to-dump/"
```
注意：Autofuse 后端会在设置的 dump 路径下生成每个融合算子的 dump 图。

编译性能诊断由 `codegen_compile_debug=true` 控制。例如：
```bash
export AUTOFUSE_DFX_FLAGS="--codegen_compile_debug=true"
```

开启后会：

- 输出每个 LLVM pass 的耗时（`-ftime-report=per-pass`）；
- 生成编译器时间线 JSON 文件，默认保存到 `~/.cache/autofuse_compile_trace`，终端会输出 `[CompileTrace] <文件路径>`。

Host 编译会复用已有 PCH，缓存未命中时尝试创建。PCH 缓存目录为 `~/.cache/autofuse_pch_cache`；PCH 缓存或创建失败时会自动回退到普通 Host 编译。

### 结果分析 & 调测输出分析
用户开启 `TORCH_COMPILE_DEBUG` 后，调试信息会输出到当前执行目录下的 `torch_compile_debug` 子目录。其中，以 `autofused_` 为前缀的目录是 `torch_npu` AscendC 后端生成的融合算子产物，其余目录为 PyTorch Inductor 生成的原生产物。每个以 `autofused_` 为前缀的目录对应一个融合算子的白盒结构，可用于查看融合范围和代码生成结果。如果未生成以 `autofused_` 为前缀的目录，则说明当前编译过程中没有产生融合算子。此时，可以根据终端输出中的 `Fallback aten.xxxx $reason: xx原因` 信息分析未发生融合的原因。

用户也可以通过 Profiling 相关配置，观察使能自动融合后的算子性能收益。对于上述 Sample 用例，可以注释整个 `torch.compile(...)` 代码块，使模型以非编译模式执行，作为未使能自动融合的对照场景。
```python
# model = torch.compile(
#     model,
#     dynamic=False,
#     fullgraph=True,
#     options={"npu_backend": "ascendc"},
# )
```
分别采集未使能自动融合和使能自动融合两种场景的 Profiling 数据，并对比相同计算范围内所有相关算子的总耗时。

详细的Profiling性能分析工具的使用方法，可参见[Profiling性能分析工具指南](https://hiascend.com/document/redirect/CannCommunityToolProfiling)。

需要注意的是，不是模型里所有的算子都能被融合，对于在 Inductor 层未被 lowering 的算子，最后仍然以单算子形式存在。融合提升比，等于 (融合前所有算子耗时-融合后所有算子耗时)/融合前所有算子耗时。更进一步的，可以观察融合算子相比于单算子的 aiv_mte2_time（输入搬运耗时）和 aiv_mte3_time（输出搬运耗时）的提升情况。

对于精度的分析，详细的精度调试工具的使用方法，可参见[精度调试工具指南](https://hiascend.com/document/redirect/CannCommunityToolAccucacy)。

### 复杂网络使能
用户在网络中使能 AutoFuse 时，无需单独导入 `inductor_npu_ext`，
只需在 `torch.compile` 中指定 AscendC 后端：

```python
model = torch.compile(
    model,
    dynamic=False,
    fullgraph=True,
    options={"npu_backend": "ascendc"},
)
```
