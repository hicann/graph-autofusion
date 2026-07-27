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
├── autofuse               # config 配置
├── codegen                # kernel 代码生成 模块
├── common                 # 通用工具方法
├── compiler               # 对外API 接口
├── examples               # 示例脚本，演示典型用法
├── graph_metadef          # 基本图接口
├── inc                    # 供 GE 调用接口
├── optimize               # 调度切分 模块
├── scripts                # 脚本路径
├── v35                    # 昇腾950 芯片相关优化
├── CMakeLists.txt         # CMake 配置文件
├── blacklist.txt          # 工程配置文件
├── README.md
```

## 构建与安装

参考[执行构建](../docs/zh/build.md)。

## 上板验证指导
用户如果想在昇腾设备上体验Autofuse 的功能与性能，可以先参考[快速安装](../docs/zh/quick_install.md)准备环境。无论是没有昇腾设备的开发者，还是已有昇腾设备的开发者，都可以快速搭建好环境。在此基础上，按照上一步[构建与安装](../docs/zh/build.md)，增量安装了graph-autofusion仓编译生成的cann包。

此处指导如何搭建 Pytorch 环境，创建脚本，跑通 Inductor + Autofuse场景，并可视化生成的自动融合算子，以及观察最后的kernel性能。

当前自动融合支持elementwise类型+element类型，element类型+broadcast类型，element类型+reduce类型算子的融合。更多融合场景的支持（concat，gather等等）逐步开放中。


### 安装依赖

#### 安装 torch_npu
```bash
pip3 install numpy
pip3 install pyyaml
pip3 install setuptools
pip3 install torch_npu==2.8  # 通过pip 安装 torch_npu 时，会自动安装依赖的torch 版本
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


### sample 用例
autofuse 提供了丰富的 sample 用例，可以参考[Autofuse样例](./examples/README.md)。

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
注意：在设置的dump图路径下，生成 Autofuse 后端，对于每个融合算子的dump图。

### 结果分析 & 调测输出分析
用户开启 TORCH_COMPILE_DEBUG 后，调试信息输出位于执行目录下的torch_compile_debug子目录，带有 autofused_ 前缀的目录为 inductor-npu-ext 相关产物，其余均为 inductor 原生产物。每一个autofused_ 前缀的目录，都表示一个融合算子的白盒结构。如果没有融合算子产生（即未发生融合，需要通过打屏的 "Fallback aten.xxxx $reason: xx原因" 信息去判断原因。具体可参考[inductor-npu-ext使用手册](https://gitcode.com/Ascend/torchair/blob/master/experimental/_inductor_npu_ext/docs/manuals.md)。

用户也可以通过profiling的相关配置，观察使能自动融合后，算子性能收益情况。对于上面的sample用例，可以注释 "model = torch.compile(model, dynamic=False, fullgraph=True)" 这一行，即可走单算子流程。然后对比profiling里，单算子场景所有算子的总耗时，与使能 Inductor+Autofuse，融合算子的总耗时。详细的Profling性能分析工具的使用方法，可参见[Profiling性能分析工具指南](https://hiascend.com/document/redirect/CannCommunityToolProfiling)。

需要注意的是，不是模型里所有的算子都能被融合，对于在 Inductor 层未被 lowering 的算子，最后仍然以单算子形式存在。融合提升比，等于 (融合后所有算子耗时-融合前所有算子耗时)/融合前所有算子耗时。更进一步的，可以观察融合算子的 aiv_mte2_time（输入搬运耗时）和 aiv_mte3_time（输出搬运耗时）的提升情况。

对于精度的分析，详细的精度调试工具的使用方法，可参见[精度调试工具指南](https://hiascend.com/document/redirect/CannCommunityToolAccucacy)。

### 复杂网络使能
用户如果想在网络里，使能 Autofuse 功能，只需要在模型文件的开头，导入torch后面，加上 import inductor_npu_ext 即可。
