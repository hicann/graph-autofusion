# Autofuse

## Introduction

AutoFuse is an automatic fusion framework based on Ascend C, supporting automatic fusion scope identification, automatic operator code generation, Auto Tiling optimization, dynamic shape, and mixed precision features. In algorithm networks, due to numerous Vector calculations, substantial memory transfers occur between Vector calculations, causing Memory Bound issues. AutoFuse automatically fuses multiple operators into a single operator, reducing the number of operators and memory transfers in the network, thereby alleviating Memory Bound issues, releasing Ascend computing power, and improving model execution performance.

For detailed introduction, refer to "[Autofuse Automatic Fusion](https://www.hiascend.com/document/detail/zh/canncommercial/850/graph/autofuse)".

## Autofuse Directory Structure

```text
autofuse/
├── ascendc                # ascendc api definitions
├── ascir                  # operator registration ascir
├── att                    # automatic tiling generation module
├── autofuse               # config configuration
├── cmake                  # third-party library related configuration
├── codegen                # kernel code generation module
├── common                 # common utility methods
├── compiler               # external API interface
├── examples               # example scripts demonstrating typical usage
├── graph_metadef          # basic graph interface
├── inc                    # interface for GE calls
├── optimize               # scheduling and partitioning module
├── scripts                # script path
├── v35                    # Ascend 950 chip related optimization
├── CMakeLists.txt         # CMake configuration file
├── blacklist.txt          # project configuration file
├── build_third_party.sh   # third-party library installation script
├── README.md
```

## Build and Installation

Refer to [Build Instructions](../docs/en/build.md).

## On-Device Verification Guide

Users who wish to experience AutoFuse functionality and performance on Ascend devices can first refer to [Quick Installation](../docs/en/quick_install.md) to prepare the environment. Whether developers have Ascend devices or not, they can quickly set up the environment. On this basis, following the previous [Build and Installation](../docs/en/build.md), incrementally install the CANN package compiled from the graph-autofusion repository.

This section guides how to set up a PyTorch environment, create scripts, run through the Inductor + AutoFuse scenario, visualize generated auto-fusion operators, and observe final kernel performance.

Currently, auto-fusion supports fusion of elementwise + element type, element + broadcast type, and element + reduce type operators. Support for more fusion scenarios (concat, gather, and so on) is gradually being released.

### Install Dependencies

#### Install torch_npu

```bash
pip3 install numpy
pip3 install pyyaml
pip3 install setuptools
pip3 install torch_npu==2.8  # Installing torch_npu via pip automatically installs the dependent torch version
```

#### Install inductor-npu-ext (AutoFuse enabling framework in Inductor)

```bash
git clone https://gitcode.com/Ascend/torchair.git
cd torchair/experimental/_inductor_npu_ext/
pip3 install -e ./python/
```

#### Other Environment Dependencies

```bash
CMake >= 3.16.0
GCC >= 7.3.0
```
On openEuler systems, you can install through the following commands:
```bash
sudo yum install cmake gcc
```
On Ubuntu systems, you can install through the following commands:
```bash
sudo apt-get install cmake gcc
```

### Sample Use Cases

AutoFuse provides abundant sample use cases. Refer to [AutoFuse Samples](./examples/pytorch/README_en.md).

### Set Environment Variables

Before executing use cases, set the following environment variables to configure the NPU device:
```bash
# Your own driver package installation path
source /usr/local/Ascend/driver/bin/setenv.sh
# Your own CANN package installation path
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# Assume running on card 0, keep consistent with script
export ASCEND_DEVICE_ID=0
```

### Execute Use Cases

Assume the use case name is test.py, execute directly:

```bash
python3 test.py
```

### More Debugging Related Environment Variables

#### TORCH_COMPILE_DEBUG

Purpose: Native torch environment variable that enables detailed debug logging and saving of compilation intermediate artifacts.

Usage:
```bash
export TORCH_COMPILE_DEBUG=1
```
Note: Multiple executions of the same script may skip compilation due to cache. Can use with TORCHINDUCTOR_FORCE_DISABLE_CACHES to force recompilation each execution.

#### TORCHINDUCTOR_FORCE_DISABLE_CACHES

Purpose: Native torch environment variable that disables Inductor cache, forcing recompilation each execution.

Usage:
```bash
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
```
Note: Significantly increases graph startup time. Do not use this environment variable in actual deployment.

#### Optional: ASCEND_LAUNCH_BLOCKING

Purpose: Native torch_npu environment variable that enables Ascend kernel synchronous execution. Each kernel launch waits for completion, facilitating identification of the first erroneous kernel.

Usage:
```bash
export ASCEND_LAUNCH_BLOCKING=1
```
Note: Significantly reduces launch performance. Do not use this environment variable in actual deployment.

#### Optional: AUTOFUSE_DFX_FLAGS

Purpose: AutoFuse DFX environment variable that dumps internal fusion graph structure for each auto-fusion operator. pbtxt files can be opened with netron.app.

Usage:
```bash
export AUTOFUSE_DFX_FLAGS="--codegen_compile_debug=true;--debug_dir=/path-to-dump/"
```
Note: Generates dump graphs for each fused operator from AutoFuse backend in the specified dump path.

### Result Analysis & Debug Output Analysis

After enabling TORCH_COMPILE_DEBUG, debug information output is located in the torch_compile_debug subdirectory under the execution directory. Directories prefixed with autofused_ are artifacts related to inductor-npu-ext, others are native Inductor artifacts. Each autofused_ prefix directory represents a white-box structure of a fusion operator. If no fusion operator is generated (that is, no fusion occurs), check the printed "Fallback aten.xxxx $reason: xx reason" information to determine the cause. Refer to [inductor-npu-ext User Manual](https://gitcode.com/Ascend/torchair/blob/master/experimental/_inductor_npu_ext/docs/manuals.md).

Users can also observe operator performance gains after enabling auto-fusion through profiling configuration. For the sample use cases above, comment the "model = torch.compile(model, dynamic=False, fullgraph=True)" line to run single-operator flow. Then compare the total time of all operators in single-operator scenario in profiling with the total time of fusion operators when enabling Inductor + AutoFuse. For detailed Profiling performance analysis tool usage, refer to [Profiling Performance Analysis Tool Guide](https://hiascend.com/document/redirect/CannCommunityToolProfiling).

Note that not all operators in the model can be fused. Operators not lowered at the Inductor layer still exist as single operators. Fusion improvement ratio equals (total operator time after fusion - total operator time before fusion) / total operator time before fusion. Further, observe the improvement of aiv_mte2_time (input transfer time) and aiv_mte3_time (output transfer time) for fusion operators.

For precision analysis, refer to [Precision Debugging Tool Guide](https://hiascend.com/document/redirect/CannCommunityToolAccucacy).

### Enabling in Complex Networks

Users who wish to enable AutoFuse in networks only need to import inductor_npu_ext after importing torch at the beginning of the model file.
