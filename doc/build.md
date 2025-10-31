# 一键式构建指南

## 前置环境准备
本项目支持一键式源码构建，在构建前，请根据如下步骤完成相关环境准备，确保系统满足构建要求。

### 安装构建工具

以下所列为构建所需的工具，请注意版本要求：

- Python3 >= 3.7.0, <= 3.11.4

- CMake >= 3.16.0  (建议使用3.20.0版本)
  ```shell
  # Ubuntu/Debian操作系统安装命令示例如下，其他操作系统请自行安装
  sudo apt-get install cmake
  ```
### 安装社区版cann-toolkit包

根据实际环境，下载对应`Ascend-cann-toolkit_${cann_version}_linux-${arch}.run`包，下载链接为[toolkit x86_64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20251030_test/Ascend-cann-toolkit_8.5.0.alpha001_linux-x86_64.run)、[toolkit aarch64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20251030_test/Ascend-cann-toolkit_8.5.0.alpha001_linux-aarch64.run)。

```shell
# 确保安装包具有可执行权限
chmod +x Ascend-cann-toolkit_${cann_version}_linux-${arch}.run
# 安装命令
./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --full --force --install-path=${ascend_install_path}
```
- \$\{cann\_version\}：表示 CANN 包版本号。
- \$\{arch\}：表示CPU架构，如 aarch64、x86_64。
- \$\{ascend\_install\_path\}：表示指定安装路径，不指定则默认安装在`/usr/local/Ascend`（root 用户）或`${HOME}/Ascend`（非 root 用户）目录。
- 安装过程如有其他问题请参阅：CANN 社区版-环境准备-[软件安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/softwareinst/instg/instg_quick.html?Mode=PmIns&OS=Debian&Software=cannToolKit)。 


### 安装社区版CANN ops包

   根据产品型号和环境架构，下载对应`cann-${soc_name}-ops-${cann_version}_linux-${arch}.run`包，下载链接如下：
   
   - Atlas A3 训练系列产品/Atlas A3 推理系列产品：[ops x86_64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20251030_test/Ascend-cann-A3-ops_8.5.0.alpha001_linux-x86_64.run)、[ops aarch64包](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20251030_test/Ascend-cann-A3-ops_8.5.0.alpha001_linux-arch64.run)。
   
   ```bash
   # 确保安装包具有可执行权限
   chmod +x Ascend-cann-${soc_name}-ops-${cann_version}_linux-${arch}.run
   # 安装命令
   ./Ascend-cann-${soc_name}-ops-${cann_version}_linux-${arch}.run --full --install-path=${ascend_install_path}
   ```
   
   - \$\{soc\_name\}：表示NPU型号名称，即\$\{soc\_version\}删除“ascend”后剩余的内容。
   - \$\{ascend\_install\_path\}：表示指定安装路径，需要与toolkit包安装在相同路径，默认安装在`/usr/local/Ascend`目录。

### 源码下载

```shell
git clone https://gitcode.com/cann/graph-autofusion.git
 ```

### 安装 python 依赖

1. 建议使用 Python 虚拟环境：
    ```shell
    python3 -m venv venv
    source venv/bin/activate
    ```
2. 安装依赖：
    ```shell
    cd graph-autofusion
    pip3 install -r super_kernel/requirements-dev.txt
    ```

## 构建

确保已完成[前置环境准备](#前置环境准备)。  

### 配置环境变量

根据前面[安装社区版cann-toolkit包](#安装社区版cann-toolkit包)的安装路径，执行合适的命令。
``` shell
    # 如果之前未指定路径安装，则：
    # root 用户，ascend_install_path=/usr/local/Ascend
    # 非 root 用户，ascend_install_path=${HOME}/Ascend
    source ${ascend_install_path}/latest/bin/setenv.bash
```

### 执行构建

执行以下命令：
```shell
  bash build.sh --pkg
```
执行成功后会在`build_out`目录下生成`cann-graph-autofusion_${cann_version}_linux-${arch}.run`。
- --pkg 表示构建 run 包
- ${cann_version} 表示 cann 版本号。
- ${arch} 表示 CPU 架构，如 aarch64、x86_64。
- 更多选项功能可以用 -h 查看。  
``` shell
  bash build.sh -h
```

## 测试验证
在执行本章节操作之前，确保已完成[前置环境准备](#前置环境准备)。

### UT 验证

```bash
  bash build.sh -u
```
执行完成后根据输出日志查看UT测试执行情况，用例执行成功会打印`passed`并且无`failed`打印，确认所有测试用例通过。

### ST 验证

```bash
  bash build.sh -s
```
执行完成后根据输出日志查看ST测试执行情况，用例执行成功会打印`passed`并且无`failed`打印，确认所有测试用例通过。

### 覆盖率验证
```bash
  bash build.sh -u -c # UT 覆盖率
  bash build.sh -s -c # ST 覆盖率
  bash build.sh -c    # UT 覆盖率 + ST 覆盖率
```
执行完成后根据输出日志查看覆盖率情况，确认所有测试用例通过。

## 安装与卸载

`cann-graph-autofusion_${cann_version}_linux-${arch}.run`来自前面[执行构建](#执行构建)环节构建生成。
```shell
# 安装，如果需要指定安装路径，则加上 --install-path=${ascend_install_path}
./cann-graph-autofusion_${cann_version}_linux-${arch}.run --full --quiet --pylocal

# 卸载，如果是安装到指定路径情况，则加上 --install-path=${ascend_install_path}
./cann-graph-autofusion_${cann_version}_linux-${arch}.run --uninstall
```
- 说明，此处的安装路径（无论默认还是指定）需与前面安装 cann-toolkit 包时的路径保持一致。  
- --full          全量模式安装。
- --install-path  指定安装路径，不指定则默认安装在`/usr/local/Ascend`（root 用户）或`${HOME}/Ascend`（非 root 用户）目录。
- --quiet         静默安装，跳过人机交互环节。  
- --pylocal       安装 run 包时，是否将包内的 .whl 跟随 run 包安装路径来安装。  
  - 若选择该参数，则 .whl 安装在`${ascend_install_path}/latest/python/site-packages`路径下。
  - 若不选择该参数，则 .whl 安装在本地 python 路径下，例如`/usr/local/python3.7.5/lib/python3.7/site-packages`。
- 更多安装选项请用 --help 选项查看。  

**安装完成后可参考[样例运行](../super_kernel/examples/README.md)尝试运行样例**。  
