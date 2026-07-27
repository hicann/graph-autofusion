# TensorFlow 场景用例演示

## 功能描述

使用 autofuse 完成 TensorFlow 网络下的算子融合。通过 GE（Graph Engine）中的 Autofuse fusion pass 自动识别可融合算子并完成融合。

## 目录结构

```text
├── README.md                              # 本文档
└── af_tf_eleandele/                       # elementwise 类型算子融合的样例
    ├── README.md                          # 样例说明
    └── test_abs_relu_exp.py               # 通过 autofuse 完成 abs + relu + exp 三个 elementwise 算子的融合
```

## 前置说明

运行本用例前，需依次完成以下步骤：

1. 通过 [安装指导](../../../docs/zh/quick_install.md) 正确安装 toolkit 和 ops 包，并配置环境变量
2. 通过 [环境编译部署](../../../docs/env_install/tensorflow/env_tf.md) 搭建 TensorFlow 环境（x86_64 可直接 pip 安装，aarch64 需源码编译）
3. 也可使用一键配置脚本自动搭建环境（**仅 x86_64 架构可用**）：

   ```bash
   bash scripts/env_install/setup_tf_env.sh
   ```

   脚本完成后激活环境：

   ```bash
   source scripts/env_install/env/activate_tf1.sh    # TF 1.15
   # 或
   source scripts/env_install/env/activate_tf2.sh    # TF 2.6.5
   ```

   > **aarch64 架构不支持此脚本**：aarch64 请按 [aarch64 架构 TF 源码编译](../../../docs/env_install/tensorflow/build_tf_aarch64.md) 手动编译。

## 设置环境变量

```bash
# cann包安装路径变量定义，值根据实际安装位置设置
export CANN_INSTALL_PATH=/usr/local/Ascend
# 激活cann包中驱动相关环境变量
source $CANN_INSTALL_PATH/driver/bin/setenv.sh
# 激活cann包中toolkit相关环境变量
source $CANN_INSTALL_PATH/ascend-toolkit/set_env.sh

# 假设跑在 device0
export ASCEND_DEVICE_ID=0

# 开启自动融合
export AUTOFUSE_FLAGS="--enable_autofuse=true"
```

## 执行用例

```bash
cd af_tf_eleandele

# TF 1.15 环境
python3 test_abs_relu_exp.py --mode tf1

# TF 2.6.5 环境（兼容模式）
python3 test_abs_relu_exp.py --mode tf2-compat
```

## 预期执行结果

脚本执行 100 步推理，无报错即表示融合算子执行成功。可通过 Dump 图或 Profiling 进一步验证融合效果。

## 参考

- [Autofuse 简介](../../README.md)
- [Autofuse 业务流程](../../../docs/zh/component_workflow.md)
- [环境编译部署](../../../docs/env_install/tensorflow/env_tf.md)
- [aarch64 架构 TF 源码编译](../../../docs/env_install/tensorflow/build_tf_aarch64.md)
- [精度调试工具指南](https://hiascend.com/document/redirect/CannCommunityToolAccucacy)
- [Profiling 性能分析工具指南](https://hiascend.com/document/redirect/CannCommunityToolProfiling)
