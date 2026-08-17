# PyTorch Inductor 场景用例演示

## 功能描述

使用 `torch.compile` 完成 PyTorch 网络下的算子融合。

当前包含以下三个用例：

- `add + ge`：将加法和比较算子融合为一个算子；
- `mul + reducesum`：将乘法和求和归约算子融合为一个算子；
- `gather + add`：构造索引取数和逐元素加法图模式

注：当前暂不支持gather融合能力，等待[ issue175 ](https://gitcode.com/cann/graph-autofusion/issues/175)这个issue完成后gather可以和add进行融合。

三个用例均开启 NPU Profiling，可通过生成的性能分析文件查看算子执行情况和融合结果。

## 目录结构

```text
pytorch
├── README.md
├── README_en.md
├── af_pointwise
│   ├── README.md
│   ├── README_en.md
│   └── af_add_ge.py              # 融合 add + ge
├── af_reduce
│   ├── README.md
│   ├── README_en.md
│   └── af_mul_reducesum.py       # 融合 mul + reducesum
└── af_gather
    ├── README.md
    ├── README_en.md
    └── af_gather_add.py          # gather + add 图模式
```

## 前置说明

运行本用例前，请先认真阅读[ PyTorch环境安装说明 ](../../../docs/env_install/pytorch/env_pytorch.md)。需完成以下步骤：
1. CANN 包版本要求为 `9.0.0` 及以上，通过 [CANN 快速安装](https://www.hiascend.com/cann/download?versionId=745&ids=d802%2Ch0501%2Ch0602%2Ch0701) 正确安装 toolkit 和 ops 包，可以参考[ 安装指导 ](../../../docs/zh/quick_install.md)。
2. `torch_npu` 版本要求为 `2.9.0` 及以上，可以根据 [环境快速安装脚本](../../../scripts/env_install/pytorch/setup_torch_npu_daily.sh) 快速安装python环境和 `torch_npu` 。

## 设置环境变量

每次新开终端后，执行：

```bash
# 环境激活
source /mnt/workspace/env/venv/torch210_daily/bin/activate

# CANN 包安装路径根据实际安装位置确定。
export CANN_INSTALL_PATH=/home/developer/Ascend

# 加载 CANN 相关环境变量
source $CANN_INSTALL_PATH/cann/set_env.sh

#假设跑在 device0
export ASCEND_DEVICE_ID=0
```

## 执行用例

### add + ge 融合

```bash
cd af_pointwise
python af_add_ge.py
```

### mul + reducesum 融合

```bash
cd af_reduce
python af_mul_reducesum.py
```

### gather + add 图模式

```bash
cd af_gather
python af_gather_add.py
```

## 预期执行结果

程序执行完成后，当前目录下会生成 `profiling` 目录。

可在以下目录中查看算子执行详情：

```text
profiling/PROF_时间戳/mindstudio_profiler_output
```

打开其中的：

```text
op_summary_时间戳.csv
```

如果算子列表中存在名称以 `autofused_` 开头的 Kernel，表示相关算子已经成功融合为一个融合算子。

## 参考

- [Autofuse 简介与快速上手](../../README.md)
- [Profiling 性能分析工具指南](https://hiascend.com/document/redirect/CannCommunityToolProfiling)
- [精度调试工具指南](https://hiascend.com/document/redirect/CannCommunityToolAccucacy)
