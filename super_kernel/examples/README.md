# super_kernel样例使用指导

## 功能描述

使用super_kernel完成算子融合

## 目录结构
```text
examples/
├── super_kernel_base							# 基础功能的样例
│   └── superkernel_scope.py					# 通过super_kernel完成算子融合
├── super_kernel_profiling						# 展示profiling的样例
│   └── superkernel_compare.py					# 使用super_kernel与不使用super_kernel的数据进行对比
└── super_kernel_runtime_ascendc_only			# 极简super_kernel样例
    └── superkernel_runtime_ascendc_basic.py	# 通过ascendc编译super_kernel完成算子融合，并使用runtime运行时环境执行
```
## 前置说明
请务必参考[《源码构建指南》](../../docs/zh/build.md)完成前置环境准备。

## 依赖安装

样例执行所需的Python依赖已写入[requirements.txt](requirements.txt)，可通过以下命令安装：
```shell
pip3 install -r requirements.txt
```

## 用例演示

[用例1](super_kernel_base/README.md)

[用例2](super_kernel_profiling/README.md)

[用例3](super_kernel_runtime_ascendc_only/README.md)

## 参考

请参考[《Ascend Extension for PyTorch》](https://www.hiascend.com/document/redirect/pytorchuserguide)中“套件与三方库 > PyTorch图模式使用(TorchAir) > API参考 > torchair.scope > super_kernel”的相关内容。
