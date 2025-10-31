# super_kernel样例使用指导

## 功能描述

使用super_kernel完成算子融合

## 目录结构
```
├── super_kernel_base             # 基础功能的样例
│  └── superkernel_scope.py       # 通过super_kernel完成算子融合
└── super_kernel_profiling        # 展示profiling的样例
  └── superkernel_compare.py      # 使用super_kernel与不使用super_kernel的数据进行对比
```
## 前置说明
确保已根据[《构建指南》](../../doc/build.md)完成前置环境准备和环境变量配置，确保已构建出`cann-graph-autofusion_${cann_version}_linux-${arch}.run`包。

## 依赖安装

### **安装驱动与固件**
编译源码、执行UT/ST时无需安装驱动与固件。运行样例时需要依赖device环境，必须安装相应驱动与固件，安装指导详见[《CANN软件安装指南》](https://www.hiascend.com/document/redirect/CannCommunityInstSoftware)。

### **安装toolkit包与算子包**
参考 [《构建指南-安装社区版cann-toolkit包》](../../doc/build.md#安装社区版cann-toolkit包) 章节安装社区版cann-toolkit包；
参考 [《构建指南-安装算子包》](../../doc/build.md#安装算子包) 章节安装社区版算子包。



### **安装graph-autofusion包**
参考 [《构建指南-安装与卸载》](../../doc/build.md#安装与卸载)章节安装编译好的graph-autofusion包。

### **安装pytorch、torch_npu插件**
当前用例执行时是通过pytorch框架中调用相关算子并执行，因此需要安装pytorch和torch_npu插件。
安装PyTorch框架和torch_npu插件，请参考[《Ascend Extension for PyTorch 软件安装指南》](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0001.html)，请保证与CANN相关包的版本匹配（参见[《版本说明》](https://www.hiascend.com/document/detail/zh/Pytorch/710/releasenote/releasenote_0002.html)），否则功能可能无法正常使用。

> 说明:
>
> PyTorch与PyTorchAdapter（Ascend Extension for PyTorch）版本当前只支持python3.8+（具体详见PyTorchAdapter版本说明），为此，如果需要执行example，需要保证当前python版本满足PyTorch安装要求

## 用例演示

[用例1](super_kernel_base/README.md)

[用例2](super_kernel_profiling/README.md)

## 参考

[super_kernel使用方法介绍](https://www.hiascend.com/document/detail/zh/Pytorch/710/modthirdparty/torchairuseguide/torchair_00090.html)