# Graph-autofusion

## 🔥Latest News

- [2025/10] Graph-autofusion 项目开源！
  在昇腾芯片上提供 SuperKernel 自动融合功能，可以减少任务调度等待时间和调度开销，优化算子执行头开销。

## 🚀概述

Graph-autofusion 是一个面向昇腾（Ascend）芯片的轻量级、解耦式组件集合，旨在通过自动融合技术加速模型执行。
目前已开源 SuperKernel 组件，未来将持续开放更多自动融合相关模块。

组件特点：

- **专注自动融合加速技术**：基于 codegen 的 JIT 编译机制实现高效融合与加速。
- **模块化与解耦**：各组件之间独立，可按需选用；底层依赖极少，仅依赖 AscendC 与 runtime 环境。

## ⚡️快速入门

- 若您想体验 Graph-autofusion 的完整构建、测试与样例运行流程，请参考：[构建验证](doc/build.md)
- 若您希望了解 SuperKernel 组件的原理与使用，请参考： [SuperKernel 简介](super_kernel/README.md)。

## 🔍目录结构

```
├── build.sh                       # 一键式项目工程编译脚本
├── cmake                          # 项目工程编译目录
├── CMakeLists.txt                 # 项目 CMakeLists
├── doc                            # 项目整体文档
│  ├── build.md                    # 一键式构建脚本文档
│  ├── ...                         # 其他有关该项目的文档说明
├── super_kernel                   # SuperKernel 组件，SuperKernel 源代码、测试、文档均在该子目录中
├── ...                            # 未来规划的组件
├── README.md                      # graph-autofusion项目整体功能介绍
├── scripts                        # 脚本路径
   └── package
```

## 📝相关信息

- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)

