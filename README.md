# Graph-autofusion

## 🔥Latest News

- [2026/04] Autofuse 组件开源！
  在昇腾芯片上提供 Autofuse 自动融合功能，可以自动将相邻算子融合为1个，消除输入输出的搬运耗时，降低算子数量，优化算子总时长。
- [2025/10] Graph-autofusion 项目开源！
  在昇腾芯片上提供 SuperKernel 融合功能，可以减少任务调度等待时间和调度开销，优化算子执行头开销。

## 🚀概述

Graph-autofusion 是一个面向昇腾（Ascend）芯片的轻量级、解耦式组件集合，旨在通过各种融合相关技术，加速模型执行。
目前已开源 SuperKernel 组件和 Autofuse 组件，未来将持续开放更多融合相关模块。

组件特点：

- **专注融合加速技术**：基于 codegen 的 JIT 编译机制实现高效融合与加速。
- **模块化与解耦**：各组件之间独立，可按需选用；底层依赖极少，仅依赖 AscendC 与 runtime 环境。

## ⚡️快速入门

- 若您想体验 Graph-autofusion 的完整构建、测试与样例运行流程，请参考：[构建验证](docs/zh/build.md)
- 若您希望了解 SuperKernel 组件的原理与使用，请参考： [SuperKernel 简介](super_kernel/README.md)。
- 若您希望了解 Autofuse 组件的原理与使用，请参考： [Autofuse 简介与快速上手](autofuse/README.md)。

## 🔍目录结构

```text
graph-autofusion/
├── autofuse                                  # Autofuse 组件，Autofuse 源代码、测试、文档均在该子目录中
├── build.sh                                  # 一键式项目工程编译脚本
├── cmake                                     # 项目工程编译目录
├── CMakeLists.txt                            # 项目 CMakeLists
├── docs                                      # 项目整体文档
│   ├── zh                                    # 中文文档
│   │   ├── build.md                          # 一键式构建脚本文档
│   │   └── ...                               # 其他中文文档
│   └── en                                    # 英文文档
│       ├── build.md                          # 一键式构建脚本文档
│       └── ...                               # 其他英文文档
├── scripts                                   # 构建、环境安装和测试脚本
│   ├── env_install                           # 环境安装脚本
│   ├── package                               # 打包脚本
│   ├── test                                  # 测试脚本
│   ├── check_env.sh                          # 环境检查脚本
│   ├── init_env.sh                           # 环境初始化脚本
│   ├── oat_check.sh                          # OAT 合规性检查脚本
│   └── support_multiple_versions_of_lcov.sh  # 多版本 lcov 兼容脚本
├── super_kernel                              # SuperKernel 组件，SuperKernel 源代码、测试、文档均在该子目录中
├── README.md                                 # graph-autofusion 项目整体功能介绍
└── README_en.md                              # graph-autofusion 项目英文介绍
```

## 📝相关信息

- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)
