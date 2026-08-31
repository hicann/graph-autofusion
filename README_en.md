# Graph-autofusion

## 🔥Latest News

- [2026/04] Autofuse component is open sourced!
  The Autofuse automatic fusion function is provided on Ascend chips. It can automatically fuse adjacent operators into one, eliminating the time consumption of input and output transfer, reducing the number of operators, and optimizing the total operator execution time.
- [2025/10] Graph-autofusion project is open sourced!
  SuperKernel fusion function is provided on Ascend chips. It can reduce task scheduling waiting time and scheduling overhead, optimizing operator execution overhead.

## 🚀Overview

Graph-autofusion is a lightweight, decoupled component collection for Ascend chips. It aims to accelerate model execution through various fusion-related technologies. Currently, the SuperKernel component and Autofuse component are open sourced, and more fusion-related modules will be continuously released in the future.

Component features:

- **Focus on fusion acceleration technology**: Efficient fusion and acceleration are implemented based on codegen JIT compilation mechanism.
- **Modular and decoupled**: Components are independent and can be selected as needed. The underlying dependencies are minimal, relying only on AscendC and the runtime environment.

## ⚡️Quick Start

- To experience the complete build, test, and sample running process of Graph-autofusion, see [Build Verification](docs/en/build.md).
- To understand the principle and usage of the SuperKernel component, see [SuperKernel Introduction](super_kernel/README.md).
- To understand the principle and usage of the Autofuse component, see [Autofuse Introduction and Quick Start](autofuse/README.md).

## 📚 Documents

If you want to understand the architecture, module functions, and skills of Graph-autofusion, please refer to the following documents:

- [AutoFuse Architecture Description](docs/en/autofuse/design/architecture.md): Introduces the overall architecture, key technical solutions, processing flow and module responsibilities of AutoFuse.
- [Contribution Guide](CONTRIBUTING.md): Describes how to contribute to the project, submit Issues and Pull Requests.
- [Skills Management Guide](docs/zh/opencode-skill-management.md): Introduces the default Skills used in the repository and their management methods.
- [Skill Reuse Guide](docs/zh/skill-reuse-guide.md): Introduces how to reuse Skills and use Agent to assist code reading, development and problem location.

## 🌐 Ecosystem Integration

AutoFuse can be used as an automatic fusion backend for upper-layer graph compilers and deep learning frameworks, providing Ascend C fused operator generation capabilities for model compilation and execution. Currently the main integration paths are:

- **GE**: As the automatic fusion backend for GE. [GE Project](https://gitcode.com/cann/ge)
- **PyTorch**: As the AscendC backend for PyTorch Inductor, use AutoFuse via `torch.compile`. [TorchAir Project](https://gitcode.com/Ascend/torchair)
- **TensorFlow**: As the automatic fusion backend after TensorFlow Adapter connects to GE. [TensorFlow Adapter Project](https://gitcode.com/cann/tensorflow)

The above integration paths are subject to the actual support of the corresponding CANN version and upper-layer components.

## 🔍Directory Structure

```text
graph-autofusion/
├── autofuse                                  # Autofuse component. Autofuse source code, tests, and documentation are in this subdirectory.
├── build.sh                                  # One-click project build script
├── cmake                                     # Project build directory
├── CMakeLists.txt                            # Project CMakeLists
├── docs                                      # Project documentation
│   ├── zh                                    # Chinese documentation
│   │   ├── build.md                          # One-click build script documentation
│   │   └── ...                               # Other Chinese documentation
│   └── en                                    # English documentation
│       ├── build.md                          # One-click build script documentation
│       └── ...                               # Other English documentation
├── scripts                                   # Build, environment installation, and test scripts
│   ├── env_install                           # Environment installation scripts
│   ├── package                               # Packaging scripts
│   ├── test                                  # Test scripts
│   ├── check_env.sh                          # Environment check script
│   ├── init_env.sh                           # Environment initialization script
│   ├── oat_check.sh                          # OAT compliance check script
│   └── support_multiple_versions_of_lcov.sh  # Multi-version lcov compatibility script
├── super_kernel                              # SuperKernel component. SuperKernel source code, tests, and documentation are in this subdirectory.
├── README.md                                 # Chinese graph-autofusion project introduction
└── README_en.md                              # English graph-autofusion project introduction
```

## 📝Related Information

- [Security Statement](SECURITY_en.md)
- [License](LICENSE)
