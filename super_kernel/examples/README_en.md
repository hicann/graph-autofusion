# super_kernel Sample Usage Guide

## Function Description

Use super_kernel to complete operator fusion.

## Directory Structure
```text
examples/
├── super_kernel_base							# basic functionality sample
│   └── superkernel_scope.py					# complete operator fusion through super_kernel
├── super_kernel_profiling						# profiling demonstration sample
│   └── superkernel_compare.py					# compare data using super_kernel vs not using super_kernel
└── super_kernel_runtime_ascendc_only			# minimal super_kernel sample
    └── superkernel_runtime_ascendc_basic.py	# compile super_kernel through ascendc for operator fusion, execute using runtime environment
```

## Prerequisites

Please refer to "[Source Build Guide](../../docs/en/build.md)" to complete prerequisite environment preparation.

## Dependency Installation

Python dependencies required for sample execution are written in [requirements.txt](requirements.txt), can install through:
```shell
pip3 install -r requirements.txt
```

## Use Case Demonstrations

[Use Case 1](super_kernel_base/README_en.md)

[Use Case 2](super_kernel_profiling/README_en.md)

[Use Case 3](super_kernel_runtime_ascendc_only/README_en.md)

## Reference

Please refer to relevant content in "[Ascend Extension for PyTorch](https://www.hiascend.com/document/redirect/pytorchuserguide)" under "Suite and Third-party Libraries > PyTorch Graph Mode Usage (TorchAir) > API Reference > torchair.scope > super_kernel".
