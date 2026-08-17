# super_kernel Minimal Sample

## Use Case Function:

This sample demonstrates how to use super_kernel to complete operator fusion, including operator fusion definition, compilation, execution, and other steps.
Core features:
- Simple dependencies, only depends on AscendC and runtime environment.
- Uses Python wrapper for underlying C interfaces, simplifying development workflow.

## Directory Structure
```text
super_kernel_runtime_ascendc_only/
├── README.md                                     # Chinese documentation
├── README_en.md                                  # English documentation
├── compile_sk.py                                 # Compile sub-kernel and SuperKernel operators
├── superkernel_runtime_ascendc_basic.py          # Main entry for sub-kernel compilation, SuperKernel compilation, memory allocation, loading, and execution
└── utils.py                                      # Utility functions
```

## Use Case Introduction

```mermaid
graph TB
    %% Overall view: two inputs -> superkernel -> output
    input1[input1] --> sk[superkernel]
    input2[input2] --> sk
    sk --> output[output]
    %% Enlarged internal view: subgraph shows superkernel internal structure
    subgraph superkernel_internal_structure
        inner1_in(inner_input1) --> pow(pow)
        inner2_in(inner_input2)  --> pow
        pow --> isinf(isinf)
        isinf --> inner_out(inner_output)
    end
```

This use case demonstrates super_kernel basic functionality through compile-time dependency on ascendc and runtime dependency. Main steps:
- 1. Initialization
- 2. Compile sub_kernel, compile super_kernel, set sub-kernel topology relationship in super kernel for memory allocation
- 3. Memory allocation, input data construction
- 4. Kernel loading
- 5. Launch execution, including args layout and so on
- 6. Print output, output result validation
- 7. Resource cleanup and release, including memory, kernel, stream, and so on

> Explanation:
> 1. Sub-kernel topology relationship is represented through strings. For example, if pow output is isinf input, then pow output and isinf input use the same string representation.
> 2. When allocating memory, use strings to express same memory addresses.
> 3. When launching args, layout follows [pow_in1, pow_in2, pow_ws, isinf_in1, isinf_out1, isinf_ws]

## Execution Command

```bash
python3 superkernel_runtime_ascendc_basic.py
```

## Expected Execution Result

After execution, print shows success:
```text
execute sample success
```
