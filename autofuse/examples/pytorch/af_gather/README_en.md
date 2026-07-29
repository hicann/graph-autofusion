# Autofuse Use Case Demonstration

## Use Case Function:

This example constructs a gather + add graph pattern. It first selects values along dimension 1 through `torch.gather(x, 1, indices)`, and then performs an element-wise addition with tensor `y` of the same shape. The add operation is an elewise computation.

## Execution Command

```bash
python3 af_gather_add.py
```

## Expected Execution Result

Profiling files are generated in the profiling directory under the current directory. Open op_summary_timestampxx.csv under PROF_000001_timestampxx/mindstudio_profiler_output to view operator execution details. If the current PyTorch, torch_npu, and AscendC backend versions support Gather frontend conversion, the indexed selection and addition should appear as one kernel whose name starts with autofused_, without a standalone GatherElementsV2 or add. If a standalone GatherElementsV2 is still present, Gather has fallen back to ACLNN and this pattern is not fused in the current environment.
