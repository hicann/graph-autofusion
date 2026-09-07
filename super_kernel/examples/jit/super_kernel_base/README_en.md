# super_kernel Use Case Demonstration

## Use Case Function:

sk1 fuses GroupedMatmul+GroupedMatmul+MoeGatingTopK three operators.

## Use super_kernel to Fuse Operators

Use the following with statement block (super_kernel). Operators within the statement block are all fused into one super kernel for computation:
```python
with torchair.scope.super_kernel("sk1"):
```
For detailed function introduction, see [Mark SuperKernel Scope in Graph](https://www.hiascend.com/document/redirect/PytorchTorchairSuperKernel).

## Execution Command

```bash
python3 superkernel_scope.py
```

## Expected Execution Result

After execution, print shows success:
```text
execute sample success
```
