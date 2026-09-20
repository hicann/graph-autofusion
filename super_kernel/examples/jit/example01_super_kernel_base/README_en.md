# super_kernel Use Case Demonstration

## Use Case Function:

sk1 fuses GroupedMatmul+GroupedMatmul+MoeGatingTopK three operators.

## Use super_kernel to Fuse Operators

Use the following with statement block (super_kernel). Operators within the statement block are all fused into one super kernel for computation:
```python
with torchair.scope.super_kernel("sk1"):
```
For detailed function introduction, see [Mark SuperKernel Scope in Graph](https://www.hiascend.com/document/redirect/PytorchTorchairSuperKernel).

## Environment Requirements

This sample supports the following product models:

- Atlas A3 training series products/Atlas A3 inference series products
- Atlas A2 training series products/Atlas A2 inference series products

Follow the [source build guide](../../../../docs/en/build.md), then install PyTorch 2.7.1 and
TorchNPU 2.7.1.post10 (CANN 9.1.0 or later, Python 3.9.x or later) as officially released matching
versions. For version compatibility and installation, see the
[Ascend Extension for PyTorch user guide](https://www.hiascend.com/document/redirect/pytorchuserguide).
Finally, install the sample's [Python dependencies](../../requirements.txt):

```bash
pip install -r super_kernel/examples/requirements.txt
```

## Execution Command

In the sample directory, run:

```bash
python3 superkernel_scope.py
```

## Expected Result

On success, the sample prints the following key log:
```text
execute sample success
```
