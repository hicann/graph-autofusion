# autofuse 用例演示

## 用例功能：

本用例构造 gather + add 图模式：先通过 `torch.gather(x, 1, indices)` 沿第 1 轴取数，再将取数结果与同形状张量 `y` 逐元素相加。其中 add 属于 elewise 计算。

## 执行命令

```bash
python3 af_gather_add.py
```

## 预期执行结果

当前目录下的 profiling 目录下，有生成的 profiling 文件。其中 PROF_000001_时间戳xx/mindstudio_profiler_output 下面，可以打开 op_summary_时间戳xx.csv 文件，查看算子执行详情。若当前 PyTorch、torch_npu 和 AscendC 后端支持 Gather 前端转换，取数与加法应表现为一个名称以 autofused_ 开头的 kernel，且不再出现独立的 GatherElementsV2 和独立的 add；若仍出现独立 GatherElementsV2，则说明 Gather 已回退到 ACLNN，本用例在当前环境未发生融合。
