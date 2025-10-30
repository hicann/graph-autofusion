# super_kernel 用例演示 

## 用例功能：

sk1 融合 GroupedMatmul+GroupedMatmul+MoeGatingTopK 三个算子

## 使用super_kernel融合算子

使用如下with语句块（super_kernel），语句块内算子均被融合为一个超级Kernel进行计算
```
with torchair.scope.super_kernel("sk1"): 
```
详细功能介绍见[图内标定SuperKernel范围](https://www.hiascend.com/document/detail/zh/Pytorch/710/modthirdparty/torchairuseguide/torchair_00035.html)

## 执行命令

```
python3 superkernel_scope.py
```

## 预期执行结果

执行后打印显示success
```
execute sample success
```