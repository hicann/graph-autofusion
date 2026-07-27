# af-perf-modeler

`af-perf-modeler` 是 graph-autofusion 的 V2 节点性能建模SKILL。

## 功能

- Cast / Reduce / Compare 等算子的性能公式建模
- ATT 性能建模、MicroAPI 成本、repeat_time / call_count 分析
- codegen 与性能公式对齐、tiling 参数透传

## 触发场景

- 用户提到 Cast / Reduce / Compare 等算子的性能公式
- 用户提到 ATT 性能建模、MicroAPI 成本、repeat_time / call_count
- 用户提到 codegen 与性能公式对齐、tiling 参数透传

## 验证方法

在 opencode 中输入以下指令验证 Skill 是否生效：
- "计算cast算子的性能公式"
- "分析compare算子性能公式"
