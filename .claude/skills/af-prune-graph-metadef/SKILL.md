---
name: af-prune-graph-metadef
description: Use when removing redundant or unused functions, definitions, classes, aliases, macros, or includes from an Autofuse graph_metadef header file in the graph-autofusion repository.
---

# 精简 Graph Metadef

## 原则

用仓内静态证据识别无用候选，用删除后的完整 UT 验证行为未回归。黑名单是不可修改的硬边界；除此之外，不因 public API/ABI 身份单独阻止删除。

## 流程

### 第一步：分析并确认

1. 要求用户给出一个仓内现存头文件。运行 `git status --porcelain`；输出非空时立即停止，请用户先提交、暂存到仓外或以其他方式保存并清理工作区。不要代用户 stash、commit 或暂存。
2. 在任何分析或编辑前检查用户给出的目标：

   ```bash
   skill_dir=.claude/skills/af-prune-graph-metadef
   bash "$skill_dir/scripts/check-blacklist.sh" check-target <header>
   ```

   命中黑名单时立即停止；不得通过改名、移动或修改黑名单文件绕过检查。
3. 阅读目标头文件，对其中每个声明、定义、宏等内容，逐项给出是否删除的处置结论，不得因内容类别未在 skill 中列举而跳过。检查源码、测试、CMake、脚本、Python/C++ 绑定、注册/序列化、字符串引用、条件编译和生成代码等所有可能使用该内容的位置。不要把“无直接调用”等同于“无用”；同时检查间接使用和隐式行为。
4. 判断逻辑使用，而不只统计文本引用：

   - 仅有 `#include`、但包含方未使用该头文件提供的符号、宏、注册或编译期副作用，视为无用候选。
   - 若包含方只依赖该头文件传递引入的其他声明，改为直接 include 真正依赖的头文件，并把原 include 视为无用候选。
   - 所有 include 均无实质使用，且目标头文件也没有独立注册、生成或构建用途时，可把整个头文件及对应实现列为删除候选。
   - 构造函数、析构函数、复制/移动构造函数和复制/移动赋值运算符不属于本重构范围，不得列为候选。保留其 `= default`、`= delete` 或自定义实现；这些声明表达类型生命周期、所有权和可复制/可移动性约束，不能用引用计数或编译器可能生成的隐式成员替代。

   不要仅凭 include 数量保留内容；也不要忽略宏展开、静态注册等无显式符号调用的真实行为。
5. 向用户提交候选清单：每项包含符号/定义、无用证据、需要修改的文件和不确定性。**停止并等待用户明确确认；未确认不得删除。**

### 第二步：删除并验证

1. 收到确认后，再次运行 `git status --porcelain`。工作区非空时停止，请用户处理。
2. 从依赖叶子开始删除已确认项。只修改目标头文件及删除所必需的直接关联文件；同步清理因此失效的实现、include、测试或构建项。不得扩大候选范围或清理无关历史代码。
3. 审阅最终 diff。完整 UT 较慢，正常流程只在所有删除和静态检查完成后运行一次。依次执行：

   ```bash
   bash "$skill_dir/scripts/check-blacklist.sh" check-diff
   git diff --check
   bash build.sh -u -j 8
   ```

4. 验证通过后，按仓库规范创建本地提交。
5. 报告删除项、判定证据、实际测试结果、commit 信息和残余不确定项。UT 失败时只修复本次删除引入的问题；修复后必须重新运行完整 UT。

## 完成门槛

- 黑名单校验与 `git diff --check` 通过。
- `bash build.sh -u -j 8` 全部通过；只跑模块 UT 不满足最终门槛。
- 用户已明确确认第一步给出的候选清单，实际删除未超出确认范围。
- 无法证明无用的内容保留并报告；不得靠编译碰运气批量删除。
- 任一黑名单文件发生新增、删除、移动或内容变化，立即停止且不得宣称完成。
