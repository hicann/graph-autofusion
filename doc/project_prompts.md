# Project Prompts

## 📘 文件说明

本文件用于保存 **与 Graph-autofusion 项目相关的 AI 辅助开发指令（prompts）**。  
这些指令可直接复制到 VSCode Chat（GitHub Copilot Chat / ChatGPT Codex）中使用，  
帮助开发者快速初始化环境、运行测试、或与智能 Agent 协同开发。

> 💡 说明：
> - 该文件仅用于记录可复用的智能 Agent 指令，不会在项目构建或运行过程中被执行。
> - Agent 不会自动读取此文件，只有在人工引用（例如输入 “阅读 #file:doc/project_prompts.md”）时才会被加载。
> - 当项目未来增加新的开发流程、测试任务或组件（如 superkernel、autofuse-core 等），可在本文件追加对应的 prompt 模板。

---

## 🧩 环境初始化与测试 Prompt

此 prompt 用于在 VSCode Copilot / ChatGPT Codex 中快速初始化 Graph-autofusion 项目的开发环境，  
并验证 pytest 测试是否能正常通过。

```prompt
# Graph-autofusion 项目开发环境初始化

请阅读以下文件，理解项目结构与开发流程：
- #file:README.md
- #file:developer_guide.md

设置环境变量：
  ASCEND_INSTALL_PATH=<!!!注意配置你的环境变量!!!>

使用项目根目录下的 Python 虚拟环境：
  venv

---

## 初始任务（Initial Task）

1. **读取并理解项目结构与依赖**（read and understand the project structure and dependencies）  
2. **执行 pytest 测试**（run `pytest` in project root）  
   - 若测试失败，请列出失败的测试项、错误堆栈（traceback）与可能原因。  
   - 不要修改任何代码。  
3. **报告结果**（summarize test results）  
   - 成功时说明测试数量与耗时；  
   - 失败时重点说明可能的依赖或环境问题。

---

## 后续计划（Next Step）

当 pytest 全部通过后，我们将：
- 一起开发和调试 **superkernel** 组件；
- 根据测试结果优化自动融合逻辑；
- 在此过程中，你保持中文交流，我来指令下一步要开发的内容。

---

## 规则（Rules）
- 所有系统命令（如 `pytest`、`pip install`、`python setup.py build`）请在项目根目录执行。  
- 仅在我明确指示时修改源代码。  
- 输出报告时请使用中文解释，代码与命令保持英文。  

```