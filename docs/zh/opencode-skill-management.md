# opencode Skill 管理机制

## 概述

graph-autofusion 项目采用三层 Skill 架构（与 GE 项目对齐），结合本地定制、远程自动更新和第三方插件，为 AI 辅助开发提供丰富的领域能力。项目在此基础上扩展了 3 个项目专属 Skills，并提供面向开源社区的外部贡献者指引。

## 架构总览

```
opencode 启动
  │
  ├── 扫描 .claude/skills/*/SKILL.md ──────────────┐
  ├── 扫描 .opencode/skills/*/SKILL.md ────────────┤── 注册为可用 Skills
  ├── 加载插件系统（superpowers 等）────────────────┤
  └── 执行 plugins/install-default-skills.ts       │
        └── 调用 install-default-skills.sh         │
              └── 从远程仓库拉取 → _remote/ ────────┘
                    └── 创建符号链接到一级目录
```

## 三层 Skill 来源

### 第一层：本地自定义 Skills

| 属性 | 说明 |
|-----------|-------------|
| 路径 | `.claude/skills/<name>/SKILL.md` |
| 管理方式 | 随项目 git 提交，团队共享 |
| 适用场景 | 项目强相关的定制能力 |

当前包含：

| Skill | 功能 |
|-------|------|
| `cann-toolkit-installer` | 自动下载安装 CANN Toolkit |
| `default-skills` | 默认 Skills 安装入口（触发远程安装脚本） |
| `af-build-runner` | 编译构建辅助（build.sh、CMake 配置、编译错误解析） |
| `af-test-developer` | UT/ST 测试开发辅助（gtest/mockcpp、测试编写与运行） |
| `af-code-reviewer` | 代码审查/贡献规范（CONTRIBUTING 检查、代码风格、DCO） |

### 第二层：远程自动安装 Skills

| 属性 | 说明 |
|------|------|
| 路径 | `.claude/skills/_remote/<name>/SKILL.md`（符号链接到一级目录） |
| 管理方式 | 启动时自动从远程仓库拉取更新，`.gitignore` 忽略 |
| 远程仓库 | `https://gitcode.com/cann-agent/skills.git` |
| 适用场景 | 跨项目共享的工具能力，独立仓库维护 |

当前包含：

| Skill | 功能 |
|-------|------|
| `gitcode-pr` | 创建 PR、获取评论、查看讨论 |
| `gitcode-issue` | 读取 Issue 详情和评论 |
| `gitcode-pipeline` | 触发流水线并监控状态 |
| `api-doc-generator` | 生成 API 接口文档 |

**目录结构：**

```
.claude/skills/
├── af-build-runner/               # 本地 skill（git 管控）
│   ├── SKILL.md
│   └── README.md
├── af-test-developer/             # 本地 skill（git 管控）
│   ├── SKILL.md
│   └── README.md
├── af-code-reviewer/              # 本地 skill（git 管控）
│   ├── SKILL.md
│   └── README.md
├── cann-toolkit-installer/SKILL.md
├── default-skills/
│   ├── SKILL.md
│   └── scripts/install-default-skills.sh
├── _remote/                       # 远程 skills 实际存放目录（git 忽略）
│   ├── gitcode-pr/SKILL.md
│   ├── gitcode-issue/SKILL.md
│   ├── gitcode-pipeline/SKILL.md
│   └── api-doc-generator/SKILL.md
├── gitcode-pr -> _remote/gitcode-pr          # 符号链接（git 忽略）
├── gitcode-issue -> _remote/gitcode-issue    # 符号链接（git 忽略）
├── gitcode-pipeline -> _remote/gitcode-pipeline
└── api-doc-generator -> _remote/api-doc-generator
```

### 第三层：第三方插件 Skills

| 属性 | 说明 |
|------|------|
| 路径 | `~/.cache/opencode/node_modules/superpowers/skills/` |
| 管理方式 | 在 `.opencode/opencode.json` 中配置 git 仓库，opencode 自动安装到缓存目录 |
| 适用场景 | 通用开发方法论，社区维护，按需引入 |

配置方式（`.opencode/opencode.json`）：

```json
{
  "plugin": ["superpowers@git+https://github.com/obra/superpowers.git"]
}
```

当前包含 14 个通用工作流 Skills：

| 类别 | Skills |
|------|--------|
| 流程控制 | `brainstorming`, `writing-plans`, `executing-plans` |
| 开发方法 | `test-driven-development`, `systematic-debugging` |
| 代码审查 | `requesting-code-review`, `receiving-code-review` |
| 并行执行 | `dispatching-parallel-agents`, `subagent-driven-development` |
| Git 工作流 | `using-git-worktrees`, `finishing-a-development-branch` |
| 元技能 | `using-superpowers`, `verification-before-completion`, `writing-skills` |

## 自动发现机制

opencode 启动时自动扫描以下路径，将所有发现的 `SKILL.md` 注册为可用 Skills：

1. **`.claude/skills/*/SKILL.md`** — 兼容 Claude Code 约定（包括符号链接）
2. **`.opencode/skills/*/SKILL.md`** — opencode 原生路径
3. **插件系统** — 通过 `opencode.json` 中配置的 plugin 自动安装和发现

## 远程 Skills 自动更新流程

```
opencode 启动
  │
  ▼
加载 .opencode/plugins/install-default-skills.ts
  │
  ▼
检测 bash 环境（Windows 无 bash 则跳过，提示手动安装）
  │
  ▼
执行 .claude/skills/default-skills/scripts/install-default-skills.sh
  │
  ├── 1. 检查网络连通性（curl/wget 访问 gitcode.com）
  ├── 2. 克隆远程仓库到临时目录（depth=1，超时 20s）
  ├── 3. 拷贝 skills 到 .claude/skills/_remote/
  ├── 4. 创建符号链接到 .claude/skills/ 一级目录
  ├── 5. 更新 .gitignore 忽略符号链接和 _remote 目录
  └── 6. 清理临时目录
  │
  ▼
对比安装前后文件 MD5，如有变更则提示"重启 opencode 才能完全生效"
```

## .gitignore 配置（白名单模式）

采用**默认忽略 + 白名单放行**策略，确保只有经过审核的本地 Skills 纳入 git 管理，防止开发者自建 Skill 被意外提交。

忽略规则放在 `.claude/skills/.gitignore` 中（而非项目根目录），职责内聚，便于维护：

```gitignore
# .claude/skills/.gitignore
# 默认忽略所有内容，仅白名单内的 Skill 纳入 git 管理
# 新增本地 Skill 需在此添加白名单条目: !<skill-name>/
*
!cann-toolkit-installer/
!default-skills/
!af-build-runner/
!af-test-developer/
!af-code-reviewer/
!.gitignore
```

**效果：**

| 内容 | 状态 | 说明 |
|------|------|------|
| 白名单内 5 个本地 Skills | 跟踪 | git 管控，团队共享 |
| 远程 Skills（`_remote/` + 符号链接） | 忽略 | 启动时自动安装，不提交 |
| 开发者自建 Skill | 忽略 | 需显式添加白名单才能提交 |

**新增本地 Skill 时**，需在 `.claude/skills/.gitignore` 中添加一行：

```gitignore
!<new-skill-name>/
```

## 模块化 Skill 包结构

每个本地 Skill 采用模块化包结构，便于维护者和外部贡献者理解：

```
.claude/skills/<skill-name>/
├── SKILL.md          # Skill 定义文件（必须）
├── README.md         # 使用说明，面向贡献者和使用者（推荐）
└── examples/         # 示例交互文档（可选）
    └── *.md
```

### SKILL.md 标准格式

```markdown
---
name: skill-name
description: |
  简短描述功能。
  **必须触发的场景**：列出关键词和触发场景。
---

## 功能说明

描述该 Skill 的核心功能和使用方法。

## 使用步骤

1. 步骤一
2. 步骤二

## 约束

- 约束条件一
- 约束条件二
```

## 新增 Skill 指南

### 新增本地 Skill

1. 在 `.claude/skills/` 下创建模块化 Skill 包：

```bash
mkdir -p .claude/skills/my-skill
```

2. 编写 `SKILL.md`，遵循标准格式（见上方"模块化 Skill 包结构"章节）

3. 编写 `README.md`，说明 Skill 的用途、触发场景和使用示例

4. 在 `.claude/skills/.gitignore` 中添加白名单条目：

```gitignore
!my-skill/
```

5. 提交到 git 即可团队共享

### 新增远程 Skill

1. 在远程仓库 `https://gitcode.com/cann-agent/skills.git` 中添加 skill
2. 在 `install-default-skills.sh` 的 `DEFAULT_SKILLS` 数组中添加 skill 名称
3. 用户下次启动 opencode 时自动安装（远程 Skills 已被 `.claude/skills/.gitignore` 的 `*` 规则默认忽略，无需额外配置）

### 新增插件 Skill

在 `.opencode/opencode.json` 的 `plugin` 数组中添加插件地址：

```json
{
  "plugin": [
    "superpowers@git+https://github.com/obra/superpowers.git",
    "my-plugin@git+https://github.com/user/my-plugin.git"
  ]
}
```

## 开源贡献者指引

### 贡献流程

```
外部贡献者创建 Skill
  │
  ├── 1. Fork graph-autofusion 仓库
  ├── 2. 在 .claude/skills/ 下创建模块化 Skill 包
  ├── 3. 编写 SKILL.md（含 frontmatter 元数据，遵循标准格式）
  ├── 4. 编写 README.md 说明用途和使用方法
  ├── 5. 提交 PR → 触发 CI 检查
  │      ├── SKILL.md 格式校验（frontmatter 完整性）
  │      ├── 无敏感信息/密钥泄露
  │      └── 描述清晰度和触发场景合理性
  └── 6. Review 通过后合入
```

### 贡献规范

| 规则 | 说明 |
|------|------|
| **命名** | 小写字母 + 连字符，项目专属 Skill 使用 `af-` 前缀（如 `af-build-runner`），通用 Skill 不加前缀 |
| **作用域** | 必须与 graph-autofusion 项目相关 |
| **无副作用** | Skill 不应修改用户文件系统（除明确告知的操作） |
| **语言** | SKILL.md 建议中英双语，至少包含中文 |
| **依赖** | 不得引入需要额外安装的第三方工具 |
| **测试** | 建议在 README.md 中包含验证该 Skill 可用性的方法 |

### Skill 升级路径

本地 Skills 如果被广泛使用且适合跨项目共享，维护者可以将其迁移到远程仓库 `https://gitcode.com/cann-agent/skills.git`，升级为远程层管理：

1. 将 Skill 目录移动到远程仓库
2. 在 `install-default-skills.sh` 的 `DEFAULT_SKILLS` 数组中添加该 Skill 名称
3. 在 `.gitignore` 中添加对应的忽略规则
4. 从项目 git 中移除该 Skill 目录
5. 用户下次启动 opencode 时自动从远程安装

## 各层对比

| | 本地 Skills | 远程 Skills | 插件 Skills |
|---|---|---|---|
| **存放位置** | `.claude/skills/` | `.claude/skills/_remote/` | `~/.cache/opencode/` |
| **版本管控** | 项目 git | 远程仓库独立 git | 插件仓库独立 git |
| **更新方式** | 手动 git pull | 启动时自动拉取 | opencode 自动安装 |
| **作用范围** | 仅当前项目 | 仅当前项目 | 全局所有项目 |
| **维护者** | 项目团队 + 社区贡献者 | 工具团队 | 社区/插件作者 |
| **适用场景** | 项目定制能力 | 跨项目共享工具 | 通用开发方法论 |
| **外部贡献** | 接受 PR 贡献 | 需迁移到远程仓库 | 向插件仓库贡献 |
