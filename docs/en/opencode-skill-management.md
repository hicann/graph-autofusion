# opencode Skill Management

## Overview

The graph-autofusion project adopts a three-layer Skill architecture (aligned with the GE project). Combined with local customization, remote auto-update, and third-party plugins, it provides rich domain capabilities for AI-assisted development. The project extends this with 3 project-specific Skills and provides guidelines for external open-source contributors.

## Architecture Overview

```
opencode startup
  │
  ├── Scans .claude/skills/*/SKILL.md ─────────────┐
  ├── Scans .opencode/skills/*/SKILL.md ───────────┤── Registers as available Skills
  ├── Loads plugin system (superpowers, etc.) ─────┤
  └── Executes plugins/install-default-skills.ts    │
        └── Calls install-default-skills.sh        │
              └── Pulls from remote repo → _remote/ ┘
                    └── Creates symlinks to first-level directory
```

## Three Layers of Skill Sources

### Layer 1: Local Custom Skills

| Attribute | Description |
|-----------|-------------|
| Path | `.claude/skills/<name>/SKILL.md` |
| Management | Committed with the project git, shared by the team |
| Use case | Project-specific custom capabilities |

Currently includes:

| Skill | Function |
|-------|----------|
| `cann-toolkit-installer` | Automatically downloads and installs CANN Toolkit |
| `default-skills` | Default Skills installation entry (triggers the remote install script) |
| `af-build-runner` | Build assistance (build.sh, CMake configuration, compile error analysis) |
| `af-test-developer` | UT/ST test development assistance (gtest/mockcpp, test writing and running) |
| `af-code-reviewer` | Code review/contribution guidelines (CONTRIBUTING checks, code style, DCO) |

### Layer 2: Remote Auto-installed Skills

| Attribute | Description |
|-----------|-------------|
| Path | `.claude/skills/_remote/<name>/SKILL.md` (symlinked to first-level directory) |
| Management | Automatically pulled from the remote repository at startup, ignored by `.gitignore` |
| Remote repo | `https://gitcode.com/cann-agent/skills.git` |
| Use case | Cross-project shared tool capabilities, maintained in an independent repository |

Currently includes:

| Skill | Function |
|-------|----------|
| `gitcode-pr` | Create PRs, fetch comments, view discussions |
| `gitcode-issue` | Read Issue details and comments |
| `gitcode-pipeline` | Trigger pipelines and monitor status |
| `api-doc-generator` | Generate API documentation |

**Directory structure:**

```
.claude/skills/
├── af-build-runner/               # Local skill (git managed)
│   ├── SKILL.md
│   └── README.md
├── af-test-developer/             # Local skill (git managed)
│   ├── SKILL.md
│   └── README.md
├── af-code-reviewer/              # Local skill (git managed)
│   ├── SKILL.md
│   └── README.md
├── cann-toolkit-installer/SKILL.md
├── default-skills/
│   ├── SKILL.md
│   └── scripts/install-default-skills.sh
├── _remote/                       # Remote skills storage (git ignored)
│   ├── gitcode-pr/SKILL.md
│   ├── gitcode-issue/SKILL.md
│   ├── gitcode-pipeline/SKILL.md
│   └── api-doc-generator/SKILL.md
├── gitcode-pr -> _remote/gitcode-pr          # Symlink (git ignored)
├── gitcode-issue -> _remote/gitcode-issue    # Symlink (git ignored)
├── gitcode-pipeline -> _remote/gitcode-pipeline
└── api-doc-generator -> _remote/api-doc-generator
```

### Layer 3: Third-party Plugin Skills

| Attribute | Description |
|-----------|-------------|
| Path | `~/.cache/opencode/node_modules/superpowers/skills/` |
| Management | Configure the git repository in `.opencode/opencode.json`; opencode auto-installs to the cache directory |
| Use case | General development methodology, community-maintained, introduced on demand |

Configuration (`.opencode/opencode.json`):

```json
{
  "plugin": ["superpowers@git+https://github.com/obra/superpowers.git"]
}
```

Currently includes 14 general workflow Skills:

| Category | Skills |
|----------|--------|
| Flow control | `brainstorming`, `writing-plans`, `executing-plans` |
| Development methodology | `test-driven-development`, `systematic-debugging` |
| Code review | `requesting-code-review`, `receiving-code-review` |
| Parallel execution | `dispatching-parallel-agents`, `subagent-driven-development` |
| Git workflow | `using-git-worktrees`, `finishing-a-development-branch` |
| Meta-skills | `using-superpowers`, `verification-before-completion`, `writing-skills` |

## Auto-discovery Mechanism

On startup, opencode automatically scans the following paths and registers all discovered `SKILL.md` files as available Skills:

1. **`.claude/skills/*/SKILL.md`** — Compatible with Claude Code conventions (including symlinks)
2. **`.opencode/skills/*/SKILL.md`** — opencode native path
3. **Plugin system** — Automatically installed and discovered through the `plugin` configured in `opencode.json`

## Remote Skills Auto-update Flow

```
opencode startup
  │
  ▼
Load .opencode/plugins/install-default-skills.ts
  │
  ▼
Detect bash environment (skipped on Windows without bash, prompts manual installation)
  │
  ▼
Execute .claude/skills/default-skills/scripts/install-default-skills.sh
  │
  ├── 1. Check network connectivity (curl/wget to gitcode.com)
  ├── 2. Clone remote repo to temp directory (depth=1, timeout 20s)
  ├── 3. Copy skills to .claude/skills/_remote/
  ├── 4. Create symlinks to the .claude/skills/ first-level directory
  ├── 5. Update .gitignore to ignore symlinks and the _remote directory
  └── 6. Clean up temp directory
  │
  ▼
Compare file MD5 before and after installation; if changed, prompt "restart opencode for the changes to take full effect"
```

## .gitignore Configuration (Whitelist Mode)

An **ignore-by-default + whitelist** policy is used to ensure only reviewed local Skills are tracked by git, preventing developers from accidentally committing self-created Skills.

Ignore rules are placed in `.claude/skills/.gitignore` (rather than the project root) to keep responsibilities cohesive and easier to maintain:

```gitignore
# .claude/skills/.gitignore
# Ignore everything by default; only whitelisted Skills are tracked by git
# To add a new local Skill, add a whitelist entry here: !<skill-name>/
*
!cann-toolkit-installer/
!default-skills/
!af-build-runner/
!af-test-developer/
!af-code-reviewer/
!.gitignore
```

**Effect:**

| Content | Status | Description |
|---------|--------|-------------|
| 5 whitelisted local Skills | Tracked | Git managed, shared by the team |
| Remote Skills (`_remote/` + symlinks) | Ignored | Auto-installed at startup, not committed |
| Developer self-created Skills | Ignored | Must be explicitly whitelisted to commit |

**When adding a new local Skill**, add a line to `.claude/skills/.gitignore`:

```gitignore
!<new-skill-name>/
```

## Modular Skill Package Structure

Each local Skill uses a modular package structure for maintainers and external contributors to understand:

```
.claude/skills/<skill-name>/
├── SKILL.md          # Skill definition file (required)
├── README.md         # Usage instructions for contributors and users (recommended)
└── examples/         # Example interaction documents (optional)
    └── *.md
```

### SKILL.md Standard Format

```markdown
---
name: skill-name
description: |
  Short description of the function.
  **Scenarios that must trigger this**: list keywords and trigger scenarios.
---

## Function Description

Describe the core function and usage of the Skill.

## Usage Steps

1. Step one
2. Step two

## Constraints

- Constraint one
- Constraint two
```

## Guide to Adding New Skills

### Adding a Local Skill

1. Create a modular Skill package under `.claude/skills/`:

```bash
mkdir -p .claude/skills/my-skill
```

2. Write `SKILL.md` following the standard format (see the "Modular Skill Package Structure" section above)

3. Write `README.md` describing the Skill's purpose, trigger scenarios, and usage examples

4. Add a whitelist entry to `.claude/skills/.gitignore`:

```gitignore
!my-skill/
```

5. Commit to git to share with the team

### Adding a Remote Skill

1. Add the skill to the remote repository `https://gitcode.com/cann-agent/skills.git`
2. Add the skill name to the `DEFAULT_SKILLS` array in `install-default-skills.sh`
3. Users get it auto-installed on the next opencode startup (remote Skills are ignored by default via the `*` rule in `.claude/skills/.gitignore`, no extra configuration needed)

### Adding a Plugin Skill

Add the plugin URL to the `plugin` array in `.opencode/opencode.json`:

```json
{
  "plugin": [
    "superpowers@git+https://github.com/obra/superpowers.git",
    "my-plugin@git+https://github.com/user/my-plugin.git"
  ]
}
```

## Open-source Contributor Guidelines

### Contribution Flow

```
External contributor creates Skill
  │
  ├── 1. Fork the graph-autofusion repository
  ├── 2. Create a modular Skill package under .claude/skills/
  ├── 3. Write SKILL.md (with frontmatter metadata, following the standard format)
  ├── 4. Write README.md describing the purpose and usage
  ├── 5. Submit a PR → triggers CI checks
  │      ├── SKILL.md format validation (frontmatter completeness)
  │      ├── No sensitive information/key leakage
  │      └── Description clarity and trigger scenario reasonableness
  └── 6. Merged after review passes
```

### Contribution Rules

| Rule | Description |
|------|-------------|
| **Naming** | Lowercase letters + hyphens; project-specific Skills use the `af-` prefix (e.g., `af-build-runner`); general Skills use no prefix |
| **Scope** | Must be related to the graph-autofusion project |
| **No side effects** | Skills should not modify the user's file system (except for explicitly disclosed operations) |
| **Language** | SKILL.md is recommended to be bilingual (Chinese and English), at least including Chinese |
| **Dependencies** | Must not introduce third-party tools requiring additional installation |
| **Testing** | Recommended to include methods in README.md for validating the Skill's usability |

### Skill Upgrade Path

If a local Skill is widely used and suitable for cross-project sharing, maintainers can migrate it to the remote repository `https://gitcode.com/cann-agent/skills.git` and upgrade it to remote-layer management:

1. Move the Skill directory to the remote repository
2. Add the Skill name to the `DEFAULT_SKILLS` array in `install-default-skills.sh`
3. Add the corresponding ignore rule to `.gitignore`
4. Remove the Skill directory from the project git
5. Users get it auto-installed from remote on the next opencode startup

## Layer Comparison

| | Local Skills | Remote Skills | Plugin Skills |
|---|---|---|---|
| **Location** | `.claude/skills/` | `.claude/skills/_remote/` | `~/.cache/opencode/` |
| **Version control** | Project git | Independent remote repo git | Independent plugin repo git |
| **Update method** | Manual git pull | Auto-pull at startup | Auto-installed by opencode |
| **Scope** | Current project only | Current project only | Global across all projects |
| **Maintainer** | Project team + community contributors | Tool team | Community/plugin authors |
| **Use case** | Project-specific capabilities | Cross-project shared tools | General development methodology |
| **External contribution** | Accepts PR contributions | Requires migration to remote repo | Contribute to the plugin repo |
