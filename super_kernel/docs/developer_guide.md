# Developer Guide

本文档面向 SuperKernel 项目的开发人员，说明如何编写与维护 System Test（ST）用例及相关资源。

## 文档定位

- `README.md`：面向普通使用者，聚焦环境准备、打包与基础验证。
- `docs/developer_guide.md`（本文档）：面向贡献者，覆盖测试框架、依赖管理与调试技巧等内容。

## 目录结构概览

```
super_kernel/
  src/
    superkernel/    # 核心业务代码
  tests/
    fixtures/     # 运行环境配置、子内核编译等夹具
    ut/             # 单元测试
    st/
      data/         # 黄金数据样本
      scenarios/    # 端到端测试脚本
    utils/          # 通用校验与工具函数
    generated/      # 测试运行时产生的临时产物（不纳入版本库）
  docs/             # 项目文档
  examples/         # 示例或演示脚本
  scripts/          # 辅助脚本（如打包、调试工具）
  coverage/         # 覆盖率输出目录（测试后生成）
```

## 固定夹具

- `tmp_dir`（`fixtures/config.py`）：为每次测试会话提供独立的 `tests/generated/<timestamp>_<pid>` 目录，并在测试结束后清理。
- `data_dir`（`fixtures/config.py`）：定位至 `tests/st/data/`，便于场景读取黄金文件或配置。
- `soc_version`（`fixtures/config.py`）：指定默认 SoC 信息，当前默认值为 `Ascend910_9391`；如需使用其他版本，可在场景内覆盖或增加新的 fixture。
- `subkernel` 编译夹具（位于 `fixtures/sub_kernel.py`）：包含 `subkernel_is_inf_default`、`subkernel_is_inf_split_mode1`、`subkernel_is_finite_default` 等。测试用例可借助 `subkernel_inf` / `subkernel_finite` 这类间接 fixture 组合不同配置。

开发者在新增夹具时，请保持作用域、资源释放策略与现有实现一致。

## ST 场景编写约定

1. 测试脚本存放于 `st/scenarios/`，文件需以 `test_*.py` 命名，并在函数上添加 `@pytest.mark.st` 标记。
2. 所有场景应通过依赖注入使用现有夹具（例如 `tmp_dir`、`data_dir`），避免手动创建临时目录。
3. 生成的 codegen、日志等产物默认位于 `tmp_dir/<scenario>/kernel_meta/` 下，命名约定：
   - C++ 源文件：`<kernel_name>_<pid>_kernel.cpp`
   - 编译日志：`<kernel_name>_<pid>.log`

## 黄金数据与校验

- 黄金文件（如 C++ 代码、JSON）存放在 `tests/st/data/<scenario>/` 下，由用例显式引用。
- `st/utils/validators.py` 提供通用校验函数：
  - `validate_codegen_output(kernel_root, kernel_name, expected_source)`：比对生成的 C++ 与黄金文件是否一致。
  - `validate_compile_options(kernel_root, kernel_name, expected_options)`：在对应日志中定位 Bisheng 编译命令，确认包含指定选项。
- 新增校验逻辑时，建议封装成独立函数，便于多个场景复用。

## 依赖管理流程

- `pyproject.toml` 是依赖信息的单一来源：`[project.dependencies]` 用于运行时依赖，`[project.optional-dependencies.dev]` 列出开发/测试工具。
- `requirements-dev.txt` 为锁定结果，由 `pip-compile` 根据 `pyproject.toml` 生成，供需要完全可复现环境的场景（例如 CI）使用。更新方法：
  ```bash
  cd super_kernel
  pip install pip-tools         # 安装 pip-tools 工具
  pip-compile --extra dev pyproject.toml --output-file requirements-dev.txt
  ```
  运行后同名文件会被覆盖，请将最新结果提交版本库。

## 运行与调试

### 基础流程

1. 按照项目根目录的《README》完成 Ascend 工具链与 Python 环境初始化。
2. 在虚拟环境中以可编辑模式安装项目，并附带 `dev` extra 以获取测试工具：
   ```bash
   cd super_kernel
   pip install -e .[dev]
   ```
   在同一终端会话中继续执行后续命令即可。
3. 运行单元测试（UT）：
   ```bash
   pytest tests/ut -m ut
   ```
4. 运行 System Test 套件：
   ```bash
   pytest tests/st -m st
   ```
5. 运行完成后，如需保留 `tests/generated/` 中的产物，请在执行 pytest 时添加 `--keep-generated` 参数。

### 启用覆盖率

支持对 UT、ST 或组合测试进行覆盖率统计。命令示例：

- **仅 UT 覆盖率**：
  ```bash
  pytest tests/ut -m ut \
    --cov=src/superkernel \
    --cov-report=term-missing \
    --cov-report=html \
    --cov-report=xml
  ```
- **仅 ST 覆盖率**：
  ```bash
  pytest tests/st -m st \
    --cov=src/superkernel \
    --cov-report=term-missing \
    --cov-report=html \
    --cov-report=xml
  ```
- **UT + ST 综合覆盖率**：
  ```bash
  pytest \
    --cov=src/superkernel \
    --cov-report=term-missing \
    --cov-report=html \
    --cov-report=xml
  ```

运行后会生成以下覆盖率文件：
- **终端报告**：命令行打印覆盖率统计以及缺失覆盖的行号。
- **HTML 报告**：`super_kernel/coverage/html/index.html`。
- **XML 报告**：`super_kernel/coverage/coverage.xml`。
- **原始数据文件**：`super_kernel/coverage/.coverage`。

覆盖率相关配置集中在 `pyproject.toml` 的 `[tool.coverage.*]` 章节。

## 文件管理原则

### 版本控制忽略策略

项目采用**根目录**统一管理 `.gitignore` 的策略，**子目录**无例外不单独设置 `.gitignore`：避免规则分散，便于统一维护

### 临时文件管理

- 测试产生的临时文件存放在 `tests/generated/` 目录
- 覆盖率报告文件存放在项目根目录的`coverage`目录

## 贡献建议
- 阅读项目根目录的[《CONTRIBUTING.md》](../../CONTRIBUTING.md)，该文档包含了项目的贡献指南，同时在该贡献指南内部可以跳转到参与CANN所有开源项目的基础准备动作，包括但不限于代码下载、提交、流水线触发、代码检视等流程，确保在参与贡献前已经完成了所有必要的准备工作。
- 提交前请运行 `pytest` 确认所有测试用例通过。
- 建议在提交前运行覆盖率测试，确保新增代码有适当的测试覆盖。
- 新增夹具或工具时补充必要的文档与注释，保持目录结构清晰。
- 黄金文件更新需谨慎，建议先在评审中说明差异来源。

如有更复杂的调试需求，可在本指南基础上继续扩展章节或补充 FAQ。
