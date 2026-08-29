# ATT Analyze

仓内 ATT 日志分析工具，入口为：

```bash
python3 autofuse/tools/att_analyze/src/att.py --help
```

支持 `summary`、`compare`、`split-slog`、`perf-formula`、`verify-tiling` 和 `evidence` 六个命令。`summary` 默认只读解析日志；`verify-tiling` 会编译并执行代码，使用前请确认输入目录和授权。

`LogParser` 的 `OperatorSummary.parse_status` 用于标明证据是否完整：`ok` 表示日志包含完整选择信息，`inferred_graph_result` 表示仅从模板行推断 graph/result，`missing_group_case`、`missing_result_performance` 和 `missing_graph_result` 表示相应日志缺失。CSV 列保持历史含义不变。

```bash
python3 autofuse/tools/att_analyze/src/att.py summary path/to/att.log -f csv -o /tmp/summary.csv
python3 autofuse/tools/att_analyze/src/att.py compare baseline.csv candidate.csv
python3 autofuse/tools/att_analyze/src/att.py evidence path/to/att.log -o /tmp/evidence
python3 autofuse/tools/att_analyze/src/att.py verify-tiling generated/ --scene tf --preset B --aiv-num 56
```

工具不会把缺失值当作有效的 0；请在后续分析中根据 `parse_status` 决定是否需要补充日志。

## 与 ATT 模板/tiling 分析 Skill 配合

`att_analyze` 由本仓维护，Skill 调用的固定入口是
`autofuse/tools/att_analyze/src/att.py`，不需要安装或访问其他仓库。对已经采集的
数据做离线分析时，在仓库根目录执行：

```bash
python3 .claude/skills/att-template-tiling-analysis/scripts/att_analysis.py \
  analyze --run-root <run-root> --output <report-dir>
```

`run-root/default` 和 `run-root/pgo`（也支持 `base`）放入用户已执行得到的日志；
目录名称不固定时可显式传 `--default-root` 和 `--candidate-root`，脚本也会递归发现
`logs/`、`profiling/`、`kernel_meta/` 和 `dump/`。脚本只读取这些数据，不会自行
选择 case、编造执行命令或重新运行任务。Python 3.9+ 可运行基础分析，安装
`openpyxl` 后会额外生成 `summary.xlsx`。

如果需要现场执行，用户需先提供完整 case 范围和命令。本地执行需要可用的
`python3`；远端执行使用标准 `ssh`，并要求远端 checkout 能访问相同的工具脚本。
`devssh` 只能作为用户明确提供的 wrapper。编译、profiling、PGO 和
`verify-tiling` 均需用户单独确认。

原始证据与分析结论分开保存，建议布局如下：

```text
run-root/                         # 原始运行目录
  default/  pgo/
    att.log  profile/  kernel_meta/  dump/
evidence-archive/<run-name>/      # 原始文件归档
report-archive/<run-name>/        # report.md、summary.csv、root-cause.jsonl 等
```

归档脚本会为同名运行自动创建递增目录并写入 `archive-manifest.json`，不会覆盖
已有归档。详细交互契约和归档规则见 Skill 的
`references/execution-contract.md` 与 `references/archive-layout.md`。

## preset 和真实日志维护

`preset_B.json` 是 TensorFlow 动态 ABI 的示例输入，默认 `aiv_num=56`、
`ub_size=262144`，不代表所有芯片的硬件规格。执行 `verify-tiling` 时会打印
实际传入的 `aiv_num`、参数来源和动态维度；请根据目标设备核对，必要时使用
`--aiv-num` 或 `--input-json` 修改。`aiv_num` 是传给 TensorFlow tiling 的配置值，
Inductor ABI 不使用该字段。

`tests/data/` 中的日志是固定回归样例，不会自动同步现场日志。遇到新的 CANN、
TensorFlow 或 Inductor 日志格式时，请对真实日志脱敏后新增样例，并同步增加
`summary`/`evidence` 的期望结果；保留旧样例以防止已有格式回归。
