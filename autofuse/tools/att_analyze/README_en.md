# ATT Analyze

In-repository ATT log analysis utilities. Run:

```bash
python3 autofuse/tools/att_analyze/src/att.py --help
```

The CLI provides `summary`, `compare`, `split-slog`, `perf-formula`, `verify-tiling`, and `evidence`. `summary` is read-only. `verify-tiling` compiles and executes code, so review its inputs and authorization first.

`LogParser` exposes `OperatorSummary.parse_status` to make incomplete evidence explicit. `ok` means complete selection records; `inferred_graph_result` means graph/result came from template lines; `missing_group_case`, `missing_result_performance`, and `missing_graph_result` identify missing records. Existing CSV column meanings are unchanged.

## verify-tiling ABI input

Custom input JSON must include an explicit ABI contract. `tf_static` uses no shape dimensions; `tf_dynamic` and `inductor` require one or more shape dimensions. `block_dim_width` is either 32 or 64.

```json
{"dynamic_dims": [], "aiv_num": 48, "ub_size": 196608,
 "abi": {"kind": "tf_static", "shape_dims": 0, "block_dim_width": 32}}
```

```json
{"dynamic_dims": [1024, 512], "aiv_num": 56, "ub_size": 262144,
 "abi": {"kind": "tf_dynamic", "shape_dims": 2, "block_dim_width": 32}}
```

Unknown or missing ABI contracts are rejected before native code is called.

## Using the ATT template/tiling analysis skill

`att_analyze` is maintained in this repository. The skill always invokes
`autofuse/tools/att_analyze/src/att.py`; no checkout of another repository is
required. To analyze an existing run offline from the repository root:

```bash
python3 .claude/skills/att-template-tiling-analysis/scripts/att_analysis.py \
  analyze --run-root <run-root> --output <report-dir>
```

Place user-collected logs in `run-root/default` and `run-root/pgo` (or `base`),
or pass arbitrary variant directories with `--default-root` and `--candidate-root`.
The command recursively discovers logs, profiling, `kernel_meta/`, and `dump/`;
it does not require a producer-specific directory layout. The command
is read-only: it does not invent case scope or commands and does not rerun a
workload. Python 3.9+ is sufficient for the base analysis; installing
`openpyxl` additionally enables `summary.xlsx`.

For live execution, provide the exact cases and command first. Local execution
requires `python3`; remote execution uses standard `ssh` and a checkout on the
remote host. A site-specific `devssh` wrapper is accepted only when explicitly
provided by the user. Build, profiling, PGO, and `verify-tiling` actions require
separate confirmation.

Keep raw evidence and conclusions in separate archives:

```text
run-root/                         # raw run data
  default/  pgo/
    att.log  profile/  kernel_meta/  dump/
evidence-archive/<run-name>/      # raw files
report-archive/<run-name>/        # report.md, summary.csv, root-cause.jsonl, ...
```

The archive helper creates an incremented directory for duplicate run names and
writes `archive-manifest.json`; existing archives are never overwritten. See
the skill references for the execution contract and archive rules.

## Presets and real-log maintenance

`preset_B.json` is an example TensorFlow dynamic-ABI input. Its default
`aiv_num=56` and `ub_size=262144` are not guaranteed hardware specifications
for every chip. `verify-tiling` prints the effective `aiv_num`, its source, and
dynamic dimensions before compilation. Check the value against the target
device and override it with `--aiv-num` or `--input-json` when needed.
`aiv_num` is passed to the TensorFlow tiling ABI; the Inductor ABI does not use
this field.

Logs under `tests/data/` are fixed regression fixtures and are not synchronized
with live runs. When a CANN, TensorFlow, or Inductor log format changes, add a
sanitized fixture from a real run and update the expected `summary`/`evidence`
results while keeping older fixtures for compatibility coverage.
