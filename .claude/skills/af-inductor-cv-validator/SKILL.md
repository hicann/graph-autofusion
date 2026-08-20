---
name: af-inductor-cv-validator
description: |
  Use when validating Autofuse changes that may affect TorchInductor CV fusion, UBFuse, static/dynamic shapes, MM/BMM epilogue fusion, bool dtype, TopN tiling, CANN or overlay path portability, AscendC codegen, kernel compile/load/launch, or numeric correctness.
---

# Autofuse Inductor CV 功能验证

Autofuse 相关代码修改后，Inductor CV fusion 不能只用构建通过、单个 case 通过或手工 workaround 证明正确。验证必须覆盖指定矩阵，并为 codegen、AscendC 编译、kernel load、runtime launch 和数值校验留下证据。验证流程必须可迁移到不同仓库路径、CANN 安装路径和 overlay 方式，不能依赖个人目录或历史 cache。

**硬门禁**：没有完成验证矩阵和证据清单，不得声称 Inductor CV fusion 已验证通过。

## 必须触发的场景

用户提到以下任一内容时必须使用本 skill：

- Autofuse 代码修改后验证 Inductor CV fusion 功能
- TorchInductor、Inductor NPU、CV fusion、CV UBFuse、MatMul epilogue fusion
- `fusion_mode`、`mix_mode`、UB 模板、Common/fallback 模板
- AscendC codegen、`asc_kernel.py`、kernel 编译、`kernel.so` load、runtime launch
- 静态 shape、动态 shape 的 Inductor CV 回归验证
- 验证 fragment08、BMM K=1、MM sweep、CV template coverage 等 Inductor CV case

## 不适用场景

- 纯 SuperKernel 验证，不涉及 Autofuse/Inductor CV
- 普通 C++ 单元测试开发，不涉及 Inductor CV E2E
- 只解释代码或只做静态 code review，用户没有要求验证功能
- 文档修改且不会影响构建、codegen、tiling 或运行路径

## 环境门禁

验证前必须先确定唯一 CANN Toolkit 环境：

1. 首先从当前 shell 环境读取 toolkit 路径，优先检查 `ASCEND_HOME_PATH`、`ASCEND_AICPU_PATH`、`TOOLCHAIN_HOME`、`PATH`、`LD_LIBRARY_PATH` 中指向的 CANN Toolkit，并确认对应目录存在 `set_env.sh`、`version.info` 或关键 `lib64`/Python 目录。
2. 如果环境变量没有给出可用 toolkit，再探测 `/usr/local/Ascend`、`/opt/Ascend` 下的默认 toolkit，例如 `ascend-toolkit/latest` 或可用 `set_env.sh`。
3. 如果默认路径不可用，检查当前对话上下文、脚本或日志中已经使用过的 CANN 路径；这些路径只能作为候选，不能自动继承为本次正式验证路径。
4. 如果发现多个候选 CANN，列出候选路径、当前环境命中的路径、脚本中写死的路径，询问用户选择，不得自行混用。
5. 如果仍找不到 CANN，向用户索要 toolkit 路径。
6. `ASCEND_HOME_PATH`、`PATH`、`LD_LIBRARY_PATH`、Python site-packages 必须来自同一套 CANN；任何变量命中旧路径都必须先修正或停止验证。
7. 禁止在通用脚本中写死 `/workspace/xuyf`、个人 home、固定 CANN 版本或固定 overlay 路径；必须通过参数、环境变量或自动发现注入。
8. 自动发现得到的 repo、CANN、overlay、run dir 必须记录 `realpath`、来源、存在性校验结果和优先级；多个有效候选必须 fail closed 并询问用户。
9. 必须记录关键二进制、Python 包和动态库实际命中路径，至少覆盖 CANN compiler、`lib64`、Python site-packages、`pyautofuse.so`、`libaihac_codegen.so`。
10. 正式验证环境必须使用 allowlist/env manifest 重建；`PATH`、`LD_LIBRARY_PATH`、`PYTHONPATH` 中任何非本次 repo、选定 CANN 或本次 overlay 的旧路径都必须解释或移除。

禁止把以下方式当作正式验证：

- 使用已知错误或版本不明的 CANN
- 使用 `LD_PRELOAD` 绕过符号、ABI、runtime 或算子问题
- 手工修改 generated wrapper、`asc_kernel.py`、cache 内生成代码来让 case 通过
- 替换与本次构建产物不一致的 `.so`
- 清理或覆盖失败现场的 debug/cache/dump/log 后再下结论

## 运行目录和产物

默认 run 目录为仓库根目录：

```text
temp/inductor_cv_validation/<timestamp>/
```

首次使用前必须确认 `.gitignore` 包含 `/temp/`；没有则先补充。验证脚本、日志、cache、dump、生成代码、summary 放在同一个 run 目录下，便于复查。失败产物必须保留。

路径规则：

- 仓库根目录必须通过当前工作目录、`git rev-parse --show-toplevel` 或脚本参数确定，不能写死。
- `RUN_DIR`、case 目录、cache、debug、dump 都必须落在本次 run 目录下。
- 历史 `temp/inductor_cv_validation/*` 只能作为 case 设计和失败模式参考，不能复用为当前结果。
- 在新环境验证本 skill 时，应忽略或移走历史 run 目录并新建 run 目录；只有用户明确要求清理时才删除历史失败现场。
- 每次正式验证都必须使用新 shell 或等价 clean 环境，清空 `LD_PRELOAD`、`ENABLE_TILING_SHIM` 和旧 cache 相关变量；这不是只用于新环境演练的可选项。
- 清空旧 cache 变量后必须显式设置本次 per-case cache/debug/dump 目录，不能让 TorchInductor、NPU extension、Triton 或 Autofuse 回落到默认全局 cache。
- 同一任务内所有 attempt 都必须在最终报告列出；失败后重跑成功，也要保留并说明首次失败 run dir 和恢复方式。

## 构建和部署门禁

验证前必须确认本次源码修改已经进入实际运行环境。

构建规则：

```bash
source <CANN_TOOLKIT>/set_env.sh
unset ENABLE_TILING_SHIM
cmake --build build --target aihac_codegen pyautofuse -j 8
```

如果修改涉及测试目标或 C++ UT，还需要按影响范围增量编译，例如：

```bash
cmake --build build --target test_codegen -j 8
```

部署规则：

- 如果 Inductor 运行环境直接加载 build 目录产物，必须记录 `LD_LIBRARY_PATH`、`PYTHONPATH` 或 Python import 实际命中的 `.so` 路径。
- 如果需要安装到 CANN/overlay，必须记录每个源文件和目标文件，例如 `build/autofuse/libaihac_codegen.so -> <CANN>/lib64/libaihac_codegen.so`。
- 安装脚本需要 `RUN_DIR` 时，必须显式设置到本次 `temp/inductor_cv_validation/<timestamp>/`。
- 不得混用旧 `pyautofuse.so`、旧 `libaihac_codegen.so`、旧 wrapper 或旧 cache。
- 部署后至少确认 Python 侧导入路径和动态库路径来自本次构建或本次 overlay。
- 如果仓库 dirty，必须记录 `git status --short`；只修改与本次验证相关的脚本或源码，不得回滚用户已有改动。
- 必须记录 import 命中位置和动态库解析证据；至少包含 Python 模块路径、`LD_LIBRARY_PATH`、`PYTHONPATH`，必要时补充 `ldd`、`readelf -Ws`、`nm -D` 或 loader error。
- 必须记录 git SHA、dirty 状态、构建时间、关键产物 hash；运行时加载的 `.so` hash 必须能追溯到本次构建或本次 overlay。
- dirty 时必须记录变更文件列表和 diff hash；正式报告要说明产物对应的是 clean SHA 还是 dirty worktree。
- 对关键 `.so` 必须记录实际依赖链：直接依赖路径、RPATH/RUNPATH、关键依赖 hash、loader 解析结果。只证明顶层 `.so` 是新的不够。

构建或部署失败时，不得继续跑验证矩阵；必须先定位构建/部署失败原因。

## 强制验证矩阵

每次触发本 skill 后必须覆盖 MM、BMM、UB 模板、Common/fallback 模板、静态 shape、动态 shape、mixed dtype/bool 输出、extended CV/VV 边界，以及每个可 CV case 的 `vv_only_*` A/B 对照。不能只跑 fragment08、12 项压缩矩阵或单个 smoke case。

| 类别 | 必跑 case 类型 | 代表 shape / case | 必查 mode / 证据 |
|------|----------------|-------------------|------------------|
| VV 对照 | 非 CV 的 Inductor baseline | `vv_static_add_mul_mul`, `vv_dynamic_add_mul_mul`, shape `[32,64]` | 确认验证脚本和环境本身可运行，不能误判为 CV |
| 非融合对照 | 不应触发 CV fusion | `non_fused_static_mm_only`, shape `[16,18,4]` | 确认 non-fused 不被误识别为 CV |
| MM 小 shape | MM + elemwise，小 shape CV | `cv_static_mm_small_ub`, `cv_dynamic_mm_small_ub`, shape `[16,18,4]` | 当前环境预期 `fusion_mode=0`, `ub_mode=0`，static/dynamic 各跑一次 |
| MM 大 shape fallback | MM + elemwise，非 UB / fallback | `cv_static_mm_512_non_ub`, `cv_dynamic_mm_512_non_ub`, shape `[512,512,512]` | 预期 `ub_mode=0` 或 Common/fallback，static/dynamic 各跑一次 |
| MM fixpipe/epilogue | MM ReLU/fixpipe 类路径 | `cv_static_mm_relu_fixpipe`, shape `[128,128,128]` | 检查 fusion、tiling key、生成代码、数值 |
| BMM 常规 CV | BMM dynamic key1 | `cv_dynamic_bmm_key1`, shape `[4,16,64,16]` | 预期 `fusion_mode=0`, `ub_mode=0` |
| BMM K=1 mix | BMM K=1 + elemwise mix mode | `cv_static_bmm_to_mul_k1_mix2`, `cv_dynamic_bmm_to_mul_k1_mix2`, shape `[128,8,1,16]` | 当前环境预期 `fusion_mode=1`, `mix_mode=1`，static/dynamic 各跑一次 |
| MM/BMM template coverage | 历史 template/tiling key 覆盖 | `cv_static_mm_fallback_stream_k`, `cv_dynamic_bmm_high_level_iter_batch`, BMM trans/k0/k1/iter batch 等 | 记录 `expected_template`、`expected_tiling_key`，覆盖 UB、stream_k、k_equal_zero、batch_matmul_to_mul、iter/high-level 类路径 |
| mixed dtype/bool | MM epilogue mixed dtype 和 compare/where bool | `cv_static_mixed_fp32_mm_cast_fp16_mul_add`, `cv_static_mm_compare_fp32_bf16_brc` 等 | fp32->fp16 cast、fp16 baseline、fp16/bf16 RHS、bool 输出按 equality 指标校验 |
| extended CV/VV 边界 | 大 shape、尾块、broadcast、N=1/31/33/1025 等 | `cv_dynamic_vv_large_cast_chain`, `cv_static_mm_add_row_large_n1001`, `cv_boundary_n1_m1025`, `cv_bmm_k1_dynamic_boundary` | 覆盖 VV 大/动态边界、MM bias row/col/full、compare/where/cast/exp、BMM K=1 边界 |
| VV-only A/B 对照 | 同 shape/op 关闭 MatMul CV lowering | 自动生成 `vv_only_<cv_case>`，例如 `vv_only_cv_static_mm_512_non_ub` | `TORCHINDUCTOR_NPU_EXT_DEBUG` 不含 `matmul`；数值必须通过；不得出现 CV wrapper artifact 或 `fusion_mode/ub_mode/mix_mode` |
| fragment08 回归 | 真实静态 CV addmm+exp 回归 | `fragment08_mm_add_exp`: `x=[16384,26]`, `w=[26,256]`, `bias=[1,256]` | 2D `addmm+exp` 避免 reshape/view fallback；预期 CV fused 非 UB，`fusion_mode=0`, `ub_mode=0`；检查生成代码、编译、load、launch、数值 |

规则：

- 优先复用现有 `three_pr_cv_validation.py`、fragment08 或等价脚本；历史日志只能用于选 shape，不能作为当前验证结果。
- 如果用户只要求 smoke，需要明确写成“冒烟验证”，不得扩展为“Inductor CV fusion 验证通过”；正式结论必须跑完整矩阵。
- `scripts/run_inductor_cv_matrix.py` 的矩阵必须由 core、template_coverage、mixed_dtype、extended 和自动 `vv_only_*` families 组成；如果 dry-run 只显示约 12 项，说明 skill 回退到旧压缩矩阵，不能作为正式验证入口。
- 正式验证中 `--case-filter` 只能用于调试 subset；最终报告必须显示全矩阵，任何 skip、未跑、未触发、mode 未命中或证据缺失都使总体验证结论为“未完成验证”。
- 缺少任一矩阵项脚本时，必须在 `temp/inductor_cv_validation/<timestamp>/script/` 自动新增脚本。
- 每个 case 必须输出结构化结果行，例如 `CASE_RESULT {...}`，至少包含 `name`、`kind`、`dynamic`、`dtype`、`shape`、`stride`、`seed`、`fusion_mode`、`ub_mode`、`mix_mode`、`max_abs_diff`、`mean_abs_diff`、`max_rel_diff`、`fail_count`、`ok`、`stage`、`case_dir`、`command`、`exit_code`、`start_time`、`end_time`、`artifact_paths`。
- 如果某个 mode 组合无法触发，必须说明尝试的 shape、生成物证据和无法触发原因，不能静默跳过。
- 如果大 shape 因设备资源不足无法运行，必须记录为资源阻塞；不能擅自换小 shape 后声称覆盖 fallback/Common 路径。

## 无历史脚本时的脚本生成规范

如果当前环境没有可复用的 Inductor CV 验证脚本，agent 必须自动生成脚本，不能只用一次性 shell/Python 片段代替。

优先使用本 skill 自带模板：`scripts/run_inductor_cv_matrix.py`。在新环境中，把该模板复制到本次 `RUN_DIR/script/` 或直接用原路径执行；脚本会自复制到 `RUN_DIR/script/run_inductor_cv_matrix.py` 作为证据。该模板只依赖 PyTorch、`torch_npu` 和 TorchAir experimental 的 `_inductor_npu_ext/python` 包。

构造脚本时必须采用 parent/child 两层结构：parent 负责解析 `--repo-root`、`--cann-toolkit`、`--torchair-experimental`、`--autofuse-prefix`、`--run-dir`，写入 `logs/env.log`、`logs/matrix_plan.json` 并逐 case 拉起子进程；child 只运行一个 case，设置独立 cache/debug/dump 环境，导入 `torch`、`torch_npu`、`inductor_npu_ext`，构造 eager/reference 和 `torch.compile` 对照，最后打印一行 `CASE_RESULT` JSON。禁止把多 case 写在同一个 Python 进程里复用 TorchDynamo/Inductor 状态。

脚本自身也必须先验证：在不依赖真实设备的临时 repo、假 CANN、假 TorchAir experimental 目录中跑 `--mode env-probe`，确认能自复制脚本、输出 `env.log`、输出 `matrix_plan.json`、记录 `argv`/解析参数、识别 `--torchair-experimental`。只有脚本自检通过后，才能进入真实 CANN/overlay 构建和矩阵验证。

脚本必须可迁移：

- 提供 `--repo-root`、`--cann-toolkit`、`--torchair-experimental`、`--autofuse-prefix`、`--run-dir`、`--case-filter` 参数，默认值从当前环境发现。
- 不能写死个人路径、固定 CANN 版本、固定 worktree 名、固定 overlay 目录。
- 脚本启动时必须把解析后的 repo、CANN、Python、torch/torch-npu、关键环境变量写入 `logs/env.log`。
- run dir 必须保存实际执行脚本、命令行参数、脚本 hash 和环境快照；不能只在临时 shell 里执行一次性片段。
- 如果 run 包尚未安装/overlay，`--mode run` 必须阻断；`autofuse.pyautofuse`、`libaihac_codegen.so` 和 `tools/bisheng_compiler/bin/bisheng` 必须来自同一套本次安装/overlay 布局。仅使用 `build/_CPack_Packages/makeself_staging/python/site-packages` 不够，因为其相对路径可能找不到 `tools/bisheng_compiler`。
- 新环境若安装到临时 overlay，必须保证 `<autofuse-prefix>/tools/bisheng_compiler/bin/bisheng` 可用；如果 graph_autofusion run 包不携带 `tools/`，可在临时 overlay 中把 `tools` 指向所选 CANN Toolkit 的 `tools`，并在 summary 中记录。
- AscendC 编译需要 CANN host/device include 路径，runner 会注入并记录 `CPLUS_INCLUDE_PATH` / `C_INCLUDE_PATH`，至少覆盖 `x86_64-linux/asc/include/utils/tiling/platform`、`x86_64-linux/pkg_inc/base`、`x86_64-linux/include/base`、`x86_64-linux/include`，以及 `ops_nn/ascendc/common` 和包含 `arch35/mat_mul_tiling_data.h` 的 `mat_mul_v3` 父目录；缺这些目录会在 MatMul CV 编译阶段报 `fatal error: 'arch35/mat_mul_tiling_data.h' file not found` 或 `fatal error: 'cmct/block/block_scheduler_policy.h' file not found`。

新环境最小演练命令：

```bash
python <skill>/scripts/run_inductor_cv_matrix.py \
  --repo-root <graph-autofusion-repo> \
  --cann-toolkit <CANN_TOOLKIT> \
  --torchair-experimental <torchair>/experimental \
  --autofuse-prefix <graph-autofusion-overlay-or-install-root> \
  --run-dir <repo>/temp/inductor_cv_validation/<timestamp> \
  --mode env-probe

python <skill>/scripts/run_inductor_cv_matrix.py \
  --repo-root <graph-autofusion-repo> \
  --cann-toolkit <CANN_TOOLKIT> \
  --torchair-experimental <torchair>/experimental \
  --autofuse-prefix <graph-autofusion-overlay-or-install-root> \
  --run-dir <repo>/temp/inductor_cv_validation/<timestamp> \
  --mode dry-run

python <skill>/scripts/run_inductor_cv_matrix.py \
  --repo-root <graph-autofusion-repo> \
  --cann-toolkit <CANN_TOOLKIT> \
  --torchair-experimental <torchair>/experimental \
  --autofuse-prefix <graph-autofusion-overlay-or-install-root> \
  --run-dir <repo>/temp/inductor_cv_validation/<timestamp> \
  --mode run
```

其中 `<torchair>/experimental` 在当前工作区可以是 `/workspace/xuyf/torchair/experimental`，但正式脚本和文档不得写死该路径；其他环境必须通过 `--torchair-experimental` 或 `TORCHAIR_EXPERIMENTAL` 注入。

目录约定：

```text
temp/inductor_cv_validation/<timestamp>/
  script/run_inductor_cv_matrix.py
  logs/
  cases/<case_name>/
    cache/
    torch_compile_debug/
    autofuse_dump/
```

脚本必须包含以下环境设置：

```python
os.environ["RUN_DIR"] = str(run_dir)
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(case_dir / "cache" / "inductor")
os.environ["TORCHINDUCTOR_NPU_EXT_CACHE_DIR"] = str(case_dir / "cache")
os.environ["TORCHINDUCTOR_DEBUG_DIR"] = str(case_dir / "torch_compile_debug")
os.environ["TORCH_COMPILE_DEBUG_DIR"] = str(case_dir / "torch_compile_debug")
os.environ["TRITON_CACHE_DIR"] = str(case_dir / "cache" / "triton")
os.environ["AUTOFUSE_DFX_FLAGS"] = f"--codegen_compile_debug=true;--debug_dir={case_dir / 'autofuse_dump'}"
os.environ["TORCHINDUCTOR_NPU_EXT_TILING_DEBUG"] = "1"
os.environ["TORCHINDUCTOR_NPU_EXT_DISABLE_TOPN_TILING"] = "1"
if case.mode_variant == "vv_only":
    os.environ["TORCHINDUCTOR_NPU_EXT_DEBUG"] = remove_debug_option(os.environ.get("TORCHINDUCTOR_NPU_EXT_DEBUG"), "matmul")
elif case.expect_cv:
    os.environ["TORCHINDUCTOR_NPU_EXT_DEBUG"] = add_or_keep_debug_option(os.environ.get("TORCHINDUCTOR_NPU_EXT_DEBUG"), "matmul")
os.environ.setdefault("ASCEND_LAUNCH_BLOCKING", "1")
os.environ.pop("ENABLE_TILING_SHIM", None)
```

脚本结构要求：

- 每个 case 使用独立 `case_dir`，避免 cache、dump、debug 互相覆盖。
- 多 case 执行时用子进程逐个运行，避免 torch/dynamo/Inductor 状态污染。
- MM/BMM CV case 必须显式启用 TorchAir experimental MatMul lowering，即 `TORCHINDUCTOR_NPU_EXT_DEBUG` 包含 `matmul`；否则 `torch.mm`/`torch.bmm` 会 fallback 到原生 addmm/bmm，只生成独立 vector kernel，不能算 CV epilogue 验证。
- `mode_variant="vv_only"` 的 case 必须移除 `matmul` debug option，用同形状同 op 做 A/B 对照；这类 case 只证明 VV-only 路径可运行，不能计为 CV artifact 覆盖。
- 输入数据优先在 CPU 生成后搬到 NPU，避免 NPU random kernel 干扰验证。
- 每个 case 调用 `torch._dynamo.reset()`，再执行 eager/reference 和 `torch.compile(fn, dynamic=<bool>, fullgraph=True)`。
- 每个 case 后执行 `torch.npu.synchronize()`。
- case 定义必须包含 `name`、`kind`、`family`、`mode_variant`、`dynamic`、`dtype`、`shape`、`atol/rtol`、`expect_cv`、预期 `fusion_mode/ub_mode/mix_mode`，template coverage case 还必须记录 `expected_template` / `expected_tiling_key`。
- case 定义不能换成通用 conv/pool CV smoke；本 skill 的 CV 指 TorchInductor + Autofuse 的 MM/BMM/VV/fragment08 CV fusion 矩阵。
- 动态 shape case 至少使用两组实际输入 shape 或明确记录当前脚本为何只能跑一组；需要确认 guard/cache 没有复用旧 specialized kernel。
- 每个 case 必须打印一行 `CASE_RESULT` JSON，字段至少包括 `name`、`kind`、`family`、`mode_variant`、`op`、`dynamic`、`dtype`、`shape`、`stride`、`seed`、`expect_cv`、`expected_template`、`expected_tiling_key`、`fusion_mode`、`ub_mode`、`mix_mode`、`max_abs_diff`、`mean_abs_diff`、`max_rel_diff`、`fail_count`、`numeric_ok`、`structure_ok`、`ok`、`stage`、`case_dir`、`command`、`exit_code`、`start_time`、`end_time`、`artifact_paths`、`cv_artifacts`。
- `stage` 只能使用枚举：`env`、`build`、`deploy`、`fusion_codegen`、`ascendc_compile`、`kernel_load`、`launch_sync`、`numeric`、`summary`。
- 子进程必须设置 timeout；timeout、signal 或非零 `exit_code` 均为失败，不能记为 skipped。
- 默认必须继续执行所有 selected cases 并收集结果；只有调试时显式传 `--fail-fast` 才允许首个失败后停止，正式 full matrix 不得使用 `--fail-fast`。
- 汇总脚本必须校验 `artifact_paths` 中的关键产物真实存在，例如 `asc_kernel.py`、wrapper、`kernel.so`、case log、dump/debug 目录；缺失则标为 `stage=summary` 失败。
- artifact 必须位于本次 `RUN_DIR/cases/<case_name>/` 下，并包含 run id、case id、mtime 位于 case start/end 时间窗口内或 hash 被本次脚本记录；只校验“路径存在”不够。
- timeout 失败必须保留 stdout/stderr、完整命令、超时时长、最后日志片段和 case artifact 目录。
- 脚本不能在失败后删除 `cache`、`torch_compile_debug`、`autofuse_dump`。
- 如果脚本启动时要清理旧数据，只能清理本次新建 run 目录内的同名空目录，不得清理历史失败目录。

## 每个 case 的证据清单

每个矩阵 case 都必须收集并在 summary 中记录以下证据：

| 阶段 | 必须证明 | 典型证据 |
|------|----------|----------|
| Fusion/codegen | 确实进入 Inductor CV 路径 | 日志包含 `Can fuse`、`Generating kernel for fused`、`autofused_*`、case 的 graph fragment |
| 生成代码 | 生成了当前 case 的 `asc_kernel.py`、wrapper、cache 目录 | `case_dir/cache/autofused_*/<hash>/asc_kernel.py`、`inductor_wrapper.cpp`、dump graph |
| AscendC 编译 | `asc_kernel.py` 执行成功并产出 `kernel.so` | 编译命令、无 `CompileError`、存在 `kernel.so` |
| kernel load | `kernel.so` 可被 wrapper/runtime 加载 | 日志无 `Kernel load failed`；必要时用 `ctypes.CDLL(<kernel.so>)` 获取真实 loader 结果 |
| launch/sync | kernel 真实 launch 并同步完成 | 日志包含 `Start launch kernel`、`Start sync kernel`、`Succeed sync kernel` |
| mode/template | 覆盖了预期模板和 mode | wrapper/summary 中的 `fusion_mode`、`ub_mode`、`mix_mode`、tiling key、fallback 信息 |
| 数值 | 输出和 eager/reference 在 dtype 阈值内 | `CASE_RESULT` 中 `ok=true`、`max_abs_diff`、`mean_abs_diff`、`max_rel_diff`、`fail_count`、最大误差 index、`atol/rtol` |

通过标准：

- fp32/HF32 场景按脚本配置阈值判断，必须报告 `atol/rtol`。
- fp16/bf16 场景按 dtype 阈值判断，不能硬要求所有 case `max_abs_diff=0.0`。
- 如果数值失败但前面阶段都成功，结论必须是 numeric failure，不得写成 codegen 失败。
- 如果只完成 codegen 或编译，不能声称功能验证通过。
- 只有证明 eager/reference 与 kernel 使用同等精度语义后，才能调整 dtype 阈值；不得用放大 `atol/rtol` 掩盖 layout、stride、tiling 或 epilogue 错误。

## 失败分阶段定位

失败时必须先定位阶段，再提出修复假设。禁止直接按经验修改 codegen。

| 失败表现 | 阶段 | 必查项 |
|----------|------|--------|
| `Codegen generate kernel failed or abort` | `fuser.codegen` / Autofuse codegen | Autofuse dump、API call 返回、graph after optimize、是否生成 `asc_kernel.py` |
| `asc_kernel.py` 返回非 0 或 `CompileError` | AscendC host/device 编译 | 直接执行失败的 `asc_kernel.py`，读取完整 stderr，定位 host/device 源码行 |
| `Kernel load failed` | `kernel.so` 动态加载 | `ldd kernel.so`、`nm -D kernel.so`、`readelf -Ws`、`ctypes.CDLL(kernel.so)` 的原始 loader error |
| `undefined symbol` | 动态库符号/ABI | 确认符号由谁引用、谁应该导出；检查 CANN/overlay 版本是否一致 |
| launch 后 runtime error | runtime launch/sync | kernel signature、入参 shape/dtype/stride、workspace、block_dim、tiling data、stream |
| `CASE_RESULT ok=false` | 数值错误 | 保留 generated code 和输入 shape；检查 dtype 阈值、reference、tiling、offset/stride、broadcast、Nddma |

红线：

- 不能清理失败 case 的 `cache`、`torch_compile_debug`、`autofuse_dump` 后再分析。
- 不能只看最后一条 `ERR99999`；必须找最早的非派生错误。
- 不能把 `kernel.so load failed` 直接归类为 Nddma/codegen 逻辑错误；必须先拿 loader error。
- 不能把数值失败写成“编译失败”或“验证通过但 diff 较大”。
- 不能因为同类另一个 case 通过就跳过失败 case；例如 `cv_dynamic_bmm_to_mul_k1_mix2` 通过不能证明 `cv_dynamic_bmm_key1` 正确。

## 历史问题定位 Playbook

遇到以下签名时，先按表格分阶段收证，不要直接改代码：

| 签名 | 阶段 | 必查证据 | 禁止结论 |
|------|------|----------|----------|
| `MAT_MUL_*` macro 未定义、device compile 失败 | AscendC 编译 | 失败 `asc_kernel.py` stderr、包含的 `matmul*.h`、实际 CANN 头文件路径 | 不得说 fusion 没触发或 runtime 失败 |
| `arch35/mat_mul_tiling_data.h` file not found | AscendC host 编译 | `CPLUS_INCLUDE_PATH`/`C_INCLUDE_PATH` 是否包含 CANN `mat_mul_v3` 父目录、replay `asc_kernel.py` stderr | 不得靠手改 generated include 通过 |
| `cmct/block/block_scheduler_policy.h` file not found | AscendC device 编译/ops-nn 依赖 | 选定 CANN 是否含 `ops_nn/ascendc/common/cmct/...`、runner 是否注入 `ops_nn/ascendc/common`、必要时更新并安装最新 ops-nn | 不得修改 Autofuse 源码绕过 |
| `std::setw`/`std::setfill` no member named | AscendC host 编译/codegen | generated host C++ 是否缺 `<iomanip>`、replay `asc_kernel.py` stderr、对应 codegen 模板 | 不得归类为 load/launch/numeric |
| `DT_BOOL` vs `DT_UINT8`、bool compare/where/isfinite 输出不匹配 | dtype/lowering | FX graph dtype、ASC wrapper output dtype、pyascir `Output` dtype | 不得用 cast 或手改 generated output 绕过 |
| static VV `std::bad_alloc`、TopN 相关 abort | tiling/template 选择 | `TORCHINDUCTOR_NPU_EXT_DISABLE_TOPN_TILING` 是否生效、static/dynamic 对照、TopN 日志 | 不得清 cache 后直接重跑算通过 |
| `cv_dynamic_bmm_key1` compile/load/launch 成功但 diff 大 | 数值/BMM epilogue | `fusion_mode/ub_mode/mix_mode`、dynamic shape 实参、wrapper 参数顺序、stride/storage offset、tiling data、是否执行 bias/elemwise epilogue | 不得先调大容差 |
| `kernel.so load failed` 或 `undefined symbol` | 动态加载/ABI | 原始 loader error、`ldd`、`readelf -Ws`、`nm -D`、CANN/overlay 同源性 | 不得直接归因 Nddma 或 generated code |

触发 playbook 中任一签名时，对应“必查证据”必须出现在失败项里；如果证据缺失，结论只能是 `evidence_missing` / `stage=summary`，不能给根因结论。

BMM 数值失败必须额外做 A/B 证据：

- 同 shape 的 static/dynamic 对照，确认是否 dynamic 参数问题。
- 同 shape 的 CV/non-CV 或 eager/reference 对照，确认是否平台 MatMul 精度问题。
- 失败元素最大 diff 的 index、该位置 reference/compiled 值、对应 batch/M/N/K 输入切片。
- 检查 epilogue 是否执行，特别是 BMM + bias/add/mul 场景中 `AutoFusionVector::Params` 是否被对应模板消费。
- A/B 必须共享同一输入 dump、seed、shape、stride、dtype、epilogue 配置和 `atol/rtol`，否则只能作为线索，不能作为根因证据。

## 新环境演练

验证本 skill 是否好用时，必须做一次无历史依赖演练：

1. 在新 shell 或新 worktree 中执行，清空 `LD_PRELOAD`、`ENABLE_TILING_SHIM` 和旧 cache 相关变量。
2. 不使用历史 run 目录作为结果；如需避免误读，可移动或忽略旧 `temp/inductor_cv_validation/*`，但未经用户确认不得删除失败现场。
3. 只通过当前环境、用户指定参数或自动发现确定 repo/CANN/overlay 路径。
4. 生成新的 matrix 脚本和 run 目录，跑至少一个 RED 场景或已知失败 subset，确认失败能被正确分阶段分类。
5. 修复后必须在同一套 CANN 和同一套运行产物下重跑失败 subset，再跑完整矩阵。
6. 如果新环境暴露 skill 没覆盖的问题，先更新本 skill 的门禁或 playbook，再重新演练。

## Quick Reference

| 目标 | 最低要求 |
|------|----------|
| 环境 | 唯一 CANN Toolkit；环境变量同源；无 `LD_PRELOAD` workaround |
| 构建 | `aihac_codegen`、`pyautofuse` 按影响范围重编，所有构建 `-j 8` |
| 部署 | 运行环境实际加载本次构建产物或本次 overlay |
| 覆盖 | VV 对照、非融合对照、MM UB/fallback/fixpipe、BMM 常规/K=1/template、mixed dtype/bool、extended 边界、`vv_only_*` A/B、fragment08 |
| 脚本 | 参数化 repo/CANN/TorchAir/autofuse-prefix/run-dir/case-filter；无个人路径；case 独立子进程 |
| 证据 | codegen、生成代码、AscendC 编译、kernel load、launch/sync、mode、数值、seed/stride/diff 分布、产物 hash |
| 失败 | 分阶段定位，保留 cache/debug/dump/log |

## 常见错误

| 错误 | 正确做法 |
|------|----------|
| 只跑 `cmake --build` 就说验证通过 | 必须跑 Inductor E2E case 到 launch 和数值 |
| 只跑 fragment08 | 必须覆盖 MM、BMM、UB、fallback、static、dynamic |
| 用历史日志当结果 | 历史日志只能选 shape，必须当前重跑 |
| 手改 generated wrapper/cache | 只能改 generator 或正式源码 |
| load 失败直接修 Nddma | 先拿 `dlopen`/loader error |
| 清理失败目录 | 保留失败现场，另建新 run 目录 |
| 多 CANN 混用 | 停止并让用户选择唯一 toolkit |
| 脚本写死 `/workspace/xuyf` 或固定 CANN | 改为参数、环境变量或自动发现 |
| 把通用 conv/pool CV 当成 Inductor CV 矩阵 | 必须跑本 skill 的 MM/BMM/VV/fragment08 矩阵 |
| BMM 数值失败先调容差 | 先收集 diff 分布、A/B 对照、wrapper/tiling/epilogue 证据 |
| 用 `--case-filter` 跑 subset 后写通过 | subset 只能用于调试；正式结论必须展示全矩阵 |
| 只打印 `CASE_RESULT` 不校验产物 | summary 必须校验 artifact path/hash 与本次 run 绑定 |
| 重跑成功后隐藏首次失败 | 报告所有 attempt，说明失败项和恢复方式 |

## Red Flags

出现以下想法或请求时必须停止并回到本 skill 的门禁流程：

- “时间紧，先跑一个关键 case。”
- “历史日志里这个 case 过了，可以算通过。”
- “先清 cache/debug/dump 重跑，失败现场不重要。”
- “load failed 大概率是 Nddma/codegen，直接改。”
- “先用 `LD_PRELOAD` 或手改 generated wrapper 验证思路。”
- “构建过了，所以 Inductor CV 应该没问题。”
- “这个修改只影响一小块，不需要跑完整矩阵。”
- “这台机器路径不一样，先照搬历史 `/workspace/xuyf` 脚本。”
- “`cv_dynamic_bmm_key1` 只是容差问题，先放大阈值。”
- “没有历史脚本，就先跑几个 conv/pool 代表 CV。”
- “自动发现到一个 CANN，先用它，不用列候选。”
- “subset 都绿了，full matrix 以后再补也可以写通过。”
- “第一次失败后来重跑过了，不需要报告失败 run。”

这些都表示验证证据不足，不能给出通过结论。

## 最终报告格式

验证结束后必须输出：

```markdown
### Inductor CV 验证结果
- CANN Toolkit: <path/version>
- Run dir: <temp/inductor_cv_validation/...>
- 构建产物: <libaihac_codegen.so/pyautofuse.so 来源和部署位置>

### 场景矩阵
| Case | Static/Dynamic | Template/Mode | Shape | Result | Evidence |
|------|----------------|---------------|-------|--------|----------|

### 失败项
- <阶段>: <证据路径>: <根因/下一步>

### 结论
- 只有全部矩阵通过时才能写“Inductor CV fusion 验证通过”。
- 任一矩阵项未跑、未触发、失败或证据缺失，都必须写“未完成验证”并列出阻塞项。
```
