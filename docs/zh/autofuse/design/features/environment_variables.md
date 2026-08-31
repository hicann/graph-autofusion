# AutoFuse 相关环境变量参考

本文汇总 PyTorch 和 TensorFlow 框架下，AutoFuse 自动融合功能运行和调测过程中常用的环境变量及控制项。

## 目录

- [AutoFuse 共享环境变量](#autofuse-共享环境变量)
  - [`AUTOFUSE_FLAGS` 功能控制](#autofuse_flags-控制项)
  - [`AUTOFUSE_DFX_FLAGS` 调测控制](#autofuse_dfx_flags-控制项)
- [框架专属环境变量](#框架专属环境变量)
  - [PyTorch 专属](#pytorch-专属)
  - [TensorFlow 专属](#tensorflow-专属)
- [使用注意事项](#使用注意事项)

---

## AutoFuse 共享环境变量

以下环境变量为 AutoFuse 核心功能控制项，同时适用于 TensorFlow 和 PyTorch 框架：

| 环境变量               | 适用场景            | 说明                                                                                                                                                                                                                                                |
| :--------------------- | :------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `AUTOFUSE_FLAGS`     | TensorFlow、PyTorch | AutoFuse 功能控制。所有控制项格式统一，多个控制项使用英文分号分隔。<br>- **TensorFlow**：必须配置 `--enable_autofuse=true` 才能开启 AutoFuse 融合功能<br>- **PyTorch**：通过 `torch.compile` 后端配置默认开启，本环境变量仅用于配置扩展功能 |
| `AUTOFUSE_DFX_FLAGS` | TensorFlow、PyTorch | AutoFuse 调测控制，用于融合图 Dump、代码生成调测和 Auto Tiling 调测；多个控制项使用英文分号分隔。                                                                                                                                                   |

## `AUTOFUSE_FLAGS` 控制项

`AUTOFUSE_FLAGS` 用于控制 AutoFuse 功能。未开启 `--enable_autofuse=true` 时，其他 AutoFuse 功能控制项不生效。

**仅开启基础 AutoFuse 融合功能（最简配置）：**

```bash
export AUTOFUSE_FLAGS="--enable_autofuse=true"
```

下表列出所有可选控制项，可根据需要组合使用：

| 控制项                                     | 适用场景            | 说明                                                                                                                     |
| :----------------------------------------- | :------------------ | :----------------------------------------------------------------------------------------------------------------------- |
| `--enable_autofuse`                      | TensorFlow          | 取值为 `true` 或 `false`，默认关闭。控制整体自动融合功能；未开启时，其他 AutoFuse 功能控制项不生效。                  |
| `--autofuse_enable_pass`                 | TensorFlow          | 用于开启指定的扩展融合能力。目前暂时开放 `reduce` 和 `concat` 两种类型，多个取值使用英文逗号分隔；其他融合类型将在后续版本中逐步开放。默认不配置，扩展融合能力默认关闭。 |
| `--autofuse_disable_pass`                | TensorFlow          | 取值为 `reduce`、`concat`，或二者的英文逗号分隔组合。默认不配置，不能与 `--autofuse_enable_pass` 同时配置相同取值。 |
| `--autofuse_enhance_precision_blacklist` | TensorFlow          | 取值为 AscIR 算子类型字符串，多个类型使用英文逗号分隔，默认值为空。也可配置为 `all`；关闭精度提升可能影响计算精度。     |
| `--recomputation_threshold`              | TensorFlow          | 取值范围为 `0`～`255`，默认值为 `1`。设置自动融合重计算阈值。                                                       |
| `--max_fusion_size`                      | TensorFlow          | 取值范围为 `0`～`uint64_t` 最大值，默认值由实现决定；配置为 `0` 表示不融合。                                        |
| `--autofuse_enable_pgo`                  | TensorFlow、PyTorch | 取值为 `true` 或 `false`，默认关闭。开启 PGO 调优，仅支持静态图，并需要按对应版本要求准备 `mspti`。                 |
| `--experimental_enable_jit_executor_v2`  | TensorFlow          | 取值为 `true` 或 `false`，默认关闭。开启切图编译；动态分档、资源类算子及部分控制流场景不支持该功能。                  |

示例：

```bash
export AUTOFUSE_FLAGS="--enable_autofuse=true;--autofuse_enable_pass=reduce,concat"
```

## `AUTOFUSE_DFX_FLAGS` 控制项

`AUTOFUSE_DFX_FLAGS` 用于 AutoFuse 编译、Auto Tiling 和融合结果调测。

| 控制项                                 | 适用场景             | 说明                                                                                                                  |
| :------------------------------------- | :------------------- | :-------------------------------------------------------------------------------------------------------------------- |
| `--codegen_compile_debug`            | TensorFlow、PyTorch  | 取值为`true` 或 `false`，默认关闭。设置为 `true` 时保留 Kernel、Tiling、CMake 工程、编译结果并生成融合图 Dump。 |
| `--debug_dir`                        | TensorFlow、PyTorch  | 取值为有效的目录路径，默认保存到当前执行目录；需要先开启`--codegen_compile_debug=true`。                            |
| `--autofuse_att_algorithm`           | TensorFlow、PyTorch  | 取值为`AxesReorder`（默认）或试验性的 `HighPerf`。                                                                |
| `--att_accuracy_level`               | TensorFlow、PyTorch  | 取值为`0` 或 `1`，默认值为 `1`；`1` 表示高精度求解，`0` 表示低精度求解。                                    |
| `--att_enable_multicore_ub_tradeoff` | TensorFlow、PyTorch  | 取值为`true` 或 `false`，默认关闭；用于开启多核利用率与 UB 利用率权衡策略。                                       |
| `--att_ub_threshold`                 | TensorFlow、PyTorch  | 取值范围为`0`～`100`，默认值为 `20`；需要配合 `--att_enable_multicore_ub_tradeoff=true` 使用。                |
| `--att_corenum_threshold`            | TensorFlow、PyTorch  | 取值范围为`0`～`100`，默认值为 `40`；需要配合 `--att_enable_multicore_ub_tradeoff=true` 使用。                |
| `--att_profiling`                    | TensorFlow、PyTorch  | 取值为`true` 或 `false`，默认关闭；主要用于定位 Auto Tiling 耗时问题。                                            |
| `--disable_lifting`                  | TensorFlow           | 取值为`true` 或 `false`，默认开启；仅建议用于定位 AscBackend 回滚问题。                                           |
| `--autofuse_pgo_algo`                | TensorFlow           | 取值为`core_select`（默认）或 `pruning`。                                                                         |
| `--autofuse_pgo_step_max`            | TensorFlow           | 取值为`2`～`1024` 范围内的 2 的幂，默认值为 `16`；仅当 `--autofuse_pgo_algo=pruning` 时生效。                 |
| `--autofuse_pgo_topn`                | TensorFlow           | 取值为`0` 或任意正整数，默认值为 `5`；`0` 表示选择全部候选解。                                                  |
| `--skip_node_names_cfg`              | TensorFlow           | 取值为有效的`.ini` 配置文件路径；用于指定跳过融合的算子名称或算子类型。                                             |

示例：

```bash
export AUTOFUSE_DFX_FLAGS="--codegen_compile_debug=true;--debug_dir=/path/to/dump"
```

---

## 框架专属环境变量

以下环境变量为对应框架专属，仅用于特定框架的调测：

### PyTorch 专属

这些环境变量用于 PyTorch 编译或运行调测，不用于 TensorFlow 图模式：

| 环境变量                               | 说明                                                                                                                                                             | 使用方式                                        |
| :------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------- |
| `TORCH_COMPILE_DEBUG`                | 开启 PyTorch 编译调试信息，并将编译中间产物保存到当前目录的 `torch_compile_debug` 目录。以 `autofused_` 为前缀的目录通常表示 AscendC 后端生成的融合算子产物。 | `export TORCH_COMPILE_DEBUG=1`                |
| `TORCHINDUCTOR_FORCE_DISABLE_CACHES` | 禁用 Inductor 缓存，强制每次执行都重新编译。该配置会增加编译和图启动耗时，仅用于调试。                                                                           | `export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` |
| `ASCEND_LAUNCH_BLOCKING`             | 使 Ascend Kernel 同步执行，便于定位首个报错的 Kernel。该配置会降低执行性能，仅建议在问题定位时使用。                                                             | `export ASCEND_LAUNCH_BLOCKING=1`             |

### TensorFlow 专属

当前没有 TensorFlow 专属环境变量，所有 TensorFlow 相关控制项均已包含在 AutoFuse 共享环境变量中。

---

## 使用注意事项

- `AUTOFUSE_FLAGS` 和 `AUTOFUSE_DFX_FLAGS` 适用于使用 GE/AutoFuse 的图编译场景；TensorFlow 用例需要在导入 TensorFlow 和 `npu_bridge` 前设置相关环境变量。
- PyTorch 用例通过 `torch.compile(..., options={"npu_backend": "ascendc"})` 选择 AscendC 后端，通常不需要通过 `AUTOFUSE_FLAGS` 开启 AutoFuse。
- 调测环境变量会增加编译或运行开销，完成问题定位后请及时取消设置。
- 详细控制点说明请参见 [AUTOFUSE_FLAGS 环境变量控制点](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/programug/graphdevg/autofuse_1_0061.html) 和 [AUTOFUSE_DFX_FLAGS 环境变量控制点](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/programug/graphdevg/autofuse_1_0062.html)。
