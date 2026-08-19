# Inductor CV 融合工作手册

## 目标

统一指导 CV 融合的环境部署、流程验证、精度定位、性能分析、单算子比对和全量矩阵验证。

## 通用原则

1. 先确认权限，再开始下载和构建。
2. 先对齐版本，再做验证。
3. 所有验证放在 `temp/` 下，禁止污染源码树。
4. 日志、生成代码、tiling 结果都要保留。
5. 运行时日志优先于离线提取结果。

## 1. CV 环境部署

### 目标

搭好可运行、可编译、可替换的 inductor CV 环境。

### 关键约束

1. 工作目录默认使用当前会话路径下的 `af_YYYYMMDD`。
2. 每次运行前都要执行安装目录里的 `set_env.sh`。
3. `toolkit`、`950ops`、`torchair`、`torch-npu`、run 包必须版本匹配。
4. 环境不通时先修环境，不要先怀疑业务代码。

### 标准步骤

1. 先尝试访问 `https://gitcode.com/cann/graph-autofusion`。
2. 如果可以下载，说明当前账号有权限，继续下一步。
3. 如果不能下载，先提示用户提供 token 或登录信息。
4. 在当前会话路径下创建目录，例如 `af_20260813`。
5. 进入新建目录。
6. 下载最新 `graph-autofusion` 仓代码。
7. 如果用户提供了 PR 代码，就先更新 PR 代码，再重新编译 run 包。
8. 如果用户没有给 PR，就直接基于最新代码编译 autofuse run 包。
9. 基于 `https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/` 找到最新子目录。
10. 下载该子目录下匹配的 `toolkit` 和 `950ops` 两个包到 `af_*` 目录。
11. 执行安装目录中的 `set_env.sh`。
12. 按照编译出来的 autofuse run 包更新 toolkit 目录中的原始包代码。
13. 下载 `https://gitcode.com/Ascend/torchair` 到 `af_*` 目录。
14. 在 `torchair/experimental/_inductor_npu_ext` 下按仓库提示安装扩展。
15. 按 torchair 仓库对应的 skill 或说明，安装版本匹配的 `torch-npu`。
16. 如果环境还缺 `pytorch` 或其他依赖，再补齐必要包。

### 验收

- `set_env.sh` 可以正常 source。
- `graph-autofusion`、`toolkit`、`950ops`、`torchair`、`torch-npu` 版本一致。
- 能进入最小 inductor 样例验证。

## 2. CV 融合流程验证

### 目标

验证融合链路能跑通，确认从生成到执行的主路径正常。

### 推荐顺序

1. 先跑单算子，确认基础链路。
2. 再跑 VV 融合，确认向量融合链路。
3. 再跑 CV 融合，确认主目标链路。
4. 每次都单独建目录保存结果，避免相互覆盖。

### 最小验证用例

1. 构建一个简单 PyTorch inductor case，例如 `mm + mul + mul`。
2. 单独验证 VV 融合和 CV 融合都能编译、生成、运行。
3. 参考 graph-autofusion 里 CV 相关 skills 或示例脚本构造运行脚本。

### 必留产物

- `stdout.log`
- `stderr.log`
- `result.json`
- `tiling.json`
- 生成的 tiling func 文件
- 生成的 kernel 文件
- cache 目录
- 输出 tensor 或对比结果

### 运行建议

1. 开启 `ASCEND_GLOBAL_LOG_LEVEL=0`。
2. 开启 `ASCEND_SLOG_PRINT_TO_STDOUT=1`。
3. 保留每次运行的 case 目录。
4. 先看输出是否正常，再看 tiling，再看 kernel。

### 修改 autofuse 代码的快速验证

1. 如果有新 PR，先重新构建 autofuse run 包，再安装到 toolkit 目录。
2. 构建验证脚本，执行融合流程。
3. 不做性能分析时，每次运行都要把 CV 融合的 `tiling key`、`tiling data`、`tiling func` 文件、`kernel` 文件保存下来。
4. 这些中间结果按场景分目录保存，再按修改行为建子目录。
5. 如果怀疑 kernel 或 tiling func 有问题，优先修改生成出来的代码，不要先改 autofuse 源码。
6. 如果必须改 autofuse 源码，再编译修改后的 so，并替换到安装路径。

### 验收

- 能看到完整的 fusion 入口日志。
- 能找到 tiling func 和 kernel 生成结果。
- 输出能正常得到，且链路没有中断。

## 3. CV 精度功能定位

### 目标

定位数值错误来自输入、tiling、kernel 还是尾块 / 对齐问题。

### 定位顺序

1. 先看融合的 matmul 是否存在异常 tiling key 或异常 tiling data。
2. 再根据 tiling key 去核对 kernel 代码里的对应模板。
3. 再去安装目录下的 `common/cmct` 对应模板比对实现是否一致。
4. 如果前面没有问题，再看 tiling func 里对该 key 的 mode 分组是否正确。
5. 如果仍然有精度问题，再对比单算子和融合算子的输出 tensor。
6. 最后再判断是否是尾块、对齐块、边界块导致的局部异常。

### 排查重点

1. `matmul` 的 tiling key 和 tiling data 是否异常。
2. 根据 tiling key，核对 kernel 代码里的模板定义和实现位置。
3. 核对安装目录下 `common/cmct` 里的同名模板实现是否被改过，避免只看了本仓代码。
4. 重点检查 tiling key 的定义位置和模板实现位置，确认分支语义没有漂移。
5. 检查 tiling func 中对异常 key 的 mode 分组是否正确。
6. 对比单算子和融合算子的输出 tensor，观察是否有规律性偏差，还是只有部分数据异常。
7. 如果有部分异常，优先怀疑尾块 / 对齐 / 边界块。
8. 检查 `Hf32`、`isAvoidTensorApi`、`cube_tiling_key`、shape、transpose / layout 是否一致。
9. 检查对应 tiling key 的 tiling data 结构体类型和大小是否与 kernel 代码匹配。

### 验证纪律

1. 每次验证前先清缓存，避免旧结果干扰。
2. 如果只是验证假设，优先改生成代码，不要先改 autofuse 源码。
3. 如果要对比规律，优先看尾块和部分异常数据是否有固定模式。
4. 如果怀疑是缓存问题，先清理旧 cache 再跑，不要带着历史产物下结论。

### 适用场景

- 融合算子数值错误。
- 单算子正确、CV 错误。
- 怀疑 tiling key 分组、kernel 模板或尾块对齐逻辑有问题。

## 4. CV 性能分析

### 目标

拆开统计编译、host wrapper、tiling func、kernel、e2e 的性能影响。

### 统计范围

- 编译性能
- host wrapper
- tiling func
- kernel
- e2e

### 统计方法

1. 统计第一次运行性能。
2. 统计第二次运行性能。
3. 统计第 3 到第 5 次的平均性能。
4. 对比时把编译性能和运行性能分开看。

### 场景维度

每个 case 至少区分以下维度：

- 动态 shape
- 静态 shape
- 单算子
- VV 融合
- CV 融合

### 输出要求

1. 每个场景都要记录 host wrapper、tiling func、kernel、e2e 的耗时。
2. 性能统计时尽量减少额外日志对结果的影响。
3. 保留足够的中间产物，便于复现和复核。

## 5. CV 融合算子和单算子快速比对

### 目标

快速确认单算子和 CV 融合是否走了同一类 tiling / kernel 路径。

### 比对顺序

1. 先对齐输入条件。
2. 再对比 host tiling。
3. 再对比 device kernel。

### Host 侧重点

- `NeedToConvertBias`
- 输入 / 输出 shape
- `Hf32`
- `isAvoidTensorApi`
- registry 命中的分支
- `DumpTilingInfo` 的 `tilingkey`
- `workspaceSize`

### Device 侧重点

- 最终命中的 `if constexpr` 分支
- `cube_tiling_key`
- 实际 wrapper / kernel 模板

### 判断经验

1. 单算子和 CV 输入一样但 key 不同，先查 `isAvoidTensorApi`。
2. `81/82` 这种差异通常只是 `BASIC_LEVEL` / `TENSOR_LEVEL` 的区别。
3. `tiling.json` 只作辅助参考，运行日志优先。
4. 如果 host tiling 内容一致但 key 不同，优先看 launch 解析的 key 是否和运行时一致。

### 建议保留的信息

- 单算子和 CV 的 `stdout.log`
- 单算子和 CV 的 `stderr.log`
- 单算子和 CV 的 `tiling.json`
- 单算子和 CV 的生成代码和 kernel 文件

## 6. CV 全量矩阵验证

### 目标

对 static / dynamic、单算子 / VV / CV、功能 / 精度 / 性能场景做批量验证和归档。

### 矩阵维度

至少覆盖以下维度：

- 动态 shape
- 静态 shape
- 单算子
- VV 融合
- CV 融合
- 功能验证
- 精度验证
- 性能验证

### 子 skill 建议

1. 批量跑 case 和收集结果，优先用 `af-test-developer`。
2. 需要深挖异常 case 时，配合 `sk-model-analysis`。
3. 如果要重新构建 run 包或替换产物，配合 `af-build-runner`。

### CV 用例看护

1. 每个 case 都要有独立目录，按场景和日期分层。
2. 跑批时要保留日志、生成代码、tiling 文件和最终对比结果。
3. 看护时先看是否有新增失败，再按功能、精度、性能分类。
4. 如果 case 结果异常，先确认缓存和环境，再确认生成代码，再确认 kernel。
5. 不要把一个 case 的中间产物覆盖到另一个 case。

### 归档规范

推荐目录：

```text
temp/<scenario>/<date>/<case>/
```

每次运行不要覆盖上一次结果。

### 验收

- 能稳定复跑。
- 能快速定位失败 case。
- 能把功能、精度、性能问题拆开归类。
