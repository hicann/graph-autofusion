# 简介

## 目的

为 SuperKernel AOT 建立无需 NPU 的主机侧 ST。本文供测试和构建开发者实施，采用已选定的方案二：生产源码构建测试专用 `libascendsk_st.so`，UT/ST 共用外提的 stub。

实现基线：PR 1995 head `e13b3a44`。运行和扩展方法见 `super_kernel/tests/aot/README.md`。

## 范围

首批覆盖 `aclskOptimize`、`aclskScopeBegin`、`aclskScopeEnd`、`aclskScopeVerify`。包括共享桩、测试动态库、gtest 场景和脚本入口。不包括真实 kernel 数值执行、硬件性能、正式 SO 替换加载、完整 ACL capture/replay 模拟和 Autofuse 改动。

# 总体概述

## 软件概述

### 项目介绍

现有 `aot/ut/CMakeLists.txt` 将生产源码、用例和三个 stub cpp 一起编入 UT。ST 应让 AOT 内部模块完整协作，并从公开 API 驱动，观察 Runtime 边界的结果。

### 产品环境介绍

正式 `ascendsk` 依赖 ACL/Runtime/DFX 和生成的 kernel 入口；测试版替换这些外部依赖。参考 runtime 仓 `tests/depends/CMakeLists.txt` 和 `tests/depends/runtime/CMakeLists.txt` 的共享桩目标组织方式，但不引入 runtime 仓作为构建依赖。

runtime 仓中的 ST 接入并非全部有效：例如 platform 父 CMake 当前仅添加 ut，st 内仍有历史路径。本方案借鉴共享依赖边界，不照抄其 ST 入口。

## 软件功能

提供可构造、可观察的模型任务环境；从公开 API 执行优化、scope 标记和 scope 校验；检查任务修改、返回码和模型销毁时的资源释放；提供独立的 C++ ST 运行入口。

## 设计约束

- 测试版使用完整 AOT 生产源码，不替换 optimizer、graph、task builder 等内部实现。
- 生产头文件保持私有；ST 用例只能包含公开 `super_kernel.h`、外部 ACL 类型和测试支持头。
- 新测试库采用唯一名称和构建目录，不安装进 run 包或 wheel，不伪装成真实 ACL/Runtime SO。
- 构建并行度限制为 `-j 8`，继续使用已有第三方依赖机制。
- 生产 ABI、接口签名、优化算法、kernel 实现不因测试需求而改变。

## 假设和依赖关系

“无 NPU”指测试执行无需设备与驱动。第一阶段沿用仓库配置所需的 CANN/AscendC 环境和本地依赖缓存；完全移除 Toolkit 的构建依赖不在范围内。现有手写 ACL stub 头继续用于主机测试，因此本 ST 不证明其与真实 CANN ABI 完全一致。

# 需求分析与设计

## 整体介绍

```text
AOT 生产源码 ──> libascendsk_st.so ──> libsuper_kernel_aot_stub.so
                         ↑                        ↑
                 super_kernel_aot_stest ───────────┘
                    公开 API 调用          场景准备/结果查询

AOT 源码 + UT ──> super_kernel_aot_utest ──> 同一共享 stub 目标
```

UT 和 ST 是独立进程，每个进程内部仅有一份桩状态。桩负责外部对象和操作记录，融合结果必须由生产实现产生。

## 功能需求

### 功能需求 1：共享 stub 外提

1. 介绍：将 `aot/ut/stub/` 外提到 `aot/depends/`，建立 `super_kernel_aot_stub` SHARED 目标。
2. 输入：现有头文件、`aclrt_stubs.cpp`、`rt_stub.cpp`、`ut_common_stubs.cpp/.h`。
3. 处理：统一源码与 include 路径；更新 UT include；保留现有 `SkUt*` 名称、注错接口及默认语义，不做无关批量重命名。桩支持代码私有使用 kernel 参数所需的内部类型，不向 ST 传递生产内部 include 目录。
4. 输出：UT/ST 共用一个桩目标；原 UT 的直接函数 mock 能力须回归验证。

### 功能需求 2：有状态的外部模型环境

1. 介绍：支持模型、流、任务、函数元数据及调用记录，能够读取优化前后的真实差异。
2. 输入：任务类型、顺序、参数、kernel 属性、同步关系以及外部接口注错。
3. 处理：提供显式场景构造接口；模型/流查询只枚举，不重建或清空任务。`SetParams` 深拷贝 host 参数、配置属性和 opInfo，`Disable` 记录禁用状态，`Update` 区分尝试和成功次数。参数由模型持有，函数元数据在进程 registry 中保持稳定以匹配生产 binary cache 生命周期。
4. 输出：只读任务快照、launch 记录、update 计数、待释放内存和注册回调数量；这些结果用于 ST 断言。

### 功能需求 3：公开 API 系统场景

1. 介绍：全部 AOT 主机模块通过公开 API 联调。
2. 输入：无 scope、单 scope、多流模型，非法 scope，更新失败，scope 名称，以及公开 verify graph。
3. 处理：测试版库执行完整流程；只在 ACL/Runtime/DFX 和设备入口边界打桩。ST 不使用 mockcpp 替换生产内部函数。
4. 输出：返回值与外部模型变化一致，错误路径停止后续处理，销毁后资源归零。

## 非功能需求

### 可维护性

共享桩使用一个 SO。`depends/model_fixture.*` 管模型、稳定对象和观测记录；设备入口保留在原有 `ut_common_stubs.cpp` 并调用模型记录接口；原有注错控制继续复用。观测记录随模型生命周期维护，不另建 observer 状态。仅增加首批场景实际需要的单 AIV 元数据，不构建通用设备模拟器。

### 可测试性

ST 依赖 gtest 和公开头；mockcpp 仅 UT 使用。为桩自身增加必要 UT：重复查询保留修改、disable 可观察、不同模型隔离、快照深拷贝、销毁/reset 顺序正确。

### 可移植性

使用 C++17，沿用当前主机编译器与 `_GLIBCXX_USE_CXX11_ABI=0`。设备能力从可配置的 stub 查询提供，不在生产逻辑新增芯片特判。首阶段串行运行场景，不承诺线程安全设备仿真。

### 可靠性

模型销毁前保留参数和分配资源；先执行注册的销毁回调并断言，再销毁场景对象，最后 reset 控制状态。reset 不得先清回调或释放全部内存以掩盖生产泄漏。环境变量和日志工作目录由 fixture 恢复；需要重置进程静态初始化的场景用独立进程。

### 特性交叉影响

| 场景 | 适用性 | 结论 |
|---|---|---|
| SuperKernel Python 接口 | 不适用 | 不改变包和 pytest 行为；脚本分发保持原路径 |
| SuperKernel C++/AOT 接口 | 适用 | 新增公开接口 ST，生产 API/ABI 不变 |
| Autofuse 图优化 | 不适用 | 不修改图优化或 pass |
| Autofuse Codegen/Backend | 不适用 | 不修改生成代码和 backend |
| AscendC API / Runtime 交互 | 适用 | 测试桩记录外部调用和生命周期，实际 kernel 执行另测 |
| Python/C++ 混合绑定 | 不适用 | 不涉及绑定层 |
| 构建与打包 | 适用 | ST 开关、共用源码/编译约束、测试库不安装 |
| 测试与覆盖率 | 适用 | UT 回归、ST 独立报告、桩契约测试 |
| 性能与日志 | 适用 | 测试耗时与内存增加，生产路径不新增日志 |
| 兼容性 | 适用 | 保留 UT 默认行为及旧入口，新加 cpp_st 分发 |

## 性能

### 编译时长

ST 开启时多编译一次完整主机源码；UT/ST 共用桩目标，避免重复编译桩。只提取生产源码列表和公共约束，不强行共用不同头环境下的 OBJECT 文件。实现阶段记录首次和增量构建耗时；不开启 ST 时不构建测试版库。

### 执行性能

生产调度、拷贝和日志路径不改。桩只执行主机内存操作和记录，不模拟 kernel 计算、设备时间或等待线程；首批使用小图，设置测试超时，禁止忙等。

### 内存和产物大小

模型和观测快照空间随任务数与参数字节数线性增长。测试 SO 仅位于 build，生产包大小不因测试增加。深拷贝只保留断言需要的参数，销毁后检查分配计数。

## 接口设计

### 新增/修改接口描述

生产 API 原型不改。新增测试支持操作：创建/销毁模型、添加流/任务、注册函数元数据、查询任务快照和 launch 记录。创建返回稳定句柄；查询返回拥有自身数据的快照；非法句柄/容量不足必须显式失败。销毁按注册顺序调用回调后再失效句柄。具体 C++ 签名在实现时按现有 ACL 类型落地，不暴露生产内部类。

构建增加 `ENABLE_CPP_STEST`，可与 `ENABLE_CPP_UTEST` 独立开启；新增 `super_kernel_aot_stest` 和 `run_super_kernel_aot_stest`。新增脚本分发 `superkernel:cpp_st`，支持过滤器与覆盖率。

### 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|---|---|---|---|
| 接口说明 | 是否需要接口评审 | 否 | 不新增生产接口 |
| 接口说明 | 是否需要补充文档 | 是 | 测试构建、桩边界、场景新增方式 |
| 接口兼容 | 行为是否兼容 | 是 | 现有 UT 控制语义回归 |
| 接口兼容 | ABI/API 是否兼容 | 是 | 生产签名与导出宏不变，测试库单独命名 |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | 非法测试输入和缺少依赖及时失败 |
| 接口测试 | 是否需要独立接口用例 | 是 | 四个公开入口均覆盖 |

## 软件设计

### 关键数据结构

场景拥有 model → streams → tasks 的对象树，task 拥有参数存储，function registry 拥有名称/属性/地址缓冲区。句柄在整个调用与清理阶段稳定，容器扩容不移动被引用对象。observer 用 task/model 标识记录操作，不凭裸地址顺序判定结果。

现有 UT 存在自建 `TestRITask` 并转换句柄的做法。迁移时保持已有参数访问布局契约，新增观察信息使用旁表，不能在兼容任务对象尾部直接读写不存在的字段。场景查询接入现有桩，避免为每个 ST 增加一套 API 实现或 ST 模式开关。

### 关键技术/算法

桩仅维护外部 API 最小语义。当前 `aclmdlRIGetTasksByStream` 每次重置默认任务，需改为稳定存储；`aclmdlRITaskDisable` 当前为空操作，需记录结果。查询复杂度 O(任务数)，注册表通过句柄定位对象。读取桩中最终图不能重新调用 optimizer 生成“预期值”。

### 流程设计

准备场景 → 记录输入快照 → 调用公开 API → 读取更新/disable/launch 记录 → 断言结果 → 调用模型销毁 → 断言资源释放 → reset。

ScopeBegin 的生产行为为 begin、placeholder、placeholder；ScopeEnd 为 placeholder、placeholder、end。设备入口 stub 记录顺序、流和名称；集成场景可将这些 marker 追加到场景流，以便后续 Optimize 读取。

更新失败不承诺事务回滚：生产实现可能已修改 task 参数才调用 RIUpdate。ST 检查返回失败、无成功提交记录及销毁后释放资源，不凭空要求原图恢复。

### 对子模块的修改

| 路径 | 修改内容与必要性 |
|---|---|
| `super_kernel/tests/aot/ut/stub/` → `aot/depends/` | 按用户要求外提；保留原头目录布局和控制接口名 |
| `aot/depends/CMakeLists.txt` | 新增共享桩目标，仅导出测试需要的 include |
| `aot/depends/model_fixture.*`、`model_fixture_internal.h` | 稳定场景、结果观测、外部 API 适配 |
| `aot/depends/ut_common_stubs.cpp` | 保留控制接口，接入设备入口记录与共享日志缓冲 |
| `aot/ut/CMakeLists.txt`、相关 include | 链接共享桩，加入桩契约 UT |
| `super_kernel/cmake/aot_host.cmake` | 提取生产/ST 共用完整源码列表和编译/导出约束，外部链接依赖分别配置 |
| `super_kernel/CMakeLists.txt` | 使用共用配置，纳入 ST 开关；测试 target 不依赖真实设备 kernel target |
| `aot/CMakeLists.txt`、`cmake/intf.cmake` | 公共测试配置只初始化一次，分离 ABI/检测配置与 gtest/mockcpp 链接依赖 |
| `aot/st/CMakeLists.txt`、`main.cpp`、`test_optimize_st.cpp`、`test_scope_launch_st.cpp`、`test_scope_verify_st.cpp` | 测试版 SO 和公开 API 场景 |
| `build.sh`、`aot/cmake/func.cmake` | 新增 cpp_st；为报告目录增加参数，UT 保留 cpp_ut 默认，ST 使用 cpp_st |
| `aot/README.md` | 说明运行入口、边界与如何添加场景 |

ST 完整编译 `sk_dump_json.cpp`；现有 UT 用例直接 include 该 cpp，不将完整源码列表盲目追加给 UT，防止重复定义。共用配置含可见性和 `FUNC_VISIBILITY`，ST 自身不定义该宏。ST SO 对内部头使用 PRIVATE，不能沿用生产 target 当前 PUBLIC 内部 include 的传播方式。

### 错误处理

#### 系统错误

桩分配/拷贝检查长度和容量，成功分配才登记；错误注入仅发生在指定外部调用。动态库链接验证未解析符号，运行时检查实际依赖解析位置。模型失败后的资源遵循当前生产生命周期，不能用 fixture 直接释放掩盖缺陷。

#### 接口错误

检查具体公开返回值；首批 Runtime update 失败注入使用已有 ACL_ERROR_FAILURE 控制。非法 scope 检查失败且没有成功 model update。若发现生产缺陷，记录复现和修复范围，不通过桩返回成功绕过。

## 安全检查

已按编码红线逐项检查：测试不新增密钥/公网地址；索引、长度和句柄有边界检查；分配及偏移防溢出；参数内存有效到实际消费结束；显式销毁资源先于 SO 析构；不新增非开放 rt 接口；不改生产 ABI 或图改写规则；不新增芯片/框架特判；不改变高频日志；构建保持离线缓存和增量能力；运行产物位于忽略目录。

## 兼容性检查

UT 源码迁移只改必要 include；保留原函数名及注错默认。测试库无 install 规则，不能进入发布依赖闭包。Python、RDV/设备样例和 Autofuse 原有入口继续保留。PR 最新 smoke CI 仍负责设备验证，本次不将其替换为主机 ST。

## 测试设计

### 测试边界

入口为四个公开 API，出口为返回码、外部 RI 模型变化、设备入口调用及资源记录。内部 graph/split/optimizer/build/update 全部运行生产逻辑。DFX、ACL/Runtime 和生成设备入口打桩。数值正确性、硬件同步、吞吐和正式包 ABI 需继续通过设备/交付测试验证。

### 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|---|---|---|---|
| 功能 | 无 scope 标记的普通任务模型 | 按生产默认规则视为全模型 scope；检查融合入口、参数和成功 update，不能只用空图 | ST |
| 功能 | 单 scope + 两个可融合 kernel | 通过 Begin/End 记录 marker，调用 Optimize；存在有效融合入口和参数、被融合任务按预期处理、成功 update | ST |
| 功能 | 双流、多任务、跨流 record/wait | 使用同一模型，验证两条流的任务更新和同步关联，不只检查调用次数 | ST |
| 功能 | scope 内 record 与 scope 外 wait/reset | record 融入入口，外部 wait/reset 改写成同一内存地址，初值为零，销毁释放同步内存 | ST |
| 异常 | scope begin/end 不配对 | Optimize 失败，无成功提交，模型可清理 | ST |
| 异常 | RIUpdate 失败 | 注入到确实发生融合的场景，确认调用命中、失败传播、销毁释放 | ST |
| 功能 | Begin/End 参数与顺序 | 检查六次 marker 调用、stream、name；非法空名无 launch | ST |
| 功能 | Verify 无拆分 | 使用公开 graph 结构，成功且拆分数量为零 | ST |
| 功能 | Verify 有拆分 | 选取既有已验证规则的输入，核对数量、原输入节点指针、splitType/reason | ST |
| 功能 | cube、MIX 1:1、MIX 1:2 | 外部 KernelSpec 提供核类型、比例、block 数和调度模式，检查真实构建的入口名称、block 数和参数 | ST |
| 功能 | 逐算子调试、跨流同名 scope | 检查独立调试入口和跨流合并结果；STREAM_FUSION=0 按当前告警但仍合并的行为验证 | ST |
| 功能 | 调试产物与选项 | 临时目录中解析生产输出 JSON，核对持久化选项、FUNC 任务顺序与调试标志；恢复环境并禁用文件日志 | ST |
| 异常 | Runtime 枚举、入口解析、同步内存初始化失败 | 检查返回失败且不提交模型；内存初始化失败释放已分配资源 | ST |
| 边界 | Verify 容量、动态核、死锁、非法节点 | 核对所需容量及缓冲区不被覆盖、动态核改变拆分结果、死锁排除 wait、失败清空结果数量 | ST |
| 兼容性 | 同名不同核类型 | 后创建的 kernel 不改变已有 kernel 的核类型、比例、调度模式和 block 数 | UT |
| 兼容性 | 共享桩与原有 mockcpp | 迁移前后原 UT 全量运行，特别检查函数 mock 与回调 | UT |
| 功能 | 桩稳定状态/资源/快照 | 重复查询、修改后查询、多模型和深拷贝契约 | UT |
| 特性交叉 | 共享 SO 状态与公开符号 | ST 设置外部故障后 SO 内调用确实观察到；nm/readelf 检查接口和依赖 | ST/构建检查 |
| 性能 | 首次/增量构建与短场景耗时 | 记录耗时和峰值内存，不据此推断 NPU 性能 | 手工验证 |

单 scope 的精确 task 变更列表在实施首个场景时依据现有公开任务协议固定，并独立写出预期。不能接受仅 `ACL_SUCCESS` 或仅“发生一次 update”的融合成功判据。

### 测试命令

运行入口：

```bash
bash build.sh -u --module=superkernel --impl=cpp -j 8
bash build.sh -s --module=superkernel --impl=cpp -j 8
bash build.sh -s --module=superkernel --impl=cpp -c -j 8
cmake --build build --target run_super_kernel_aot_stest -j 8
```

预期：UT/ST 均通过，ST 覆盖率独立落在 `super_kernel/coverage/cpp_st`。使用实际配置的 build 目录；过滤器沿用已有入口参数。通过 `readelf -d` 和实际加载路径检查 ST 依赖仅落到测试 SO 与允许的主机库；在无 NPU 的执行环境运行全部首批 ST。不开启 ST 的正常打包和 `--no-autofuse` 打包检查测试库均未被安装。

### 主机验证结果

在首版提交 `3d2e2a63` 上增加 11 个 ST 和 1 个桩契约 UT 后，全量 21 个 ST、1086 个 UT 通过，ST 随机顺序重复 10 轮通过。清空 ST 的 gcda 后运行 `collect_coverage_data_cpp_st`，报告如下：

| 指标 | 首版 | 扩展后 |
|---|---|---|
| 生产代码行覆盖率 | 3350/8219（40.8%） | 4318/8219（52.5%） |
| 生产函数覆盖率 | 512/953（53.7%） | 598/953（62.7%） |

两次均统计同一组 39 个 `super_kernel/src/aot` 文件，不计入测试和桩。主要增量来自核类型处理、选项持久化、调试日志/JSON、Verify 校验及死锁处理。DFX 异常回调和 profiling 仍是低覆盖区域；当前结果不代表设备执行或硬件故障流程已验证。

## 验收标准

1. 外提后 UT 全量通过，旧注错及 mockcpp 不受共享库边界影响。
2. 首批场景从四个公开 API 进入；ST 无生产内部头、无内部函数 mock。
3. ST SO 使用完整生产主机源码及共用编译约束，四个公开符号可见，测试程序成功动态链接。
4. 模型查询不清除修改，禁用/更新/launch 可观察；测试进程内桩状态只有一份。
5. 单 scope 真正发生融合；多流场景验证关联；失败场景确实命中注错。
6. 销毁后回调及本次分配归零，重复运行无状态串扰；无 NPU 环境运行通过。
7. cpp_st 可独立运行、过滤和生成报告，正式包不包含测试库。

实施顺序及每步出口：

- P1：原 UT 建立基线 → 原样外提 stub 并形成共享目标 → 原 UT 全量回归。
- P2：提取生产/ST 共用源码和编译约束 → 建测试 SO 和公开 API 最小入口 → 验证符号/依赖及跨 SO 注错状态。
- P3：为稳定查询、变更记录、销毁编写桩契约 UT → 完成最小模型和观察接口 → 契约 UT 通过。
- P4：先 ScopeBegin/End 和 Verify → 再单 scope Optimize → 最后多流及失败清理；每个场景独立断言外部结果。
- P5：接入 build.sh、过滤器与独立覆盖率 → 全 UT/ST 和无 NPU 验证 → 打包隔离检查及文档更新。

### 设计文档检查结果

- [x] 跨特性交叉影响：已逐项覆盖十类场景。
- [x] 编码红线：已检查输入边界、内存/回调生命周期、ABI、构建及图改写确定性约束。
- [x] 测试设计：gtest ST 覆盖四个入口，UT 负责桩契约和原有规则回归，设备数值验证保留。
- [x] 性能评估：额外编译和快照开销仅在测试，生产调度/日志不改。

### 代码检视检查结果

- [x] 编码红线：已检查资源、参数所有权、模型销毁回调和 ABI；生产算法未改动。
- [x] 跨特性交叉影响：已检查 SuperKernel 主机测试、构建和交付边界；Autofuse/Python 行为不变。
- [x] 贡献规范：已检查 CONTRIBUTING.md；首版按用户要求本地提交为 `3d2e2a63`，未推送。覆盖率扩展保留为工作区增量。
- [x] 代码格式：新增 C++ 按用户要求采用 4 空格，使用 clang-format 18.1.8；迁移文件仅格式化本次修改处。
- [x] Pre-commit：已核对配置并执行 OAT；仍有五处原有许可证头拼写问题，未顺带修改：`super_kernel/CMakeLists.txt`、`aot/CMakeLists.txt`、`depends/aprof_pub.h`、`depends/runtime/base.h`、`depends/dump/adump_pub.h`（后三者位于 `super_kernel/tests/aot` 下）。新增文件的许可证检查通过。
- [x] 增量合规：覆盖率扩展的 9 个代码/构建文件 OAT 检查通过。
