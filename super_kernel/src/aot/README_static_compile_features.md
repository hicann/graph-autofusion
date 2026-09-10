# 模型粒度选项配置使用说明

入口为 `feature_manager.h`，命名空间为 `sk::static_compile`。本模块负责选项快照、Kernel 过滤和 Basic/SK 参数生成；模型遍历、ACLRTC 编译及 ELF 替换由调用方负责。

## 调用方式

每个模型创建一个 `FeatureManager`，初始化一次，再逐个查询实际的 `kernel_entry`。

```cpp
#include "feature_manager.h"

using sk::static_compile::Feature;
using sk::static_compile::FeatureManager;

const char *argv[] = {"-O2", "-DVALUE=1"};
aclspecOption basic{"addcustom(_.*)?", 2, argv};
aclskOption sync{};
sync.optionType = aclskOptionType::DEBUG_SYNC_ALL;
sync.debugSync.debugSyncAll = 1;
aclskOptions skOptions{&sync, 1};

aclspecOptions options{};
options.jobs = 4;
options.specOptionCount = 1;
options.specOptions = &basic;
options.skOptions = &skOptions;

FeatureManager manager;
if (!manager.Init(&options)) {
    // 配置无效，日志说明具体字段；调用方结束本次优化。
    return false;
}
const auto jobs = manager.Get<Feature::CompileJobs>();
// 在模型的 Kernel 遍历中执行下面三个查询。
const std::string_view kernelEntry = "addcustom_tiling";
const bool enabled = manager.Get<Feature::CompileEnabled>(kernelEntry);
const auto basicArgs = manager.Get<Feature::BasicCompileOptions>(kernelEntry);
const auto skArgs = manager.Get<Feature::SkCompileOptions>(kernelEntry);
// enabled=true；basicArgs={"-O2", "-DVALUE=1"}。
// skArgs={"--enable-super-kernel", "-D__DEBUG_SYNC_ALL__"}。
```

调用方先检查 `enabled`，仅为允许静态编译的 Kernel 获取参数并准备编译请求。获取参数本身不隐式执行过滤。

`kernel_spec_options.h` 暂时提供内部输入结构，未新增导出 C 接口或修改原有 `aclskOptions` ABI。后续正式 `aclSpecOptimize` 头文件落地时，应统一使用正式声明，避免两份结构定义并存。

## 规则与生命周期

| 内容 | 行为 |
| --- | --- |
| Kernel 匹配 | ECMAScript 正则，区分大小写，对完整入口执行 `regex_match`；Init 时预编译 |
| 后缀匹配 | `addcustom(_.*)?` 匹配原名及带下划线后缀；包含匹配用 `.*addcustom.*`，不能写 glob 风格的 `**addcustom**` |
| Basic 合并 | 所有命中规则按输入顺序追加参数，保留重复参数，每个字符串是一个 argv token |
| 默认过滤 | 未提供名单时全部允许；显式空 ALLOW 拒绝全部，显式空 DENY 允许全部 |
| 名单组合 | 同类型名单合并；ALLOW/DENY 同时配置直接初始化失败，即使名单为空 |
| jobs | 0 或空输入时取逻辑 CPU 数，获取失败回退 1；正数原值返回 |
| SK 开启 | 解析产生有效参数时，在 SK 列表首位添加一次 `--enable-super-kernel` |
| SK 重复与未知项 | 重复支持项、非法枚举或非法开关值失败；合法但尚不支持的类型每次 Init 按类型记录一次 INFO 后忽略 |
| 输入所有权 | Init 期间输入必须有效且不变；成功后调用方可释放输入，Get 返回值也不引用内部存储 |
| 重新初始化 | 成功替换全部配置；返回 false 或抛异常都保留旧成功配置 |
| 查询错误 | 未初始化抛 `logic_error`，空 Kernel 名抛 `invalid_argument`；分配等系统异常向上传播 |
| 并发 | 多个 Get 可以并发只读；Init、析构必须与所有查询互斥 |

SK 当前支持：

| 输入 | 生成参数 |
| --- | --- |
| `DEBUG_SYNC_ALL=1` | `-D__DEBUG_SYNC_ALL__`；0 不生成 |
| `DEBUG_PER_OP_MAX_CORE_NUM=1` | `-D__ENABLE_SUPER_KERNEL_INNER_CORE_SYNC_CHECK__`；0 不生成 |
| 出现 `DCCI_DISABLE_ON_KERNEL` | 所有 Kernel 生成 `--cce-no-dcache-flush`，不读取其 kernelNames |

两项宏沿用当前详设的暂定映射。单测验证参数生成，不代表已验证目标编译器的实际优化效果。DCCI 转换不修改资源的 SK_BIND capability。

最外层 C API 应捕获异常并映射项目既有错误码，不让 C++ 异常穿过 C ABI。

## 实现与扩展

`feature_manager.cpp` 注册 Feature、Provider 类型和查询方法。Basic/SK 绑定到同一个 `CompileOptions` 实例，每次 Init 只初始化一次。查询通过当前 State 的槽位转发，不捕获旧配置对象。

内部统一使用 `QueryFeature` 和有限类型的 `FeatureValue`，调用方仍从 `Get<F>` 直接拿到 `uint64_t`、`bool` 或参数列表。模型/Kernel 作用域与结果类型独立，新增模型 bool 或 Kernel 数字查询不需要再添加 Manager 桥接函数。`Init` 标注 `[[nodiscard]]`，忽略初始化结果时编译器会提示。

`kernel_compile_features.cpp` 实现 `ModelSettings`、`KernelFilter`、`CompileOptions` 和共用的 `KernelMatcher`。业务类不继承公共基类；Manager 内部 Holder 仅用于统一保存不同 Provider 的生命周期。

- 新增具体 SK option：实现转换函数，并将 `SK_OPTIONS` 中对应项的空函数替换为它；全新公共枚举在同一张表增加一项。表同时负责合法性识别、名称诊断和转换登记，编译期检查重复登记。当前转换器输出全局参数，后续若支持 Kernel 范围，应在 CompileOptions 内增加带 Matcher 的 SK 规则，Manager 无须修改。
- 新增查询能力：增加 `Feature`、`FeatureTraits`、Provider 查询方法及一条 `Register` 绑定。相同 Provider 自动复用。
- 新增返回类型：只在确实超出现有三种结果类型时扩展 `FeatureValue`，再增加业务查询和注册；现有无参数/Kernel 名两种查询形式共用统一转发。未预设无实际输入来源的 bool/number 配置，也没有动态注册接口。
- 修改名单或正则语义：只修改 KernelFilter 或 KernelMatcher，并补充相应测试。

## 验证入口

新增源文件位于现有 `super_kernel/CMakeLists.txt` 的 AOT 源码收集范围内。测试也接入了 `super_kernel_aot_utest`。

另有独立 Host 单测入口，复用同一份测试及仓库 ACL/dlog stub，无需构建设备 Kernel：

```bash
cmake -S super_kernel/tests/aot/ut/feature_manager -B build/feature_manager \
    -DGTest_DIR=<本机GTestConfig.cmake所在目录>
cmake --build build/feature_manager -j 8
ctest --test-dir build/feature_manager --output-on-failure -j 8
```

独立入口与原 AOT 目标一样使用 `_GLIBCXX_USE_CXX11_ABI=0`，GTest 库应使用相同 ABI。

配置阶段另外检查查询类型契约：正确调用可编译；错误作用域、查询 Count、忽略 Init 结果（启用对应 Werror）必须编译失败。这些检查使用静态库编译，避免把未链接 FeatureManager 实现误判成预期的接口错误。
