/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SK_EVENT_RECORDER_H
#define SK_EVENT_RECORDER_H

#include <atomic>
#include <condition_variable>
#include <pthread.h>
#include <cstdint>
#include <unordered_map>
#include <string>
#include <mutex>
#include <memory>
#include <vector>
#include "sk_types.h"
#include "sk_common.h"
#include "sk_file_guard.h"
#include "sk_scope_postprocess.h"

// ==================== Node 信息结构体 ====================
struct SkNodeInfo {
  std::string nodeName;
  uint32_t numBlocks{0};
};

// ==================== 常量定义 ====================
constexpr uint32_t SK_EVENT_MAX_DEVICE_NUM = 16;
constexpr uint32_t SK_EVENT_DEFAULT_CORE_SIZE = 1024 * 1024;  // 默认每个 core 1MB
constexpr uint32_t SHAPE_MAX_TENSOR_NUM = 800;

// 打点环境变量名称
static const char *ENV_SK_EVENT_RECORD = "ASCEND_PROF_SK_ON";
constexpr uint32_t SPRINT_LEN_BUFFER = 1024;

// ==================== Device 上下文 ====================
class SkEventRecorder;
// GM 内存释放函数，用于 unique_ptr 自定义 deleter
inline void GmAddrDeleter(void *ptr) {
  if (ptr != nullptr) {
    aclrtFree(ptr);
  }
}

struct SkEventDeviceCtx {
  std::atomic_int active{0};                                                       // 默认不激活
  std::unique_ptr<void, decltype(&GmAddrDeleter)> gmAddr{nullptr, GmAddrDeleter};  // GM 内存地址 (RAII)
  std::unique_ptr<uint8_t[]> hostBuf;                                              // Host 缓冲区 (RAII)
  uint32_t deviceId = 0;                                                           // Device ID
  uint32_t totalSize = 0;                                                          // 总大小
  std::string outputDir;                // 每个device的profiling输出目录路径
  FileGuard outputFp;                   // 小算子的时间信息文件的输出文件句柄 (RAII)
  std::vector<uint64_t> lastOffset;     // 每个core的上次读取位置（运行时大小）
  SkEventRecorder *recorder = nullptr;  // 回调指针
};

// ==================== 全局管理器 ====================
class SkEventRecorder {
 public:
  static SkEventRecorder &Instance();

  // 初始化（检查环境变量，如果开启则初始化）
  aclError Init();

  // 为指定 device 分配 GM 空间并返回地址
  aclError GetGmAddrForDevice(uint32_t deviceId, void *&gmAddr);

  // 停止并回收后台线程，释放 host 搬运资源
  void SkProfilingShutdown();

  // 是否启用
  bool IsEnabled() const {
    return enabled;
  }

  static uint32_t GetCoreSize() {
    return coreSize_;
  }
  static uint32_t GetTotalSize() {
    return totalSize_;
  }

  // 添加 modelId -> skId -> nodeId -> (nodeName, numBlocks) 映射（线程安全）
  void AddNodeInfoMapping(const std::string &modelId, uint32_t skId, uint32_t nodeId, const std::string &nodeName,
                          uint32_t numBlocks);

  // 获取 NodeInfo（线程安全）
  SkNodeInfo GetNodeInfo(const std::string &modelId, uint32_t skId, uint32_t nodeId) const;

  // 添加 skName 映射（线程安全）
  void AddSkNameMapping(const std::string &modelId, uint32_t skId, const std::string &skName);

  // 获取 skName（线程安全）
  std::string GetSkName(const std::string &modelId, uint32_t skId) const;

  // 注册 modelId 并返回 index（线程安全，不依赖 profiling 开关）
  uint16_t RegisterModelId(const std::string &modelId);

  // 通过 index 获取原始 modelId（线程安全，不依赖 profiling 开关）
  std::string GetModelIdByIndex(uint16_t index) const;

  // 打印所有 modelId index 映射表（线程安全）
  void PrintModelIdIndexMap() const;

 private:
  SkEventRecorder() = default;
  ~SkEventRecorder();

  // 创建输出目录 sk_meta/<pid>，返回目录路径
  static std::string CreateOutputDir();

  // 根据环境变量计算并设置 coreSize 和 totalSize，返回是否启用
  static bool ParseEnvAndSetSize();

  // 创建 device 上下文（带加锁）
  SkEventDeviceCtx *CreateDeviceCtx(uint32_t deviceId);
  bool InitDeviceOutputFile(SkEventDeviceCtx *ctx, uint32_t deviceId);

  // 后台线程入口
  static void *DumpThreadFunc(void *arg);

  void DumpModelData(SkEventRecorder *recorder);

  void ReportError(aclError error);
  // 拷贝并解析 GM 数据
  void DumpDeviceData(SkEventDeviceCtx *ctx);
  bool DumpEventRecord(SkEventDeviceCtx *ctx, const SkKernelEventRecord *record, uint32_t core);

  // 写入单个节点事件到 JSON 文件
  bool WriteNodeEventToJson(SkEventDeviceCtx *ctx, const SkKernelEventRecord *record, uint32_t core,
                            const SkNodeInfo &nodeInfo);

  // 输出 sk 级别统计事件到 JSON 文件
  bool WriteSkEventToJson(SkEventDeviceCtx *ctx, const SkKernelEventRecord *record, uint32_t core);

  // 将输出文件复制到 profiling 路径
  void CopyOutputToProfPath(SkEventDeviceCtx *ctx);

  // 设置 g_profSignal,代表profiling开启状态
  static void SetProfSignal(uint32_t val);

 private:
  std::atomic_bool enabled{false};            // 打点是否执行
  std::atomic_bool globalRunning{false};      // 全局打点解析线程运行状态
  std::atomic<aclError> error_{ACL_SUCCESS};  // 首次失败，区分正常未开启和故障停用
  std::mutex lifecycleMutex_;                 // 串行化线程创建和 join，获取顺序先于 mutex
  std::condition_variable startupCv_;
  bool startupDone_ = false;     // 由 mutex 保护，后台线程已完成 SetDevice
  bool threadStarted_ = false;   // 由 lifecycleMutex_ 保护，线程已创建但尚未 join
  pthread_t dumpThread{};        // 全局打点解析后台线程
  std::once_flag initFlag_;      // 保证 Init() 创建后台线程只执行一次
  std::mutex mutex;              // 保护设备上下文创建和 startupDone_
  std::mutex profBasePathMutex;  // 保护 profBasePath 的互斥锁
  std::string profBasePath;      // 缓存的 profiling 输出路径
  SkEventDeviceCtx deviceCtxs;   // device上下文列表
  int32_t dumpDeviceId = 0;      // dump线程的的device ID

  static uint32_t coreSize_;  // 每个 core 的profiling 记录的gm缓冲区大小（字节），由环境变量决定
  static uint32_t totalSize_;  // 总缓冲区大小 = SkRuntimeConfig::eventCoreNum * coreSize_

  // NodeInfo 映射表：modelId -> skId -> nodeId -> NodeInfo
  mutable std::mutex nodeInfoMapMutex;
  std::unordered_map<std::string, std::unordered_map<uint32_t, std::unordered_map<uint32_t, SkNodeInfo>>> nodeInfoMap;

  // SkName 映射表：modelId -> skId -> skName
  std::unordered_map<std::string, std::unordered_map<uint32_t, std::string>> skNameMap;

  // modelId index 映射表：index -> modelId（不依赖 profiling 开关，始终可用）
  mutable std::mutex modelIdIndexMapMutex;
  std::vector<std::string> modelIdIndexMap;                     // index -> modelId
  std::unordered_map<std::string, uint16_t> modelIdToIndexMap;  // modelId -> index（用于去重）
};

// ==================== sk profiling 性能分析相关函数 ====================
// 根据内核类型获取入口函数名
const char *GetEntryFuncNameByOpType(SkKernelType &opType);

// 性能分析处理
bool SkProfiling(const SuperKernelScopeInfo &scopeInfo, SkLaunchInfo &launchInfo, SuperKernelGraph &graph);

// 详细性能分析数据输出
bool DumpProfilingDetail(const std::vector<SuperKernelBaseNode *> &taskNodes, SkLaunchInfo &launchInfo,
                         const SuperKernelScopeInfo &scopeInfo, const SuperKernelGraph &graph);

std::string GetSkFuncName(const std::vector<SuperKernelBaseNode *> &nodes, uint16_t scopeId,
                          const std::string &scopeName);

inline bool CoreIsAiv(int coreId) {
  if (GetSkRuntimeConfig().kernelArch == SkKernelArch::DAV_3510) {
    return (coreId >= 18 && coreId <= 53) || (coreId >= 72 && coreId <= 107);
  }
  return coreId >= 25;
}

#endif  // SK_EVENT_RECORDER_H
