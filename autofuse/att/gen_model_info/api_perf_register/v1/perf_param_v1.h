/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_UTIL_PARAMS_V1_H_
#define ATT_UTIL_PARAMS_V1_H_

#include <string>
#include <map>
#include "gen_model_info/api_perf_register/perf_param.h"

namespace att {
class PerfParamTableV1 : public PerfParamTable {
 public:
  PerfParamTableV1();
  ~PerfParamTableV1() override = default;
  [[nodiscard]] const std::string *GetAscendCApiPerfTable() const override;
  [[nodiscard]] PipeHeadPerfFunc GetPipeHeadPerfFunc(PipeType pipe_type) const override;
  [[nodiscard]] Expr GetOpHeadCost() const override;
  [[nodiscard]] uint32_t GetMicroApiLen() const override {
    return arch_param::kV1ArchHardwareConfig.vector_len_size;
  }
  [[nodiscard]] static Expr GetMTE2PipeHead(const std::vector<NodeInfo> &node_infos,
                                            std::map<Expr, TernaryOp, ExprCmp> &ternary_ops);

 private:
  std::map<PipeType, PipeHeadPerfFunc> pipes_head_perf;
};

class TilingScheduleConfigTableV1 : public TilingScheduleConfigTable {
 public:
  [[nodiscard]] bool IsEnableBlockLoopAutoTune() const override {
    return false;
  }
  [[nodiscard]] bool IsEnableCacheLineCheck() const override {
    return false;
  }
  [[nodiscard]] TradeOffConfig GetTradeOffConfig() const override {
    return TradeOffConfig{false};
  }
  [[nodiscard]] double GetUbThresholdPerfValEffect() const override {
    constexpr double kDefaultUbThresholdPerfValEffect = 0.19;
    return kDefaultUbThresholdPerfValEffect;
  }
  [[nodiscard]] TilingScheduleConfig GetModelTilingScheduleConfig() const override {
    TilingScheduleConfig config;
    config.trade_off_config = GetTradeOffConfig();
    config.cache_line_size = GetCacheLineSize();
    config.vector_len_size = GetVectorLenSize();
    return config;
  }
  [[nodiscard]] uint32_t GetCacheLineSize() const override {
    return arch_param::kV1ArchHardwareConfig.cache_line_size;
  }
  [[nodiscard]] uint32_t GetVectorLenSize() const override {
    return arch_param::kV1ArchHardwareConfig.vector_len_size;
  }
  // 新增：V1形态不使能Reduce分核惩罚
  [[nodiscard]] bool IsCoreNumThresholdPenaltyEnable() const override {
    return false;
  }
};

class TilingScheduleConfigTableV1HeavyOp : public TilingScheduleConfigTableV1 {
 public:
  [[nodiscard]] bool IsEnableBlockLoopAutoTune() const override {
    return false;
  }
  [[nodiscard]] TradeOffConfig GetTradeOffConfig() const override {
    return {true, af::Symbol(0.1), af::Symbol(0.4)};
  }
  [[nodiscard]] TilingScheduleConfigPriority GetConfigPriority() const override {
    return TilingScheduleConfigPriority::kHeavyOpPriority;
  }
};
}  // namespace att
#endif  // ATT_UTIL_PARAMS_V1_H_
