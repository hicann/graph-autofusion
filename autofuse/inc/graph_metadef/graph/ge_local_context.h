/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_GE_LOCAL_CONTEXT_H_
#define INC_GRAPH_GE_LOCAL_CONTEXT_H_

#include <map>
#include <string>
#include "graph/ge_error_codes_af.h"
#include "graph/option/optimization_option.h"

namespace af {
class GEThreadLocalContext {
 public:
  graphStatus GetOption(const std::string &key, std::string &option);
  void SetGraphOption(std::map<std::string, std::string> options_map);
  void SetSessionOption(std::map<std::string, std::string> options_map);
  void SetGlobalOption(std::map<std::string, std::string> options_map);
  graphStatus SetOptionNameMap(const std::string &option_name_map_json);
  const std::string &GetReadableName(const std::string &key);

  void SetStreamSyncTimeout(const int32_t timeout);
  void SetEventSyncTimeout(const int32_t timeout);
  int32_t StreamSyncTimeout() const;
  int32_t EventSyncTimeout() const;
  OptimizationOption &GetOo();

  std::map<std::string, std::string> GetAllGraphOptions() const;
  std::map<std::string, std::string> GetAllSessionOptions() const;
  std::map<std::string, std::string> GetAllGlobalOptions() const;
  std::map<std::string, std::string> GetAllOptions() const;

 private:
  std::map<std::string, std::string> graph_options_;
  std::map<std::string, std::string> session_options_;
  std::map<std::string, std::string> global_options_;
  std::map<std::string, std::string> option_name_map_;
  int32_t stream_sync_timeout_ = -1;
  int32_t event_sync_timeout_ = -1;
  OptimizationOption optimization_option_;
};  // class GEThreadLocalContext

GEThreadLocalContext &GetThreadLocalContext();
}  // namespace af
#endif  // INC_GRAPH_GE_LOCAL_CONTEXT_H_
