/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_RESOURCE_CONTEXT_MGR_H_
#define INC_GRAPH_RESOURCE_CONTEXT_MGR_H_

#include <string>
#include <map>
#include <mutex>
#include "graph/resource_context_af.h"
#include "graph/ge_error_codes_af.h"
#include "graph/node.h"
#include "graph/utils/node_utils.h"

namespace af {
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ResourceContextMgr {
 public:
  ResourceContextMgr() = default;
  ~ResourceContextMgr() = default;
  /**
   * Given resource_key , return corresponding resource pointer
   * @param resource_key
   * @return orresponding resource pointer
   */
  ResourceContext *GetResourceContext(const std::string &resource_key);
  /**
   * Given resource_key , corresponding resource pointer, set resouce_context with new resource
   * @param resource_key
   * @param context
   * @return status
   */
  graphStatus SetResourceContext(const std::string &resource_key, ResourceContext *const context);
  /**
   * Given resource_key , node reiled on this resource, mgr will keep the relation
   * @param resource_key
   * @param node
   * @return status
   */
  graphStatus RegisterNodeReliedOnResource(const std::string &resource_key, NodePtr &node);
  /**
   * Given resource_key , mgr find node reiled on this reousrce.
   * @param resource_key
   * @param read_nodes
   * @return status
   */
  OrderedNodeSet &MutableNodesReliedOnResource(const std::string &resource_key);
  /**
   * Resource context need to be cleared when session finalize
   * @return status
   */
  graphStatus ClearContext();

 private:
  std::mutex ctx_mu_;
  std::map<std::string, std::unique_ptr<ResourceContext>> resource_keys_to_contexts_;
  std::map<std::string, OrderedNodeSet> resource_keys_to_read_nodes_;
};
}  // namespace af
#endif  //  INC_GRAPH_RESOURCE_CONTEXT_MGR_H_
