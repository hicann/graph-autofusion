/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "BrcBuf_tiling_data.h"
using namespace optiling;

void PrintResult(graph1TilingData &tilingData) {
  std::cout << "====================================================" << std::endl;
  auto tiling_key = tilingData.get_tiling_key();
  std::cout << "get_tiling_key" << " = " << tiling_key << std::endl;
  std::cout << "get_block_dim" << " = " << tilingData.get_block_dim() << std::endl;
  std::cout << "get_ub_size" << " = " << tilingData.get_ub_size() << std::endl;
  std::cout << "get_Z0" << " = " << tilingData.get_Z0() << std::endl;
  std::cout << "get_Z1" << " = " << tilingData.get_Z1() << std::endl;
  std::cout << "get_Z2" << " = " << tilingData.get_Z2() << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph1TilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.set_Z0(100);
  tilingData.set_Z1(200);
  tilingData.set_Z2(400);
  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
    if (tilingData.get_tiling_key() != 1101u) {
      std::cout << "tiling key 1101 (brcbuf case) should be better, but got tiling key " << tilingData.get_tiling_key()
                << "." << std::endl;
      return -1;
    }
  } else {
    std::cout << "brcbuf tiling func execute failed." << std::endl;
    return -1;
  }

  return 0;
}
