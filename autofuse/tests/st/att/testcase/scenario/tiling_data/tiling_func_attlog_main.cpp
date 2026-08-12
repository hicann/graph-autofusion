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
#include "OpTest6_tiling_data.h"
using namespace optiling;
int TestCase1(uint64_t m, uint64_t n, int32_t tilingCaseId) {
  MMTilingData tilingData;
  tilingData.set_m_size(m);
  tilingData.set_n_size(n);
  tilingData.set_block_dim(20);
  tilingData.set_l1_size(512 * 1024);
  // tilingData.z = 0;
  const auto status = GetTiling(tilingData, tilingCaseId);
  if ((status)) {
    std::cout << "Case select tiling func execute success." << std::endl;
    return 0;
  }
  std::cout << "Case select tiling func execute failed." << std::endl;
  return -1;
}

int main(int argc, char *argv[]) {
  uint64_t m = std::stoi(argv[1]);
  uint64_t n = std::stoi(argv[2]);
  int32_t tilingCaseId = std::stoi(argv[3]);
  auto ret1 = TestCase1(m, n, tilingCaseId);
  return 0;
}
