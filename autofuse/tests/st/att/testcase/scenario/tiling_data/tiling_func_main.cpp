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
#include "OpTest0_tiling_data.h"
using namespace optiling;
int TestCase0() {
  MMTilingData tilingData;
  tilingData.set_m_size(1024);
  tilingData.set_n_size(2048);
  tilingData.set_block_dim(20);
  tilingData.set_l1_size(512 * 1024);
  // tilingData.z = 0;
  const auto status = GetTiling(tilingData, 0);
  if ((status)) {
    std::cout << "stepn" << " = " << tilingData.get_stepn_size() << std::endl;
    std::cout << "stepm" << " = " << tilingData.get_stepm_size() << std::endl;
    std::cout << "tiling_key" << " = " << tilingData.get_tiling_key() << std::endl;
    std::cout << "Case0 tiling func execute success." << std::endl;
    return 0;
  }
  std::cout << "Case0 tiling func execute failed." << std::endl;
  return -1;
}

int TestCase1() {
  MMTilingData tilingData;
  tilingData.set_m_size(8192);
  tilingData.set_n_size(2048);
  tilingData.set_k_size(2048);
  tilingData.set_block_dim(20);
  tilingData.set_l1_size(512 * 1024);
  // tilingData.z = 0;
  const auto status = GetTiling(tilingData, 1);
  if ((status)) {
    std::cout << "stepka" << " = " << tilingData.get_stepka_size() << std::endl;
    std::cout << "stepkb" << " = " << tilingData.get_stepkb_size() << std::endl;
    std::cout << "tiling_key" << " = " << tilingData.get_tiling_key() << std::endl;
    std::cout << "Case1 tiling func execute success." << std::endl;
    return 0;
  }
  std::cout << "Case1 tiling func execute failed." << std::endl;
  return -1;
}
int main() {
  auto ret0 = TestCase0();
  auto ret1 = TestCase1();
  if (ret0 == 0 && ret1 == 0) {
    return 0;
  }
  return -1;
}
