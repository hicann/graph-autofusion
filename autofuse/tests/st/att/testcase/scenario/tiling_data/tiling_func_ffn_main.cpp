/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <iostream>
#include "FFN_tiling_data.h"
using namespace optiling;
bool TestCase(std::vector<int64_t> shapes) {
  int64_t maxTokens = shapes[0];
  int64_t N1 = shapes[1];
  int64_t K1 = shapes[2];
  int64_t N2 = shapes[3];
  FFNTilingData tilingData;
  tilingData.set_block_dim(48);
  tilingData.set_ub_size(240 * 1024);
  tilingData.set_btbuf_size(1 * 1024);
  tilingData.set_maxTokens(maxTokens);
  tilingData.set_N1(N1);
  tilingData.set_K1(K1);
  tilingData.set_N2(N2);
  std::cout << "maxTokens" << " = " << maxTokens << std::endl;
  std::cout << "N1" << " = " << N1 << std::endl;
  std::cout << "K1" << " = " << K1 << std::endl;
  std::cout << "N2" << " = " << N2 << std::endl;

  const auto status = GetTiling(tilingData, 0u);
  if ((status)) {
    std::cout << "ub_m" << " = " << tilingData.get_ub_m() << std::endl;
    std::cout << "base_n1" << " = " << tilingData.get_base_n1() << std::endl;
    std::cout << "base_n2" << " = " << tilingData.get_base_n2() << std::endl;
    return true;
  }
  std::cout << "ffn tiling func execute failed." << std::endl;
  return false;
}

int main() {
  bool ret = true;
  ret &= TestCase({1939, 2560, 5120, 5120});
  return ret ? 0 : -1;
}
