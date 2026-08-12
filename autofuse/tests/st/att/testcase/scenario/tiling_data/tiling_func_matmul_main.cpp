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
#include "Matmul_tiling_data.h"
using namespace optiling;
namespace optiling {
bool GetTiling(MMTilingData &tiling_data, int32_t tiling_case_id, double *perf);
}
bool TestCase(std::vector<int64_t> shapes) {
  int64_t m = shapes[0];
  int64_t k = shapes[1];
  int64_t n = shapes[3];
  MMTilingData tilingData;
  tilingData.set_block_dim(20);
  tilingData.set_l1_size(512 * 1024);
  tilingData.set_m_size(m);
  tilingData.set_k_size(k);
  tilingData.set_n_size(n);
  std::cout << "m" << " = " << m << std::endl;
  std::cout << "k" << " = " << k << std::endl;
  std::cout << "n" << " = " << n << std::endl;

  const auto status = GetTiling(tilingData, 1u, nullptr);
  if ((status)) {
    std::cout << "step_ka" << " = " << tilingData.get_stepka_size() << std::endl;
    std::cout << "step_kb" << " = " << tilingData.get_stepkb_size() << std::endl;
    return true;
  }
  std::cout << "mm tiling func execute failed." << std::endl;
  return false;
}

int main() {
  bool ret = true;
  ret &= TestCase({1536, 1536, 1536, 12288});
  ret &= TestCase({320, 1848, 1848, 1024});
  ret &= TestCase({12288, 6144, 6144, 1536});
  ret &= TestCase({1357, 8192, 8192, 5464});
  ret &= TestCase({1848, 1024, 1024, 320});
  ret &= TestCase({284, 8192, 8192, 60000});
  ret &= TestCase({10240, 576, 576, 1280});
  ret &= TestCase({3072, 6144, 6144, 8192});
  ret &= TestCase({8192, 12288, 12288, 7808});
  return ret ? 0 : -1;
}
