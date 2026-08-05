/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "tikicpulib.h"
#include "autofuse_tiling_data.h"

struct ResLimit {
  uint32_t valid_num = 0;
  uint32_t aiv_num = 0;
  uint32_t aic_num = 0;
  uint32_t ub_size = 0;
  uint32_t resv[10];
};

extern "C" int64_t GenerateTopnSolutions(const std::vector<std::map<std::string, std::string>> &input_configs,
                                         int64_t topn, std::vector<AutofuseTilingData> &tiling_datas,
                                         std::vector<int64_t> &workspaces, std::vector<int64_t> &block_dims,
                                         ResLimit *res_limit = nullptr);
extern "C" int64_t AutofuseTiling(AutofuseTilingData *tiling, uint32_t *workspaceSize, uint32_t *blockDim,
                                  ResLimit *res_limit = nullptr);
std::string GetTilingDataRepr(const AutofuseTilingData *tiling_data);

class E2EBackendInductorTopnCode : public testing::Test {};

TEST_F(E2EBackendInductorTopnCode, GenerateTopnSolutionsTop1MatchesAutofuseTiling) {
  ResLimit res_limit = {1, 48, 0, 192 * 1024, {0}};
  const std::vector<std::map<std::string, std::string>> input_configs;
  std::vector<AutofuseTilingData> tiling_datas;
  std::vector<int64_t> workspaces;
  std::vector<int64_t> block_dims;

  AutofuseTilingData default_tiling_data = {};
  uint32_t default_workspace = 0;
  uint32_t default_block_dim = 0;
  ASSERT_EQ(AutofuseTiling(&default_tiling_data, &default_workspace, &default_block_dim, &res_limit), 0);

  ASSERT_EQ(GenerateTopnSolutions(input_configs, 1, tiling_datas, workspaces, block_dims, &res_limit), 0);
  ASSERT_EQ(tiling_datas.size(), 1U);
  ASSERT_EQ(workspaces.size(), 1U);
  ASSERT_EQ(block_dims.size(), 1U);
  EXPECT_EQ(GetTilingDataRepr(&tiling_datas[0]), GetTilingDataRepr(&default_tiling_data));
  EXPECT_EQ(workspaces[0], static_cast<int64_t>(default_workspace));
  EXPECT_EQ(block_dims[0], static_cast<int64_t>(default_block_dim));
}

TEST_F(E2EBackendInductorTopnCode, GenerateTopnSolutionsRejectsInvalidTopn) {
  ResLimit res_limit = {1, 48, 0, 192 * 1024, {0}};
  const std::vector<std::map<std::string, std::string>> input_configs;
  std::vector<AutofuseTilingData> tiling_datas;
  std::vector<int64_t> workspaces;
  std::vector<int64_t> block_dims;

  EXPECT_EQ(GenerateTopnSolutions(input_configs, 0, tiling_datas, workspaces, block_dims, &res_limit), -1);
  EXPECT_TRUE(tiling_datas.empty());
  EXPECT_TRUE(workspaces.empty());
  EXPECT_TRUE(block_dims.empty());
}

TEST_F(E2EBackendInductorTopnCode, GenerateTopnSolutionsReturnsDistinctCandidates) {
  ResLimit res_limit = {1, 48, 0, 192 * 1024, {0}};
  const std::vector<std::map<std::string, std::string>> input_configs;
  std::vector<AutofuseTilingData> tiling_datas;
  std::vector<int64_t> workspaces;
  std::vector<int64_t> block_dims;

  AutofuseTilingData default_tiling_data = {};
  uint32_t default_workspace = 0;
  uint32_t default_block_dim = 0;
  ASSERT_EQ(AutofuseTiling(&default_tiling_data, &default_workspace, &default_block_dim, &res_limit), 0);

  constexpr int64_t topn = 4;
  ASSERT_EQ(GenerateTopnSolutions(input_configs, topn, tiling_datas, workspaces, block_dims, &res_limit), 0);
  ASSERT_GT(tiling_datas.size(), 1U);
  ASSERT_EQ(tiling_datas.size(), workspaces.size());
  ASSERT_EQ(tiling_datas.size(), block_dims.size());
  ASSERT_LE(static_cast<int64_t>(tiling_datas.size()), topn);

  std::vector<std::string> reprs;
  reprs.reserve(tiling_datas.size());
  for (const auto &tiling_data : tiling_datas) {
    reprs.push_back(GetTilingDataRepr(&tiling_data));
  }
  EXPECT_EQ(reprs[0], GetTilingDataRepr(&default_tiling_data));
  EXPECT_EQ(workspaces[0], static_cast<int64_t>(default_workspace));
  EXPECT_EQ(block_dims[0], static_cast<int64_t>(default_block_dim));
  for (size_t i = 1; i < reprs.size(); ++i) {
    EXPECT_NE(reprs[i], reprs[0]);
    for (size_t j = 0; j < i; ++j) {
      EXPECT_NE(reprs[i], reprs[j]);
    }
  }
}
