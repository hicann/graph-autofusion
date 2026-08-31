#include <gtest/gtest.h>
#include "tikicpulib.h"

#include "autofuse_tiling_data.h"
extern "C" __global__ __aicore__ void load_nan_out_for_store(GM_ADDR x1, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" void GetTiling(AutofuseTilingData &tiling_data);

class E2E_LoadNanOutForStore_Code : public testing::Test, public testing::WithParamInterface<std::vector<int>> {};

TEST_P(E2E_LoadNanOutForStore_Code, CalculateCorrect_Nan) {
  auto test_shape = GetParam();

  uint32_t block_dim = 48;
  int test_size = test_shape[0] * test_shape[1] * test_shape[2];

  AutofuseTilingData tiling_data;
  float *input = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  uint8_t *y = (uint8_t *)AscendC::GmAlloc(test_size * sizeof(uint8_t) + 32);
  uint8_t *expect = (uint8_t *)AscendC::GmAlloc(test_size * sizeof(uint8_t) + 32);

  // Prepare test and expect data
  srand(1);
  for (int i = 0; i < test_size; i++) {
    input[i] = rand() / (float)RAND_MAX;
    expect[i] = std::isnan(static_cast<float>(input[i]));
  }

  // Launch
  tiling_data.block_dim = block_dim;
  tiling_data.s0 = test_shape[0];
  tiling_data.s1 = test_shape[1];
  tiling_data.s2 = test_shape[2];
  tiling_data.tiling_key = 0;
  GetTiling(tiling_data);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_nan_out_for_store, tiling_data.block_dim, (uint8_t *)input, (uint8_t *)y, nullptr,
              (uint8_t *)&tiling_data);

  // Count difference
  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; i++) {
    if (y[i] != expect[i]) {
      diff_count++;
    }
  }

  EXPECT_EQ(diff_count, 0) << " of " << test_size;

  AscendC::GmFree(input);
  AscendC::GmFree(y);
  AscendC::GmFree(expect);
}

INSTANTIATE_TEST_SUITE_P(CalcWithDifferentShape, E2E_LoadNanOutForStore_Code,
                         ::testing::Values(std::vector<int>{2, 8, 8}, std::vector<int>{8, 16, 16},
                                           std::vector<int>{96, 16, 16}));
