#include <gtest/gtest.h>

#include "tikicpulib.h"
#include "autofuse_tiling_data.h"

extern "C" __global__ __aicore__ void load_scalar_brc_store_2d(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

TEST(LoadScalarAbsBrcStore2D, CalculateCorrect) {
  constexpr int s0 = 4;
  constexpr int s1 = 4;
  constexpr int test_size = s0 * s1;
  AutofuseTilingData tiling_data;
  float *x = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  float *y = (float *)AscendC::GmAlloc(test_size * sizeof(float) + 32);
  for (int i = 0; i < test_size; ++i) {
    x[i] = -1.0F;
  }

  tiling_data.block_dim = 1;
  tiling_data.s0 = s0;
  tiling_data.s1 = s1;
  tiling_data.tiling_key = 0;
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(load_scalar_brc_store_2d, tiling_data.block_dim, (uint8_t *)x, (uint8_t *)y, nullptr,
              (uint8_t *)&tiling_data);

  uint32_t diff_count = 0;
  for (int i = 0; i < test_size; ++i) {
    if (y[i] != -1.0F) {
      ++diff_count;
    }
  }
  EXPECT_EQ(diff_count, 0) << " of " << test_size;
  AscendC::GmFree(x);
  AscendC::GmFree(y);
}
