/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * test_check_bounds.cpp
 */

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_api_utils.h"
// 保持在utils.h之后
#include "api_regbase/check_bounds.h"

using namespace AscendC;

namespace af {

template <typename T>
struct CheckBoundsInputParam {
  T *sanitized{};
  uint32_t *error{};
  T *indices{};
  T *sizes{};
  T *expSanitized{};
  uint32_t *expError{};
  uint32_t calCount{0};
  uint32_t errorCapacity{0};
  bool checkLower{true};
  bool checkUpper{true};
};

class TestApiCheckBoundsUT : public testing::Test {
 protected:
  // -------------------------------------------------------
  //  Helper: invoke kernel
  // -------------------------------------------------------
  template <typename T>
  static void InvokeKernel(CheckBoundsInputParam<T> &param) {
    TPipe tpipe;
    TBuf<TPosition::VECCALC> sanitizedBuf, errorBuf, indicesBuf, sizesBuf, tmpBuf;
    // DataCopy (used by the NoCheck path) requires each tensor to be at least
    // ONE_REPEAT_BYTE_SIZE (256) bytes, and InitBuffer rejects size 0.
    // Use a single floor that satisfies both constraints for all buffers.
    constexpr uint32_t kMinElemCount = ONE_REPEAT_BYTE_SIZE / sizeof(T);
    uint32_t vlCount = GetVecLen() / sizeof(T);
    uint32_t alignedTCount = AlignUp(param.calCount, vlCount);
    if (alignedTCount < kMinElemCount) alignedTCount = kMinElemCount;
    bool checkErrorGuard = param.calCount > 0 && param.errorCapacity > param.calCount;
    uint32_t vlCountU32 = GetVecLen() / sizeof(uint32_t);
    uint32_t errorBufCount = AlignUp(checkErrorGuard ? param.errorCapacity : param.calCount, vlCountU32);
    constexpr uint32_t kMinElemCountU32 = ONE_REPEAT_BYTE_SIZE / sizeof(uint32_t);
    if (errorBufCount < kMinElemCountU32) errorBufCount = kMinElemCountU32;
    tpipe.InitBuffer(sanitizedBuf, sizeof(T) * alignedTCount);
    tpipe.InitBuffer(errorBuf, sizeof(uint32_t) * errorBufCount);
    tpipe.InitBuffer(indicesBuf, sizeof(T) * alignedTCount);
    tpipe.InitBuffer(sizesBuf, sizeof(T) * alignedTCount);
    tpipe.InitBuffer(tmpBuf, TMP_UB_SIZE);

    LocalTensor<T> l_sanitized = sanitizedBuf.Get<T>();
    LocalTensor<uint32_t> l_error = errorBuf.Get<uint32_t>();
    LocalTensor<T> l_indices = indicesBuf.Get<T>();
    LocalTensor<T> l_sizes = sizesBuf.Get<T>();
    LocalTensor<uint8_t> l_tmp = tmpBuf.Get<uint8_t>();

    GmToUb(l_indices, param.indices, param.calCount);
    GmToUb(l_sizes, param.sizes, param.calCount);
    if (checkErrorGuard) {
      GmToUb(l_error, param.error, param.errorCapacity);
    }

    CheckBoundsExtend<T>(l_sanitized, l_error, l_indices, l_sizes, l_tmp, param.calCount, param.checkLower,
                         param.checkUpper);

    UbToGm(param.sanitized, l_sanitized, param.calCount);
    UbToGm(param.error, l_error, checkErrorGuard ? param.errorCapacity : param.calCount);
  }

  // -------------------------------------------------------
  //  Helper: allocate and free gm buffers
  // -------------------------------------------------------
  template <typename T>
  static void AllocBuffers(CheckBoundsInputParam<T> &param) {
    if (param.errorCapacity == 0) {
      param.errorCapacity = param.calCount;
    }
    param.sanitized = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.calCount));
    param.error = static_cast<uint32_t *>(AscendC::GmAlloc(sizeof(uint32_t) * param.errorCapacity));
    param.indices = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.calCount));
    param.sizes = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.calCount));
    param.expSanitized = static_cast<T *>(AscendC::GmAlloc(sizeof(T) * param.calCount));
    param.expError = static_cast<uint32_t *>(AscendC::GmAlloc(sizeof(uint32_t) * param.errorCapacity));
  }

  template <typename T>
  static void FreeBuffers(CheckBoundsInputParam<T> &param) {
    AscendC::GmFree(param.sanitized);
    AscendC::GmFree(param.error);
    AscendC::GmFree(param.indices);
    AscendC::GmFree(param.sizes);
    AscendC::GmFree(param.expSanitized);
    AscendC::GmFree(param.expError);
  }

  // -------------------------------------------------------
  //  Helper: validate results
  // -------------------------------------------------------
  template <typename T>
  static uint32_t Valid(CheckBoundsInputParam<T> &param) {
    uint32_t diff_count = 0;
    for (uint32_t i = 0; i < param.errorCapacity; i++) {
      if (i < param.calCount && param.sanitized[i] != param.expSanitized[i]) {
        diff_count++;
      }
      if (param.error[i] != param.expError[i]) {
        diff_count++;
      }
    }
    return diff_count;
  }

  // -------------------------------------------------------
  //  Scenario builders
  // -------------------------------------------------------
  // Scenario 1: both checks enabled, all indices in range [0, sizes)
  template <typename T>
  static void CreateAllInBounds(CheckBoundsInputParam<T> &param) {
    AllocBuffers(param);
    for (uint32_t i = 0; i < param.calCount; i++) {
      // size is in [1, 20], index is in [0, size-1]
      param.sizes[i] = static_cast<T>((i % 20) + 1);
      param.indices[i] = static_cast<T>(i % param.sizes[i]);
      param.expSanitized[i] = param.indices[i];
      param.expError[i] = 0;
    }
  }

  // Scenario 2: both checks enabled, some indices negative (clamped to 0)
  template <typename T>
  static void CreateWithNegatives(CheckBoundsInputParam<T> &param) {
    AllocBuffers(param);
    for (uint32_t i = 0; i < param.calCount; i++) {
      param.sizes[i] = static_cast<T>((i % 25) + 1);
      if (i % 3 == 0) {
        // negative index → error and clamped to 0
        param.indices[i] = static_cast<T>(-(static_cast<int64_t>(i % 10) + 1));
        param.expSanitized[i] = static_cast<T>(0);
        param.expError[i] = 1;
      } else if (i % 5 == 0) {
        // index == 0 (boundary, valid)
        param.indices[i] = static_cast<T>(0);
        param.expSanitized[i] = static_cast<T>(0);
        param.expError[i] = 0;
      } else {
        // in range
        param.indices[i] = static_cast<T>(i % param.sizes[i]);
        param.expSanitized[i] = param.indices[i];
        param.expError[i] = 0;
      }
    }
  }

  // Scenario 3: both checks enabled, some indices >= sizes (clamped to sizes-1)
  template <typename T>
  static void CreateWithUpperOOB(CheckBoundsInputParam<T> &param) {
    AllocBuffers(param);
    for (uint32_t i = 0; i < param.calCount; i++) {
      param.sizes[i] = static_cast<T>((i % 20) + 3);
      if (i % 4 == 0) {
        // out-of-bounds upper: index >= size
        param.indices[i] = static_cast<T>(param.sizes[i] + static_cast<T>(i % 10));
        param.expSanitized[i] = static_cast<T>(param.sizes[i] - 1);
        param.expError[i] = 1;
      } else if (i % 7 == 0) {
        // index == size - 1 (boundary, valid)
        param.indices[i] = static_cast<T>(param.sizes[i] - 1);
        param.expSanitized[i] = param.indices[i];
        param.expError[i] = 0;
      } else {
        // in range
        param.indices[i] = static_cast<T>(i % param.sizes[i]);
        param.expSanitized[i] = param.indices[i];
        param.expError[i] = 0;
      }
    }
  }

  // Scenario 4: both checks enabled, mixed negative and upper OOB
  template <typename T>
  static void CreateMixedOOB(CheckBoundsInputParam<T> &param) {
    AllocBuffers(param);
    for (uint32_t i = 0; i < param.calCount; i++) {
      param.sizes[i] = static_cast<T>((i % 15) + 3);
      if (i % 3 == 0) {
        // negative
        param.indices[i] = static_cast<T>(-(static_cast<int64_t>(i % 8) + 1));
        param.expSanitized[i] = static_cast<T>(0);
        param.expError[i] = 1;
      } else if (i % 3 == 1) {
        // above upper
        param.indices[i] = static_cast<T>(param.sizes[i] + static_cast<T>(i % 7 + 1));
        param.expSanitized[i] = static_cast<T>(param.sizes[i] - 1);
        param.expError[i] = 1;
      } else {
        // in range
        param.indices[i] = static_cast<T>(i % param.sizes[i]);
        param.expSanitized[i] = param.indices[i];
        param.expError[i] = 0;
      }
    }
  }

  // Scenario 5: lower check only (checkLower=true, checkUpper=false)
  template <typename T>
  static void CreateLowerOnly(CheckBoundsInputParam<T> &param) {
    param.checkLower = true;
    param.checkUpper = false;
    AllocBuffers(param);
    for (uint32_t i = 0; i < param.calCount; i++) {
      param.sizes[i] = static_cast<T>((i % 15) + 3);
      if (i % 3 == 0) {
        // negative → error and clamped to 0
        param.indices[i] = static_cast<T>(-(static_cast<int64_t>(i % 8) + 1));
        param.expSanitized[i] = static_cast<T>(0);
        param.expError[i] = 1;
      } else if (i % 4 == 0) {
        // above upper bound but NO upper check → passes through unchanged, no error
        param.indices[i] = static_cast<T>(param.sizes[i] + static_cast<T>(i % 7 + 1));
        param.expSanitized[i] = param.indices[i];
        param.expError[i] = 0;
      } else {
        // in range
        param.indices[i] = static_cast<T>(i % param.sizes[i]);
        param.expSanitized[i] = param.indices[i];
        param.expError[i] = 0;
      }
    }
  }

  // Scenario 6: upper check only (checkLower=false, checkUpper=true).
  // Only meaningful for signed types — the "negative, no lower check" sub-scenario
  // has no unsigned equivalent because unsigned types have no negative values.
  template <typename T>
  static void CreateUpperOnly(CheckBoundsInputParam<T> &param) {
    param.checkLower = false;
    param.checkUpper = true;
    AllocBuffers(param);
    for (uint32_t i = 0; i < param.calCount; i++) {
      param.sizes[i] = static_cast<T>((i % 15) + 3);
      if (i % 3 == 0) {
        // upper OOB → error and clamped to sizes-1
        param.indices[i] = static_cast<T>(param.sizes[i] + static_cast<T>(i % 7 + 1));
        param.expSanitized[i] = static_cast<T>(param.sizes[i] - 1);
        param.expError[i] = 1;
      } else if (i % 4 == 0) {
        // negative but NO lower check → passes through unchanged, no error
        param.indices[i] = static_cast<T>(-(static_cast<int64_t>(i % 8) + 1));
        param.expSanitized[i] = param.indices[i];
        param.expError[i] = 0;
      } else {
        // in range
        param.indices[i] = static_cast<T>(i % param.sizes[i]);
        param.expSanitized[i] = param.indices[i];
        param.expError[i] = 0;
      }
    }
  }

  // Scenario 7: both checks disabled → direct copy, all error = 0
  template <typename T>
  static void CreateNoCheck(CheckBoundsInputParam<T> &param) {
    param.checkLower = false;
    param.checkUpper = false;
    AllocBuffers(param);
    for (uint32_t i = 0; i < param.calCount; i++) {
      param.sizes[i] = static_cast<T>((i % 20) + 1);
      // Include negative and out-of-bounds values — none should be clamped
      if (i % 4 == 0) {
        param.indices[i] = static_cast<T>(-(static_cast<int64_t>(i % 10) + 1));
      } else if (i % 4 == 1) {
        param.indices[i] = static_cast<T>(param.sizes[i] + static_cast<T>(i % 8 + 1));
      } else {
        param.indices[i] = static_cast<T>(i % 20);
      }
      param.expSanitized[i] = param.indices[i];
      param.expError[i] = 0;
    }
  }

  // Scenario 8: sizes=0 (triggers underflow protection path).
  // When size=0, all indices are out-of-bounds: indices >= 0 == size
  // is always true (even 0 >= 0), so error=1 for every element.
  template <typename T>
  static void CreateZeroSizes(CheckBoundsInputParam<T> &param) {
    AllocBuffers(param);
    for (uint32_t i = 0; i < param.calCount; i++) {
      param.sizes[i] = static_cast<T>(0);
      // Sanitized is always clamped to 0 (sizesMinusOne underflows to 0).
      param.expSanitized[i] = static_cast<T>(0);
      if (i % 3 == 0) {
        param.indices[i] = static_cast<T>(0);
        param.expError[i] = 1;  // 0 >= 0 → OOB
      } else if (i % 3 == 1) {
        param.indices[i] = static_cast<T>(5);
        param.expError[i] = 1;
      } else {
        param.indices[i] = static_cast<T>(-(static_cast<int64_t>(i % 5) + 1));
        param.expError[i] = 1;
      }
    }
  }

  // Scenario 9: large positive indices (stress test upper clamping)
  template <typename T>
  static void CreateLargePositive(CheckBoundsInputParam<T> &param) {
    AllocBuffers(param);
    for (uint32_t i = 0; i < param.calCount; i++) {
      param.sizes[i] = static_cast<T>((i % 10) + 1);
      if (i % 3 == 0) {
        // very large index
        param.indices[i] = static_cast<T>(1000 + i);
        param.expSanitized[i] = static_cast<T>(param.sizes[i] - 1);
        param.expError[i] = 1;
      } else if (i % 3 == 1) {
        // index == size (exactly at boundary, OOB)
        param.indices[i] = param.sizes[i];
        param.expSanitized[i] = static_cast<T>(param.sizes[i] - 1);
        param.expError[i] = 1;
      } else {
        // in range
        param.indices[i] = static_cast<T>(i % param.sizes[i]);
        param.expSanitized[i] = param.indices[i];
        param.expError[i] = 0;
      }
    }
  }

  // -------------------------------------------------------
  //  Main test runner
  // -------------------------------------------------------
  template <typename T>
  static void RunTest(CheckBoundsInputParam<T> &param) {
    auto kernel = [&param] { InvokeKernel(param); };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1);

    uint32_t diff_count = Valid(param);
    EXPECT_EQ(diff_count, 0);

    FreeBuffers(param);
  }

  template <typename T>
  static void RunScenario(uint32_t calCount, bool checkLower, bool checkUpper,
                          std::function<void(CheckBoundsInputParam<T> &)> creator) {
    CheckBoundsInputParam<T> param{};
    param.calCount = calCount;
    param.checkLower = checkLower;
    param.checkUpper = checkUpper;
    creator(param);
    RunTest(param);
  }

  // Convenience: run the scenario across small sizes (boundary / alignment coverage)
  template <typename T>
  static void RunSmallSizes(bool checkLower, bool checkUpper, std::function<void(CheckBoundsInputParam<T> &)> creator) {
    RunScenario<T>(ONE_BLK_SIZE / sizeof(T), checkLower, checkUpper, creator);
    RunScenario<T>(ONE_REPEAT_BYTE_SIZE / sizeof(T), checkLower, checkUpper, creator);
    RunScenario<T>((ONE_BLK_SIZE - sizeof(T)) / sizeof(T), checkLower, checkUpper, creator);
    RunScenario<T>((ONE_REPEAT_BYTE_SIZE - ONE_BLK_SIZE) / sizeof(T), checkLower, checkUpper, creator);
  }

  // Full-size convenience: small sizes + large repeat-stress sizes.
  // Use only for representative scenarios where large-calCount coverage matters.
  template <typename T>
  static void RunAllSizes(bool checkLower, bool checkUpper, std::function<void(CheckBoundsInputParam<T> &)> creator) {
    RunSmallSizes<T>(checkLower, checkUpper, creator);
    RunScenario<T>(MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(T), checkLower, checkUpper, creator);
    RunScenario<T>((MAX_REPEAT_NUM - 1) * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(T), checkLower, checkUpper, creator);
  }
};

// ============================================================
//  Tests with both checks enabled (default)
// ============================================================

TEST_F(TestApiCheckBoundsUT, BothChecks_AllInBounds) {
  RunAllSizes<int32_t>(true, true, CreateAllInBounds<int32_t>);
  RunAllSizes<int64_t>(true, true, CreateAllInBounds<int64_t>);
  RunAllSizes<uint32_t>(true, true, CreateAllInBounds<uint32_t>);
  RunAllSizes<uint64_t>(true, true, CreateAllInBounds<uint64_t>);
}

TEST_F(TestApiCheckBoundsUT, Int64AndUint64ErrorFlagDoesNotWritePastCalCount) {
  constexpr uint32_t kCalCount = MAX_REPEAT_NUM * ONE_REPEAT_BYTE_SIZE / 2 / sizeof(uint64_t);
  constexpr uint32_t kGuardCount = GetVecLen() / sizeof(uint32_t);
  constexpr uint32_t kGuardValue = 0xA5A5A5A5U;

  auto int64Creator = [](CheckBoundsInputParam<int64_t> &param) {
    param.errorCapacity = param.calCount + kGuardCount;
    CreateAllInBounds(param);
    for (uint32_t i = param.calCount; i < param.errorCapacity; ++i) {
      param.error[i] = kGuardValue;
      param.expError[i] = kGuardValue;
    }
  };
  RunScenario<int64_t>(kCalCount, true, true, int64Creator);

  auto uint64Creator = [](CheckBoundsInputParam<uint64_t> &param) {
    param.errorCapacity = param.calCount + kGuardCount;
    CreateAllInBounds(param);
    for (uint32_t i = param.calCount; i < param.errorCapacity; ++i) {
      param.error[i] = kGuardValue;
      param.expError[i] = kGuardValue;
    }
  };
  RunScenario<uint64_t>(kCalCount, true, true, uint64Creator);
}

TEST_F(TestApiCheckBoundsUT, BothChecks_WithNegatives) {
  RunSmallSizes<int32_t>(true, true, CreateWithNegatives<int32_t>);
  RunSmallSizes<int64_t>(true, true, CreateWithNegatives<int64_t>);
}

TEST_F(TestApiCheckBoundsUT, BothChecks_WithUpperOOB) {
  RunSmallSizes<int32_t>(true, true, CreateWithUpperOOB<int32_t>);
  RunSmallSizes<int64_t>(true, true, CreateWithUpperOOB<int64_t>);
  RunSmallSizes<uint32_t>(true, true, CreateWithUpperOOB<uint32_t>);
  RunSmallSizes<uint64_t>(true, true, CreateWithUpperOOB<uint64_t>);
}

TEST_F(TestApiCheckBoundsUT, BothChecks_MixedOOB) {
  RunSmallSizes<int32_t>(true, true, CreateMixedOOB<int32_t>);
  RunSmallSizes<int64_t>(true, true, CreateMixedOOB<int64_t>);
}

TEST_F(TestApiCheckBoundsUT, BothChecks_ZeroSizes) {
  RunAllSizes<int32_t>(true, true, CreateZeroSizes<int32_t>);
  RunAllSizes<int64_t>(true, true, CreateZeroSizes<int64_t>);
  RunAllSizes<uint32_t>(true, true, CreateZeroSizes<uint32_t>);
  RunAllSizes<uint64_t>(true, true, CreateZeroSizes<uint64_t>);
}

TEST_F(TestApiCheckBoundsUT, BothChecks_LargePositive) {
  RunAllSizes<int32_t>(true, true, CreateLargePositive<int32_t>);
  RunAllSizes<int64_t>(true, true, CreateLargePositive<int64_t>);
  RunAllSizes<uint32_t>(true, true, CreateLargePositive<uint32_t>);
  RunAllSizes<uint64_t>(true, true, CreateLargePositive<uint64_t>);
}

// ============================================================
//  Tests with lower check only
// ============================================================

TEST_F(TestApiCheckBoundsUT, LowerOnly_WithNegatives) {
  RunSmallSizes<int32_t>(true, false, CreateLowerOnly<int32_t>);
  RunSmallSizes<int64_t>(true, false, CreateLowerOnly<int64_t>);
}

// ============================================================
//  Tests with upper check only
// ============================================================

TEST_F(TestApiCheckBoundsUT, UpperOnly_WithUpperOOB) {
  RunSmallSizes<int32_t>(false, true, CreateUpperOnly<int32_t>);
  RunSmallSizes<int64_t>(false, true, CreateUpperOnly<int64_t>);
}

// ============================================================
//  Tests with both checks disabled
// ============================================================

TEST_F(TestApiCheckBoundsUT, NoCheck_DirectCopy) {
  RunSmallSizes<int32_t>(false, false, CreateNoCheck<int32_t>);
  RunSmallSizes<int64_t>(false, false, CreateNoCheck<int64_t>);
  RunSmallSizes<uint32_t>(false, false, CreateNoCheck<uint32_t>);
  RunSmallSizes<uint64_t>(false, false, CreateNoCheck<uint64_t>);
}

// ============================================================
//  Boundary value tests
// ============================================================

TEST_F(TestApiCheckBoundsUT, Boundary_IndexAtZero_Success) {
  // index == 0 is the lower bound edge — should be valid
  auto creator = [](CheckBoundsInputParam<int32_t> &param) {
    AllocBuffers(param);
    for (uint32_t i = 0; i < param.calCount; i++) {
      param.sizes[i] = static_cast<int32_t>(10);
      param.indices[i] = 0;
      param.expSanitized[i] = 0;
      param.expError[i] = 0;
    }
  };
  RunScenario<int32_t>(256, true, true, creator);
}

TEST_F(TestApiCheckBoundsUT, Boundary_IndexAtSizeMinusOne_Success) {
  // index == sizes-1 is the upper bound edge — should be valid
  auto creator = [](CheckBoundsInputParam<int32_t> &param) {
    AllocBuffers(param);
    for (uint32_t i = 0; i < param.calCount; i++) {
      param.sizes[i] = static_cast<int32_t>((i % 20) + 1);
      param.indices[i] = static_cast<int32_t>(param.sizes[i] - 1);
      param.expSanitized[i] = param.indices[i];
      param.expError[i] = 0;
    }
  };
  RunScenario<int32_t>(256, true, true, creator);
}

TEST_F(TestApiCheckBoundsUT, Boundary_IndexNegativeOne_Clamped) {
  // index == -1 → should be clamped to 0, error = 1
  auto creator = [](CheckBoundsInputParam<int32_t> &param) {
    AllocBuffers(param);
    for (uint32_t i = 0; i < param.calCount; i++) {
      param.sizes[i] = static_cast<int32_t>(10);
      param.indices[i] = -1;
      param.expSanitized[i] = 0;
      param.expError[i] = 1;
    }
  };
  RunScenario<int32_t>(256, true, true, creator);
}

TEST_F(TestApiCheckBoundsUT, Boundary_IndexAtSize_Clamped) {
  // index == size (not size-1) → OOB, should be clamped to size-1, error = 1
  auto creator = [](CheckBoundsInputParam<int32_t> &param) {
    AllocBuffers(param);
    for (uint32_t i = 0; i < param.calCount; i++) {
      param.sizes[i] = static_cast<int32_t>((i % 10) + 3);
      param.indices[i] = param.sizes[i];  // exactly equal to size
      param.expSanitized[i] = static_cast<int32_t>(param.sizes[i] - 1);
      param.expError[i] = 1;
    }
  };
  RunScenario<int32_t>(256, true, true, creator);
}

TEST_F(TestApiCheckBoundsUT, CalCountZero_NoOp) {
  // calCount == 0 should be an early return with no effect.
  // Allocate with calCount=1 to avoid nullptr from GmAlloc(0),
  // then override calCount to 0 for the actual API call.
  CheckBoundsInputParam<int32_t> param{};
  param.calCount = 1;
  param.checkLower = true;
  param.checkUpper = true;
  AllocBuffers(param);
  param.calCount = 0;

  auto kernel = [&param] { InvokeKernel(param); };
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(kernel, 1);
  // Test passes if no crash/exception occurs with calCount=0.
  FreeBuffers(param);
}

TEST_F(TestApiCheckBoundsUT, CalCountOne_SingleElement) {
  // Single element: index in-bounds
  auto creator = [](CheckBoundsInputParam<int32_t> &param) {
    AllocBuffers(param);
    param.sizes[0] = 10;
    param.indices[0] = 5;
    param.expSanitized[0] = 5;
    param.expError[0] = 0;
  };
  RunScenario<int32_t>(1, true, true, creator);

  // Single element: index out-of-bounds lower
  auto creator2 = [](CheckBoundsInputParam<int64_t> &param) {
    AllocBuffers(param);
    param.sizes[0] = 10;
    param.indices[0] = -3;
    param.expSanitized[0] = 0;
    param.expError[0] = 1;
  };
  RunScenario<int64_t>(1, true, true, creator2);

  // Single element: index out-of-bounds upper
  auto creator3 = [](CheckBoundsInputParam<uint32_t> &param) {
    AllocBuffers(param);
    param.sizes[0] = 5;
    param.indices[0] = 7;
    param.expSanitized[0] = 4;
    param.expError[0] = 1;
  };
  RunScenario<uint32_t>(1, true, true, creator3);
}

TEST_F(TestApiCheckBoundsUT, UintTypes_NegativeValueIsWrapped) {
  // For unsigned types, "negative" int values wrap to large positive values
  // which are treated as upper-OOB.
  auto creator = [](CheckBoundsInputParam<uint32_t> &param) {
    AllocBuffers(param);
    for (uint32_t i = 0; i < param.calCount; i++) {
      param.sizes[i] = static_cast<uint32_t>(20 + (i % 10));
      if (i % 2 == 0) {
        // A deliberately large uint32_t value that exceeds all sizes
        param.indices[i] = static_cast<uint32_t>(0xFFFFFFFEU);
        param.expSanitized[i] = static_cast<uint32_t>(param.sizes[i] - 1);
        param.expError[i] = 1;
      } else {
        param.indices[i] = static_cast<uint32_t>(i % param.sizes[i]);
        param.expSanitized[i] = param.indices[i];
        param.expError[i] = 0;
      }
    }
  };
  RunScenario<uint32_t>(ONE_BLK_SIZE / sizeof(uint32_t), true, true, creator);
}

}  // namespace af
