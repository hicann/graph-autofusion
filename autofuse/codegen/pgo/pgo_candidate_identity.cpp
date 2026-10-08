/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

#include "pgo_candidate_identity.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <sstream>

namespace codegen::pgo {
namespace {

// Self-written 128-bit non-cryptographic digest for candidate identity keys.
// Four independent multiply/xorshift lanes with arbitrary odd multipliers keep
// the implementation free of third-party algorithm tables (open-source
// compliance); the digest only needs to be deterministic and collision-free
// for identity purposes, not cryptographically strong.
constexpr size_t kDigestLaneCount = 4U;
constexpr size_t kDigestSize = 16U;
constexpr size_t kDigestBytesPerLane = 4U;
constexpr size_t kDigestBitsPerByte = 8U;
constexpr int kDigestHexWidth = 2;
constexpr size_t kCandidateKeyLength = 32U;
constexpr uint32_t kLaneMixShift = 13U;
constexpr uint32_t kLaneAvalancheShift = 16U;
constexpr uint32_t kLaneFinalShift = 15U;
constexpr std::array<uint32_t, kDigestLaneCount> kLaneSeed = {0x3c5f7a1bU, 0x6e2d9c4fU, 0x159d3b7eU, 0x7f4a1c6dU};
constexpr std::array<uint32_t, kDigestLaneCount> kLaneMultiplier = {0x0d1b73a1U, 0x2c4f6e13U, 0x59a7de35U, 0x8e3f9c57U};
constexpr uint32_t kCounterStep = 0x1d9e4f3bU;
constexpr uint32_t kFinalMultiplierA = 0x4d9c3a5bU;
constexpr uint32_t kFinalMultiplierB = 0x1f8d6b39U;

std::array<uint8_t, kDigestSize> Digest128(const std::string &payload) {
  std::array<uint32_t, kDigestLaneCount> lanes = kLaneSeed;
  uint32_t counter = 0;
  for (char ch : payload) {
    const uint32_t value = static_cast<uint32_t>(static_cast<unsigned char>(ch)) + counter;
    for (size_t lane = 0; lane < lanes.size(); ++lane) {
      lanes[lane] ^= value + static_cast<uint32_t>(lane);
      lanes[lane] *= kLaneMultiplier[lane];
      lanes[lane] ^= lanes[lane] >> kLaneMixShift;
    }
    counter += kCounterStep;
  }
  for (uint32_t &lane : lanes) {
    lane ^= lane >> kLaneAvalancheShift;
    lane *= kFinalMultiplierA;
    lane ^= lane >> kLaneFinalShift;
    lane *= kFinalMultiplierB;
    lane ^= lane >> kLaneAvalancheShift;
  }
  std::array<uint8_t, kDigestSize> digest{};
  for (size_t lane = 0; lane < kDigestLaneCount; ++lane) {
    for (size_t byte = 0; byte < kDigestBytesPerLane; ++byte) {
      digest[lane * kDigestBytesPerLane + byte] =
          static_cast<uint8_t>((lanes[lane] >> (byte * kDigestBitsPerByte)) & 0xffU);
    }
  }
  return digest;
}

}  // namespace

std::string BuildCandidateKey(const std::string &graph_name, const std::string &tiling_repr, uint32_t workspace_size,
                              uint32_t block_dim) {
  const std::string payload =
      graph_name + "|" + tiling_repr + "|" + std::to_string(workspace_size) + "|" + std::to_string(block_dim);
  const auto digest = Digest128(payload);
  std::ostringstream key;
  key << std::hex << std::setfill('0');
  for (uint8_t byte : digest) {
    key << std::setw(kDigestHexWidth) << static_cast<uint32_t>(byte);
  }
  return key.str();
}

bool IsValidCandidateKey(const std::string &candidate_key) {
  return candidate_key.size() == kCandidateKeyLength &&
         std::all_of(candidate_key.begin(), candidate_key.end(), [](unsigned char value) {
           return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
         });
}

}  // namespace codegen::pgo
