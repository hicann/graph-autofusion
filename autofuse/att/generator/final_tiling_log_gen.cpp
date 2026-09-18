/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

#include "final_tiling_log_gen.h"

#include <sstream>

namespace att::final_tiling {
namespace {

void AppendFinalTilingStructs(std::stringstream &code) {
  code << R"(
struct FinalTilingPipeEstimate {
  std::string name;
  double value;
  bool valid;
};

struct FinalTilingContext {
  uint32_t schema;
  std::string source;
  std::string selection_mode;
  std::string op;
  uint32_t graph;
  uint32_t result;
  uint32_t group;
  int32_t case_id;
  uint32_t tiling_key;
  int32_t score;
  std::string sub_case_tag;
  std::string template_name;
  std::vector<FinalTilingPipeEstimate> pipe_estimates;
  std::string tiling_repr;
  std::string repr_kind;
};

struct FinalTilingGroupSelection {
  uint32_t group;
  int32_t case_id;
  uint32_t tiling_key;
  int32_t score;
  std::string sub_case_tag;
  std::string template_name;
};

)";
}

void AppendFinalTilingSummaryStruct(std::stringstream &code, const bool null_log) {
  code << R"(
struct FinalTilingSummaryContext {
  uint32_t schema;
  std::string source;
  std::string selection_mode;
  std::string op;
  uint32_t graph;
  uint32_t result;
  std::vector<FinalTilingGroupSelection> groups;
};

inline bool ShouldEmitFinalTiling() noexcept {
)";
  code << "  return " << (null_log ? "false" : "CheckCachedLogLevel(DLOG_INFO)") << R"(;
}

inline thread_local bool g_final_tiling_observe_enabled = false;
)";
}

void AppendFinalTilingQuoteHelpers(std::stringstream &code) {
  code << R"(
inline std::string FinalTilingShellQuote(const std::string &value) {
  std::string quoted;
  quoted.reserve(value.size() + 2U);
  quoted.push_back('\"');
  for (const char ch : value) {
    if (ch == '\\' || ch == '\"') { quoted.push_back('\\'); }
    if (ch == '\n') { quoted.append("\\n"); } else if (ch == '\r') { quoted.append("\\r"); } else {
      quoted.push_back(ch);
    }
  }
  quoted.push_back('\"');
  return quoted;
}

inline std::string FinalTilingJsonQuote(const std::string &value) {
  std::string quoted;
  quoted.reserve(value.size() + 2U);
  quoted.push_back('\"');
  for (const unsigned char ch : value) {
    switch (ch) {
      case '\\': quoted.append("\\\\"); break;
      case '\"': quoted.append("\\\""); break;
      case '\n': quoted.append("\\n"); break;
      case '\r': quoted.append("\\r"); break;
      case '\t': quoted.append("\\t"); break;
      default: quoted.push_back(static_cast<char>(ch)); break;
    }
  }
  quoted.push_back('\"');
  return quoted;
}
)";
}

void AppendFinalTilingChecksumHelper(std::stringstream &code) {
  code << R"(
inline std::string FinalTilingChecksum(const std::string &input) {
  uint64_t hash = 0x6d2b79f5ULL;
  for (const unsigned char value : input) {
    hash ^= static_cast<uint64_t>(value) + 0x9e3779b9ULL + (hash << 6U) + (hash >> 2U);
    hash = (hash << 13U) | (hash >> 51U);
  }
  std::ostringstream output;
  output << std::hex << std::setfill('0') << std::setw(16) << hash;
  return output.str();
}
)";
}

void AppendFinalTilingIdentityHelpers(std::stringstream &code) {
  code << R"(
inline std::string FinalTilingStableIdPart(const std::string &value) {
  return std::to_string(value.size()) + ":" + value;
}

inline std::string FinalTilingStableId(const FinalTilingContext &context) {
  return FinalTilingStableIdPart(context.op) + "|" + std::to_string(context.graph) + "|" +
         std::to_string(context.result) + "|" + std::to_string(context.group) + "|" +
         std::to_string(context.case_id) + "|" + std::to_string(context.tiling_key) + "|" +
         FinalTilingStableIdPart(context.sub_case_tag);
}

inline std::string FinalTilingNumber(const double value) {
  return std::isfinite(value) ? std::to_string(value) : std::string("null");
}

inline std::string FinalTilingPipeJson(const FinalTilingContext &context) {
  std::string pipe = "{";
  for (size_t index = 0U; index < context.pipe_estimates.size(); ++index) {
    const auto &estimate = context.pipe_estimates[index];
    if (index != 0U) { pipe.push_back(','); }
    pipe += FinalTilingJsonQuote(estimate.name);
    pipe.push_back(':');
    pipe += (estimate.valid && std::isfinite(estimate.value)) ? std::to_string(estimate.value) : "null";
  }
  pipe.push_back('}');
  return pipe;
}

inline std::string FinalTilingCommon(const FinalTilingContext &context) {
  return std::string("schema=") + std::to_string(context.schema) + " source=" + FinalTilingShellQuote(context.source) +
      " selection_mode=" + FinalTilingShellQuote(context.selection_mode) + " operator=" +
      FinalTilingShellQuote(context.op) + " graph=" + std::to_string(context.graph) +
      " result=" + std::to_string(context.result) + " group=" + std::to_string(context.group) +
      " case_id=" + std::to_string(context.case_id) + " tiling_key=" + std::to_string(context.tiling_key) +
      " score=" + std::to_string(context.score) + " sub_case_tag=" + FinalTilingShellQuote(context.sub_case_tag) +
      " template=" + FinalTilingShellQuote(context.template_name) + " repr_kind=" +
      FinalTilingShellQuote(context.repr_kind) + " pipe_estimates=" +
      FinalTilingShellQuote(FinalTilingPipeJson(context));
}

inline std::string FinalTilingChunkPrefix(const std::string &id, const size_t seq) {
  return std::string("[ATT][FINAL_TILING_CHUNK] id=") + FinalTilingShellQuote(id) +
         " seq=" + std::to_string(seq) + " data=";
}
)";
}

void AppendFinalTilingSplitPayload(std::stringstream &code) {
  code << R"(
inline std::string FinalTilingSummaryChunkPrefix(const std::string &id, const size_t seq) {
  return std::string("[ATT][FINAL_TILING_SUMMARY_CHUNK] id=") + FinalTilingShellQuote(id) +
         " seq=" + std::to_string(seq) + " data=";
}

inline std::vector<std::string> FinalTilingSplitPayload(const std::string &payload, const std::string &chunk_prefix) {
  constexpr size_t kLineBudget = 700U;
  std::vector<std::string> chunks;
  size_t begin = 0U;
  while (begin < payload.size()) {
    size_t end = begin;
    while (end < payload.size()) {
      size_t next = end + 1U;
      while (next < payload.size() && (static_cast<unsigned char>(payload[next]) & 0xc0U) == 0x80U) { ++next; }
      const std::string candidate = payload.substr(begin, next - begin);
      const std::string probe = chunk_prefix + FinalTilingShellQuote(candidate);
      if (probe.size() > kLineBudget && end != begin) { break; }
      end = next;
      if (probe.size() > kLineBudget) { break; }
    }
    if (end == begin) {
      end = begin + 1U;
      while (end < payload.size() && (static_cast<unsigned char>(payload[end]) & 0xc0U) == 0x80U) { ++end; }
    }
    chunks.emplace_back(payload.substr(begin, end - begin));
    begin = end;
  }
  return chunks;
}
)";
}

void AppendFinalTilingRecordFormatter(std::stringstream &code) {
  code << R"(
inline std::vector<std::string> FormatFinalTilingRecord(const FinalTilingContext &context) {
  const std::string common = FinalTilingCommon(context);
  const std::string single = std::string("[ATT][FINAL_TILING] ") + common + " tiling_repr=" +
      FinalTilingShellQuote(context.tiling_repr);
  if (single.size() <= 700U) { return {single}; }
  const std::string id = FinalTilingStableId(context);
  const std::string hash = FinalTilingChecksum(context.tiling_repr);
  const auto chunks = FinalTilingSplitPayload(context.tiling_repr, FinalTilingChunkPrefix(id, std::numeric_limits<size_t>::max()));
  std::vector<std::string> lines;
  lines.emplace_back(std::string("[ATT][FINAL_TILING_BEGIN] ") + common + " id=" + FinalTilingShellQuote(id) +
                     " chunks=" + std::to_string(chunks.size()) + " len=" + std::to_string(context.tiling_repr.size()) +
                     " hash_alg=att_mix64_v1 hash=" + hash);
  for (size_t seq = 0U; seq < chunks.size(); ++seq) {
    lines.emplace_back(FinalTilingChunkPrefix(id, seq) + FinalTilingShellQuote(chunks[seq]));
  }
  lines.emplace_back(std::string("[ATT][FINAL_TILING_END] id=") + FinalTilingShellQuote(id) +
                     " chunks=" + std::to_string(chunks.size()) + " len=" + std::to_string(context.tiling_repr.size()) +
                     " hash_alg=att_mix64_v1 hash=" + hash);
  return lines;
}
)";
}

void AppendFinalTilingSummaryFormatter(std::stringstream &code) {
  code << R"(
inline std::string FinalTilingGroupsJson(const std::vector<FinalTilingGroupSelection> &groups) {
  std::string value = "{";
  for (size_t index = 0U; index < groups.size(); ++index) {
    const auto &group = groups[index];
    if (index != 0U) { value.push_back(','); }
    value += FinalTilingJsonQuote(std::to_string(group.group)) + ":{\"case_id\":" +
        std::to_string(group.case_id) + ",\"tiling_key\":" + std::to_string(group.tiling_key) +
        ",\"score\":" + std::to_string(group.score) + ",\"sub_case_tag\":" +
        FinalTilingJsonQuote(group.sub_case_tag) + ",\"template\":" + FinalTilingJsonQuote(group.template_name) +
        "}";
  }
  value.push_back('}');
  return value;
}

inline std::vector<std::string> FormatFinalTilingSummary(const FinalTilingSummaryContext &context) {
  const std::string groups = FinalTilingGroupsJson(context.groups);
  const std::string line = std::string("[ATT][FINAL_TILING_SUMMARY] schema=") + std::to_string(context.schema) +
      " source=" + FinalTilingShellQuote(context.source) + " selection_mode=" +
      FinalTilingShellQuote(context.selection_mode) + " operator=" + FinalTilingShellQuote(context.op) +
      " graph=" + std::to_string(context.graph) + " result=" + std::to_string(context.result) +
      " groups=" + FinalTilingShellQuote(groups);
  if (line.size() <= 700U) { return {line}; }
  const std::string id = FinalTilingStableIdPart(context.op) + "|" + std::to_string(context.graph) + "|" + std::to_string(context.result);
  const std::string hash = FinalTilingChecksum(groups);
  const auto chunks = FinalTilingSplitPayload(groups, FinalTilingSummaryChunkPrefix(id, std::numeric_limits<size_t>::max()));
  std::vector<std::string> lines;
  lines.emplace_back(std::string("[ATT][FINAL_TILING_SUMMARY_BEGIN] schema=") + std::to_string(context.schema) +
      " source=" + FinalTilingShellQuote(context.source) + " selection_mode=" +
      FinalTilingShellQuote(context.selection_mode) + " operator=" + FinalTilingShellQuote(context.op) +
      " graph=" + std::to_string(context.graph) + " result=" + std::to_string(context.result) +
      " id=" + FinalTilingShellQuote(id) +
      " chunks=" + std::to_string(chunks.size()) + " len=" + std::to_string(groups.size()) +
      " hash_alg=att_mix64_v1 hash=" + hash);
  for (size_t seq = 0U; seq < chunks.size(); ++seq) {
    lines.emplace_back(FinalTilingSummaryChunkPrefix(id, seq) + FinalTilingShellQuote(chunks[seq]));
  }
  lines.emplace_back(std::string("[ATT][FINAL_TILING_SUMMARY_END] id=") + FinalTilingShellQuote(id) +
      " chunks=" + std::to_string(chunks.size()) + " len=" + std::to_string(groups.size()) +
      " hash_alg=att_mix64_v1 hash=" + hash);
  return lines;
}
)";
}

void AppendFinalTilingEmitters(std::stringstream &code) {
  code << R"(
inline void EmitFinalTilingLines(const std::vector<std::string> &lines) noexcept {
  try {
    for (const auto &line : lines) { ATT_FINAL_TILING_LOGI(OP_NAME, "%s", line.c_str()); }
  } catch (...) {
    // Observability must not change the tiling result.
  }
}

inline void EmitFinalTilingSummary(const FinalTilingSummaryContext &context) noexcept {
  try {
    EmitFinalTilingLines(FormatFinalTilingSummary(context));
  } catch (...) {
    // Observability must not change the tiling result.
  }
}

inline void EmitFinalTilingRecord(const FinalTilingContext &context) noexcept {
  try {
    EmitFinalTilingLines(FormatFinalTilingRecord(context));
  } catch (...) {
    // Formatting and allocation failures are also observational only.
  }
}
)";
}

}  // namespace

std::string GenFinalTilingLogHelpers(const bool null_log) {
  std::stringstream code;
  AppendFinalTilingStructs(code);
  AppendFinalTilingSummaryStruct(code, null_log);
  AppendFinalTilingQuoteHelpers(code);
  AppendFinalTilingChecksumHelper(code);
  AppendFinalTilingIdentityHelpers(code);
  AppendFinalTilingSplitPayload(code);
  AppendFinalTilingRecordFormatter(code);
  AppendFinalTilingSummaryFormatter(code);
  AppendFinalTilingEmitters(code);
  return code.str();
}

}  // namespace att::final_tiling
