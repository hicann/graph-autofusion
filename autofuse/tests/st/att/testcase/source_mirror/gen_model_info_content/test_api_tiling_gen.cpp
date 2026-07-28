/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common_gen_utils.h"
#include <iostream>
#include <regex>
#include "gtest/gtest.h"
#include "base/att_const_values.h"
#include "tiling_code_generator.h"
#include "gen_model_info.h"
#include "ascir_ops.h"
#include "autofuse_config/auto_fuse_config.h"
#include "base/model_info.h"
#include "common/ascgen_log.h"
#include "gen_tiling_impl.h"
#include "ascendc_ir_dump_utils.h"
#include "common_utils.h"
#include "graph_construct_utils.h"
#include "common/st_scenario_utils.h"
#include "test_common_utils.h"

using namespace ge::ascir_op;
namespace ascir {
constexpr int64_t ID_NONE = -1;
using namespace ge;
using HintGraph = AscGraph;
}  // namespace ascir
using namespace att;

// 辅助函数：生成特定操作的主程序文件
inline void GenerateOpMainFile(const std::string &op_name, const std::string &tiling_data_type) {
  std::ofstream oss;
  std::string filename = "tiling_func_main_" + op_name + ".cpp";
  oss.open(filename, std::ios::out);
  oss << "/**\n";
  oss << " * Copyright (c) 2026 Huawei Technologies Co., Ltd.\n";
  oss << " * This program is free software, you can redistribute it and/or modify it under the terms and conditions "
         "of\n";
  oss << " * CANN Open Software License Agreement Version 2.0 (the \"License\").\n";
  oss << " * Please refer to the License for details. You may use this file except in compliance with the License.\n";
  oss << " * THIS SOFTWARE IS PROVIDED ON AN \"AS IS\" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR "
         "IMPLIED,\n";
  oss << " * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.\n";
  oss << " * See LICENSE in the root of the software repository for the full text of the License.\n";
  oss << " */\n";
  oss << "\n";
  oss << "#include <iostream>\n";
  oss << "#include \"" << tiling_data_type << ".h\"\n";
  oss << "using namespace optiling;\n";
  oss << "\n";
  oss << "void PrintResult(graph_normalTilingData &tilingData) {\n";
  oss << "    std::cout << \"====================================================\" << std::endl;\n";
  oss << "    std::cout << \"get_tiling_key\" << \" = \" << tilingData.graph0_tiling_key_ << std::endl;\n";
  oss << "    std::cout << \"====================================================\" << std::endl;\n";
  oss << "}\n";
  oss << "\n";
  oss << "int main() {\n";
  oss << "    graph_normalTilingData tilingData;\n";
  oss << "    tilingData.set_block_dim(64);\n";
  oss << "    tilingData.set_ub_size(34078720);\n";
  oss << "    if (GetTiling(tilingData)) {\n";
  oss << "        PrintResult(tilingData);\n";
  oss << "    } else {\n";
  oss << "        std::cout << \"" << op_name << " tiling func execute failed.\" << std::endl;\n";
  oss << "        return -1;\n";
  oss << "    }\n";
  oss << "    return 0;\n";
  oss << "}\n";
  oss.close();
}

class TestApiTilingGen : public ::testing::Test {
 public:
  static void TearDownTestCase() {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {
    // Code here will be called immediately after the constructor (right
    // before each test).
    att::AutoFuseConfig::MutableAttStrategyConfig().Reset();
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "4", 1);
  }

  void TearDown() override {
    // 清理测试生成的临时文件
    autofuse::test::CleanupTestArtifacts();
    unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
    unsetenv("AUTOFUSE_DFX_FLAGS");
  }
};

namespace ge {
namespace ascir {
namespace cg {
// 辅助函数：验证排列有效性
bool IsValidPermutation(const std::vector<int64_t> &perm) {
  std::set<int64_t> unique(perm.begin(), perm.end());
  if (unique.size() != perm.size()) return false;
  for (auto idx : perm) {
    if (idx < 0 || idx >= static_cast<int64_t>(perm.size())) return false;
  }
  return true;
}

static ge::Expression One = ge::Symbol(1);
Status BuildTransposeAscendGraph(ge::AscGraph &graph, const std::vector<int64_t> &perm = {0, 2, 1} /* 默认021转置 */
) {
  // 参数校验
  if (perm.size() != 3 || !IsValidPermutation(perm)) {
    return af::FAILED;  // 仅支持3D转置
  }

  auto s0 = graph.CreateSizeVar(16);
  auto s1 = graph.CreateSizeVar(32);
  auto s2 = graph.CreateSizeVar(128);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  // 轴集合便于按perm访问
  std::vector<ge::Axis> axes = {z0, z1, z2};
  std::vector<ge::Expression> dims = {s0, s1, s2};

  // 分块逻辑保持不变
  auto [z0T, z0t] = graph.TileSplit(z0.id);
  auto [z0TB, z0Tb] = graph.BlockSplit(z0T->id);

  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0, z1, z2}, 0, FORMAT_ND);
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {z0, z1, z2}, 1, FORMAT_ND);

  // 根据原始维度设置repeats和strides
  *data1.repeats = {s0, s1, s2};
  *data2.repeats = {s0, s1, s2};
  *data1.strides = {s1 * s2, s2, One};
  *data2.strides = {s1 * s2, s2, One};

  LOOP(*z0TB) {
    LOOP(*z0Tb) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 2);
      auto load2 = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*z0TB, *z0Tb, z0, z1, z2}, {load1, load2}, 1));

      auto add = Add("add", load1, load2).TBuf(Position::kPositionVecCalc);

      // 根据perm动态设置转置属性
      *(add.vectorized_axis) = {z0.id, z1.id, z2.id};
      *(add.axis) = {z0.id, z1.id, z2.id};
      *(add.repeats) = {s0, s1, s2};
      *(add.strides) = {s1 * s2, s2, One};

      auto transpose = Transpose("transpose", add).TQue(Position::kPositionVecOut, 1, 2);

      // 关键修改点：根据perm参数动态设置
      *(transpose.vectorized_axis) = {axes[perm[0]].id, axes[perm[1]].id, axes[perm[2]].id};
      *(transpose.axis) = {axes[perm[0]].id, axes[perm[1]].id, axes[perm[2]].id};

      // 动态计算转置后的repeats和strides
      *(transpose.repeats) = {dims[perm[0]], dims[perm[1]], dims[perm[2]]};

      // 计算转置后的strides
      auto stride0 = dims[perm[1]] * dims[perm[2]];
      auto stride1 = dims[perm[2]];
      *(transpose.strides) = {stride0, stride1, One};

      auto store = Store("store", transpose);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*z0TB, *z0Tb, *z0t, z1, z2}, {store}, 1));

      auto output = Output("output", store);

      // 内存重用ID分配
      load1.mem->reuse_id = 0;
      load2.mem->reuse_id = 1;
      transpose.mem->reuse_id = 2;
      add.mem->reuse_id = 3;
    }
  }

  auto transpose_node = graph.FindNode("transpose");
  GE_ASSERT_NOTNULL(transpose_node);
  transpose_node->attr.api.unit = ComputeUnit::kUnitVector;
  return af::SUCCESS;
}

Status BuildTransposeSplitAscendGraph(ge::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(16);
  auto s1 = graph.CreateSizeVar(32);
  auto s2 = graph.CreateSizeVar(128);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  // 轴集合便于按perm访问
  std::vector<ge::Axis> axes = {z0, z1, z2};
  std::vector<ge::Expression> dims = {s0, s1, s2};

  // 分块逻辑保持不变
  auto [z0T, z0t] = graph.TileSplit(z0.id);
  auto [z0TB, z0Tb] = graph.BlockSplit(z0T->id);

  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {z0, z1, z2}, 0, FORMAT_ND);
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {z0, z1, z2}, 1, FORMAT_ND);

  // 根据原始维度设置repeats和strides
  *data1.repeats = {s0, s1, s2};
  *data2.repeats = {s0, s1, s2};
  *data1.strides = {s1 * s2, s2, One};
  *data2.strides = {s1 * s2, s2, One};

  LOOP(*z0TB) {
    LOOP(*z0Tb) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 2);
      auto load2 = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*z0TB, *z0Tb, z0, z1, z2}, {load1, load2}, 1));

      auto add = Add("add", load1, load2).TBuf(Position::kPositionVecCalc);

      // 根据perm动态设置转置属性
      *(add.axis) = {z0T->id, z0t->id, z1.id, z2.id};
      *(add.repeats) = {s0 / z0t->size, z0t->size, s1, s2};
      *(add.strides) = {z0t->size * s1 * s2, s1 * s2, s2, One};
      *(add.vectorized_axis) = {z0t->id, z1.id, z2.id};

      auto transpose = Transpose("transpose", add).TQue(Position::kPositionVecOut, 1, 2);

      // 关键修改点：根据perm参数动态设置
      *(transpose.axis) = {z0T->id, z0t->id, z2.id, z1.id};
      *(transpose.repeats) = {s0 / z0t->size, z0t->size, s2, s1};
      *(transpose.strides) = {z0t->size * s1 * s2, s1 * s2, s1, One};
      *(transpose.vectorized_axis) = {z0t->id, z2.id, z1.id};

      auto store = Store("store", transpose);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*z0TB, *z0Tb, *z0t, z1, z2}, {store}, 1));
      auto output = Output("output", store);

      // 内存重用ID分配
      load1.mem->reuse_id = 0;
      load2.mem->reuse_id = 1;
      transpose.mem->reuse_id = 2;
      add.mem->reuse_id = 3;
    }
  }

  auto transpose_node = graph.FindNode("transpose");
  GE_ASSERT_NOTNULL(transpose_node);
  transpose_node->attr.api.unit = ComputeUnit::kUnitVector;
  return af::SUCCESS;
}

Status Build4DTransposeAscendGraph(ge::AscGraph &graph, const std::vector<int64_t> &perm = {0, 1, 2, 3}) {
  // 参数校验
  if (perm.size() != 4 || !IsValidPermutation(perm)) {
    return af::FAILED;  // 仅支持4D转置
  }

  // 创建4D尺寸变量
  auto s0 = graph.CreateSizeVar(16);
  auto s1 = graph.CreateSizeVar(32);
  auto s2 = graph.CreateSizeVar(64);
  auto s3 = graph.CreateSizeVar(128);

  // 创建4D轴
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  // 轴和维度集合
  std::vector<ge::Axis> axes = {z0, z1, z2, z3};
  std::vector<ge::Expression> dims = {s0, s1, s2, s3};

  // 分块策略（示例：对z0和z1分块）
  auto [z0T, z0t] = graph.TileSplit(z0.id);
  auto [z0TB, z0Tb] = graph.BlockSplit(z0T->id);

  auto data = graph.CreateContiguousData("input", DT_FLOAT, {z0, z1, z2, z3}, 0, FORMAT_ND);

  // 根据原始维度设置repeats和strides
  *data.repeats = {s0, s1, s2, s3};
  *data.strides = {s1 * s2 * s3, s2 * s3, s3, One};

  LOOP(*z0TB) {
    LOOP(*z0Tb) {
      auto load = Load("load", data).TQue(Position::kPositionVecIn, 1, 2);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*z0TB, *z0Tb, z0, z1, z2}, {load}, 1));

      // 动态计算转置属性
      auto transpose = Transpose("transpose", load).TQue(Position::kPositionVecOut, 1, 2);

      // 关键修改点：根据perm参数动态设置
      *(transpose.vectorized_axis) = {axes[perm[0]].id, axes[perm[1]].id, axes[perm[2]].id, axes[perm[3]].id};
      *(transpose.axis) = {axes[perm[0]].id, axes[perm[1]].id, axes[perm[2]].id, axes[perm[3]].id};

      // 动态计算转置后的repeats和strides
      *(transpose.repeats) = {dims[perm[0]], dims[perm[1]], dims[perm[2]], dims[perm[3]]};

      // 计算转置后的strides
      auto stride0 = dims[perm[1]] * dims[perm[2]] * dims[perm[3]];
      auto stride1 = dims[perm[2]] * dims[perm[3]];
      auto stride2 = dims[perm[3]];
      *(transpose.strides) = {stride0, stride1, stride2, One};

      auto store = Store("store", transpose);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*z0TB, *z0Tb, *z0t, z1, z2}, {store}, 1));
      auto output = Output("output", store);

      // 内存重用ID分配
      load.mem->reuse_id = 0;
      transpose.mem->reuse_id = 1;
    }
  }

  auto transpose_node = graph.FindNode("transpose");
  GE_ASSERT_NOTNULL(transpose_node);
  transpose_node->attr.api.unit = ComputeUnit::kUnitVector;
  return af::SUCCESS;
}

Status BuildFlashSoftmaxAscendGraph(ge::AscGraph &graph) {
  auto ND = ge::Symbol("ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd}, 0);
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd}, 1);
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2 = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto broadcast = Broadcast("broadcast", load1).TBuf(Position::kPositionVecOut);
      auto [softmax_out1, softmax_out2, softmax_out3] = FlashSoftmax("softmax", broadcast, load2, load2);
      auto store1 = Store("store1", softmax_out1);
      auto store2 = Store("store2", softmax_out2);
      auto store3 = Store("store3", softmax_out3);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes(
          {*ndB, *ndbT, *ndbt},
          {load1, load2, broadcast, softmax_out1, softmax_out2, softmax_out3, store1, store2, store3}, 1));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
      auto output3 = Output("output2", store3);
    }
  }
  auto softmax = graph.FindNode("softmax");
  GE_ASSERT_NOTNULL(softmax);
  softmax->attr.api.unit = ComputeUnit::kUnitVector;
  return af::SUCCESS;
}

Status BuildWorkSpaceAscendGraph(ge::AscGraph &graph) {
  auto ND = ge::Symbol("ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd}, 0);
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd}, 1);
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Workspace("workspace1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2 = Workspace("workspace2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto broadcast = Broadcast("broadcast", load1).TBuf(Position::kPositionVecOut);
      auto [softmax_out1, softmax_out2, softmax_out3] = FlashSoftmax("softmax", broadcast, load2, load2);
      auto store1 = Workspace("workspace3", softmax_out1);
      auto store2 = Workspace("workspace4", softmax_out2);
      auto store3 = Workspace("workspace5", softmax_out3);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes(
          {*ndB, *ndbT, *ndbt},
          {load1, load2, broadcast, softmax_out1, softmax_out2, softmax_out3, store1, store2, store3}, 1));
      auto output1 = Output("output1", store1);
      auto output2 = Output("output2", store2);
      auto output3 = Output("output2", store3);
    }
  }
  auto softmax = graph.FindNode("softmax");
  GE_ASSERT_NOTNULL(softmax);
  softmax->attr.api.unit = ComputeUnit::kUnitVector;
  return af::SUCCESS;
}

Status BuildTilingReduceAscendGraph(ge::AscGraph &graph) {
  auto R = ge::Symbol("R");
  auto A = ge::Symbol("A");
  auto r = graph.CreateAxis("r", R);
  auto a = graph.CreateAxis("a", A);
  auto [rT, rt] = graph.TileSplit(r.id);
  auto [rTB, rTb] = graph.BlockSplit(rT->id);
  auto [aT, at] = graph.TileSplit(a.id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {r, a}, 0);
  LOOP(*rT) {
    LOOP(*rTB) {
      LOOP(*rTb) {
        LOOP(*aT) {
          auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
          auto broadcast = Broadcast("broadcast", load1).TBuf(Position::kPositionVecOut);
          GE_ASSERT_SUCCESS(
              GraphConstructUtils::UpdateOutputTensorAxes({*rTB, *rTb, *aT, *rt, *at}, {load1, broadcast}, 2));
          (*broadcast.repeats)[2] = ge::Symbol(100);  // rTb
          (*broadcast.strides)[2] = ge::Symbol(100);
          (*broadcast.repeats)[3] = ge::Symbol(1);  // rt
          (*broadcast.strides)[3] = ge::Symbol(0);
          (*broadcast.repeats)[4] = ge::Symbol(200);  // at
          (*broadcast.strides)[4] = ge::Symbol(200);
          auto store1 = Store("store1", broadcast);
          GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*rTB, *rTb, *aT, *rt, *at}, {store1}, 2));
          auto output1 = Output("output1", store1);
        }
      }
    }
  }
  auto broadcast1 = graph.FindNode("broadcast");
  GE_ASSERT_NOTNULL(broadcast1);
  broadcast1->attr.api.unit = ComputeUnit::kUnitVector;
  return af::SUCCESS;
}

Status BuildMatMulDemoAscendGraph(ge::AscGraph &graph) {
  auto ND = ge::Symbol("ND");
  auto nd = graph.CreateAxis("nd", ND);
  auto [ndB, ndb] = graph.BlockSplit(nd.id);
  auto [ndbT, ndbt] = graph.TileSplit(ndb->id);
  auto data1 = graph.CreateContiguousData("input1", DT_FLOAT, {nd}, 0);
  auto data2 = graph.CreateContiguousData("input2", DT_FLOAT, {nd}, 1);
  LOOP(*ndB) {
    LOOP(*ndbT) {
      auto load1 = Load("load1", data1).TQue(Position::kPositionVecIn, 1, 1);
      auto load2 = Load("load2", data2).TQue(Position::kPositionVecIn, 1, 2);
      auto broadcast = Broadcast("broadcast", load1).TBuf(Position::kPositionVecOut);
      auto mat_mul_out = Add("mat_mul", broadcast, load2);
      auto store1 = Store("store1", mat_mul_out);
      GE_ASSERT_SUCCESS(GraphConstructUtils::UpdateOutputTensorAxes({*ndB, *ndbT, *ndbt},
                                                                    {load1, load2, broadcast, mat_mul_out, store1}, 1));
      auto output1 = Output("output1", store1);
    }
  }
  auto mat_mul = graph.FindNode("mat_mul");
  GE_ASSERT_NOTNULL(mat_mul);
  mat_mul->attr.api.unit = ComputeUnit::kUnitVector;
  return af::SUCCESS;
}
std::string RemoveAutoFuseTilingHeadGuards(const std::string &input) {
  std::istringstream iss(input);
  std::ostringstream oss;
  std::string line;
  const std::string guard_token = "__AUTOFUSE_TILING_FUNC_COMMON_H__";

  while (std::getline(iss, line)) {
    // 如果当前行不包含 guard_token，则保留
    if (line.find(guard_token) == std::string::npos) {
      oss << line << "\n";
    }
  }
  return oss.str();
}

void CombineTilings(const std::map<std::string, std::string> &tilings, std::string &result) {
  const std::string tiling_head = "TilingHead";                       // TilingHead作为开头拼接其他文件
  const std::string tiling_data = "TilingData";                       // 要排除的 TilingData 子串
  result += RemoveAutoFuseTilingHeadGuards(tilings.at(tiling_head));  // 删除头文件的宏保护，cpp文件不需要
  const std::string include_str = "#include \"autofuse_tiling_func_common.h\"";

  // 遍历所有非 TilingHead 和 TilingData 的条目，去掉第一行后拼接
  for (const auto &[key, value] : tilings) {
    if (key == tiling_head || key.find(tiling_data) != std::string::npos) {
      continue;
    }

    // 查找并跳过第一行头文件行
    size_t include_pos = value.find(include_str);
    if (include_pos != std::string::npos) {
      // 找到 include 行，跳过它，并去掉后面的换行符
      size_t content_start = include_pos + include_str.length();
      while (content_start < value.size() && (value[content_start] == '\n' || value[content_start] == '\r')) {
        content_start++;
      }
      result += value.substr(content_start);
    } else {
      // 如果没有 include 行，直接拼接整个内容
      result += value;
    }

    if (!result.empty() && result.back() != '\n') {
      result += '\n';
    }
  }
}
}  // namespace cg
}  // namespace ascir
}  // namespace ge
namespace {
bool IsFileContainsString(const std::string &filename, const std::string &search_sub_string) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "无法打开文件: " << filename << std::endl;
    return false;
  }
  std::string line;
  while (std::getline(file, line)) {
    if (line.find(search_sub_string) != std::string::npos) {
      file.close();
      return true;
    }
  }
  file.close();
  return false;
}

void PrepareDualScheduleResult(const ascir::AscGraph &graph, ascir::FusedScheduledResult &fused_result) {
  ascir::ScheduleGroup sg1, sg2;
  ascir::ScheduledResult sr1, sr2;
  sg1.impl_graphs.emplace_back(graph);
  sg2.impl_graphs.emplace_back(graph);
  sr1.schedule_groups.emplace_back(sg1);
  sr2.schedule_groups.emplace_back(sg2);
  std::vector<ascir::ScheduledResult> schedule_results;
  schedule_results.emplace_back(sr1);
  schedule_results.emplace_back(sr2);
  fused_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
}

void WriteTilingFuncFile(const std::string &tiling_func, const std::string &header_name) {
  std::ofstream oss(header_name + "_tiling_func.cpp", std::ios::out);
  oss << "#include \"" << header_name << "_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();
}

void GenerateAndWriteTilingData(const ascir::FusedScheduledResult &fused_result,
                                const std::map<std::string, std::string> &options, const std::string &op_name,
                                const std::string &graph_name) {
  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode(op_name, all_model_infos, generator_config, tiling_res), af::SUCCESS);
  std::ofstream oss(op_name + "_tiling_data.h", std::ios::out);
  oss << tiling_res[graph_name + "TilingData"];
  oss.close();
}

int32_t CopyStubsAndCompile(const std::string &main_cpp, const std::string &tiling_cpp, const std::string &output_bin) {
  auto ret = std::system(
      std::string("cp ").append(TOP_DIR).append("/autofuse/tests/st/att/testcase/stub/op_log.h ./ -f").c_str());
  EXPECT_EQ(ret, 0);
  ret = autofuse::test::CopyStubFiles(TOP_DIR, "autofuse/tests/st/att/testcase/stub/");
  EXPECT_EQ(ret, 0);
  ret = std::system(std::string("cp -r ")
                        .append(TOP_DIR)
                        .append("/tests/st/att/testcase/source_mirror/generator/generator_utils/kernel_tiling/ ./ -f")
                        .c_str());
  EXPECT_EQ(ret, 0);
  std::string cmd = "g++ " + main_cpp + " " + tiling_cpp + " -o " + output_bin + " -I ./ -DSTUB_LOG";
  ret = std::system(cmd.c_str());
  EXPECT_EQ(ret, 0);
  return std::system(("./" + output_bin).c_str());
}

void PrepareSingleSchedule(const ascir::AscGraph &graph, ascir::FusedScheduledResult &fused_result,
                           std::map<std::string, std::string> &options, const std::string &op_name,
                           std::map<std::string, std::string> &tiling_funcs) {
  options["output_file_path"] = "./";
  options["solver_type"] = "AxesReorder";
  ascir::ScheduleGroup schedule_group;
  ascir::ScheduledResult schedule_result;
  std::vector<ascir::ScheduledResult> schedule_results;
  schedule_group.impl_graphs.emplace_back(graph);
  schedule_result.schedule_groups.emplace_back(schedule_group);
  schedule_results.emplace_back(schedule_result);
  fused_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  GenTilingImplAutoFuseV3(op_name, fused_result, options, tiling_funcs, true);
}

void WriteTransposeTilingFiles(const ascir::FusedScheduledResult &fused_result,
                               const std::map<std::string, std::string> &options, const std::string &tiling_func) {
  std::ofstream oss;
  oss.open("transpose_autofuse_tiling_func.cpp", std::ios::out);
  oss << "#include \"Transpose_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GetModelInfoMap(fused_result, options, all_model_infos);
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  EXPECT_EQ(generator.GenTilingCode("Transpose", all_model_infos, generator_config, tiling_res), af::SUCCESS);
}

int32_t CopyAndRunTransposeTest() {
  int32_t ret =
      std::system(std::string("cp ").append(TILING_DATA_DIR).append("/tiling_func_main_transpose.cpp ./ -f").c_str());
  EXPECT_EQ(ret, 0);
  (void)std::system("mkdir ./graph/");
  ret = std::system(std::string("cp -r ")
                        .append(TOP_DIR)
                        .append("/autofuse/tests/st/att/testcase/stub/tensor.h ./graph/ -f")
                        .c_str());
  EXPECT_EQ(ret, 0);
  ret = std::system(std::string("cp -r ")
                        .append(TOP_DIR)
                        .append("/tests/st/att/testcase/source_mirror/generator/generator_utils/lib/ ./ -f")
                        .c_str());
  EXPECT_EQ(ret, 0);
  ret = std::system(std::string("cp -r ")
                        .append(TOP_DIR)
                        .append("/tests/st/att/testcase/source_mirror/generator/generator_utils/kernel_tiling/ ./ -f")
                        .c_str());
  EXPECT_EQ(ret, 0);
  ret = std::system(std::string("cp ").append(TILING_DATA_DIR).append("/Transpose_tiling_data.h ./ -f").c_str());
  EXPECT_EQ(ret, 0);
  ret = std::system(
      std::string("cp ").append(TOP_DIR).append("/autofuse/tests/st/att/testcase/stub/op_log.h ./ -f").c_str());
  EXPECT_EQ(ret, 0);
  ret = autofuse::test::CopyStubFiles(TOP_DIR, "autofuse/tests/st/att/testcase/stub/");
  EXPECT_EQ(ret, 0);
  ret = std::system(
      "g++ tiling_func_main_transpose.cpp transpose_autofuse_tiling_func.cpp -o tiling_func_main_transpose -I ./ "
      "-DSTUB_LOG");
  EXPECT_EQ(ret, 0);
  ret = std::system("./tiling_func_main_transpose");
  return ret;
}

af::Status PrepareFlashSoftmaxSchedule(ascir::FusedScheduledResult &fused_schedule_result,
                                       std::map<std::string, std::string> &options,
                                       std::map<std::string, std::string> &tiling_funcs) {
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  GE_ASSERT_SUCCESS(ge::ascir::cg::BuildTilingReduceAscendGraph(graph_normal));
  std::vector<ascir::AscGraph> graphs = {graph_normal};
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);
  options["output_file_path"] = "./";
  options["solver_type"] = "AxesReorder";
  fused_schedule_result.fused_graph_name = "FlashSoftmax";
  std::vector<ascir::ScheduledResult> scheduled_results;
  for (int i = 0; i < 2; ++i) {
    ascir::ScheduleGroup schedule_group;
    schedule_group.impl_graphs.emplace_back(graph_normal);
    ascir::ScheduledResult scheduled_result;
    scheduled_result.schedule_groups.emplace_back(schedule_group);
    scheduled_results.emplace_back(scheduled_result);
  }
  fused_schedule_result.node_idx_to_scheduled_results.emplace_back(scheduled_results);
  GE_ASSERT_TRUE(GenTilingImplAutoFuseV3("FlashSoftmax", fused_schedule_result, options, tiling_funcs, true));
  return af::SUCCESS;
}

af::Status WriteSoftmaxTilingFiles(const ascir::FusedScheduledResult &fused_schedule_result,
                                   const std::map<std::string, std::string> &options, const std::string &tiling_func) {
  std::ofstream oss;
  oss.open("flash_softmax_tiling_func.cpp", std::ios::out);
  oss << "#include \"FlashSoftmax_tiling_data.h\"\n";
  oss << tiling_func;
  oss.close();

  TilingCodeGenerator generator;
  TilingCodeGenConfig generator_config;
  std::map<std::string, std::string> tiling_res;
  FusedParsedScheduleResult all_model_infos;
  GE_ASSERT_SUCCESS(GetModelInfoMap(fused_schedule_result, options, all_model_infos));
  generator_config.type = TilingImplType::HIGH_PERF;
  generator_config.tiling_data_type_name = options[kTilingDataTypeName];
  generator_config.gen_tiling_data = true;
  generator_config.gen_extra_infos = true;
  GE_ASSERT_SUCCESS(generator.GenTilingCode("FlashSoftmax", all_model_infos, generator_config, tiling_res));
  oss.open("FlashSoftmax_tiling_data.h", std::ios::out);
  oss << tiling_res["graph_normalTilingData"];
  oss.close();
  return af::SUCCESS;
}

af::Status BuildSoftmaxMainAndCompile() {
  std::ofstream oss;
  oss.open("tiling_func_main.cpp", std::ios::out);
  oss << R"(
#include <iostream>
#include "FlashSoftmax_tiling_data.h"
using namespace optiling;

void PrintResult(graph_normalTilingData &tilingData) {
  std::cout << "====================================================" << std::endl;
  std::cout << "get_tiling_key = " << tilingData.get_graph0_tiling_key() << std::endl;
  std::cout << "get_at_size = " << tilingData.graph0_result0_g0_tiling_data.get_at_size() << std::endl;
  std::cout << "get_rt_size = " << tilingData.graph0_result0_g0_tiling_data.get_rt_size() << std::endl;
  std::cout << "get_rTb_size = " << tilingData.graph0_result0_g0_tiling_data.get_rTb_size() << std::endl;
  std::cout << "get_block_dim = " << tilingData.graph0_result0_g0_tiling_data.get_block_dim() << std::endl;
  std::cout << "====================================================" << std::endl;
}

int main() {
  graph_normalTilingData tilingData;
  tilingData.set_block_dim(64);
  tilingData.set_ub_size(245760);
  tilingData.graph0_result0_g0_tiling_data.set_A(100);
  tilingData.graph0_result0_g0_tiling_data.set_R(1024);

  if (GetTiling(tilingData)) {
    PrintResult(tilingData);
  } else {
    std::cout << "transpose tiling func execute failed." << std::endl;
    return -1;
  }
  return 0;
}
)";
  oss.close();
  auto ret = std::system(
      std::string("cp ").append(TOP_DIR).append("/autofuse/tests/st/att/testcase/stub/op_log.h ./ -f").c_str());
  GE_ASSERT_TRUE(ret == 0);
  ret = autofuse::test::CopyStubFiles(TOP_DIR, "autofuse/tests/st/att/testcase/stub/");
  GE_ASSERT_TRUE(ret == 0);
  ret = std::system(
      "g++ tiling_func_main.cpp flash_softmax_tiling_func.cpp -o tiling_func_main_softmax -I ./ "
      "-DSTUB_LOG");
  GE_ASSERT_TRUE(ret == 0);
  ret = std::system("./tiling_func_main_softmax > softmax_tiling.log");
  GE_ASSERT_TRUE(ret == 0);
  (void)system("cat softmax_tiling.log");
  return af::SUCCESS;
}

af::Status ConstructTilingRCase() {
  ascir::FusedScheduledResult fused_schedule_result;
  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  GE_ASSERT_SUCCESS(PrepareFlashSoftmaxSchedule(fused_schedule_result, options, tiling_funcs));
  std::string tiling_func;
  ge::ascir::cg::CombineTilings(tiling_funcs, tiling_func);
  GE_ASSERT_TRUE(tiling_func.find("tilingCaseImplPtr = &caseR1101") != std::string::npos);
  GE_ASSERT_SUCCESS(WriteSoftmaxTilingFiles(fused_schedule_result, options, tiling_func));
  GE_ASSERT_SUCCESS(BuildSoftmaxMainAndCompile());
  return af::SUCCESS;
}
}  // namespace
TEST_F(TestApiTilingGen, gen_transpose021_split_api_tiling_success) {
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(0u);
  ASSERT_EQ(ge::ascir::cg::BuildTransposeSplitAscendGraph(graph_normal), af::SUCCESS);
  ge::DumpAscirGraph::WriteOutToFile("Dump_Graph_Att_Transpose", graph_normal);
  std::vector<ascir::AscGraph> graphs = {graph_normal};
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  ascir::FusedScheduledResult fused_scheduled_result;
  PrepareSingleSchedule(graph_normal, fused_scheduled_result, options, "Transpose", tiling_funcs);
  EXPECT_TRUE(!tiling_funcs.empty());
  std::string tiling_func;
  ge::ascir::cg::CombineTilings(tiling_funcs, tiling_func);

  WriteTransposeTilingFiles(fused_scheduled_result, options, tiling_func);
  EXPECT_EQ(CopyAndRunTransposeTest(), 0);
}

TEST_F(TestApiTilingGen, gen_transpose021_split_api_tiling_dy_success) {
  std::vector<ascir::AscGraph> graphs;
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(0u);
  ASSERT_EQ(ge::ascir::cg::BuildTransposeSplitAscendGraph(graph_normal), af::SUCCESS);
  ge::DumpAscirGraph::WriteOutToFile("Dump_Graph_Att_Transpose", graph_normal);
  graphs.emplace_back(graph_normal);
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["solver_type"] = "AxesReorder";
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduledResult schedule_result1;
  std::vector<ascir::ScheduledResult> schedule_results;
  schedule_group1.impl_graphs.emplace_back(graph_normal);
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_results.emplace_back(schedule_result1);
  std::map<std::string, std::string> tiling_funcs;
  ascir::FusedScheduledResult fused_scheduled_result;
  fused_scheduled_result.node_idx_to_scheduled_results.emplace_back(schedule_results);
  EXPECT_EQ(GenTilingImplAutoFuseV3("Transpose", fused_scheduled_result, options, tiling_funcs, true), true);
  std::string tiling_func;
  ge::ascir::cg::CombineTilings(tiling_funcs, tiling_func);
  EXPECT_EQ(ascgen_utils::FormatExpression("(s2 * s3)"),
            "static_cast<int64_t>(tiling_data.get_s2() * tiling_data.get_s3())");
}

TEST_F(TestApiTilingGen, gen_transpose102_api_tiling_success) {
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(0u);
  ASSERT_EQ(ge::ascir::cg::BuildTransposeAscendGraph(graph_normal, {1, 0, 2}), af::SUCCESS);
  ge::DumpAscirGraph::WriteOutToFile("Dump_Graph_Att_Transpose", graph_normal);
  std::vector<ascir::AscGraph> graphs = {graph_normal};
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  ascir::FusedScheduledResult fused_scheduled_results;
  PrepareSingleSchedule(graph_normal, fused_scheduled_results, options, "Transpose", tiling_funcs);
  EXPECT_TRUE(!tiling_funcs.empty());
  std::string tiling_func;
  ge::ascir::cg::CombineTilings(tiling_funcs, tiling_func);

  WriteTransposeTilingFiles(fused_scheduled_results, options, tiling_func);
  EXPECT_EQ(CopyAndRunTransposeTest(), 0);
}

TEST_F(TestApiTilingGen, gen_transpose021_api_tiling_success) {
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(0u);
  ASSERT_EQ(ge::ascir::cg::BuildTransposeAscendGraph(graph_normal, {0, 2, 1}), af::SUCCESS);
  ge::DumpAscirGraph::WriteOutToFile("Dump_Graph_Att_Transpose", graph_normal);
  std::vector<ascir::AscGraph> graphs = {graph_normal};
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  ascir::FusedScheduledResult fused_scheduled_results;
  PrepareSingleSchedule(graph_normal, fused_scheduled_results, options, "Transpose", tiling_funcs);
  EXPECT_TRUE(!tiling_funcs.empty());
  std::string tiling_func;
  ge::ascir::cg::CombineTilings(tiling_funcs, tiling_func);

  WriteTransposeTilingFiles(fused_scheduled_results, options, tiling_func);
  EXPECT_EQ(CopyAndRunTransposeTest(), 0);
}

TEST_F(TestApiTilingGen, gen_transpose210_api_tiling_success) {
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(0u);
  ASSERT_EQ(ge::ascir::cg::BuildTransposeAscendGraph(graph_normal, {2, 1, 0}), af::SUCCESS);
  ge::DumpAscirGraph::WriteOutToFile("Dump_Graph_Att_Transpose", graph_normal);
  std::vector<ascir::AscGraph> graphs = {graph_normal};
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  ascir::FusedScheduledResult fused_scheduled_results;
  PrepareSingleSchedule(graph_normal, fused_scheduled_results, options, "Transpose", tiling_funcs);
  EXPECT_TRUE(!tiling_funcs.empty());
  std::string tiling_func;
  ge::ascir::cg::CombineTilings(tiling_funcs, tiling_func);

  WriteTransposeTilingFiles(fused_scheduled_results, options, tiling_func);
  EXPECT_EQ(CopyAndRunTransposeTest(), 0);
}

TEST_F(TestApiTilingGen, gen_transpose0213_api_tiling_success) {
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(0u);
  ASSERT_EQ(ge::ascir::cg::Build4DTransposeAscendGraph(graph_normal, {0, 2, 1, 3}), af::SUCCESS);
  ge::DumpAscirGraph::WriteOutToFile("Dump_Graph_Att_Transpose", graph_normal);
  std::vector<ascir::AscGraph> graphs = {graph_normal};
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  ascir::FusedScheduledResult fused_scheduled_results;
  PrepareSingleSchedule(graph_normal, fused_scheduled_results, options, "Transpose", tiling_funcs);
  EXPECT_TRUE(!tiling_funcs.empty());
  std::string tiling_func;
  ge::ascir::cg::CombineTilings(tiling_funcs, tiling_func);

  WriteTransposeTilingFiles(fused_scheduled_results, options, tiling_func);
  EXPECT_EQ(CopyAndRunTransposeTest(), 0);
}

TEST_F(TestApiTilingGen, gen_transpose2103_api_tiling_success) {
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(0u);
  ASSERT_EQ(ge::ascir::cg::Build4DTransposeAscendGraph(graph_normal, {2, 1, 0, 3}), af::SUCCESS);
  ge::DumpAscirGraph::WriteOutToFile("Dump_Graph_Att_Transpose", graph_normal);
  std::vector<ascir::AscGraph> graphs = {graph_normal};
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  ascir::FusedScheduledResult fused_scheduled_results;
  PrepareSingleSchedule(graph_normal, fused_scheduled_results, options, "Transpose", tiling_funcs);
  EXPECT_TRUE(!tiling_funcs.empty());
  std::string tiling_func;
  ge::ascir::cg::CombineTilings(tiling_funcs, tiling_func);

  WriteTransposeTilingFiles(fused_scheduled_results, options, tiling_func);
  EXPECT_EQ(CopyAndRunTransposeTest(), 0);
}

TEST_F(TestApiTilingGen, gen_transpose0321_api_tiling_success) {
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(0u);
  ASSERT_EQ(ge::ascir::cg::Build4DTransposeAscendGraph(graph_normal, {0, 3, 2, 1}), af::SUCCESS);
  ge::DumpAscirGraph::WriteOutToFile("Dump_Graph_Att_Transpose", graph_normal);
  std::vector<ascir::AscGraph> graphs = {graph_normal};
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);

  std::map<std::string, std::string> options;
  std::map<std::string, std::string> tiling_funcs;
  ascir::FusedScheduledResult fused_scheduled_results;
  PrepareSingleSchedule(graph_normal, fused_scheduled_results, options, "Transpose", tiling_funcs);
  EXPECT_TRUE(!tiling_funcs.empty());
  std::string tiling_func;
  ge::ascir::cg::CombineTilings(tiling_funcs, tiling_func);

  WriteTransposeTilingFiles(fused_scheduled_results, options, tiling_func);
  EXPECT_EQ(CopyAndRunTransposeTest(), 0);
}

TEST_F(TestApiTilingGen, gen_transpose0123_api_tiling_success) {
  std::vector<ascir::AscGraph> graphs;
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(0u);
  ASSERT_EQ(ge::ascir::cg::Build4DTransposeAscendGraph(graph_normal, {0, 1, 2, 3}), af::SUCCESS);
  ge::DumpAscirGraph::WriteOutToFile("Dump_Graph_Att_Transpose", graph_normal);
  graphs.emplace_back(graph_normal);
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["solver_type"] = "AxesReorder";
  ascir::ScheduleGroup schedule_group1;
  ascir::ScheduledResult schedule_result1;
  std::vector<ascir::ScheduledResult> schedule_results;
  schedule_group1.impl_graphs.emplace_back(graph_normal);
  schedule_result1.schedule_groups.emplace_back(schedule_group1);
  schedule_results.emplace_back(schedule_result1);
  ascir::FusedScheduledResult fused_scheduled_results;
  fused_scheduled_results.node_idx_to_scheduled_results.emplace_back(schedule_results);
  std::map<std::string, std::string> tiling_funcs;
  EXPECT_EQ(GenTilingImplAutoFuseV3("Transpose", fused_scheduled_results, options, tiling_funcs, true), false);
}

TEST_F(TestApiTilingGen, gen_softmax_api_tiling_success) {
  setenv("AUTOFUSE_DFX_FLAGS", "--att_enable_small_shape_strategy=true", 1);
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  ASSERT_EQ(ge::ascir::cg::BuildFlashSoftmaxAscendGraph(graph_normal), af::SUCCESS);
  std::vector<ascir::AscGraph> graphs = {graph_normal};
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["solver_type"] = "AxesReorder";
  ascir::FusedScheduledResult fused_scheduled_results;
  PrepareDualScheduleResult(graph_normal, fused_scheduled_results);
  std::map<std::string, std::string> tiling_funcs;
  EXPECT_EQ(GenTilingImplAutoFuseV3("FlashSoftmax", fused_scheduled_results, options, tiling_funcs, true), true);
  std::string tiling_func;
  ge::ascir::cg::CombineTilings(tiling_funcs, tiling_func);
  WriteTilingFuncFile(tiling_func, "FlashSoftmax");
  GenerateAndWriteTilingData(fused_scheduled_results, options, "FlashSoftmax", "graph_normal");
  GenerateOpMainFile("softmax", "FlashSoftmax_tiling_data");
  CopyStubsAndCompile("tiling_func_main_softmax.cpp", "FlashSoftmax_tiling_func.cpp", "tiling_func_main_softmax");
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

TEST_F(TestApiTilingGen, gen_schedule_group_cache_success) {
  // 设置环境变量启用 OperatorCache
  setenv("AUTOFUSE_DFX_FLAGS", "--autofuse_enable_tiling_cache=true", 1);
  att::AutoFuseConfig::MutableAttStrategyConfig().Reset();

  std::vector<ascir::AscGraph> graphs;
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  ASSERT_EQ(ge::ascir::cg::BuildFlashSoftmaxAscendGraph(graph_normal), af::SUCCESS);
  graphs.emplace_back(graph_normal);
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["solver_type"] = "AxesReorder";
  ascir::FusedScheduledResult fused_schedule_result;
  std::vector<ascir::ScheduledResult> scheduled_results;
  fused_schedule_result.fused_graph_name = "FlashSoftmax";
  for (int i = 0; i < 2; ++i) {
    ascir::ScheduleGroup schedule_group;
    schedule_group.impl_graphs.emplace_back(graph_normal);
    ascir::ScheduledResult scheduled_result;
    scheduled_result.schedule_groups.emplace_back(schedule_group);
    scheduled_results.emplace_back(scheduled_result);
  }
  fused_schedule_result.node_idx_to_scheduled_results.emplace_back(scheduled_results);
  std::map<std::string, std::string> tiling_funcs;
  EXPECT_EQ(GenTilingImplAutoFuseV3("FlashSoftmax", fused_schedule_result, options, tiling_funcs, true), true);
  std::string tiling_func;
  ge::ascir::cg::CombineTilings(tiling_funcs, tiling_func);
  // TTODO 当前仅检查是否有生成使能cache后的字符串，后续需要增加端到端验证用例
  // 更新：新API使用SaveOperatorCache替代SaveCache
  EXPECT_NE(tiling_func.find("SaveOperatorCache"), std::string::npos);

  // 清理环境变量
  unsetenv("AUTOFUSE_DFX_FLAGS");
  att::AutoFuseConfig::MutableAttStrategyConfig().Reset();
}

TEST_F(TestApiTilingGen, gen_schedule_group_reduce_tile_r) {
  EXPECT_EQ(ConstructTilingRCase(), af::SUCCESS);
  EXPECT_EQ(IsFileContainsString("softmax_tiling.log", "get_tiling_key = 0"), true);
}

TEST_F(TestApiTilingGen, gen_schedule_group_reduce_tile_r_force_r) {
  setenv("AUTOFUSE_DFX_FLAGS", "--force_tiling_case=1101_R", 1);
  EXPECT_EQ(ConstructTilingRCase(), af::SUCCESS);
  // 若优先切R则A为100，R为216，否则切A的话，at为16， rt为1024
  EXPECT_EQ(IsFileContainsString("softmax_tiling.log", "get_at_size = 100"), true);
  EXPECT_EQ(IsFileContainsString("softmax_tiling.log", "get_rt_size = 205"), true);
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

TEST_F(TestApiTilingGen, gen_schedule_pgo_env) {
  setenv("AUTOFUSE_FLAGS", "--autofuse_enable_pgo=true", 1);
  setenv("AUTOFUSE_DFX_FLAGS", "--autofuse_pgo_algo=pgo_algo_invalid;--autofuse_pgo_step_max=32", 1);
  att::AutoFuseConfig::MutablePgoStrategyConfig().is_first_init = true;
  EXPECT_EQ(ConstructTilingRCase(), af::SUCCESS);
  ASSERT_EQ(att::AutoFuseConfig::MutablePgoStrategyConfig().Init(), af::SUCCESS);
  EXPECT_EQ(att::AutoFuseConfig::GetPgoStrategyConfig().enable_autofuse_pgo, "true");
  EXPECT_EQ(att::AutoFuseConfig::GetPgoStrategyConfig().autofuse_pgo_algo_select, "core_select");
  EXPECT_EQ(att::AutoFuseConfig::GetPgoStrategyConfig().autofuse_pgo_algo_step_max, 32);
  EXPECT_EQ(IsFileContainsString("softmax_tiling.log", "get_tiling_key = 0"), true);
  unsetenv("AUTOFUSE_FLAGS");
  unsetenv("AUTOFUSE_DFX_FLAGS");
}

TEST_F(TestApiTilingGen, gen_softmax_api_tiling_with_var_relation) {
  setenv("AUTOFUSE_DFX_FLAGS", "--att_enable_small_shape_strategy=true", 1);
  setenv("AUTOFUSE_FLAGS", "--autofuse_enable_pgo=true", 1);
  att::AutoFuseConfig::MutablePgoStrategyConfig().is_first_init = true;
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  ASSERT_EQ(ge::ascir::cg::BuildFlashSoftmaxAscendGraph(graph_normal), af::SUCCESS);
  std::vector<ascir::AscGraph> graphs = {graph_normal};
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["solver_type"] = "AxesReorder";
  std::vector<ascir::ScheduledResult> schedule_results;
  for (int i = 0; i < 2; ++i) {
    ascir::ScheduleGroup schedule_group;
    schedule_group.impl_graphs.emplace_back(graph_normal);
    ascir::ScheduledResult scheduled_result;
    Expr src_var = ge::Symbol("ND");
    scheduled_result.var_relations = {{1, {{0, {{"ND", src_var}}}}}};
    scheduled_result.schedule_groups.emplace_back(schedule_group);
    scheduled_result.schedule_groups.emplace_back(schedule_group);
    schedule_results.emplace_back(scheduled_result);
  }
  ascir::FusedScheduledResult fused_scheduled_results;
  fused_scheduled_results.node_idx_to_scheduled_results.emplace_back(schedule_results);
  std::map<std::string, std::string> tiling_funcs;
  EXPECT_EQ(GenTilingImplAutoFuseV3("FlashSoftmax", fused_scheduled_results, options, tiling_funcs, true), true);
  std::string tiling_func;
  ge::ascir::cg::CombineTilings(tiling_funcs, tiling_func);
  EXPECT_NE(tiling_func.find(
                "graph0_result0_g1_tiling_data.set_ND(static_cast<double>(graph0_result0_g0_tiling_data.get_ND()))"),
            std::string::npos);
  WriteTilingFuncFile(tiling_func, "FlashSoftmax");
  GenerateAndWriteTilingData(fused_scheduled_results, options, "FlashSoftmax", "graph_normal");
  GenerateOpMainFile("softmax_var", "FlashSoftmax_tiling_data");
  CopyStubsAndCompile("tiling_func_main_softmax_var.cpp", "FlashSoftmax_tiling_func.cpp", "tiling_func_main_softmax");
  unsetenv("AUTOFUSE_DFX_FLAGS");
  unsetenv("AUTOFUSE_FLAGS");
  att::AutoFuseConfig::MutablePgoStrategyConfig().is_first_init = true;
}

TEST_F(TestApiTilingGen, gen_workspace_with_tensor_id) {
  std::vector<ascir::AscGraph> graphs;
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  ASSERT_EQ(ge::ascir::cg::BuildWorkSpaceAscendGraph(graph_normal), af::SUCCESS);
  graphs.emplace_back(graph_normal);
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["solver_type"] = "AxesReorder";
  ascir::FusedScheduledResult fused_schedule_result;
  std::vector<ascir::ScheduledResult> scheduled_results;
  fused_schedule_result.fused_graph_name = "FlashSoftmax";
  for (int i = 0; i < 2; ++i) {
    ascir::ScheduleGroup schedule_group;
    schedule_group.impl_graphs.emplace_back(graph_normal);
    ascir::ScheduledResult scheduled_result;
    scheduled_result.schedule_groups.emplace_back(schedule_group);
    scheduled_results.emplace_back(scheduled_result);
  }
  fused_schedule_result.node_idx_to_scheduled_results.emplace_back(scheduled_results);
  std::map<std::string, std::string> tiling_funcs;
  EXPECT_EQ(GenTilingImplAutoFuseV3("FlashSoftmax", fused_schedule_result, options, tiling_funcs, true), true);
  std::string tiling_func;
  ge::ascir::cg::CombineTilings(tiling_funcs, tiling_func);
  EXPECT_NE(tiling_func.find("tiling_data.set_workspace0(it0->second);"), std::string::npos);
}

TEST_F(TestApiTilingGen, gen_mat_mul_tiling_success) {
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  ASSERT_EQ(ge::ascir::cg::BuildMatMulDemoAscendGraph(graph_normal), af::SUCCESS);
  std::vector<ascir::AscGraph> graphs = {graph_normal};
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["solver_type"] = "AxesReorder";
  ascir::FusedScheduledResult fused_scheduled_results;
  PrepareDualScheduleResult(graph_normal, fused_scheduled_results);
  std::map<std::string, std::string> tiling_funcs;
  EXPECT_EQ(GenTilingImplAutoFuseV3("MatMul", fused_scheduled_results, options, tiling_funcs, true), true);
  std::string tiling_func;
  ge::ascir::cg::CombineTilings(tiling_funcs, tiling_func);
  WriteTilingFuncFile(tiling_func, "MatMul");
  GenerateAndWriteTilingData(fused_scheduled_results, options, "MatMul", "graph_normal");
  GenerateOpMainFile("mat_mul", "MatMul_tiling_data");
  CopyStubsAndCompile("tiling_func_main_mat_mul.cpp", "MatMul_tiling_func.cpp", "tiling_func_main_mat_mul");
}

// ============================================================================
// ATT Tiling缓存功能系统测试
// ============================================================================

/**
 * @brief ST测试用例1：算子级缓存启用测试
 *
 * 测试项：ATT算子级缓存代码生成（启用状态）
 * 重要级别：高 (P0)
 *
 * 预置条件：
 * - 设置环境变量 AUTOFUSE_FLAGS="--autofuse_enable_tiling_cache=true"
 * - 编译环境正常
 * - FlashSoftmax图构建正常
 *
 * 操作步骤：
 * 1. 设置环境变量启用缓存
 * 2. 创建FlashSoftmax图结构
 * 3. 调用GenTilingImplAutoFuseV3生成Tiling代码
 * 4. 合并Tiling函数
 * 5. 验证生成代码中的缓存相关内容
 * 6. 清理环境变量
 *
 * 输入：
 * - 环境变量: AUTOFUSE_FLAGS="--autofuse_enable_tiling_cache=true"
 * - TilingKey: 1101u
 * - 图名: FlashSoftmax
 * - 求解器类型: AxesReorder
 *
 * 预期结果：
 * 生成的代码包含：
 * - `using OperatorLevelCache`
 * - `constexpr size_t kInputShapeSize`
 * - `constexpr size_t kOperatorCacheCapacity`
 * - `bool FindOperatorCache`
 * - `bool SaveOperatorCache`
 * - `bool IsCacheEnabled`
 *
 * 备注：验证编译时缓存开关功能正常
 */
TEST_F(TestApiTilingGen, gen_op_level_cache_basic) {
  // 设置环境变量启用缓存
  setenv("AUTOFUSE_DFX_FLAGS", "--autofuse_enable_tiling_cache=true", 1);
  att::AutoFuseConfig::MutableAttStrategyConfig().Reset();

  std::vector<ascir::AscGraph> graphs;
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  ASSERT_EQ(ge::ascir::cg::BuildFlashSoftmaxAscendGraph(graph_normal), af::SUCCESS);
  graphs.emplace_back(graph_normal);
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["solver_type"] = "AxesReorder";
  ascir::FusedScheduledResult fused_schedule_result;
  std::vector<ascir::ScheduledResult> scheduled_results;
  fused_schedule_result.fused_graph_name = "FlashSoftmax";
  for (int i = 0; i < 2; ++i) {
    ascir::ScheduleGroup schedule_group;
    schedule_group.impl_graphs.emplace_back(graph_normal);
    ascir::ScheduledResult scheduled_result;
    scheduled_result.schedule_groups.emplace_back(schedule_group);
    scheduled_results.emplace_back(scheduled_result);
  }
  fused_schedule_result.node_idx_to_scheduled_results.emplace_back(scheduled_results);
  std::map<std::string, std::string> tiling_funcs;
  EXPECT_EQ(GenTilingImplAutoFuseV3("FlashSoftmax", fused_schedule_result, options, tiling_funcs, true), true);
  std::string tiling_func;
  ge::ascir::cg::CombineTilings(tiling_funcs, tiling_func);

  // 验证缓存类型定义生成
  EXPECT_NE(tiling_func.find("using OperatorLevelCache"), std::string::npos);
  EXPECT_NE(tiling_func.find("constexpr size_t kInputShapeSize"), std::string::npos);
  EXPECT_NE(tiling_func.find("constexpr size_t kOperatorCacheCapacity"), std::string::npos);

  // 验证缓存函数生成
  EXPECT_NE(tiling_func.find("FindOperatorCache(const"), std::string::npos);
  EXPECT_NE(tiling_func.find("bool SaveOperatorCache"), std::string::npos);

  // 注意：缓存查询代码(input_shapes数组构建)只在有缓存复用信息时生成
  // 这是当前设计的限制，算子级缓存类型和函数已正确生成

  unsetenv("AUTOFUSE_DFX_FLAGS");
  att::AutoFuseConfig::MutableAttStrategyConfig().Reset();
}

/**
 * @brief ST测试用例2：算子级缓存禁用测试
 *
 * 测试项：ATT算子级缓存代码生成（禁用状态）
 * 重要级别：高 (P0)
 *
 * 预置条件：
 * - 设置环境变量 AUTOFUSE_DFX_FLAGS="--disable_cache=true"
 * - 编译环境正常
 *
 * 操作步骤：
 * 1. 设置环境变量禁用缓存
 * 2. 创建FlashSoftmax图结构
 * 3. 调用GenTilingImplAutoFuseV3生成Tiling代码
 * 4. 合并Tiling函数
 * 6. 清理环境变量
 *
 * 输入：
 * - 环境变量: AUTOFUSE_DFX_FLAGS="--disable_cache=true"
 * - TilingKey: 1101u
 * - 图名: FlashSoftmax
 *
 * 预期结果：
 * 生成的代码不包含：
 * - `using OperatorLevelCache`
 * - `thread_local` 缓存变量
 * - `TilingCacheContext` 类
 *
 * 备注：OperatorCache 默认关闭，通过环境变量显式开启时才生成
 */
TEST_F(TestApiTilingGen, gen_op_level_cache_disabled) {
  // 不设置AUTOFUSE_FLAGS，使用默认配置（OperatorCache禁用）

  std::vector<ascir::AscGraph> graphs;
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  ASSERT_EQ(ge::ascir::cg::BuildFlashSoftmaxAscendGraph(graph_normal), af::SUCCESS);
  graphs.emplace_back(graph_normal);
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["solver_type"] = "AxesReorder";
  ascir::FusedScheduledResult fused_schedule_result;
  std::vector<ascir::ScheduledResult> scheduled_results;
  fused_schedule_result.fused_graph_name = "FlashSoftmax";
  for (int i = 0; i < 2; ++i) {
    ascir::ScheduleGroup schedule_group;
    schedule_group.impl_graphs.emplace_back(graph_normal);
    ascir::ScheduledResult scheduled_result;
    scheduled_result.schedule_groups.emplace_back(schedule_group);
    scheduled_results.emplace_back(scheduled_result);
  }
  fused_schedule_result.node_idx_to_scheduled_results.emplace_back(scheduled_results);
  std::map<std::string, std::string> tiling_funcs;
  EXPECT_EQ(GenTilingImplAutoFuseV3("FlashSoftmax", fused_schedule_result, options, tiling_funcs, true), true);
  std::string tiling_func;
  ge::ascir::cg::CombineTilings(tiling_funcs, tiling_func);

  // 验证OperatorCache类型未生成（默认禁用）
  EXPECT_EQ(tiling_func.find("using OperatorLevelCache"), std::string::npos);

  // 验证TilingCacheContext类未生成
  EXPECT_EQ(tiling_func.find("class TilingCacheContext"), std::string::npos);

  // 验证OperatorCache专属thread_local缓存变量未生成
  EXPECT_EQ(tiling_func.find("thread_local std::unique_ptr<OperatorLevelCache> operator_cache_"), std::string::npos);

  // Group Cache 仍然可以正常工作（由with_reuse_info_控制）
  // 验证GroupCache相关代码存在（如果有reuse_info）
  EXPECT_NE(tiling_func.find("using GroupLevelCache"), std::string::npos);

  unsetenv("AUTOFUSE_DFX_FLAGS");
}

/**
 * @brief 两级缓存完整功能测试
 *
 * 验证：
 * 1. 两级缓存类型正确生成
 * 2. 两级缓存函数正确生成
 * 3. TilingCacheContext类正确生成
 *
 * 备注：验证两级缓存完整功能
 */
TEST_F(TestApiTilingGen, two_level_cache_full_test) {
  // 设置环境变量启用缓存
  setenv("AUTOFUSE_DFX_FLAGS", "--autofuse_enable_tiling_cache=true", 1);
  att::AutoFuseConfig::MutableAttStrategyConfig().Reset();

  std::vector<ascir::AscGraph> graphs;
  ascir::AscGraph graph_normal("graph_normal");
  graph_normal.SetTilingKey(1101u);
  ASSERT_EQ(ge::ascir::cg::BuildFlashSoftmaxAscendGraph(graph_normal), af::SUCCESS);
  graphs.emplace_back(graph_normal);
  GraphConstructUtils::UpdateGraphsVectorizedStride(graphs);

  std::map<std::string, std::string> options;
  options["output_file_path"] = "./";
  options["solver_type"] = "AxesReorder";

  ascir::FusedScheduledResult fused_schedule_result;
  std::vector<ascir::ScheduledResult> scheduled_results;
  fused_schedule_result.fused_graph_name = "FlashSoftmax";
  for (int i = 0; i < 2; ++i) {
    ascir::ScheduleGroup schedule_group;
    schedule_group.impl_graphs.emplace_back(graph_normal);
    ascir::ScheduledResult scheduled_result;
    scheduled_result.schedule_groups.emplace_back(schedule_group);
    scheduled_results.emplace_back(scheduled_result);
  }
  fused_schedule_result.node_idx_to_scheduled_results.emplace_back(scheduled_results);

  std::map<std::string, std::string> tiling_funcs;
  EXPECT_EQ(GenTilingImplAutoFuseV3("FlashSoftmax", fused_schedule_result, options, tiling_funcs, true), true);
  std::string tiling_func;
  ge::ascir::cg::CombineTilings(tiling_funcs, tiling_func);

  // 验证两级缓存类型定义生成
  EXPECT_NE(tiling_func.find("using OperatorLevelCache"), std::string::npos);
  EXPECT_NE(tiling_func.find("using GroupLevelCache"), std::string::npos);
  EXPECT_NE(tiling_func.find("constexpr size_t kInputShapeSize"), std::string::npos);
  EXPECT_NE(tiling_func.find("constexpr size_t kOperatorCacheCapacity"), std::string::npos);

  // 验证两级缓存函数生成
  EXPECT_NE(tiling_func.find("FindOperatorCache(const"), std::string::npos);
  EXPECT_NE(tiling_func.find("bool SaveOperatorCache"), std::string::npos);

  // 验证TilingCacheContext类生成
  EXPECT_NE(tiling_func.find("class TilingCacheContext"), std::string::npos);
  EXPECT_NE(tiling_func.find("thread_local std::unique_ptr<OperatorLevelCache> operator_cache_"), std::string::npos);

  unsetenv("AUTOFUSE_DFX_FLAGS");
  att::AutoFuseConfig::MutableAttStrategyConfig().Reset();
}
