/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "base/base_types.h"
#include "util/base_types_printer.h"
#define private public
#include "generator/solver_pass_gen/general_solver/general_solver_gen.h"

using namespace att;

class UTTEST_GENERAL_SOLVER_GEN_01 : public ::testing::Test {
 public:
  static void TearDownTestCase() {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(UTTEST_GENERAL_SOLVER_GEN_01, test_set_search_args) {
  GeneralSolverGen *solver;
  std::vector<Expr> search_args;
  Expr x0 = CreateExpr("x0");
  Expr x1 = CreateExpr("x1");
  Expr x2 = CreateExpr("x2");
  solver = new GeneralSolverGen("Case0", "TilingData");

  search_args = {x0};
  solver->search_args_.clear();
  solver->SetSearchArgs(search_args);
  EXPECT_EQ(solver->search_args_.size(), 1);
  EXPECT_EQ(solver->search_args_, search_args);

  search_args = {x0, x1};
  solver->search_args_.clear();
  solver->SetSearchArgs(search_args);
  EXPECT_EQ(solver->search_args_.size(), 2);
  EXPECT_EQ(solver->search_args_, search_args);

  search_args = {x0, x1, x2};
  solver->search_args_.clear();
  solver->SetSearchArgs(search_args);
  EXPECT_EQ(solver->search_args_.size(), 3);
  EXPECT_EQ(solver->search_args_, search_args);
}

TEST_F(UTTEST_GENERAL_SOLVER_GEN_01, test_set_solved_args) {
  GeneralSolverGen *solver;
  std::vector<Expr> solved_args;
  Expr x0 = CreateExpr("x0");
  Expr x1 = CreateExpr("x1");
  Expr x2 = CreateExpr("x2");
  solver = new GeneralSolverGen("Case0", "TilingData");

  solver->solved_args_.clear();
  solved_args = {x0, x1};
  solver->SetSolvedArgs(solved_args);
  EXPECT_EQ(solver->solved_args_.size(), 2);

  solver->solved_args_.clear();
  solved_args = {x0, x2};
  solver->SetSolvedArgs(solved_args);
  EXPECT_EQ(solver->solved_args_.size(), 2);
}

TEST_F(UTTEST_GENERAL_SOLVER_GEN_01, test_set_const_args) {
  GeneralSolverGen *solver;
  std::vector<Expr> args;
  ExprUintMap const_args;
  Expr x0 = CreateExpr("x0");
  Expr x1 = CreateExpr("x1");
  Expr x2 = CreateExpr("x2");
  solver = new GeneralSolverGen("Case0", "TilingData");

  const_args.clear();
  solver->const_args_.clear();
  const_args[x0] = 1;
  solver->SetConstArgs(const_args);
  EXPECT_EQ(solver->const_args_[x0], 1);

  const_args.clear();
  solver->const_args_.clear();
  const_args[x1] = 2;
  solver->SetConstArgs(const_args);
  EXPECT_EQ(solver->const_args_[x1], 2);
}

TEST_F(UTTEST_GENERAL_SOLVER_GEN_01, test_set_obj) {
  GeneralSolverGen *solver;
  Expr x0 = CreateExpr("x0");
  Expr x1 = CreateExpr("x1");
  std::map<PipeType, Expr> obj;
  obj[PipeType::AIC_MTE1] = x0 + x1;
  solver = new GeneralSolverGen("Case0", "TilingData");

  solver->SetObj(obj);
  EXPECT_EQ(Str(solver->obj_["AIC_MTE1"]), Str(obj[PipeType::AIC_MTE1]));
}

TEST_F(UTTEST_GENERAL_SOLVER_GEN_01, test_set_value) {
  GeneralSolverGen *solver;
  ExprExprMap max_value;
  ExprExprMap min_value;
  ExprExprMap init_value;
  Expr x0 = CreateExpr("x0");
  Expr x1 = CreateExpr("x1");
  Expr x2 = CreateExpr("x2");
  Expr res = (x1 * x2);
  Expr val = af::sym::kSymbolOne;
  solver = new GeneralSolverGen("Case0", "TilingData");

  max_value[x0] = res;
  min_value[x0] = val;
  solver->SetMaxValue(max_value);
  solver->SetMinValue(min_value);
  EXPECT_EQ(solver->max_value_[x0], res);
  EXPECT_EQ(solver->min_value_[x0], val);

  init_value[x0] = res;
  solver->SetInitValue(init_value);
  EXPECT_EQ(solver->init_value_[x0], res);
}

TEST_F(UTTEST_GENERAL_SOLVER_GEN_01, test_fix_var) {
  GeneralSolverGen *solver;
  solver = new GeneralSolverGen("Case0", "TilingData");
  std::vector<Expr> search_args;
  Expr x0 = CreateExpr("x0");
  search_args = {x0};
  solver->SetSearchArgs(search_args);
  solver->FixVar(0, 2);
  EXPECT_EQ(solver->fixed_args_[0], 2);
}

TEST_F(UTTEST_GENERAL_SOLVER_GEN_01, test_fix_range) {
  GeneralSolverGen *solver;
  solver = new GeneralSolverGen("Case0", "TilingData");
  std::vector<Expr> search_args;
  Expr x0 = CreateExpr("x0");
  search_args = {x0};
  solver->SetSearchArgs(search_args);
  solver->FixRange(0, 1, 5);
  EXPECT_EQ(Str(solver->max_value_[x0]), std::to_string(5));
  EXPECT_EQ(Str(solver->min_value_[x0]), std::to_string(1));
}

TEST_F(UTTEST_GENERAL_SOLVER_GEN_01, test_set_buff_cons) {
  GeneralSolverGen *solver;
  std::map<HardwareDef, Expr> buffer_cons;
  std::vector<Expr> search_args;
  Expr x0 = CreateExpr("x0");
  Expr x1 = CreateExpr("x1");
  Expr gm = CreateExpr("hbm_size");
  Expr capacity = (x0 * x1);
  solver = new GeneralSolverGen("Case0", "TilingData");

  buffer_cons[HardwareDef::GM] = capacity;
  search_args = {x0, x1};
  solver->SetSearchArgs(search_args);
  solver->SetBufferCons(buffer_cons);

  Expr cons_expr = (capacity - gm);

  EXPECT_EQ(Str(solver->buffer_cost_[BaseTypeUtils::DumpHardware(HardwareDef::GM)]), Str(cons_expr));

  EXPECT_EQ(solver->hardware_args_.size(), 1);
  EXPECT_EQ(Str(solver->hardware_args_[0]), Str(gm));
}

TEST_F(UTTEST_GENERAL_SOLVER_GEN_01, test_set_core_num_cons_with_input_args) {
  GeneralSolverGen solver("Case0", "TilingData");
  const Expr input_axis = CreateExpr("input_axis");
  solver.SetInputArgs({input_axis});
  solver.SetBufferCons({{HardwareDef::CORENUM, input_axis * CreateExpr(2)}});

  ASSERT_EQ(solver.leqs_.size(), 1U);
  EXPECT_NE(Str(solver.leqs_.front()).find("block_dim"), std::string::npos);
}

TEST_F(UTTEST_GENERAL_SOLVER_GEN_01, test_set_cut_cons) {
  GeneralSolverGen *solver;
  std::vector<Expr> cut_cons;
  std::vector<Expr> search_args;
  Expr x0 = CreateExpr("x0");
  Expr x1 = CreateExpr("x1");
  Expr cons_expr = ((x0 + x1) - CreateExpr(3));
  solver = new GeneralSolverGen("Case0", "TilingData");

  cut_cons = {cons_expr};
  search_args = {x0, x1};
  solver->SetSearchArgs(search_args);
  solver->SetCutCons(cut_cons);

  EXPECT_EQ(Str(solver->leq_cost_["leq1_cost"]), Str(cons_expr));
  EXPECT_EQ(solver->leqs_, cut_cons);
}

TEST_F(UTTEST_GENERAL_SOLVER_GEN_01, test_set_expr_relation) {
  GeneralSolverGen *solver;
  Expr x0 = CreateExpr("x0");
  Expr x1 = CreateExpr("x1");
  Expr x2 = CreateExpr("x2");
  solver = new GeneralSolverGen("Case0", "TilingData");
  std::vector<Expr> search_args;
  std::vector<Expr> innest_dim;
  std::vector<bool> res;

  search_args = {x0, x1, x2};
  innest_dim = {x1};
  solver->innest_dim_.clear();
  solver->search_args_.clear();
  solver->SetSearchArgs(search_args);
  solver->SetInnestDim(innest_dim);
  res = {false, true, false};
  EXPECT_EQ(solver->innest_dim_, res);

  search_args = {x0, x1};
  innest_dim = {};
  solver->innest_dim_.clear();
  solver->search_args_.clear();
  solver->SetSearchArgs(search_args);
  solver->SetInnestDim(innest_dim);
  res = {false, false};
  EXPECT_EQ(solver->innest_dim_, res);
}

TEST_F(UTTEST_GENERAL_SOLVER_GEN_01, test_get_double_vars) {
  GeneralSolverGen *solver = new GeneralSolverGen("Case0", "TilingData");
  Expr x0 = CreateExpr("x0");
  std::vector<Expr> search_args = {x0};
  solver->SetSearchArgs(search_args);
  std::string codes;
  std::string indent;

  indent = "    ";
  codes = solver->GetDoubleVars(indent, {x0});
  std::string expect_codes = indent + "double x0 = static_cast<double>(vars[x0_idx]);\n";
  EXPECT_EQ(codes, expect_codes);
}

TEST_F(UTTEST_GENERAL_SOLVER_GEN_01, test_gen_class_def) {
  GeneralSolverGen *solver = new GeneralSolverGen("Case0", "TilingData");
  Expr a = CreateExpr("a");
  Expr x0 = CreateExpr("x0");
  solver->search_args_ = {x0};
  std::vector<Expr> input_args = {a};
  solver->SetInputArgs(input_args);

  solver->GenClassDef();
  std::string expect_codes = "";
  expect_codes += "/*\n";
  expect_codes += "用户可以在派生类中重载Run函数,构造自定义的求解算法,即\n";
  expect_codes += "  void bool Run(int32_t &solution_num, uint64_t *solutions) override;\n";
  expect_codes += "其中:\n";
  expect_codes += "  solution_num:int32_t类型的参数,用来输出实际得到的解的个数\n";
  expect_codes += "  solutions:uint64_t类型的数组,指向一块num_var * top_num的内存,算法将可行解放入该空间\n";
  expect_codes += "Run函数可以使用下述函数辅助求解:\n";
  expect_codes += "  bool CheckValid()\n";
  expect_codes += "    用于检测当前解是否为可行解\n";
  expect_codes += "  bool UpdateCurVarVal(uint64_t value, int32_t idx)\n";
  expect_codes += "    将下标为idx的待求解变量改为value,同时更新cons_info_->leqs中的值\n";
  expect_codes += "  bool RecordBestVarVal()\n";
  expect_codes += "    待求解变量的当前值所对应的目标函数寻优\n";
  expect_codes += "Run函数可以使用下述参数辅助求解:\n";
  expect_codes += "  cons_info_->leqs, double类型的数组, 用于记录不等式约束的函数值, 其下标含义如下:\n";
  expect_codes += "  var_info_->cur_vars, uint64_t类型的数组, 用于记录待求解变量的当前值, 其下标含义如下:\n";
  expect_codes += "  var_info_->upper_bound, uint64_t类型的数组, 用于记录待求解变量的上界\n";
  expect_codes += "  var_info_->lower_bound, uint64_t类型的数组, 用于记录待求解变量的下界\n";
  expect_codes += "*/\n";
  expect_codes += "class GeneralSolverCase0 : public GeneralSolver<GeneralSolverCase0>\n";
  expect_codes += "{\n";
  expect_codes += "    public:\n";
  expect_codes += "        explicit GeneralSolverCase0(SolverConfig& config, TilingData& tiling_data) {\n";
  expect_codes += "            case_id_ = \"Case0\";\n";
  expect_codes += "            solver_config_ = config;\n";
  expect_codes += "            a = tiling_data.get_a();\n";
  expect_codes += "        }\n\n";
  expect_codes += "        double GetObj(uint64_t* vars);\n";
  expect_codes += "        double GetSmoothObj(uint64_t* vars);\n";
  expect_codes += "        double GetBuffCost(uint64_t* vars);\n";
  expect_codes += "        bool CheckLocalValid(double* leqs, int32_t idx);\n";
  expect_codes += "        void DisplayVarVal(uint64_t* vars);\n";
  expect_codes += "        void UpdateLeqs(uint64_t* vars, int32_t idx, double* leqs);\n";
  expect_codes += "        double GetBuffDiff(uint64_t* vars, double* weight);\n";
  expect_codes += "        double GetLeqDiff(uint64_t* vars, double* weight);\n";
  expect_codes += "        void MapVarVal(uint64_t* vars, TilingData& tiling_data);\n";
  expect_codes += "        void GetResult(int32_t solution_num, uint64_t* solution, TilingData& tiling_data);\n";
  expect_codes += "        bool Init(const SolverInput &input);\n";
  expect_codes += "    private:\n";
  expect_codes += "        const int64_t x0_idx = 0;\n";
  expect_codes += "        uint64_t a;\n";
  expect_codes += "};\n";
  EXPECT_EQ(solver->impl_codes_, expect_codes);
}
