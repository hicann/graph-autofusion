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
#define private public
#include "generator/solver_pass_gen/general_solver/general_solver_gen.h"

using namespace att;

class UTTEST_GENERAL_SOLVER_GEN_02 : public ::testing::Test {
 public:
  static void TearDownTestCase() {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {
    Expr x0 = af::Symbol("x0");
    Expr x1 = af::Symbol("x1");
    Expr a = af::Symbol("a");

    solver_ = new GeneralSolverGen("Case0", "TilingData");
    solver_->SetSearchArgs({x0, x1});

    ExprExprMap expr_relation;
    ExprExprMap vars_relation;
    expr_relation[x0] = x0;
    vars_relation[x0] = x0;
    vars_relation[x1] = x1;
    solver_->SetExprRelation(expr_relation, vars_relation);

    ExprUintMap const_args;
    const_args[a] = 0;
    solver_->SetConstArgs(const_args);

    std::map<PipeType, Expr> obj;
    obj[PipeType::AIC_MTE1] = (x0 + x1);
    solver_->SetObj(obj);

    std::map<HardwareDef, Expr> buffer_cons;
    buffer_cons[HardwareDef::GM] = (x0 + x1);
    solver_->SetBufferCons(buffer_cons);

    solver_->SetCutCons({((x0 + x1) - a)});

    ExprExprMap max_value;
    ExprExprMap min_value;
    max_value[x0] = (CreateExpr(2) * a);
    max_value[x1] = (CreateExpr(2) * a);
    min_value[x0] = af::sym::kSymbolOne;
    min_value[x1] = af::sym::kSymbolOne;
    solver_->SetMaxValue(max_value);
    solver_->SetMinValue(min_value);

    solver_->SetInnestDim({x0});
    solver_->SetHeadCost(CreateExpr(1));
  }

  void TearDown() override {}
  GeneralSolverGen *solver_;
};

TEST_F(UTTEST_GENERAL_SOLVER_GEN_02, test_gen_buffer_cost) {
  std::string expect_codes = "";
  solver_->impl_codes_ = "";
  solver_->GenBuffFunc();
  expect_codes += "/*\n";
  expect_codes += "函数名:Gethbm_sizeCost(重要函数)\n";
  expect_codes += "功能描述:\n";
  expect_codes += "  根据待求解变量值hbm_size缓存占用信息(occupy-buff)\n";
  expect_codes += "输入参数:\n";
  expect_codes += "  vars:一个长度为num_var的数组,对应了待求解变量\n";
  expect_codes += "*/\n";
  expect_codes += "inline double GeneralSolverCase0::Gethbm_sizeCost(uint64_t* vars)\n";
  expect_codes += "{\n";
  expect_codes += "    double x0 = static_cast<double>(vars[x0_idx]);\n";
  expect_codes += "    double x1 = static_cast<double>(vars[x1_idx]);\n";
  expect_codes += "    return (x0 + x1 - hbm_size);\n";
  expect_codes += "}\n";
  expect_codes += "\n";

  expect_codes += "/*\n";
  expect_codes += "函数名:GetSmoothhbm_sizeCost(重要函数)\n";
  expect_codes += "功能描述:\n";
  expect_codes += "  根据待求解变量值hbm_size的平滑化缓存占用信息\n";
  expect_codes += "  与Gethbm_sizeCost函数相比,整除运算被替换为浮点数的除法运算\n";
  expect_codes += "输入参数:\n";
  expect_codes += "  vars:一个长度为num_var的数组,对应了待求解变量\n";
  expect_codes += "*/\n";
  expect_codes += "inline double GeneralSolverCase0::GetSmoothhbm_sizeCost(uint64_t* vars)\n";
  expect_codes += "{\n";
  expect_codes += "    double x0 = static_cast<double>(vars[x0_idx]);\n";
  expect_codes += "    double x1 = static_cast<double>(vars[x1_idx]);\n";
  expect_codes += "    return (x0 + x1 - hbm_size);\n";
  expect_codes += "}\n";
  expect_codes += "\n";
  EXPECT_EQ(solver_->impl_codes_, expect_codes);
}

TEST_F(UTTEST_GENERAL_SOLVER_GEN_02, test_gen_get_func) {
  std::string expect_codes;
  std::string indent = "    ";

  solver_->impl_codes_ = "";
  solver_->GenGetFunc(FuncType::OBJ);
  expect_codes = "/*\n";
  expect_codes += "函数名:GetObj(重要函数)\n";
  expect_codes += "功能描述:\n";
  expect_codes += "  根据待求解变量值输出目标函数\n";
  expect_codes += "输入参数:\n";
  expect_codes += "  vars:一个长度为num_var的数组,对应了待求解变量\n";
  expect_codes += "*/\n";
  expect_codes += "inline double GeneralSolverCase0::GetObj(uint64_t* vars)\n";
  expect_codes += "{\n";
  expect_codes += "    double block_dim = 1;\n";
  expect_codes += "    double x0 = static_cast<double>(vars[x0_idx]);\n";
  expect_codes += "    double x1 = static_cast<double>(vars[x1_idx]);\n";
  expect_codes += "    double AIC_MTE1 = (x0 + x1);\n";
  expect_codes += "    OP_LOGD(OP_NAME, \"AIC_MTE1 = %f\", AIC_MTE1);\n";
  expect_codes += "    OP_LOGD(OP_NAME, \"The expression of AIC_MTE1 is (x0 + x1)\");\n";
  expect_codes += "    return (1 + AIC_MTE1);\n";
  expect_codes += "}\n";
  EXPECT_EQ(solver_->impl_codes_, expect_codes);

  solver_->impl_codes_ = "";
  solver_->GenGetFunc(FuncType::BUFFER);
  expect_codes = "/*\n";
  expect_codes += "函数名:GetBuffCost(重要函数)\n";
  expect_codes += "功能描述:\n";
  expect_codes += "  根据待求解变量值输出缓存占用信息的罚函数(sigma(min(0, occupy-buff)^2))\n";
  expect_codes += "  该函数用于量化解在缓存占用方面的质量\n";
  expect_codes += "输入参数:\n";
  expect_codes += "  vars:一个长度为num_var的数组,对应了待求解变量\n";
  expect_codes += "*/\n";
  expect_codes += "inline double GeneralSolverCase0::GetBuffCost(uint64_t* vars)\n";
  expect_codes += "{\n";
  expect_codes += "    double hbm_size_cost = Gethbm_sizeCost(vars);\n";
  expect_codes += "    return (Min(0, hbm_size_cost) * Min(0, hbm_size_cost));\n";
  expect_codes += "}\n";
  EXPECT_EQ(solver_->impl_codes_, expect_codes);
}

TEST_F(UTTEST_GENERAL_SOLVER_GEN_02, test_get_leq_info) {
  std::string local_valid_func;
  std::string update_func;
  std::string expect_valid_func = "";
  std::string expect_update_func = "";

  expect_valid_func += "    if (idx == x0_idx) {\n";
  expect_valid_func += "        return leqs[0] <= 0 && leqs[1] <= 0;\n";
  expect_update_func += "    if (idx == x0_idx) {\n";
  expect_update_func += "        leqs[0] = (x0 + x1 - hbm_size);\n";
  expect_update_func += "        leqs[1] = (x0 + x1 - a);\n";
  solver_->GetLeqInfo(0, local_valid_func, update_func);
  EXPECT_EQ(local_valid_func, expect_valid_func);
  EXPECT_EQ(update_func, expect_update_func);

  expect_valid_func += "    } else if (idx == x1_idx) {\n";
  expect_valid_func += "        return leqs[0] <= 0 && leqs[1] <= 0;\n";
  expect_update_func += "    } else if (idx == x1_idx) {\n";
  expect_update_func += "        leqs[0] = (x0 + x1 - hbm_size);\n";
  expect_update_func += "        leqs[1] = (x0 + x1 - a);\n";
  solver_->GetLeqInfo(1, local_valid_func, update_func);
  EXPECT_EQ(local_valid_func, expect_valid_func);
  EXPECT_EQ(update_func, expect_update_func);
}

TEST_F(UTTEST_GENERAL_SOLVER_GEN_02, test_gen_leq_info) {
  std::string expect_codes = "";

  expect_codes += "inline bool GeneralSolverCase0::CheckLocalValid(double* leqs, int32_t idx)\n";
  expect_codes += "{\n";
  expect_codes += "    if (idx == x0_idx) {\n";
  expect_codes += "        return leqs[0] <= 0 && leqs[1] <= 0;\n";
  expect_codes += "    } else if (idx == x1_idx) {\n";
  expect_codes += "        return leqs[0] <= 0 && leqs[1] <= 0;\n";
  expect_codes += "    }\n";
  expect_codes += "    return true;\n";
  expect_codes += "}\n";
  expect_codes += "\n";

  expect_codes += "inline void GeneralSolverCase0::UpdateLeqs(uint64_t* vars, int32_t idx, double* leqs)\n";
  expect_codes += "{\n";
  expect_codes += "    double x0 = static_cast<double>(vars[x0_idx]);\n";
  expect_codes += "    double x1 = static_cast<double>(vars[x1_idx]);\n";
  expect_codes += "    if (idx == x0_idx) {\n";
  expect_codes += "        leqs[0] = (x0 + x1 - hbm_size);\n";
  expect_codes += "        leqs[1] = (x0 + x1 - a);\n";
  expect_codes += "    } else if (idx == x1_idx) {\n";
  expect_codes += "        leqs[0] = (x0 + x1 - hbm_size);\n";
  expect_codes += "        leqs[1] = (x0 + x1 - a);\n";
  expect_codes += "    } else if (idx == -1) {\n";
  expect_codes += "        leqs[0] = (x0 + x1 - hbm_size);\n";
  expect_codes += "        leqs[1] = (x0 + x1 - a);\n";
  expect_codes += "    }\n";
  expect_codes += "}\n";
  expect_codes += "\n";

  solver_->GenLeqInfo();
  EXPECT_EQ(solver_->impl_codes_, expect_codes);
}

TEST_F(UTTEST_GENERAL_SOLVER_GEN_02, test_map_varval) {
  std::string expect_codes = "";
  std::string indent = "    ";

  expect_codes += "inline void GeneralSolverCase0::MapVarVal(uint64_t* vars, TilingData& tiling_data)\n";
  expect_codes += "{\n";
  expect_codes += solver_->GetUIntVars(indent);
  expect_codes += "    OP_LOGD(OP_NAME, \"The output of the solver for tilingCaseId Case0 is:\");\n";
  expect_codes += "    tiling_data.set_x0(static_cast<uint64_t>(x0));\n";
  expect_codes += "    OP_LOGD(OP_NAME, \"x0 = %u\", tiling_data.get_x0());\n";
  expect_codes += "}\n";
  expect_codes += "\n";

  solver_->GenMapVarVal();
  EXPECT_EQ(solver_->impl_codes_, expect_codes);
}

TEST_F(UTTEST_GENERAL_SOLVER_GEN_02, test_display_varval) {
  std::string expect_codes = "";
  std::string indent = "    ";
  expect_codes += "inline void GeneralSolverCase0::DisplayVarVal(uint64_t* vars)\n";
  expect_codes += "{\n";
  expect_codes += solver_->GetUIntVars(indent);
  expect_codes += "    OP_LOGD(OP_NAME, \"x0 = %lu\", static_cast<uint64_t>(x0));\n";
  expect_codes += "}\n";
  expect_codes += "\n";

  solver_->GenDisplayVarVal();
  EXPECT_EQ(solver_->impl_codes_, expect_codes);
}

TEST_F(UTTEST_GENERAL_SOLVER_GEN_02, test_gen_get_result) {
  std::string expect_codes = "";

  expect_codes +=
      "inline void GeneralSolverCase0::GetResult(int32_t solution_num, uint64_t* solution, TilingData& "
      "tiling_data)\n{\n";
  expect_codes += "    if (solution_num > 0) {\n";
  expect_codes += "        OP_LOGD(OP_NAME, \"Filling tilingdata for Case0.\");\n";
  expect_codes += "        OP_LOGD(OP_NAME, \"Estimate the occupy.\");\n";
  expect_codes +=
      "        OP_LOGD(OP_NAME, \"hbm_size = %ld\", static_cast<uint64_t>(Gethbm_sizeCost(solution) + hbm_size));\n";
  expect_codes += "        OP_LOGD(OP_NAME, \"Simulate the cost.\");\n";
  expect_codes += "        OP_LOGD(OP_NAME, \"Objective value for Case0 is %f.\", GetObj(solution));\n";
  expect_codes += "        MapVarVal(solution, tiling_data);\n";
  expect_codes += "    }\n";
  expect_codes += "}\n\n";
  solver_->GenGetResult();
  EXPECT_EQ(solver_->impl_codes_, expect_codes);
}

TEST_F(UTTEST_GENERAL_SOLVER_GEN_02, test_create_input) {
  std::string expect_codes = "";
  expect_codes += "    // 以下参数若未注明是可修改参数,则不建议修改\n";
  expect_codes += "    // 由modelinfo传入的待求解变量个数\n";
  expect_codes += "    int32_t num_var = 2;\n";
  expect_codes += "    // 由modelinfo传入的不等式约束个数\n";
  expect_codes += "    int32_t num_leq = 2;\n";
  expect_codes +=
      "    OP_LOGD(OP_NAME, \"The number of variable is %d(x0, x1), the number of constraints is %d.\", num_var, "
      "num_leq);\n";
  expect_codes += "    // 初始化解的个数为0\n";
  expect_codes += "    int32_t solution_num = 0;\n";
  expect_codes += "    size_t uint_size = 6 * static_cast<size_t>(num_var) * sizeof(uint64_t);\n";
  expect_codes += "    size_t double_size = 2 * static_cast<size_t>(num_leq + num_var) * sizeof(double);\n";
  expect_codes += "    size_t bool_size = 2 * static_cast<size_t>(num_var) * sizeof(bool);\n";
  expect_codes += "    size_t VarVal_size = sizeof(VarVal) + (sizeof(uint64_t) * static_cast<size_t>(num_var));\n";
  expect_codes += "    size_t total_VarVal_size = static_cast<size_t>(2 * cfg_top_num + 1) * VarVal_size;\n";
  expect_codes += "    size_t ret_size = static_cast<size_t>(num_var * cfg_top_num) * sizeof(uint64_t);\n";
  expect_codes += "    size_t visited_size = static_cast<size_t>(num_var * cfg_iterations) * sizeof(uint64_t);\n";
  expect_codes +=
      "    void* memory_pool = calloc(1, uint_size + double_size + bool_size"
      " + sizeof(VarInfo) + sizeof(ConsInfo) + sizeof(Momentum)"
      " + total_VarVal_size + sizeof(Result) + ret_size"
      " + visited_size + sizeof(VisitedNode));\n";
  expect_codes += "    if (memory_pool == nullptr) {\n";
  expect_codes += "        OP_LOGE(OP_NAME, \"Failed to allocate memory pool for solver.\");\n";
  expect_codes += "        return false;\n";
  expect_codes += "    }\n";
  expect_codes += "    size_t offset_uint = 0;\n";
  expect_codes += "    size_t offset_double = offset_uint + uint_size;\n";
  expect_codes += "    size_t offset_bool = offset_double + double_size;\n";
  expect_codes += "    size_t offset_var_info = offset_bool + bool_size;\n";
  expect_codes += "    size_t offset_cons_info = offset_var_info + sizeof(VarInfo);\n";
  expect_codes += "    size_t offset_momentum = offset_cons_info + sizeof(ConsInfo);\n";
  expect_codes += "    size_t offset_varVal = offset_momentum + sizeof(Momentum);\n";
  expect_codes += "    size_t offset_temp = offset_varVal + VarVal_size;\n";
  expect_codes += "    size_t offset_solution = offset_temp + cfg_top_num * VarVal_size;\n";
  expect_codes += "    size_t offset_result = offset_solution + cfg_top_num * VarVal_size;\n";
  expect_codes += "    size_t offset_ret = offset_result + sizeof(Result);\n";
  expect_codes += "    size_t offset_visited = offset_ret + ret_size;\n";
  expect_codes += "    size_t offset_node = offset_visited + visited_size;\n";
  expect_codes += "    uint64_t* uint_space = (uint64_t*)((char*)memory_pool + offset_uint);\n";
  expect_codes += "    double* double_space = (double*)((char*)memory_pool + offset_double);\n";
  expect_codes += "    bool* bool_space = (bool*)((char*)memory_pool + offset_bool);\n";
  expect_codes +=
      "    // 可修改参数:待求解变量的上界,过大的上界将导致搜索范围与耗时增加,过小的上界更有可能获得较差的局部最优解\n";
  expect_codes += "    uint_space[0] = static_cast<uint64_t>((2 * a));\n";
  expect_codes += "    uint_space[1] = static_cast<uint64_t>((2 * a));\n";
  expect_codes +=
      "    // 可修改参数:待求解变量的下界,过小的下界将导致搜索范围与耗时增加,过大的下界更有可能获得较差的局部最优解\n";
  expect_codes += "    uint_space[2] = static_cast<uint64_t>(1);\n";
  expect_codes += "    if (static_cast<uint64_t>(1) > static_cast<uint64_t>((2 * a))) {\n";
  expect_codes += "        OP_LOGW(OP_NAME, \"Lower_bound[0] is larger than upper_bound[0].\");\n";
  expect_codes += "        return false;\n";
  expect_codes += "    }\n";
  expect_codes += "    uint_space[3] = static_cast<uint64_t>(1);\n";
  expect_codes += "    if (static_cast<uint64_t>(1) > static_cast<uint64_t>((2 * a))) {\n";
  expect_codes += "        OP_LOGW(OP_NAME, \"Lower_bound[1] is larger than upper_bound[1].\");\n";
  expect_codes += "        return false;\n";
  expect_codes += "    }\n";
  expect_codes += "    // 可修改参数:待求解变量的初始值,算法趋向于求初始值附近的局部最优解\n";
  expect_codes += "    uint_space[8] = static_cast<uint64_t>((2 * a));\n";
  expect_codes += "    uint_space[9] = static_cast<uint64_t>((2 * a));\n";
  expect_codes += "    uint64_t* upper_bound = uint_space;\n";
  expect_codes += "    uint64_t* lower_bound = uint_space + 2;\n";
  expect_codes += "    uint64_t* init_vars = uint_space + 8;\n";
  expect_codes += "    // 可修改参数:最后更新的待求解变量,设置为true的对应变量会更接近初始值\n";
  expect_codes += "    bool_space[0] = true;\n";
  expect_codes += "    bool_space[1] = false;\n";
  expect_codes += "    VarInfo* var_info = (VarInfo*)((char*)memory_pool + offset_var_info);\n";
  expect_codes += "    ConsInfo* cons_info = (ConsInfo*)((char*)memory_pool + offset_cons_info);\n";
  expect_codes += "    Momentum* momentum = (Momentum*)((char*)memory_pool + offset_momentum);\n";
  expect_codes += "    VarVal* varval;\n";
  expect_codes += "    size_t offset;\n";
  expect_codes += "    for (uint64_t i = 0u; i < 2 * cfg_top_num + 1; i++) {\n";
  expect_codes += "        offset = offset_varVal + i * VarVal_size;\n";
  expect_codes += "        varval = (VarVal*)((char*)memory_pool + offset);\n";
  expect_codes += "        varval->var_num = num_var;\n";
  expect_codes += "        varval->vars = (uint64_t*)((char*)memory_pool + offset + sizeof(VarVal));\n";
  expect_codes += "    }\n";
  expect_codes += "    Result* result = (Result*)((char*)memory_pool + offset_result);\n";
  expect_codes += "    uint64_t* solution = (uint64_t*)((char*)memory_pool + offset_ret);\n";
  expect_codes += "    uint64_t* visited_head = (uint64_t*)((char*)memory_pool + offset_visited);\n";
  expect_codes += "    VisitedNode* visited_node = (VisitedNode*)((char*)memory_pool + offset_node);\n";
  expect_codes += "    var_info->SetVarInfo(num_var, uint_space, bool_space);\n";
  expect_codes += "    cons_info->SetConsInfo(num_leq, double_space);\n";
  expect_codes += "    momentum->SetMomentum(num_var, num_leq, double_space, bool_space);\n";
  expect_codes +=
      "    result->SetResult(cfg_top_num, num_var, (VarVal*)((char*)memory_pool + offset_varVal),((char*)memory_pool + "
      "offset_temp), ((char*)memory_pool + offset_solution));\n";
  expect_codes += "    visited_node->SetVisitedNode(num_var, visited_head);\n";
  expect_codes += "    // 通用求解器的输入参数\n";
  expect_codes += "    SolverInput input;\n";
  expect_codes += "    input.corenum = corenum_;\n";
  expect_codes += "    input.var_info = var_info;\n";
  expect_codes += "    input.cons_info = cons_info;\n";
  expect_codes += "    input.momentum = momentum;\n";
  expect_codes += "    input.result = result;\n";
  expect_codes += "    input.visited_node = visited_node;\n";
  expect_codes +=
      "    OP_LOGD(OP_NAME, \"x0->init value: %lu, range: [%lu, %lu].\", init_vars[0], lower_bound[0], "
      "upper_bound[0]);\n";
  expect_codes +=
      "    OP_LOGD(OP_NAME, \"x1->init value: %lu, range: [%lu, %lu].\", init_vars[1], lower_bound[1], "
      "upper_bound[1]);\n";
  expect_codes += "\n";

  solver_->CreateInput();
  EXPECT_EQ(solver_->invoke_codes_, expect_codes);
}

TEST_F(UTTEST_GENERAL_SOLVER_GEN_02, test_run_solver) {
  std::string expect_codes = "";

  expect_codes +=
      "    std::shared_ptr<GeneralSolverCase0> solver = std::make_shared<GeneralSolverCase0>(cfg, tiling_data);\n";
  expect_codes += "    if (solver != nullptr) {\n";
  expect_codes += "        // 导入通用求解器的输入参数并完成初始化\n";
  expect_codes += "        OP_LOGD(OP_NAME, \"Start initializing the input.\");\n";
  expect_codes += "        if (solver -> Init(input)) {\n";
  expect_codes += "            // 运行通用求解器并获取算法的解\n";
  expect_codes += "            OP_LOGD(OP_NAME, \"Initialization finished, start running the solver.\");\n";
  expect_codes += "            if (solver -> Run(solution_num, solution)) {\n";
  expect_codes += "                solver -> GetResult(solution_num, solution, tiling_data);\n";
  expect_codes += "                free(memory_pool);\n";
  expect_codes += "                OP_LOGD(OP_NAME, \"The solver executed successfully.\");\n";
  expect_codes += "                return true;\n";
  expect_codes += "            }\n";
  expect_codes += "            OP_LOGW(OP_NAME, \"Failed to find any solution.\");\n";
  expect_codes += "        }\n";
  expect_codes += "    }\n";
  expect_codes += "    free(memory_pool);\n";
  solver_->RunSolver();
  EXPECT_EQ(solver_->invoke_codes_, expect_codes);
}

TEST_F(UTTEST_GENERAL_SOLVER_GEN_02, test_create_config) {
  std::string expect_codes = "";

  expect_codes += "    SolverConfig cfg;\n";
  expect_codes += "    cfg.top_num = cfg_top_num;\n";
  expect_codes += "    cfg.search_length = cfg_search_length;\n";
  expect_codes += "    cfg.iterations = cfg_iterations;\n";
  expect_codes += "    cfg.simple_ver = cfg_simple_ver;\n";
  expect_codes +=
      "    cfg.momentum_factor = cfg_momentum_factor > 1 ? 1 : (cfg_momentum_factor < 0 ? 0 : cfg_momentum_factor);\n";
  expect_codes += "    OP_LOGD(OP_NAME, \"Record a maximum of %lu solutions.\", cfg.top_num);\n";
  expect_codes += "    OP_LOGD(OP_NAME, \"The searching range covers %lu unit(s).\", cfg.search_length);\n";
  expect_codes += "    OP_LOGD(OP_NAME, \"The maximum number of iterations is %lu.\", cfg.iterations);\n";
  expect_codes += "    if (cfg.simple_ver) {\n";
  expect_codes += "        OP_LOGD(OP_NAME, \"Using high-efficiency version.\");\n";
  expect_codes += "    } else {\n";
  expect_codes += "        OP_LOGD(OP_NAME, \"Using high-performance version.\");\n";
  expect_codes += "    }\n";
  expect_codes += "    OP_LOGD(OP_NAME, \"The momentum factor is %f.\", cfg.momentum_factor);\n";
  expect_codes += "\n";

  solver_->CreateConfig();
  EXPECT_EQ(solver_->invoke_codes_, expect_codes);
}
