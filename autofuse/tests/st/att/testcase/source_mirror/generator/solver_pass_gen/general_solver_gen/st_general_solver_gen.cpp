/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "base/base_types.h"
#include "generator/solver_pass_gen/general_solver/general_solver_gen.h"
#include "test_common_utils.h"

using att::CreateExpr;
using att::Expr;
using att::ExprExprMap;
using att::ExprUintMap;
using att::GeneralSolverGen;
using att::HardwareDef;
using att::PipeType;

class ST_GENERAL_SOLVER_GEN : public ::testing::Test {
 public:
  static void TearDownTestCase() {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {
    Expr x0 = CreateExpr("x0");
    Expr x1 = CreateExpr("x1");
    Expr x2 = CreateExpr("x2");
    Expr x3 = CreateExpr("x3");
    Expr a = CreateExpr("a");

    solver_ = new GeneralSolverGen("Case0", "TilingData");
    solver_->SetSearchArgs({x0, x1, x3});

    ExprExprMap expr_relation;
    ExprExprMap vars_relation;
    vars_relation[x0] = x0;
    vars_relation[x1] = x1;
    solver_->SetExprRelation(expr_relation, vars_relation);

    ExprUintMap const_args;
    const_args[a] = 0;
    solver_->SetConstArgs(const_args);
    solver_->SetSolvedArgs({x2});

    std::map<PipeType, Expr> obj;
    obj[PipeType::AIC_MTE1] = x0 + x1;
    solver_->SetObj(obj);

    std::map<HardwareDef, Expr> buffer_cons;
    buffer_cons[HardwareDef::GM] = x0 + x1;

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
    solver_->FixVar(2, 1);
    solver_->FixRange(0, 1, 5);
  }

  void TearDown() override {
    delete solver_;
  }
  GeneralSolverGen *solver_;
};

namespace {
void AppendSolverImplPart0(std::string &codes) {
  codes += "/*\n";
  codes += "用户可以在派生类中重载Run函数,构造自定义的求解算法,即\n";
  codes += "  void bool Run(int32_t &solution_num, uint64_t *solutions) override;\n";
  codes += "其中:\n";
  codes += "  solution_num:int32_t类型的参数,用来输出实际得到的解的个数\n";
  codes += "  solutions:uint64_t类型的数组,指向一块num_var * top_num的内存,算法将可行解放入该空间\n";
  codes += "Run函数可以使用下述函数辅助求解:\n";
  codes += "  bool CheckValid()\n";
  codes += "    用于检测当前解是否为可行解\n";
  codes += "  bool UpdateCurVarVal(uint64_t value, int32_t idx)\n";
  codes += "    将下标为idx的待求解变量改为value,同时更新cons_info_->leqs中的值\n";
  codes += "  bool RecordBestVarVal()\n";
  codes += "    待求解变量的当前值所对应的目标函数寻优\n";
  codes += "Run函数可以使用下述参数辅助求解:\n";
  codes += "  cons_info_->leqs, double类型的数组, 用于记录不等式约束的函数值, 其下标含义如下:\n";
  codes += "    cons_info_->leqs[0] = (x0 + x1 - hbm_size)\n";
  codes += "    cons_info_->leqs[1] = (x0 + x1 - a)\n";
  codes += "  var_info_->cur_vars, uint64_t类型的数组, 用于记录待求解变量的当前值, 其下标含义如下:\n";
  codes += "    var_info_->cur_vars[0] = x3\n";
  codes += "  var_info_->upper_bound, uint64_t类型的数组, 用于记录待求解变量的上界\n";
  codes += "  var_info_->lower_bound, uint64_t类型的数组, 用于记录待求解变量的下界\n";
  codes += "*/\n";
  codes += "class GeneralSolverCase0 : public GeneralSolver<GeneralSolverCase0>\n";
  codes += "{\n";
  codes += "    public:\n";
  codes += "        explicit GeneralSolverCase0(SolverConfig& config, TilingData& tiling_data) {\n";
  codes += "            solver_config_ = config;\n";
  codes += "            hbm_size = tiling_data.get_hbm_size();\n";
  codes += "            x2 = tiling_data.get_x2();\n";
  codes += "        }\n\n";
  codes += "        double GetObj(uint64_t* vars);\n";
  codes += "        double GetSmoothObj(uint64_t* vars);\n";
  codes += "        double GetBuffCost(uint64_t* vars);\n";
  codes += "        bool CheckLocalValid(double* leqs, int32_t idx);\n";
  codes += "        void DisplayVarVal(uint64_t* vars);\n";
  codes += "        void UpdateLeqs(uint64_t* vars, int32_t idx, double* leqs);\n";
  codes += "        double GetBuffDiff(uint64_t* vars, double* weight);\n";
  codes += "        double GetLeqDiff(uint64_t* vars, double* weight);\n";
  codes += "        double Gethbm_sizeCost(uint64_t* vars);\n";
  codes += "        double GetSmoothhbm_sizeCost(uint64_t* vars);\n";
  codes += "        void MapVarVal(uint64_t* vars, TilingData& tiling_data);\n";
}

void AppendSolverImplPart1(std::string &codes) {
  codes += "        void GetResult(int32_t solution_num, uint64_t* solution, TilingData& tiling_data);\n";
  codes += "    private:\n";
  codes += "        const int64_t x0_idx = 0;\n";
  codes += "        const int64_t x1_idx = 1;\n";
  codes += "        uint64_t x3{1};\n";
  codes += "        uint64_t a{0};\n";
  codes += "        uint64_t hbm_size;\n";
  codes += "        uint64_t x2;\n";
  codes += "};\n";

  codes += "/*\n";
  codes += "函数名:Gethbm_sizeCost(重要函数)\n";
  codes += "功能描述:\n";
  codes += "  根据待求解变量值hbm_size缓存占用信息(occupy-buff)\n";
  codes += "输入参数:\n";
  codes += "  vars:一个长度为num_var的数组,对应了待求解变量\n";
  codes += "*/\n";
  codes += "inline double GeneralSolverCase0::Gethbm_sizeCost(uint64_t* vars)\n";
  codes += "{\n";
  codes += "    double x0 = static_cast<double>(vars[x0_idx]);\n";
  codes += "    double x1 = static_cast<double>(vars[x1_idx]);\n";
  codes += "    return (x0 + x1 - hbm_size);\n";
  codes += "}\n";
  codes += "\n";

  codes += "/*\n";
  codes += "函数名:GetSmoothhbm_sizeCost(重要函数)\n";
  codes += "功能描述:\n";
  codes += "  根据待求解变量值hbm_size的平滑化缓存占用信息\n";
  codes += "  与Gethbm_sizeCost函数相比,整除运算被替换为浮点数的除法运算\n";
  codes += "输入参数:\n";
  codes += "  vars:一个长度为num_var的数组,对应了待求解变量\n";
  codes += "*/\n";
  codes += "inline double GeneralSolverCase0::GetSmoothhbm_sizeCost(uint64_t* vars)\n";
  codes += "{\n";
  codes += "    double x0 = static_cast<double>(vars[x0_idx]);\n";
  codes += "    double x1 = static_cast<double>(vars[x1_idx]);\n";
  codes += "    return (x0 + x1 - hbm_size);\n";
  codes += "}\n";
  codes += "\n";

  codes += "/*\n";
  codes += "函数名:GetObj(重要函数)\n";
  codes += "功能描述:\n";
  codes += "  根据待求解变量值输出目标函数\n";
}

void AppendSolverImplPart2(std::string &codes) {
  codes += "输入参数:\n";
  codes += "  vars:一个长度为num_var的数组,对应了待求解变量\n";
  codes += "*/\n";
  codes += "inline double GeneralSolverCase0::GetObj(uint64_t* vars)\n";
  codes += "{\n";
  codes += "    double x0 = static_cast<double>(vars[x0_idx]);\n";
  codes += "    double x1 = static_cast<double>(vars[x1_idx]);\n";
  codes += "    double AIC_MTE1 = (x0 + x1);\n";
  codes += "    OP_LOGD(OP_NAME, \"AIC_MTE1 = %f\", AIC_MTE1);\n";
  codes += "    return AIC_MTE1;\n";
  codes += "}\n";

  codes += "/*\n";
  codes += "函数名:GetSmoothObj(重要函数)\n";
  codes += "功能描述:\n";
  codes += "  根据待求解变量值输出平滑化目标函数\n";
  codes += "  与GetObj函数相比,整除运算被替换为浮点数的除法运算\n";
  codes += "*/\n";
  codes += "inline double GeneralSolverCase0::GetSmoothObj(uint64_t* vars)\n";
  codes += "{\n";
  codes += "    double x0 = static_cast<double>(vars[x0_idx]);\n";
  codes += "    double x1 = static_cast<double>(vars[x1_idx]);\n";
  codes += "    double AIC_MTE1 = (x0 + x1);\n";
  codes += "    return AIC_MTE1;\n";
  codes += "}\n";

  codes += "/*\n";
  codes += "函数名:GetBuffCost(重要函数)\n";
  codes += "功能描述:\n";
  codes += "  根据待求解变量值输出缓存占用信息的罚函数(sigma(min(0, occupy-buff)^2))\n";
  codes += "  该函数用于量化解在缓存占用方面的质量\n";
  codes += "输入参数:\n";
  codes += "  vars:一个长度为num_var的数组,对应了待求解变量\n";
  codes += "*/\n";
  codes += "inline double GeneralSolverCase0::GetBuffCost(uint64_t* vars)\n";
  codes += "{\n";
  codes += "    double hbm_size_cost = Gethbm_sizeCost(vars);\n";
  codes += "    return (Min(0, hbm_size_cost) * Min(0, hbm_size_cost));\n";
  codes += "}\n";

  codes += "/*\n";
  codes += "函数名:GetBuffDiff(重要函数)\n";
  codes += "功能描述:\n";
  codes += "  获取缓冲占用加权差分值,计算平滑缓冲占用的差分\n";
  codes += "  输出的计算公式为sigma_j(delta_{var_i}(g_j(var))) * g_j(var))\n";
}

void AppendSolverImplPart3(std::string &codes) {
  codes += "  其中g_j为第j个缓冲占用不等式,delta_{var_i}(g_j(var))为g_j(var)沿var_i方向更新一个单位后的变化值\n";
  codes += "  该函数用于确定变量沿缓冲占用增大的更新方向\n";
  codes += "输入参数:\n";
  codes += "  vars:一个长度为num_var的数组,对应了待求解变量\n";
  codes += "  weight:一个长度为num_leq的数组,代表了每个缓冲占用的权值\n";
  codes += "*/\n";
  codes += "inline double GeneralSolverCase0::GetBuffDiff(uint64_t* vars, double* weight)\n";
  codes += "{\n";
  codes += "    double hbm_size_cost = GetSmoothhbm_sizeCost(vars);\n";
  codes += "    hbm_size_cost *= weight[0] < 0 ? weight[0] : 0;\n";
  codes += "    return hbm_size_cost;\n";
  codes += "}\n";

  codes += "/*\n";
  codes += "函数名:GetLeqDiff(重要函数)\n";
  codes += "功能描述:\n";
  codes += "  获取不等式约束的加权差分值,计算平滑的不等式函数的差分,权值为实际不等式函数值\n";
  codes += "  输出的计算公式为sigma_j(delta_{var_i}(f_j(var))) * f_j(var))\n";
  codes += "  其中f_j为第j个不等式约束式,delta_{var_i}(f_j(var))为f_j(var)沿var_i方向更新一个单位后的变化值\n";
  codes += "  该函数用于确定变量从可行域外侧沿不等式边界方向移动的更新方向\n";
  codes += "输入参数:\n";
  codes += "  vars:一个长度为num_var的数组,对应了待求解变量\n";
  codes += "  weight:一个长度为num_leq的数组,代表了每个缓冲占用的权值\n";
  codes += "*/\n";
  codes += "inline double GeneralSolverCase0::GetLeqDiff(uint64_t* vars, double* weight)\n";
  codes += "{\n";
  codes += "    double x0 = static_cast<double>(vars[x0_idx]);\n";
  codes += "    double x1 = static_cast<double>(vars[x1_idx]);\n";
  codes += "    double hbm_size_cost = GetSmoothhbm_sizeCost(vars);\n";
  codes += "    hbm_size_cost *= weight[0] > 0 ? weight[0] : 0;\n";
  codes += "    double leq1_cost = (x0 + x1 - a);\n";
  codes += "    leq1_cost *= weight[1] > 0 ? weight[1] : 0;\n";
  codes += "    return hbm_size_cost + leq1_cost;\n";
  codes += "}\n";

  codes += "inline bool GeneralSolverCase0::CheckLocalValid(double* leqs, int32_t idx)\n";
  codes += "{\n";
  codes += "    if (idx == x0_idx) {\n";
  codes += "        return leqs[0] <= 0 && leqs[1] <= 0;\n";
  codes += "    } else if (idx == x1_idx) {\n";
  codes += "        return leqs[0] <= 0 && leqs[1] <= 0;\n";
  codes += "    }\n";
  codes += "    return true;\n";
  codes += "}\n";
  codes += "\n";
}

void AppendSolverImplPart4(std::string &codes) {
  codes += "inline void GeneralSolverCase0::UpdateLeqs(uint64_t* vars, int32_t idx, double* leqs)\n";
  codes += "{\n";
  codes += "    double x0 = static_cast<double>(vars[x0_idx]);\n";
  codes += "    double x1 = static_cast<double>(vars[x1_idx]);\n";
  codes += "    if (idx == x0_idx) {\n";
  codes += "        leqs[0] = (x0 + x1 - hbm_size);\n";
  codes += "        leqs[1] = (x0 + x1 - a);\n";
  codes += "    } else if (idx == x1_idx) {\n";
  codes += "        leqs[0] = (x0 + x1 - hbm_size);\n";
  codes += "        leqs[1] = (x0 + x1 - a);\n";
  codes += "    } else if (idx == -1) {\n";
  codes += "        leqs[0] = (x0 + x1 - hbm_size);\n";
  codes += "        leqs[1] = (x0 + x1 - a);\n";
  codes += "    }\n";
  codes += "}\n";
  codes += "\n";

  codes += "inline void GeneralSolverCase0::DisplayVarVal(uint64_t* vars)\n";
  codes += "{\n";
  codes += "    uint64_t x0 = vars[x0_idx];\n";
  codes += "    uint64_t x1 = vars[x1_idx];\n";
  codes += "    OP_LOGD(OP_NAME, \"x3 = %lu\", static_cast<uint64_t>(1));\n";
  codes += "}\n";
  codes += "\n";

  codes += "inline void GeneralSolverCase0::MapVarVal(uint64_t* vars, TilingData& tiling_data)\n";
  codes += "{\n";
  codes += "    uint64_t x0 = vars[x0_idx];\n";
  codes += "    uint64_t x1 = vars[x1_idx];\n";
  codes += "    OP_LOGD(OP_NAME, \"The output of the solver for tilingCaseId Case0 is:\");\n";
  codes += "    tiling_data.set_x3(static_cast<uint64_t>(1));\n";
  codes += "    OP_LOGD(OP_NAME, \"x3 = %u\", tiling_data.get_x3());\n";
  codes += "}\n";
  codes += "\n";

  codes +=
      "inline void GeneralSolverCase0::GetResult(int32_t solution_num, uint64_t* solution, TilingData& "
      "tiling_data)\n{\n";
  codes += "    if (solution_num > 0) {\n";
  codes += "        OP_LOGD(OP_NAME, \"Filling tilingdata for Case0.\");\n";
  codes += "        OP_LOGD(OP_NAME, \"Estimate the occupy.\");\n";
  codes +=
      "        OP_LOGD(OP_NAME, \"hbm_size = %ld\", static_cast<uint64_t>(Gethbm_sizeCost(solution) + hbm_size));\n";
  codes += "        OP_LOGD(OP_NAME, \"Simulate the cost.\");\n";
  codes += "        OP_LOGD(OP_NAME, \"Objective value for Case0 is %f.\", GetObj(solution));\n";
}

void AppendSolverImplPart5(std::string &codes) {
  codes += "        MapVarVal(solution, tiling_data);\n";
  codes += "    }\n";
  codes += "}\n\n";

  codes += "bool ExecuteCase0GeneralSolver(TilingData& tiling_data)\n";
  codes += "{\n";

  codes += "    SolverConfig cfg;\n";
  codes += "    cfg.top_num = cfg_top_num;\n";
  codes += "    cfg.search_length = cfg_search_length;\n";
  codes += "    cfg.iterations = cfg_iterations;\n";
  codes += "    cfg.simple_ver = cfg_simple_ver;\n";
  codes +=
      "    cfg.momentum_factor = cfg_momentum_factor > 1 ? 1 : (cfg_momentum_factor < 0 ? 0 : cfg_momentum_factor);\n";
  codes += "    OP_LOGD(OP_NAME, \"Record a maximum of %lu solutions.\", cfg.top_num);\n";
  codes += "    OP_LOGD(OP_NAME, \"The searching range covers %lu unit(s).\", cfg.search_length);\n";
  codes += "    OP_LOGD(OP_NAME, \"The maximum number of iterations is %lu.\", cfg.iterations);\n";
  codes += "    if (cfg.simple_ver) {\n";
  codes += "        OP_LOGD(OP_NAME, \"Using high-efficiency version.\");\n";
  codes += "    } else {\n";
  codes += "        OP_LOGD(OP_NAME, \"Using high-performance version.\");\n";
  codes += "    }\n";
  codes += "    OP_LOGD(OP_NAME, \"The momentum factor is %f.\", cfg.momentum_factor);\n";
  codes += "\n";

  codes += "    // 以下参数若未注明是可修改参数,则不建议修改\n";
  codes += "    // 由modelinfo传入的待求解变量个数\n";
  codes += "    int32_t num_var = 2;\n";
  codes += "    // 由modelinfo传入的不等式约束个数\n";
  codes += "    int32_t num_leq = 2;\n";
  codes +=
      "    OP_LOGD(OP_NAME, \"The number of variable is %d(x0, x1), the number of constraints is %d.\", num_var, "
      "num_leq);\n";
  codes += "    // (可修改参数) 待求解变量的初始值,算法趋向于求初始值附近的局部最优解\n";
  codes += "    uint64_t init_vars[num_var] = {static_cast<uint64_t>(5), static_cast<uint64_t>((2 * a))};\n";
  codes +=
      "    // (可修改参数) "
      "待求解变量的上界,过大的上界将导致搜索范围与耗时增加,过小的上界更有可能获得较差的局部最优解\n";
  codes += "    uint64_t upper_bound[num_var] = {static_cast<uint64_t>(5), static_cast<uint64_t>((2 * a))};\n";
  codes +=
      "    // (可修改参数) "
      "待求解变量的下界,过小的下界将导致搜索范围与耗时增加,过大的下界更有可能获得较差的局部最优解\n";
  codes += "    uint64_t lower_bound[num_var] = {static_cast<uint64_t>(1), static_cast<uint64_t>(1)};\n";
  codes += "    // (可修改参数) 最后更新的待求解变量,设置为true的对应变量会更接近初始值\n";
  codes += "    bool update_last[num_var] = {true, false};\n";
  codes += "    // 初始化解的个数为0\n";
}

void AppendSolverImplPart6(std::string &codes) {
  codes += "    int32_t solution_num = 0;\n";
  codes += "    // 为求解器的输出分配内存\n";
  codes += "    uint64_t* solution = new(std::nothrow) uint64_t[num_var * cfg.top_num];\n";
  codes += "    if (solution == nullptr)\n";
  codes += "    {\n";
  codes += "        OP_LOGW(OP_NAME, \"Create solution failed.\");\n";
  codes += "        return false;\n";
  codes += "    }\n";
  codes += "    // 通用求解器的输入参数\n";
  codes += "    SolverInput input;\n";
  codes += "    input.var_num = num_var;\n";
  codes += "    input.leq_num = num_leq;\n";
  codes += "    input.cur_vars = init_vars;\n";
  codes += "    input.upper_bound = upper_bound;\n";
  codes += "    input.lower_bound = lower_bound;\n";
  codes += "    input.update_last = update_last;\n";
  codes +=
      "    OP_LOGD(OP_NAME, \"x0->init value: %lu, range: [%lu, %lu].\", init_vars[0], lower_bound[0], "
      "upper_bound[0]);\n";
  codes +=
      "    OP_LOGD(OP_NAME, \"x1->init value: %lu, range: [%lu, %lu].\", init_vars[1], lower_bound[1], "
      "upper_bound[1]);\n";
  codes += "\n";

  codes += "    GeneralSolverCase0* solver = new(std::nothrow) GeneralSolverCase0(cfg, tiling_data);\n";
  codes += "    if (solver != nullptr) {\n";
  codes += "        // 导入通用求解器的输入参数并完成初始化\n";
  codes += "        OP_LOGD(OP_NAME, \"Start initializing the input.\");\n";
  codes += "        if (solver -> Init(input)) {\n";
  codes += "            // 运行通用求解器并获取算法的解\n";
  codes += "            OP_LOGD(OP_NAME, \"Initialization finished, start running the solver.\");\n";
  codes += "            if (solver -> Run(solution_num, solution)) {\n";
  codes += "                solver -> GetResult(solution_num, solution, tiling_data);\n";
  codes += "                delete solver;\n";
  codes += "                delete[] solution;\n";
  codes += "                OP_LOGD(OP_NAME, \"The solver executed successfully.\");\n";
  codes += "                return true;\n";
  codes += "            }\n";
  codes += "            OP_LOGW(OP_NAME, \"Failed to find any solution.\");\n";
  codes += "        }\n";
  codes += "    }\n";
  codes += "    if (solver != nullptr) {\n";
  codes += "        delete solver;\n";
  codes += "    }\n";
  codes += "    if (solution != nullptr) {\n";
}

void AppendSolverImplPart7(std::string &codes) {
  codes += "        delete[] solution;\n";
  codes += "    }\n";

  codes += "    OP_LOGW(OP_NAME, \"The solver executed failed.\");\n";
  codes += "    return false;\n";
  codes += "}\n";
  codes += "\n";
}

std::string GetExpectedSolverImplCodes() {
  std::string codes;
  AppendSolverImplPart0(codes);
  AppendSolverImplPart1(codes);
  AppendSolverImplPart2(codes);
  AppendSolverImplPart3(codes);
  AppendSolverImplPart4(codes);
  AppendSolverImplPart5(codes);
  AppendSolverImplPart6(codes);
  AppendSolverImplPart7(codes);
  return codes;
}

void AppendSolverInvokePart0(std::string &codes) {
  codes += "bool ExecuteCase0GeneralSolver(TilingData& tiling_data)\n";
  codes += "{\n";

  codes += "    SolverConfig cfg;\n";
  codes += "    cfg.top_num = cfg_top_num;\n";
  codes += "    cfg.search_length = cfg_search_length;\n";
  codes += "    cfg.iterations = cfg_iterations;\n";
  codes += "    cfg.simple_ver = cfg_simple_ver;\n";
  codes +=
      "    cfg.momentum_factor = cfg_momentum_factor > 1 ? 1 : (cfg_momentum_factor < 0 ? 0 : cfg_momentum_factor);\n";
  codes += "    OP_LOGD(OP_NAME, \"Record a maximum of %lu solutions.\", cfg.top_num);\n";
  codes += "    OP_LOGD(OP_NAME, \"The searching range covers %lu unit(s).\", cfg.search_length);\n";
  codes += "    OP_LOGD(OP_NAME, \"The maximum number of iterations is %lu.\", cfg.iterations);\n";
  codes += "    if (cfg.simple_ver) {\n";
  codes += "        OP_LOGD(OP_NAME, \"Using high-efficiency version.\");\n";
  codes += "    } else {\n";
  codes += "        OP_LOGD(OP_NAME, \"Using high-performance version.\");\n";
  codes += "    }\n";
  codes += "    OP_LOGD(OP_NAME, \"The momentum factor is %f.\", cfg.momentum_factor);\n";
  codes += "\n";

  codes += "    // 以下参数若未注明是可修改参数,则不建议修改\n";
  codes += "    // 由modelinfo传入的待求解变量个数\n";
  codes += "    int32_t num_var = 2;\n";
  codes += "    // 由modelinfo传入的不等式约束个数\n";
  codes += "    int32_t num_leq = 2;\n";
  codes +=
      "    OP_LOGD(OP_NAME, \"The number of variable is %d(x0, x1), the number of constraints is %d.\", num_var, "
      "num_leq);\n";
  codes += "    // (可修改参数) 待求解变量的初始值,算法趋向于求初始值附近的局部最优解\n";
  codes += "    uint64_t init_vars[num_var] = {static_cast<uint64_t>(5), static_cast<uint64_t>((2 * a))};\n";
  codes +=
      "    // (可修改参数) "
      "待求解变量的上界,过大的上界将导致搜索范围与耗时增加,过小的上界更有可能获得较差的局部最优解\n";
  codes += "    uint64_t upper_bound[num_var] = {static_cast<uint64_t>(5), static_cast<uint64_t>((2 * a))};\n";
  codes +=
      "    // (可修改参数) "
      "待求解变量的下界,过小的下界将导致搜索范围与耗时增加,过大的下界更有可能获得较差的局部最优解\n";
  codes += "    uint64_t lower_bound[num_var] = {static_cast<uint64_t>(1), static_cast<uint64_t>(1)};\n";
  codes += "    // (可修改参数) 最后更新的待求解变量,设置为true的对应变量会更接近初始值\n";
  codes += "    bool update_last[num_var] = {true, false};\n";
  codes += "    // 初始化解的个数为0\n";
  codes += "    int32_t solution_num = 0;\n";
}

void AppendSolverInvokePart1(std::string &codes) {
  codes += "    // 为求解器的输出分配内存\n";
  codes += "    uint64_t* solution = new(std::nothrow) uint64_t[num_var * cfg.top_num];\n";
  codes += "    if (solution == nullptr)\n";
  codes += "    {\n";
  codes += "        OP_LOGW(OP_NAME, \"Create solution failed.\");\n";
  codes += "        return false;\n";
  codes += "    }\n";
  codes += "    // 通用求解器的输入参数\n";
  codes += "    SolverInput input;\n";
  codes += "    input.var_num = num_var;\n";
  codes += "    input.leq_num = num_leq;\n";
  codes += "    input.cur_vars = init_vars;\n";
  codes += "    input.upper_bound = upper_bound;\n";
  codes += "    input.lower_bound = lower_bound;\n";
  codes += "    input.update_last = update_last;\n";
  codes +=
      "    OP_LOGD(OP_NAME, \"x0->init value: %lu, range: [%lu, %lu].\", init_vars[0], lower_bound[0], "
      "upper_bound[0]);\n";
  codes +=
      "    OP_LOGD(OP_NAME, \"x1->init value: %lu, range: [%lu, %lu].\", init_vars[1], lower_bound[1], "
      "upper_bound[1]);\n";
  codes += "\n";

  codes += "    GeneralSolverCase0* solver = new(std::nothrow) GeneralSolverCase0(cfg, tiling_data);\n";
  codes += "    if (solver != nullptr) {\n";
  codes += "        // 导入通用求解器的输入参数并完成初始化\n";
  codes += "        OP_LOGD(OP_NAME, \"Start initializing the input.\");\n";
  codes += "        if (solver -> Init(input)) {\n";
  codes += "            // 运行通用求解器并获取算法的解\n";
  codes += "            OP_LOGD(OP_NAME, \"Initialization finished, start running the solver.\");\n";
  codes += "            if (solver -> Run(solution_num, solution)) {\n";
  codes += "                solver -> GetResult(solution_num, solution, tiling_data);\n";
  codes += "                delete solver;\n";
  codes += "                delete[] solution;\n";
  codes += "                OP_LOGD(OP_NAME, \"The solver executed successfully.\");\n";
  codes += "                return true;\n";
  codes += "            }\n";
  codes += "            OP_LOGW(OP_NAME, \"Failed to find any solution.\");\n";
  codes += "        }\n";
  codes += "    }\n";
  codes += "    if (solver != nullptr) {\n";
  codes += "        delete solver;\n";
  codes += "    }\n";
  codes += "    if (solution != nullptr) {\n";
  codes += "        delete[] solution;\n";
}

void AppendSolverInvokePart2(std::string &codes) {
  codes += "    }\n";

  codes += "    OP_LOGW(OP_NAME, \"The solver executed failed.\");\n";
  codes += "    return false;\n";
  codes += "}\n";
  codes += "\n";
}

std::string GetExpectedSolverInvokeCodes() {
  std::string codes;
  AppendSolverInvokePart0(codes);
  AppendSolverInvokePart1(codes);
  AppendSolverInvokePart2(codes);
  return codes;
}
}  // namespace

TEST_F(ST_GENERAL_SOLVER_GEN, test_gen_solver_impl) {
  std::string codes = solver_->GenSolverClassImpl();
  std::string expect_codes = GetExpectedSolverImplCodes();
  EXPECT_NE(codes, "");
}

TEST_F(ST_GENERAL_SOLVER_GEN, test_gen_solver_invoke) {
  std::string codes = solver_->GenSolverFuncInvoke();
  std::string expect_codes = GetExpectedSolverInvokeCodes();
  EXPECT_NE(codes, "");
}
