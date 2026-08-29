/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_GENERAL_SOLVER_H_
#define ATT_GENERAL_SOLVER_H_
#include <string>
#include "util/base_types_printer.h"
#include "util/duration.h"

namespace att {
inline std::string GenConstVars() {
  std::string strs = "";
  std::string general_solver = "";
  strs += "(可修改变量)用于控制通用求解器求解质量的超参数\n";
  strs += "cfg_top_num:保留目标函数最优的前top_num个解,用户可以打印这些解并从中选取较优项(默认值为5)\n";
  strs += "cfg_search_length:在可行域内执行局部搜索的搜索范围,当搜索范围内存在更优的解时会将该解视为候选\n";
  strs += "  搜索范围越大,越有可能获取更优的解,但求解耗时更长(默认值为1)\n";
  strs += "cfg_iterations:启发式求解算法的迭代轮次上限,算法最多执行iterations次,并在满足早停逻辑时提前退出\n";
  strs += "  在不满足早停逻辑的前提下,设置更大的iterations算法有机会取得更好的解,但求解耗时更长(默认值为500)\n";
  strs += "cfg_simple_ver:用户可以选择使用的求解器版本(高效率版/高性能版)\n";
  strs += "  高效率版采用二分搜索逻辑搜索更优解,变量求解顺序相对简单\n";
  strs += "  高性能版会检查搜索范围内所有的可行解,同时采用更精细的变量求解顺序\n";
  strs += "  高性能版的耗时相对更长,但是可能取到比高效率版更优的解(默认采用高效率版)\n";
  strs += "cfg_momentum_factor:更新变量信息时所采用的动量因子\n";
  strs += "  在选取变量时,变量的动量值为momentum * momentum_factor + update_value * (1 - momentum_factor)\n";
  strs += "  动量因子越大,求解器越可能反复选取同一个变量进行更新(默认值为0.9)\n";
  strs += "  当用户取大于1的数时取1,取小于0的数时取0\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "static const uint64_t cfg_top_num = 5;\n";
  general_solver += "static const uint64_t cfg_search_length = 1;\n";
  general_solver += "static const uint64_t cfg_iterations = 100;\n";
  general_solver += "static const bool cfg_simple_ver = true;\n";
  general_solver += "static const double cfg_momentum_factor = 0.9;\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenLocality() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "Locality:定域过程中待求解变量的优先级\n";
  strs += "  GLOBALVALID:更新该变量会使待求解变量走入可行域,即直接获取一个可行解\n";
  strs += "  LOCALVALID:更新该变量能满足该变量相关的约束\n";
  strs += "  CROSSREGION:更新该变量会跨越可行域,即由可行域的一侧到达另一侧\n";
  strs += "  INVALID:仅更新该变量无法获取可行域内的解,即定义域内不存在可行域\n";
  strs +=
      "  "
      "ALTERNATIVE:(仅在高性能版本中生效)该变量的预期落点是曾搜索得到的解,"
      "尝试跨越可行域获取另一侧边界的解作为备选方案\n";
  strs += "  REJECT:该变量的落点为上轮迭代中的实际落点,即出现了反复震荡\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "enum class Locality\n";
  general_solver += "{\n";
  general_solver += "    GLOBALVALID = 0,\n";
  general_solver += "    LOCALVALID = 1,\n";
  general_solver += "    CROSSREGION = 2,\n";
  general_solver += "    INVALID = 3,\n";
  general_solver += "    ALTERNATIVE = 4,\n";
  general_solver += "    REJECT = 5,\n";
  general_solver += "};\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenTunePriority() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "TunePriority:微调过程中待求解变量的优先级\n";
  strs += "  HARMLESS:更新该变量会获得一个目标函数更优的可行解(即存在无损更新)\n";
  strs += "  DILATED:更新该变量会获得一个目标函数不变,距离缓存占用边界更近的可行解(即存在膨胀更新)\n";
  strs += "  NORMAL:沿着目标函数的优化方向进行更新会走出可行域\n";
  strs += "  OTHER:更新变量会走出可行域并获得一个更差的解\n";
  strs += "  TABU:该变量的落点为上轮迭代中的实际落点,即出现了反复震荡\n";
  strs += "  REFUSE:更新后会在可行域内获得一个更差的解\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "enum class TunePriority\n";
  general_solver += "{\n";
  general_solver += "    HARMLESS = 0,\n";
  general_solver += "    DILATED = 1,\n";
  general_solver += "    NORMAL = 2,\n";
  general_solver += "    OTHER = 3,\n";
  general_solver += "    TABU = 4,\n";
  general_solver += "    REFUSE = 5,\n";
  general_solver += "};\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenFuncInfo() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "FuncInfo:函数信息\n";
  strs += "  LEQ:不等式约束所对应的罚函数\n";
  strs += "  BUFFER:缓存占用约束所对应的罚函数\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "enum class FuncInfo\n";
  general_solver += "{\n";
  general_solver += "    LEQ = 0,\n";
  general_solver += "    BUFFER = 1,\n";
  general_solver += "};\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenUpdateDirection() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "UpdateDirection:变量的更新方向\n";
  strs += "  POSITIVE:沿正方向更新\n";
  strs += "  NONE:不存在更新方向\n";
  strs += "  POSITIVE:沿负方向更新\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "enum class UpdateDirection\n";
  general_solver += "{\n";
  general_solver += "    POSITIVE = 0,\n";
  general_solver += "    NONE = 1,\n";
  general_solver += "    NEGATIVE = 2,\n";
  general_solver += "};\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenUpdateInfo() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "UpdateInfo:变量的更新信息\n";
  strs += "  idx:变量的索引值\n";
  strs += "  thres:沿着更新方向变量的更新阈值\n";
  strs += "  update_direction:变量的更新方向\n";
  strs += "  init_obj:更新前变量的目标函数值\n";
  strs += "  init_cons:更新前变量的缓存占用冗余\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "struct UpdateInfo\n";
  general_solver += "{\n";
  general_solver += "    int32_t idx{0};\n";
  general_solver += "    uint64_t thres{0u};\n";
  general_solver += "    UpdateDirection update_direction{UpdateDirection::NONE};\n";
  general_solver += "    double init_obj{0};\n";
  general_solver += "    double init_cons{0};\n";
  general_solver +=
      "    UpdateInfo(int32_t idx, uint64_t thres, UpdateDirection direction, double obj = 0, double cons = 0) : "
      "idx(idx), thres(thres), update_direction(direction), init_obj(obj), init_cons(cons) {}\n";
  general_solver += "};\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenNode() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "Node:用于记录待求解变量的数据结构,以{x0,x1}为例,假设当前指向x0\n";
  strs += "  value:x0的值\n";
  strs += "  next_val:x0的下一个值\n";
  strs += "  next_var:当前x0的value所对应的解中x1的第一个值\n";
  strs += "  next_node:指向下一个node对象的指针\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "struct Node\n";
  general_solver += "{\n";
  general_solver += "    uint64_t value{0u};\n";
  general_solver += "    bool searched{false};\n";
  general_solver += "    Node *next_val{nullptr};\n";
  general_solver += "    Node *next_var{nullptr};\n";
  general_solver += "    Node *next_node{nullptr};\n";
  general_solver += "    explicit Node(uint64_t val) : value(val) {}\n";
  general_solver += "};\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenVisitedNode() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "VisitedNode:用于记录已搜索到的可行解\n";
  strs += "  depth:待求解变量的个数\n";
  strs += "  head:首个node节点(为值为0)\n";
  strs += "  tail:最后一个node节点\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "struct VisitedNode\n";
  general_solver += "{\n";
  general_solver += "    void SetVisitedNode(int32_t num_var, uint64_t *head_ptr) {\n";
  general_solver += "        rec_num = 0u;\n";
  general_solver += "        var_num = num_var;\n";
  general_solver += "        head = head_ptr;\n";
  general_solver += "    }\n";
  general_solver += "    int32_t Cmp(uint64_t idx, uint64_t *vars);\n";
  general_solver += "    bool SearchVars(uint64_t *vars, bool insert_vars);\n";
  general_solver += "    uint64_t rec_num{0u};\n";
  general_solver += "    uint64_t var_num{0u};\n";
  general_solver += "    uint64_t *head{nullptr};\n";
  general_solver += "};\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenSolverInput() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "SolverInput:求解器所需的输入信息\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "struct SolverInput\n";
  general_solver += "{\n";
  general_solver += "    uint64_t corenum{0u};\n";
  general_solver += "    VarInfo *var_info{nullptr};\n";
  general_solver += "    ConsInfo *cons_info{nullptr};\n";
  general_solver += "    Momentum *momentum{nullptr};\n";
  general_solver += "    Result *result{nullptr};\n";
  general_solver += "    VisitedNode *visited_node{nullptr};\n";
  general_solver += "};\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenSolverConfig() {
  std::string general_solver = "";
  general_solver += "struct SolverConfig\n";
  general_solver += "{\n";
  general_solver += "    uint64_t top_num{5u};\n";
  general_solver += "    uint64_t search_length{1u};\n";
  general_solver += "    uint64_t iterations{500u};\n";
  general_solver += "    bool simple_ver{false};\n";
  general_solver += "    double momentum_factor{0.9f};\n";
  general_solver += "};\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenVarVal() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "VarVal:用于输出至Result的中间信息\n";
  strs += "  var_num_:待求解变量的个数\n";
  strs += "  obj_:解的目标函数值\n";
  strs += "  cons_:解的缓存占用冗余值\n";
  strs += "  vars_:可行解的指针\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "struct VarVal\n";
  general_solver += "{\n";
  general_solver += "    void SetVarVal(double var_obj, double var_cons, uint64_t *varval) {\n";
  general_solver += "        obj = var_obj;\n";
  general_solver += "        cons = var_cons;\n";
  general_solver += "        for (int32_t i = 0; i < var_num; i++)\n";
  general_solver += "        {\n";
  general_solver += "            vars[i] = varval[i];\n";
  general_solver += "        }\n";
  general_solver += "    }\n";
  general_solver += "    inline void CopyVarVal(VarVal* from_var) {\n";
  general_solver += "        SetVarVal(from_var->obj, from_var->cons, from_var->vars);\n";
  general_solver += "    }\n";
  general_solver += "    int32_t var_num{0};\n";
  general_solver += "    double obj{0};\n";
  general_solver += "    double cons{0};\n";
  general_solver += "    uint64_t *vars{nullptr};\n";
  general_solver += "};\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string SetResult() {
  std::string general_solver = "";
  general_solver +=
      "    void SetResult(int32_t top_num, int32_t var_num, VarVal *var_space, char *temp_space, char "
      "*solution_space)\n";
  general_solver += "    {\n";
  general_solver += "        solution_num_ = 0;\n";
  general_solver += "        top_n_ = top_num;\n";
  general_solver += "        var_num_ = var_num;\n";
  general_solver += "        new_var_ = var_space;\n";
  general_solver += "        temp_ = temp_space;\n";
  general_solver += "        solution_ = solution_space;\n";
  general_solver += "    }\n";
  return general_solver;
}

inline std::string GenResult() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "Result:最终输出的解信息\n";
  strs += "  top_n_:最多可以记录的可行解个数\n";
  strs += "  var_num_:待求解变量的个数\n";
  strs += "  solution_num_:输出的可行解个数(不会大于top_n)\n";
  strs += "  solution_:输出的可行解(占用空间的尺寸为top_n*var_num_,有效元素个数为solution_num_*var_num_)\n";
  strs += "    其中,第i组解可通过访问[(i-1)*var_num_, i*var_num_)范围内的元素获取\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "class Result\n";
  general_solver += "{\n";
  general_solver += "public:\n";
  general_solver += SetResult();
  general_solver += "    bool AddVarVal(uint64_t *vars, double obj, double cons);\n";
  general_solver += "    bool GetResult(int32_t &solution_num, uint64_t *solution);\n";
  general_solver += "    VarVal *GetTemp(size_t idx);\n";
  general_solver += "    VarVal *GetSolution(size_t idx);\n";
  general_solver += "\n";
  general_solver += "private:\n";
  general_solver += "    uint32_t top_n_{0};\n";
  general_solver += "    uint32_t var_num_{0};\n";
  general_solver += "    uint32_t solution_num_{0};\n";
  general_solver += "    VarVal *new_var_{nullptr};\n";
  general_solver += "    char *temp_{nullptr};\n";
  general_solver += "    char *solution_{nullptr};\n";
  general_solver += "};\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string ConstructVarInfo() {
  std::string general_solver = "";
  general_solver += "    void SetVarInfo(int32_t num_var, uint64_t *uint_space, bool *bool_space) {\n";
  general_solver += "        var_num = num_var;\n";
  general_solver += "        upper_bound = uint_space;\n";
  general_solver += "        lower_bound = uint_space + var_num;\n";
  general_solver += "        history_vars = uint_space + 2 * var_num;\n";
  general_solver += "        rec_vars = uint_space + 3 * var_num;\n";
  general_solver += "        cur_vars = uint_space + 4 * var_num;\n";
  general_solver += "        target_val = uint_space + 5 * var_num;\n";
  general_solver += "        update_last = bool_space;\n";
  general_solver += "    }\n";
  return general_solver;
}

inline std::string GenVarInfo() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "VarInfo:求解过程中的中间参数\n";
  strs += "  var_num:待求解变量个数\n";
  strs += "  chosen_var_idx:本轮迭代过程中待更新的变量下标\n";
  strs += "  upper_bound:待求解变量的上界(var_num个)\n";
  strs += "  history_vars:上轮迭代过程启动前待求解变量的值(var_num个)\n";
  strs += "  rec_vars:执行本轮迭代时待求解变量的值(var_num个)\n";
  strs += "  cur_vars:待求解变量的当前值(var_num个)\n";
  strs += "  target_val:待求解变量在本轮迭代过程中的预期值(var_num个)\n";
  strs += "  update_last:用于标记待求解变量,指明该变量是否需要最后切分\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "struct VarInfo\n";
  general_solver += "{\n";
  general_solver += "    int32_t var_num{0};\n";
  general_solver += "    int32_t chosen_var_idx{-1};\n";
  general_solver += "    uint64_t *upper_bound{nullptr};\n";
  general_solver += "    uint64_t *lower_bound{nullptr};\n";
  general_solver += "    uint64_t *history_vars{nullptr};\n";
  general_solver += "    uint64_t *rec_vars{nullptr};\n";
  general_solver += "    uint64_t *cur_vars{nullptr};\n";
  general_solver += "    uint64_t *target_val{nullptr};\n";
  general_solver += "    bool *update_last{nullptr};\n";
  general_solver += ConstructVarInfo();
  general_solver += "};\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string ConstructConsInfo() {
  std::string general_solver = "";
  general_solver += "    void SetConsInfo(int32_t num_leq, double *double_space)\n";
  general_solver += "    {\n";
  general_solver += "        leq_num = num_leq;\n";
  general_solver += "        leqs = double_space;\n";
  general_solver += "        weight = double_space + leq_num;\n";
  general_solver += "    }\n";
  return general_solver;
}

inline std::string GenConsInfo() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "ConsInfo:不等式约束信息\n";
  strs += "  leq_num:不等式约束个数\n";
  strs += "  leqs:不等式约束的函数值\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "struct ConsInfo\n";
  general_solver += "{\n";
  general_solver += "    int32_t leq_num{0};\n";
  general_solver += "    double *leqs{nullptr};\n";
  general_solver += "    double *weight{nullptr};\n";
  general_solver += ConstructConsInfo();
  general_solver += "};\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string ConstructMomentum() {
  std::string general_solver = "";
  general_solver +=
      "    void SetMomentum(int32_t var_num, int32_t leq_num, double *double_space, bool *bool_space) {\n";
  general_solver += "        momentum = double_space + 2 * leq_num;\n";
  general_solver += "        cur_value = double_space + 2 * leq_num + var_num;\n";
  general_solver += "        is_valid = bool_space + var_num;\n";
  general_solver += "    }\n";
  return general_solver;
}

inline std::string GenMomentum() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "Momentum:动量信息\n";
  strs += "  momentum:上轮迭代的动量值\n";
  strs += "  cur_value:本轮迭代的动量信息\n";
  strs += "  is_valid:用于判断是否为有效动量\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "struct Momentum\n";
  general_solver += "{\n";
  general_solver += "    double *momentum{nullptr};\n";
  general_solver += "    double *cur_value{nullptr};\n";
  general_solver += "    bool *is_valid{nullptr};\n";
  general_solver += ConstructMomentum();
  general_solver += "};\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenPublicFunc() {
  std::string general_solver = "";
  general_solver += "    bool Init(const SolverInput &input);\n";
  general_solver += "    virtual bool Run(int32_t &solution_num, uint64_t *solutions);\n";
  general_solver += "\n";
  general_solver += "    int32_t GetVarNum() const;\n";
  general_solver += "\n";
  general_solver += "    double GetFuncVal(uint64_t *vars, FuncInfo func_info);\n";
  general_solver += "    UpdateDirection GetDescent(uint64_t *vars, int32_t idx, FuncInfo func_info);\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenPrivateFunc() {
  std::string general_solver = "";
  general_solver += "    bool SetSolverInput(const SolverInput &input);\n";
  general_solver += "    bool SearchVars(uint64_t *vars) const;\n";
  general_solver += "    bool UpdateCurVarVal(uint64_t value, int32_t idx);\n";
  general_solver += "\n";
  general_solver += "    Locality GetLocality(int32_t idx, UpdateDirection update_direction);\n";
  general_solver += "    bool GetCoarseLoc(const UpdateInfo &update_info, uint64_t &step, Locality &cur_locality);\n";
  general_solver += "    bool GetFineLoc(const UpdateInfo &update_info, uint64_t &step, Locality &cur_locality);\n";
  general_solver += "    bool GetPeerLoc(const UpdateInfo &update_info, Locality &cur_locality);\n";
  general_solver +=
      "    bool LocateLoc(const UpdateInfo &update_info, uint64_t &step, Locality &cur_locality, Locality "
      "&best_locality);\n";
  general_solver += "    bool TryLocate(int32_t idx, double init_obj, Locality &best_locality);\n";
  general_solver += "\n";
  general_solver += "    TunePriority GetTunePriority(int32_t idx, double rec_obj, double &cur_obj);\n";
  general_solver +=
      "    bool SearchLoc(const UpdateInfo &update_info, uint64_t &step, double &cur_obj, TunePriority "
      "&cur_priority);\n";
  general_solver += "    bool GetHarmlessLoc(const UpdateInfo &update_info, uint64_t &step, double &cur_obj);\n";
  general_solver += "    bool GetDilatedLoc(const UpdateInfo &update_info, uint64_t &step);\n";
  general_solver +=
      "    bool TuneLoc(const UpdateInfo &update_info, double cur_obj, uint64_t &step, TunePriority &cur_priority, "
      "TunePriority &best_priority);\n";
  general_solver +=
      "    bool TryTune(int32_t idx, UpdateDirection update_direction, double init_obj, double init_cons, TunePriority "
      "&best_priority);\n";
  general_solver += "\n";
  general_solver += "    bool CheckValid() const;\n";
  general_solver += "    void ResetMomentum();\n";
  general_solver +=
      "    void UpdateMomentum(int32_t idx, double update_value, Locality cur_locality, Locality &best_locality);\n";
  general_solver +=
      "    void UpdateMomentum(int32_t idx, double update_value, TunePriority cur_priority, TunePriority "
      "&best_priority);\n";
  general_solver += "    bool GetBestChoice();\n";
  general_solver += "    bool UpdateBestVar();\n";
  general_solver += "\n";
  general_solver += "    void Initialize(int32_t iter);\n";
  general_solver += "    bool LocateRegion();\n";
  general_solver += "    bool FineTune();\n";
  general_solver += "    bool RecordBestVarVal();\n";
  general_solver += "    bool is_feasible_{false};\n";
  general_solver += "    bool has_feasible_{false};\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenClassDef() {
  std::string general_solver = "";
  general_solver += "template <typename SpecificCase>\n";
  general_solver += "class GeneralSolver\n";
  general_solver += "{\n";
  general_solver += "public:\n";
  general_solver += GenPublicFunc();
  general_solver += "    SolverConfig solver_config_;\n";
  general_solver += "    string case_id_;\n";
  general_solver += "private:\n";
  general_solver += GenPrivateFunc();
  general_solver += "    Result *result_{nullptr};\n";
  general_solver += "    VarInfo *var_info_{nullptr};\n";
  general_solver += "    ConsInfo *cons_info_{nullptr};\n";
  general_solver += "    Momentum *momentum_info_{nullptr};\n";
  general_solver += "    VisitedNode *visited_node_{nullptr};\n";
  general_solver += "};\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenDef() {
  std::string general_solver = "";
  general_solver += GenLocality();
  general_solver += GenTunePriority();
  general_solver += GenFuncInfo();
  general_solver += GenUpdateDirection();
  general_solver += GenUpdateInfo();
  general_solver += GenNode();
  general_solver += GenVisitedNode();
  general_solver += GenSolverConfig();
  general_solver += GenVarVal();
  general_solver += GenResult();
  general_solver += GenVarInfo();
  general_solver += GenConsInfo();
  general_solver += GenMomentum();
  general_solver += GenSolverInput();
  general_solver += GenClassDef();
  return general_solver;
}

inline std::string GenToolFunc() {
  std::string general_solver = "";
  general_solver += "inline int32_t GetValue(UpdateDirection update_direction)\n";
  general_solver += "{\n";
  general_solver += "    const int32_t positive = 1;\n";
  general_solver += "    const int32_t none = 0;\n";
  general_solver += "    const int32_t negative = -1;\n";
  general_solver += "    if (update_direction == UpdateDirection::POSITIVE) {\n";
  general_solver += "        return positive;\n";
  general_solver += "    } else if (update_direction == UpdateDirection::NEGATIVE) {\n";
  general_solver += "        return negative;\n";
  general_solver += "    }\n";
  general_solver += "    return none;\n";
  general_solver += "}\n";
  general_solver += "\n";
  general_solver +=
      "inline uint64_t Bound(uint64_t upper_bound, uint64_t lower_bound, uint64_t val, uint64_t step, UpdateDirection "
      "direction)\n";
  general_solver += "{\n";
  general_solver += "    if (direction == UpdateDirection::POSITIVE)\n";
  general_solver += "    {\n";
  general_solver += "        return (step + val > upper_bound) ? upper_bound : (step + val);\n";
  general_solver += "    }\n";
  general_solver +=
      "    return (step > val) ? lower_bound : ((val - step < lower_bound) ? lower_bound : (val - step));\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenCmp() {
  std::string general_solver = "";
  general_solver += "inline int32_t VisitedNode::Cmp(uint64_t idx, uint64_t *vars) {\n";
  general_solver += "    uint64_t* cur_var = head + idx * var_num;\n";
  general_solver += "    for (uint32_t i = 0u; i < var_num; i++) {\n";
  general_solver += "        if (cur_var[i] > vars[i]) {\n";
  general_solver += "            return 1;\n";
  general_solver += "        } else if (cur_var[i] < vars[i]) {\n";
  general_solver += "            return -1;\n";
  general_solver += "        }\n";
  general_solver += "    }\n";
  general_solver += "    return 0;\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string AddVarValAnotation() {
  std::string strs = "";
  strs += "函数名:AddVarVal\n";
  strs += "功能描述:将一组可行解vars传入Result\n";
  strs += "  若这组可行解的质量较差(目标函数值较大或距离约束边界较远),则舍弃\n";
  strs += "  若这组可行解可以被排进前top_n_,则保留该组可行解\n";
  strs += "  temp: 最大容量为top_n的备选可行解集\n";
  strs += "  先将solution_复制到temp中\n";
  strs += "  然后比较new_vars的目标值与temp中元素的目标值\n";
  strs += "  自小到大地将可行解填入solution_\n";
  strs += "输入参数:\n";
  strs += "  vars:一组可行解\n";
  strs += "  obj:该可行解所对应的目标函数值\n";
  strs += "  cons:可行解距约束边界的距离\n";
  return AddAnotationBlock(strs);
}

inline std::string ReorderVarVal() {
  std::string general_solver = "";
  general_solver += "    for (uint64_t i = 0; i < rec_num; i++)\n";
  general_solver += "    {\n";
  general_solver += "        GetTemp(i)->CopyVarVal(GetSolution(i));\n";
  general_solver += "    }\n";
  general_solver += "\n";
  general_solver += "    while ((cnt_num < solution_num_) && (temp_idx < rec_num))\n";
  general_solver += "    {\n";
  general_solver += "        auto temp = GetTemp(temp_idx);\n";
  general_solver += "        if (!has_add && (obj < temp->obj || (IsEqual(obj, temp->obj) && cons < temp->cons))) {\n";
  general_solver += "            has_add = true;\n";
  general_solver += "            GetSolution(cnt_num++)->CopyVarVal(new_var_);\n";
  general_solver += "        } else {\n";
  general_solver += "            GetSolution(cnt_num++)->CopyVarVal(temp);\n";
  general_solver += "            ++temp_idx;\n";
  general_solver += "        }\n";
  general_solver += "    }\n";
  general_solver += "\n";
  general_solver += "    if ((!has_add) && (cnt_num < solution_num_))\n";
  general_solver += "    {\n";
  general_solver += "        GetSolution(cnt_num++)->CopyVarVal(new_var_);\n";
  general_solver += "        has_add = true;\n";
  general_solver += "    }\n";
  return general_solver;
}

inline std::string GenAddVarVal() {
  std::string general_solver = "";
  general_solver += AddVarValAnotation();
  general_solver += "inline bool Result::AddVarVal(uint64_t *vars, double obj, double cons)\n";
  general_solver += "{\n";
  general_solver += "    uint64_t rec_num = solution_num_;\n";
  general_solver += "    if (rec_num > MAX_SOLUTION) {\n";
  general_solver += "        OP_LOGW(OP_NAME, \"Too many solutions.\");\n";
  general_solver += "        return false;\n";
  general_solver += "    }\n";
  general_solver += "    uint32_t cnt_num = 0;\n";
  general_solver += "    uint32_t temp_idx = 0;\n";
  general_solver += "    bool has_add = false;\n";
  general_solver += "    solution_num_ = Min(solution_num_ + 1, top_n_);\n";
  general_solver += "    new_var_->SetVarVal(obj, cons, vars);\n";
  general_solver += "    if (rec_num == 0) {\n";
  general_solver += "        GetSolution(0)->CopyVarVal(new_var_);\n";
  general_solver += "        return true;\n";
  general_solver += "    }\n";
  general_solver += ReorderVarVal();
  general_solver += "    return cnt_num == solution_num_;\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenGetResult() {
  std::string general_solver = "";
  general_solver += "inline bool Result::GetResult(int32_t &solution_num, uint64_t *solution)\n";
  general_solver += "{\n";
  general_solver += "    for (uint32_t i = 0u; i < solution_num_; i++)\n";
  general_solver += "    {\n";
  general_solver += "        for (uint32_t j = 0u; j < var_num_; j++) {\n";
  general_solver += "            solution[i * var_num_ + j] = GetSolution(i)->vars[j];\n";
  general_solver += "        }\n";
  general_solver += "    }\n";
  general_solver += "    solution_num = solution_num_;\n";
  general_solver += "    return true;\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenGetVarPointer() {
  std::string general_solver = "";
  general_solver += "inline VarVal *Result::GetTemp(size_t idx) {\n";
  general_solver += "    return (VarVal*)(temp_ + idx * (sizeof(VarVal) + (sizeof(uint64_t) * var_num_)));\n";
  general_solver += "}\n";
  general_solver += "\n";
  general_solver += "inline VarVal *Result::GetSolution(size_t idx) {\n";
  general_solver += "    return (VarVal*)(solution_ + idx * (sizeof(VarVal) + (sizeof(uint64_t) * var_num_)));\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenResultFunc() {
  std::string general_solver = "";
  general_solver += GenAddVarVal();
  general_solver += GenGetResult();
  general_solver += GenGetVarPointer();
  return general_solver;
}

inline std::string GenGetFuncVal() {
  std::string general_solver = "";
  general_solver += "template <typename SpecificCase>\n";
  general_solver += "inline double GeneralSolver<SpecificCase>::GetFuncVal(uint64_t *vars, FuncInfo func_info)\n";
  general_solver += "{\n";
  general_solver += "    if (func_info == FuncInfo::BUFFER)\n";
  general_solver += "    {\n";
  general_solver += "        return static_cast<SpecificCase*>(this)->GetBuffDiff(vars, cons_info_->weight);\n";
  general_solver += "    }\n";
  general_solver += "    else if (func_info == FuncInfo::LEQ)\n";
  general_solver += "    {\n";
  general_solver += "        return static_cast<SpecificCase*>(this)->GetLeqDiff(vars, cons_info_->weight);\n";
  general_solver += "    }\n";
  general_solver += "    return 0;\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenGetDescent() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "函数名:GetDescent\n";
  strs += "功能描述:获取“缓存占用函数/不等式约束的罚函数”的下降方向\n";
  strs += "输入参数:\n";
  strs += "  vars:当前待求解参数的下降方向\n";
  strs += "  idx:关于某参数下降方向中,某参数的下标\n";
  strs += "  func_info:用于指明计算下降方向的函数(FuncInfo::BUFFER/FuncInfo::LEQ)\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "template <typename SpecificCase>\n";
  general_solver +=
      "inline UpdateDirection GeneralSolver<SpecificCase>::GetDescent(uint64_t *vars, int32_t idx, FuncInfo "
      "func_info)\n";
  general_solver += "{\n";
  general_solver += "    if ((idx < 0) || (idx >= var_info_->var_num)) {\n";
  general_solver +=
      "        OP_LOGW(OP_NAME, \"idx = %d, var_info_->var_num = %d, idx illegal.\", idx, var_info_->var_num);\n";
  general_solver += "        return UpdateDirection::NONE;\n";
  general_solver += "    }\n";
  general_solver += "    static_cast<SpecificCase*>(this)->UpdateLeqs(vars, -1, cons_info_->weight);\n";
  general_solver += "    double cur_val = GetFuncVal(vars, func_info);\n";

  general_solver += "    vars[idx] += 1;\n";
  general_solver += "    double next_val = GetFuncVal(vars, func_info);\n";
  general_solver += "    vars[idx] -= 1;\n";
  general_solver += "    if (!IsEqual(cur_val, next_val))\n";
  general_solver += "    {\n";
  general_solver += "        return (cur_val > next_val) ? UpdateDirection::POSITIVE : UpdateDirection::NEGATIVE;\n";
  general_solver += "    }\n";

  general_solver += "    if (vars[idx] >= 1)\n";
  general_solver += "    {\n";
  general_solver += "        vars[idx] -= 1;\n";
  general_solver += "        double pre_val = GetFuncVal(vars, func_info);\n";
  general_solver += "        vars[idx] += 1;\n";
  general_solver += "        if (!IsEqual(cur_val, pre_val))\n";
  general_solver += "        {\n";
  general_solver += "            return (pre_val > cur_val) ? UpdateDirection::POSITIVE : UpdateDirection::NEGATIVE;\n";
  general_solver += "        }\n";
  general_solver += "    }\n";
  general_solver += "    return UpdateDirection::NONE;\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenInit() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "函数名:Init\n";
  strs += "功能描述:初始化通用求解器,导入待求解变量的先验信息,分配求解器所需的空间\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "template <typename SpecificCase>\n";
  general_solver += "inline bool GeneralSolver<SpecificCase>::Init(const SolverInput &input)\n";
  general_solver += "{\n";
  general_solver += "    var_info_ = input.var_info;\n";
  general_solver += "    cons_info_ = input.cons_info;\n";
  general_solver += "    momentum_info_ = input.momentum;\n";
  general_solver += "    result_ = input.result;\n";
  general_solver += "    visited_node_ = input.visited_node;\n";
  general_solver += "    return true;\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenUpdateCurVarVal() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "函数名:UpdateCurVarVal\n";
  strs += "功能描述:更新cur_var中某个待求解变量的值,并同步更新不等式约束的值\n";
  strs += "输入参数:\n";
  strs += "  value:待求解变量被更新成为的值\n";
  strs += "  idx:更新的待求解变量的下标\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "template <typename SpecificCase>\n";
  general_solver += "inline bool GeneralSolver<SpecificCase>::UpdateCurVarVal(uint64_t value, int32_t idx)\n";
  general_solver += "{\n";
  general_solver += "    if (idx < 0 || idx >= var_info_->var_num) {\n";
  general_solver += "        return false;\n";
  general_solver += "    }\n";
  general_solver += "    var_info_->cur_vars[idx] = value;\n";
  general_solver += "    static_cast<SpecificCase*>(this)->UpdateLeqs(var_info_->cur_vars, idx, cons_info_->leqs);\n";
  general_solver += "    return true;\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenSearchVars() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "函数名:SearchVars\n";
  strs += "功能描述:用于判断某组解是否曾被搜索过\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "inline bool VisitedNode::SearchVars(uint64_t *vars, bool insert_vars) {\n";
  general_solver += "    int32_t cmp;\n";
  general_solver += "    uint64_t mid;\n";
  general_solver += "    int32_t left = 0;\n";
  general_solver += "    int32_t right = rec_num;\n";
  general_solver += "    while (left < right) {\n";
  general_solver += "        mid = (left + right) >> 1;\n";
  general_solver += "        cmp = Cmp(mid, vars);\n";
  general_solver += "        if (cmp == 0) {\n";
  general_solver += "            return true;\n";
  general_solver += "        } else if (cmp > 0) {\n";
  general_solver += "            right = mid - 1;\n";
  general_solver += "        } else {\n";
  general_solver += "            left = mid + 1;\n";
  general_solver += "        }\n";
  general_solver += "    }\n";
  general_solver += "    uint32_t uleft = static_cast<uint32_t>(left);\n";
  general_solver += "    if (uleft < rec_num && Cmp(uleft, vars) == 0) {\n";
  general_solver += "        return true;\n";
  general_solver += "    }\n";
  general_solver += "    if (rec_num < cfg_iterations && insert_vars) {\n";
  general_solver += "        for (uint32_t i = 0u; i < (rec_num - left) * var_num; ++i) {\n";
  general_solver += "            head[(rec_num + 1) * var_num - i - 1] = head[rec_num * var_num - i - 1];\n";
  general_solver += "        }\n";
  general_solver += "        for (uint32_t i = 0u; i < var_num; i++) {\n";
  general_solver += "            head[left * var_num + i] = vars[i];\n";
  general_solver += "        }\n";
  general_solver += "        ++rec_num;\n";
  general_solver += "    }\n";
  general_solver += "    return false;\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenCheckValid() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "函数名:CheckValid\n";
  strs += "功能描述:用于判断cur_var所对应的解是否为可行解\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "template <typename SpecificCase>\n";
  general_solver += "inline bool GeneralSolver<SpecificCase>::CheckValid() const\n";
  general_solver += "{\n";
  general_solver += "    for (int32_t i = 0; i < cons_info_->leq_num; i++)\n";
  general_solver += "    {\n";
  general_solver += "        if (cons_info_->leqs[i] > 0)\n";
  general_solver += "        {\n";
  general_solver += "            return false;\n";
  general_solver += "        }\n";
  general_solver += "    }\n";
  general_solver += "    return true;\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenResetMomentum() {
  std::string general_solver = "";
  general_solver += "template <typename SpecificCase>\n";
  general_solver += "inline void GeneralSolver<SpecificCase>::ResetMomentum()\n";
  general_solver += "{\n";
  general_solver += "    for (int32_t i = 0; i < var_info_->var_num; i++)\n";
  general_solver += "    {\n";
  general_solver += "        momentum_info_->is_valid[i] = false;\n";
  general_solver += "    }\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenInitialize() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "函数名:Initialize\n";
  strs += "功能描述:用于在每一轮迭代开始执行前进行初始化操作\n";
  strs += "  在此过程中会重置var_info_中的部分参数\n";
  strs += "  并根据当前状态的cur_vars信息更新不等式约束值\n";
  strs += "输入参数:\n";
  strs += "  iter:迭代轮次\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "template <typename SpecificCase>\n";
  general_solver += "inline void GeneralSolver<SpecificCase>::Initialize(int32_t iter)\n";
  general_solver += "{\n";
  general_solver += "    ResetMomentum();\n";
  general_solver += "    var_info_->chosen_var_idx = -1;\n";
  general_solver += "    static_cast<SpecificCase*>(this)->UpdateLeqs(var_info_->cur_vars, -1, cons_info_->leqs);\n";
  general_solver += "    is_feasible_ = CheckValid();\n";
  general_solver += "    has_feasible_ = has_feasible_ || is_feasible_;\n";
  general_solver += "    for (int32_t i = 0; i < var_info_->var_num; i++)\n";
  general_solver += "    {\n";
  general_solver +=
      "        var_info_->history_vars[i] = (iter == 1) ? (var_info_->cur_vars[i]) : (var_info_->rec_vars[i]);\n";
  general_solver += "        var_info_->rec_vars[i] = var_info_->cur_vars[i];\n";
  general_solver += "    }\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenGetLocality() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "函数名:GetLocality\n";
  strs += "功能描述:用来检测定域操作过程中所选变量的优先级\n";
  strs += "输入参数:\n";
  strs += "  idx:变量的下标\n";
  strs += "  update_direction:变量在当前位置的下降方向\n";
  strs += "输出参数:\n";
  strs += "  Locality类型的优先级指标\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "template <typename SpecificCase>\n";
  general_solver +=
      "inline Locality GeneralSolver<SpecificCase>::GetLocality(int32_t idx, UpdateDirection update_direction)\n";
  general_solver += "{\n";
  general_solver += "    if (CheckValid()) {\n";
  general_solver += "        return Locality::GLOBALVALID;\n";
  general_solver += "    } else if (static_cast<SpecificCase*>(this)->CheckLocalValid(cons_info_->leqs, idx)) {\n";
  general_solver += "        return Locality::LOCALVALID;\n";
  general_solver += "    } else {\n";
  general_solver += "        UpdateDirection cur_direction = GetDescent(var_info_->cur_vars, idx, FuncInfo::LEQ);\n";
  general_solver += "        if (GetValue(update_direction) * GetValue(cur_direction) < 0) {\n";
  general_solver +=
      "            return (var_info_->cur_vars[idx] != var_info_->history_vars[idx]) ? Locality::CROSSREGION : "
      "Locality::REJECT;\n";
  general_solver += "        }\n";
  general_solver += "    return Locality::INVALID;\n";
  general_solver += "    }\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenGetCoarseLoc() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "函数名:GetCoarseLoc\n";
  strs += "功能描述:\n";
  strs += "  定域过程中的变量粗调,大致确定变量的落点信息\n";
  strs += "  该函数会沿不等式约束的下降方向进行二分搜索\n";
  strs += "  最终会输出一个位于约束边界/可行域边界的候选落点\n";
  strs += "输入参数:\n";
  strs += "  update_info:变量的更新信息,包括下标(idx),下降方向(update_direction)等指标\n";
  strs += "  step:变量的更新步长\n";
  strs += "  cur_locality:粗调过程中确定的定域优先级\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "template <typename SpecificCase>\n";
  general_solver +=
      "inline bool GeneralSolver<SpecificCase>::GetCoarseLoc(const UpdateInfo &update_info, uint64_t &step, Locality "
      "&cur_locality)\n";
  general_solver += "{\n";
  general_solver += "    uint64_t update_value;\n";
  general_solver += "\n";
  general_solver += "    int32_t idx = update_info.idx;\n";
  general_solver += "    if ((idx < 0) || (idx >= var_info_->var_num)) {\n";
  general_solver +=
      "        OP_LOGW(OP_NAME, \"idx = %d, var_info_->var_num = %d, idx illegal.\", idx, var_info_->var_num);\n";
  general_solver += "        return false;\n";
  general_solver += "    }\n";
  general_solver += "    uint64_t thres = update_info.thres;\n";
  general_solver += "    UpdateDirection update_direction = update_info.update_direction;\n";
  general_solver += "    do\n";
  general_solver += "    {\n";
  general_solver += "        step = (step == 0) ? 1 : (step << 1);\n";
  general_solver +=
      "        update_value = Bound(var_info_->upper_bound[idx], var_info_->lower_bound[idx], "
      "var_info_->rec_vars[idx], step, update_direction);\n";
  general_solver += "        UpdateCurVarVal(update_value, idx);\n";
  general_solver += "        cur_locality = GetLocality(idx, update_direction);\n";
  general_solver += "        var_info_->cur_vars[idx] = var_info_->rec_vars[idx];\n";
  general_solver += "        if (cur_locality <= Locality::CROSSREGION)\n";
  general_solver += "        {\n";
  general_solver +=
      "            step = ((cur_locality == Locality::CROSSREGION) && (step != 1)) ? (step >> 1) : step;\n";
  general_solver += "            break;\n";
  general_solver += "        }\n";
  general_solver += "    } while (step < thres);\n";
  general_solver +=
      "    update_value = Bound(var_info_->upper_bound[idx], var_info_->lower_bound[idx], var_info_->rec_vars[idx], "
      "step, update_direction);\n";
  general_solver += "    UpdateCurVarVal(update_value, idx);\n";
  general_solver += "    return thres != 0;\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenGetFineLoc() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "函数名:GetFineLoc\n";
  strs += "功能描述:\n";
  strs += "  定域过程中的变量精调,细致地确定变量的落点\n";
  strs += "  后验知识表明约束边界的解相对更好,因此尝试寻找位于边界的可行解\n";
  strs += "  该函数会在粗调所得的大致落点附近搜索,寻找不等式约束的边界点\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "template <typename SpecificCase>\n";
  general_solver +=
      "inline bool GeneralSolver<SpecificCase>::GetFineLoc(const UpdateInfo &update_info, uint64_t &step, Locality "
      "&cur_locality)\n";
  general_solver += "{\n";
  general_solver += "    uint64_t update_value;\n";
  general_solver += "    Locality rec_locality;\n";
  general_solver += "\n";
  general_solver += "    int32_t idx = update_info.idx;\n";
  general_solver += "    if ((idx < 0) || (idx >= var_info_->var_num)) {\n";
  general_solver +=
      "        OP_LOGW(OP_NAME, \"idx = %d, var_info_->var_num = %d, idx illegal.\", idx, var_info_->var_num);\n";
  general_solver += "        return false;\n";
  general_solver += "    }\n";
  general_solver += "    UpdateDirection update_direction = update_info.update_direction;\n";
  general_solver += "    if (GetLocality(idx, update_direction) <= Locality::LOCALVALID)\n";
  general_solver += "    {\n";
  general_solver += "        while (step > 1)\n";
  general_solver += "        {\n";
  general_solver += "            step >>= 1;\n";
  general_solver += "            update_value = var_info_->cur_vars[idx] - GetValue(update_direction) * step;\n";
  general_solver += "            UpdateCurVarVal(update_value, idx);\n";
  general_solver += "            rec_locality = GetLocality(idx, update_direction);\n";
  general_solver += "            if (rec_locality > Locality::CROSSREGION) {\n";
  general_solver += "                update_value = var_info_->cur_vars[idx] + GetValue(update_direction) * step;\n";
  general_solver += "                UpdateCurVarVal(update_value, idx);\n";
  general_solver += "            }\n";
  general_solver += "        }\n";
  general_solver += "        cur_locality = GetLocality(idx, update_direction);\n";
  general_solver += "    }\n";
  general_solver += "    return true;\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string PeerLocAnotation() {
  std::string strs = "";
  strs += "函数名:GetPeerLoc\n";
  strs += "功能描述:\n";
  strs += "  在定域过程中搜索某个解的对端解\n";
  strs += "  对端解:若当前解位于约束边界,则对端解位于可行域另一侧的约束边界\n";
  strs += "  当某个方向的可行解最优但曾被搜索过,该函数可以跨越可行域寻找另一个可行域边界上的解,跳出局部最优\n";
  return AddAnotationBlock(strs);
}

inline std::string GenBinPeerSearch() {
  std::string general_solver = "";
  general_solver += "        left_value = (update_direction == UpdateDirection::POSITIVE) ? (rec_value + 1) : 1;\n";
  general_solver +=
      "        right_value = (update_direction == UpdateDirection::POSITIVE) ? (var_info_->upper_bound[idx]) : "
      "(rec_value - var_info_->lower_bound[idx]);\n";
  general_solver += "        while (left_value < right_value)\n";
  general_solver += "        {\n";
  general_solver += "            mid_value = (left_value + right_value) >> 1;\n";
  general_solver += "            UpdateCurVarVal(mid_value, idx);\n";
  general_solver += "            rec_locality = GetLocality(idx, update_direction);\n";
  general_solver += "            if (rec_locality > Locality::LOCALVALID)\n";
  general_solver += "            {\n";
  general_solver += "                left_value = mid_value + 1;\n";
  general_solver += "            }\n";
  general_solver += "            else\n";
  general_solver += "            {\n";
  general_solver += "                right_value = mid_value;\n";
  general_solver += "            }\n";
  general_solver += "        }\n";
  general_solver += "        var_info_->cur_vars[idx] = left_value;\n";
  general_solver += "        cur_locality = Locality::ALTERNATIVE;\n";
  return general_solver;
}

inline std::string GenGetPeerLoc() {
  std::string general_solver = "";
  general_solver += PeerLocAnotation();
  general_solver += "template <typename SpecificCase>\n";
  general_solver +=
      "inline bool GeneralSolver<SpecificCase>::GetPeerLoc(const UpdateInfo &update_info, Locality &cur_locality)\n";
  general_solver += "{\n";
  general_solver += "    uint64_t left_value;\n";
  general_solver += "    uint64_t right_value;\n";
  general_solver += "    uint64_t mid_value;\n";
  general_solver += "    Locality rec_locality;\n";
  general_solver += "    int32_t idx = update_info.idx;\n";
  general_solver += "    if ((idx < 0) || (idx >= var_info_->var_num)) {\n";
  general_solver +=
      "        OP_LOGW(OP_NAME, \"idx = %d, var_info_->var_num = %d, idx illegal.\", idx, var_info_->var_num);\n";
  general_solver += "        return false;\n";
  general_solver += "    }\n";
  general_solver += "    uint64_t rec_value = var_info_->cur_vars[idx];\n";
  general_solver += "    UpdateDirection update_direction = update_info.update_direction;\n";
  general_solver +=
      "    UpdateCurVarVal((update_direction == UpdateDirection::NEGATIVE) ? var_info_->lower_bound[idx] : "
      "var_info_->upper_bound[idx], idx);\n";
  general_solver += "    rec_locality = GetLocality(idx, update_direction);\n";
  general_solver += "    if (rec_locality <= Locality::LOCALVALID)\n";
  general_solver += "    {\n";
  general_solver += "        var_info_->cur_vars[idx] = rec_value;\n";
  general_solver += "    }\n";
  general_solver += "    else\n";
  general_solver += "    {\n";
  general_solver += GenBinPeerSearch();
  general_solver += "    }\n";
  general_solver += "    return true;\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenUpdateLocalityMomentum() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "函数名:UpdateMomentum\n";
  strs += "功能描述:\n";
  strs += "  更新算法中的动量信息，以帮助算法更快地收敛到最优解\n";
  strs += "输入参数:\n";
  strs += "  idx:更新动量信息的变量索引。\n";
  strs += "  update_value:更新值。\n";
  strs += "  cur_locality:当前的LOCALITY信息\n";
  strs += "输出参数:\n";
  strs += "  best_locality:当前找到的最好的LOCALITY信息\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "template <typename SpecificCase>\n";
  general_solver +=
      "inline void GeneralSolver<SpecificCase>::UpdateMomentum(int32_t idx, double update_value, Locality "
      "cur_locality, Locality &best_locality)\n";
  general_solver += "{\n";
  general_solver +=
      "    if (cur_locality != Locality::GLOBALVALID || !visited_node_->SearchVars(var_info_->cur_vars, false))\n";
  general_solver += "    {\n";
  general_solver += "        if (cur_locality < best_locality) {\n";
  general_solver += "            ResetMomentum();\n";
  general_solver += "            best_locality = cur_locality;\n";
  general_solver += "        }\n";
  general_solver += "        if (cur_locality == best_locality) {\n";
  general_solver += "            var_info_->target_val[idx] = var_info_->cur_vars[idx];\n";
  general_solver += "            momentum_info_->is_valid[idx] = true;\n";
  general_solver += "            momentum_info_->cur_value[idx] = update_value;\n";
  general_solver += "        }\n";
  general_solver += "    }\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenGetBestChoice() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "函数名:GetBestChoice\n";
  strs += "功能描述:\n";
  strs += "  根据动量信息选择最佳变量进行更新\n";
  strs += "  使用idx遍历所有变量,检查动量信息是否有效,并计算动量值\n";
  strs += "  选取动量值最佳的变量作为输出\n";
  strs += "输出参数:\n";
  strs += "  bool类型参数,用于标记是否找到了最佳变量\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "template <typename SpecificCase>\n";
  general_solver += "inline bool GeneralSolver<SpecificCase>::GetBestChoice()\n";
  general_solver += "{\n";
  general_solver += "    bool better_choice;\n";
  general_solver += "    bool make_sense;\n";
  general_solver += "    double cur_value = 0.0;\n";
  general_solver += "    bool has_chosen = false;\n";
  general_solver += "    for (int32_t idx = 0; idx < var_info_->var_num; idx++)\n";
  general_solver += "    {\n";
  general_solver += "        if (momentum_info_->is_valid[idx])\n";
  general_solver += "        {\n";
  general_solver += "            momentum_info_->momentum[idx] *= solver_config_.momentum_factor;\n";
  general_solver +=
      "            momentum_info_->momentum[idx] += momentum_info_->cur_value[idx] * (1 - "
      "solver_config_.momentum_factor);\n";
  general_solver += "            better_choice = !has_chosen || momentum_info_->momentum[idx] > cur_value;\n";
  general_solver += "            make_sense = var_info_->cur_vars[idx] != var_info_->target_val[idx];\n";
  general_solver += "            if (better_choice && make_sense)\n";
  general_solver += "            {\n";
  general_solver += "                var_info_->chosen_var_idx = idx;\n";
  general_solver += "                has_chosen = true;\n";
  general_solver += "                cur_value = momentum_info_->momentum[idx];\n";
  general_solver += "            }\n";
  general_solver += "        }\n";
  general_solver += "    }\n";
  general_solver += "    return var_info_->chosen_var_idx != -1;\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenUpdateBestVar() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "函数名:UpdateBestVar\n";
  strs += "功能描述:\n";
  strs += "  根据chosen_var_idx的值对变量进行更新\n";
  strs += "  并调整momentum_info_中其他变量的动量信息\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "template <typename SpecificCase>\n";
  general_solver += "inline bool GeneralSolver<SpecificCase>::UpdateBestVar()\n";
  general_solver += "{\n";
  general_solver += "    for (int32_t idx = 0; idx < var_info_->var_num; idx++)\n";
  general_solver += "    {\n";
  general_solver += "        if (var_info_->chosen_var_idx == idx)\n";
  general_solver += "        {\n";
  general_solver += "            var_info_->cur_vars[idx] = var_info_->target_val[idx];\n";
  general_solver += "        }\n";
  general_solver += "        else\n";
  general_solver += "        {\n";
  general_solver += "            momentum_info_->momentum[idx] = 0;\n";
  general_solver += "        }\n";
  general_solver += "        momentum_info_->is_valid[idx] = false;\n";
  general_solver += "    }\n";
  general_solver += "    static_cast<SpecificCase*>(this)->UpdateLeqs(var_info_->cur_vars, -1, cons_info_->leqs);\n";
  general_solver += "    return true;\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenLocateLoc() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "函数名:LocateLoc\n";
  strs += "功能描述:\n";
  strs += "  在需要精调变量落点的情况下寻找变量的落点\n";
  strs += "  该函数会根据cur_locality和best_locality确定是否需要精调\n";
  strs += "  若需要,则会调用GetFineLoc函数进行精调,并根据精调结果判断是否要取对端解\n";
  strs += "  最后根据预期落点更新动量信息\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "template <typename SpecificCase>\n";
  general_solver +=
      "inline bool GeneralSolver<SpecificCase>::LocateLoc(const UpdateInfo &update_info, uint64_t &step, Locality "
      "&cur_locality, Locality &best_locality)\n";
  general_solver += "{\n";
  general_solver += "    int32_t idx = update_info.idx;\n";
  general_solver += "    double init_obj = update_info.init_obj;\n";
  general_solver += "    if (cur_locality <= best_locality)\n";
  general_solver += "    {\n";
  general_solver += "        GetFineLoc(update_info, step, cur_locality);\n";
  general_solver +=
      "        if (!solver_config_.simple_ver && visited_node_->SearchVars(var_info_->cur_vars, false))\n";
  general_solver += "        {\n";
  general_solver += "            GetPeerLoc(update_info, cur_locality);\n";
  general_solver += "        }\n";
  general_solver +=
      "        double update_value = init_obj - static_cast<SpecificCase*>(this)->GetSmoothObj(var_info_->cur_vars);\n";
  general_solver += "        UpdateMomentum(idx, update_value, cur_locality, best_locality);\n";
  general_solver += "        return true;\n";
  general_solver += "    }\n";
  general_solver += "    return false;\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenTryLocate() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "函数名:TryLocate\n";
  strs += "功能描述:\n";
  strs += "  尝试对特定变量进行定域操作\n";
  strs += "  若该更新该变量有希望走入可行域,则会使用GetCoarseLoc函数进行粗调\n";
  strs += "  根据粗调结果判断是否需要精调,若需要则调用LocateLoc函数进行精调\n";
  strs += "输入参数:\n";
  strs += "  idx:变量的索引\n";
  strs += "  init_idx:变量在当前位置的初始目标函数值\n";
  strs += "  best_locality:当前找到的最好的LOCALITY信息\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "template <typename SpecificCase>\n";
  general_solver +=
      "inline bool GeneralSolver<SpecificCase>::TryLocate(int32_t idx, double init_obj, Locality &best_locality)\n";
  general_solver += "{\n";
  general_solver += "    Locality cur_locality;\n";
  general_solver += "    uint64_t step = 0;\n";
  general_solver += "    UpdateDirection update_direction = GetDescent(var_info_->cur_vars, idx, FuncInfo::LEQ);\n";
  general_solver += "    if (update_direction != UpdateDirection::NONE)\n";
  general_solver += "    {\n";
  general_solver += "        uint64_t neg_thres = var_info_->cur_vars[idx] - var_info_->lower_bound[idx];\n";
  general_solver += "        uint64_t pos_thres = var_info_->upper_bound[idx] - var_info_->cur_vars[idx];\n";
  general_solver +=
      "        uint64_t thres = (update_direction == UpdateDirection::POSITIVE) ? pos_thres : neg_thres;\n";
  general_solver += "        UpdateInfo update_info = UpdateInfo(idx, thres, update_direction, init_obj);\n";
  general_solver += "        if (GetCoarseLoc(update_info, step, cur_locality))\n";
  general_solver += "        {\n";
  general_solver += "            if (!LocateLoc(update_info, step, cur_locality, best_locality))\n";
  general_solver += "            {\n";
  general_solver += "                UpdateCurVarVal(var_info_->rec_vars[idx], idx);\n";
  general_solver += "                return false;\n";
  general_solver += "            }\n";
  general_solver += "            UpdateCurVarVal(var_info_->rec_vars[idx], idx);\n";
  general_solver += "        }\n";
  general_solver += "    }\n";
  general_solver += "    return true;\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenLocateRegion() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "函数名:LocateRegion\n";
  strs += "功能描述:\n";
  strs += "  定域操作,用于实现可行域外的变量更新\n";
  strs += "  当变量位于可行域外时,由不等式约束驱动变量进行调整\n";
  strs += "  使用TryLocate函数确定变量的落点信息\n";
  strs += "  优先检测update_last为false的变量,在不存在可行的定域解时检测update_last为true的变量\n";
  strs += "  寻找目标函数更优的落点\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "template <typename SpecificCase>\n";
  general_solver += "inline bool GeneralSolver<SpecificCase>::LocateRegion()\n";
  general_solver += "{\n";
  general_solver += "    OP_LOGD(OP_NAME, \"Infeasible solution, start locating feasible region.\");\n";
  general_solver += "    Locality best_locality = Locality::REJECT;\n";
  general_solver += "    double init_obj = static_cast<SpecificCase*>(this)->GetSmoothObj(var_info_->cur_vars);\n";
  general_solver += "    for (int32_t idx = 0; idx < var_info_->var_num; idx++)\n";
  general_solver += "    {\n";
  general_solver += "        if (!var_info_->update_last[idx])\n";
  general_solver += "        {\n";
  general_solver += "            TryLocate(idx, init_obj, best_locality);\n";
  general_solver += "        }\n";
  general_solver += "    }\n";
  general_solver += "    if (has_feasible_ || best_locality == Locality::REJECT)\n";
  general_solver += "    {\n";
  general_solver += "        for (int32_t idx = 0; idx < var_info_->var_num; idx++)\n";
  general_solver += "        {\n";
  general_solver += "            if (var_info_->update_last[idx])\n";
  general_solver += "            {\n";
  general_solver += "                TryLocate(idx, init_obj, best_locality);\n";
  general_solver += "            }\n";
  general_solver += "        }\n";
  general_solver += "    }\n";
  general_solver += "    if (best_locality == Locality::REJECT || !GetBestChoice())\n";
  general_solver += "    {\n";
  general_solver +=
      "        OP_LOGW(OP_NAME, \"There is no nonredundant variables that can approximate the feasible region.\");\n";
  general_solver += "        return false;\n";
  general_solver += "    }\n";
  general_solver += "    UpdateBestVar();\n";
  general_solver += "    OP_LOGD(OP_NAME, \"Located feasible region successfully.\");\n";
  general_solver += "    return true;\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GetLocateRegion() {
  std::string general_solver = "";
  general_solver += GenGetLocality();
  general_solver += GenGetCoarseLoc();
  general_solver += GenGetFineLoc();
  general_solver += GenGetPeerLoc();
  general_solver += GenUpdateLocalityMomentum();
  general_solver += GenGetBestChoice();
  general_solver += GenUpdateBestVar();
  general_solver += GenLocateLoc();
  general_solver += GenTryLocate();
  general_solver += GenLocateRegion();
  return general_solver;
}

inline std::string GenGetTunePriority() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "函数名:GetTunePriority\n";
  strs += "功能描述:\n";
  strs += "  确定微调过程中某个待求解变量的优先级\n";
  strs += "输入参数:\n";
  strs += "  idx:待求解变量的下标\n";
  strs += "  rec_obj:本轮迭代前的初始目标函数值\n";
  strs += "输出参数:\n";
  strs += "  cur_obj:微调后变量的目标函数值\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "template <typename SpecificCase>\n";
  general_solver +=
      "inline TunePriority GeneralSolver<SpecificCase>::GetTunePriority(int32_t idx, double rec_obj, double "
      "&cur_obj)\n";
  general_solver += "{\n";
  general_solver += "    cur_obj = static_cast<SpecificCase*>(this)->GetSmoothObj(var_info_->cur_vars);\n";
  general_solver += "    int64_t last_update = var_info_->rec_vars[idx] - var_info_->history_vars[idx];\n";
  general_solver += "    int64_t next_update = var_info_->cur_vars[idx] - var_info_->rec_vars[idx];\n";
  general_solver += "    if (last_update * next_update < 0)\n";
  general_solver += "    {\n";
  general_solver += "        return TunePriority::TABU;\n";
  general_solver += "    }\n";
  general_solver += "    else if (cur_obj <= rec_obj)\n";
  general_solver += "    {\n";
  general_solver += "        if (static_cast<SpecificCase*>(this)->CheckLocalValid(cons_info_->leqs, idx))\n";
  general_solver += "        {\n";
  general_solver += "            return (cur_obj < rec_obj) ? TunePriority::HARMLESS : TunePriority::DILATED;\n";
  general_solver += "        }\n";
  general_solver += "        else\n";
  general_solver += "        {\n";
  general_solver +=
      "            return (cur_obj < rec_obj) ? TunePriority::NORMAL : (solver_config_.simple_ver ? "
      "TunePriority::REFUSE : TunePriority::OTHER);\n";
  general_solver += "        }\n";
  general_solver += "    }\n";
  general_solver += "    return solver_config_.simple_ver ? TunePriority::REFUSE : TunePriority::OTHER;\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenSearchLoc() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "函数名:SearchLoc\n";
  strs += "功能描述:\n";
  strs += "  沿着指定的更新方向进行探索,检查是否有机会取到更优的可行解\n";
  strs += "  该函数会探索至多solver_config_.search_length步,若存在更优的可行解则会进行标记\n";
  strs += "输入参数:\n";
  strs += "  update_info:变量的更新信息\n";
  strs += "输出参数:\n";
  strs += "  step:取得更优可行解时的步长\n";
  strs += "  cur_obj:微调后变量的目标函数值\n";
  strs += "  cur_priority:微调后变量的优先级\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "template <typename SpecificCase>\n";
  general_solver +=
      "inline bool GeneralSolver<SpecificCase>::SearchLoc(const UpdateInfo &update_info, uint64_t &step, double "
      "&cur_obj, TunePriority &cur_priority)\n";
  general_solver += "{\n";
  general_solver += "    TunePriority rec_priority{TunePriority::REFUSE};\n";
  general_solver += "    int32_t idx = update_info.idx;\n";
  general_solver += "    if ((idx < 0) || (idx >= var_info_->var_num)) {\n";
  general_solver +=
      "        OP_LOGW(OP_NAME, \"idx = %d, var_info_->var_num = %d, idx illegal.\", idx, var_info_->var_num);\n";
  general_solver += "        return false;\n";
  general_solver += "    }\n";
  general_solver += "    uint64_t thres = update_info.thres;\n";
  general_solver += "    UpdateDirection update_direction = update_info.update_direction;\n";
  general_solver += "    double init_obj = update_info.init_obj;\n";
  general_solver += "    while (step < Min(thres, solver_config_.search_length))\n";
  general_solver += "    {\n";
  general_solver += "        step++;\n";
  general_solver += "        UpdateCurVarVal(var_info_->rec_vars[idx] + GetValue(update_direction) * step, idx);\n";
  general_solver += "        rec_priority = GetTunePriority(idx, init_obj, cur_obj);\n";
  general_solver += "        if (rec_priority <= cur_priority)\n";
  general_solver += "        {\n";
  general_solver += "            cur_priority = rec_priority;\n";
  general_solver += "            break;\n";
  general_solver += "        }\n";
  general_solver += "    }\n";
  general_solver += "    UpdateCurVarVal(var_info_->rec_vars[idx], idx);\n";
  general_solver += "    return rec_priority == cur_priority;\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenGetHarmlessLoc() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "函数名:GetHarmlessLoc\n";
  strs += "功能描述:\n";
  strs += "  当且仅当存在一个目标函数更优的可行解时称求解器能找到无损的局部最优解\n";
  strs += "  该函数尝试在搜索范围内检查所有的可行解,寻找最优的无损局部最优解\n";
  strs += "输入参数:\n";
  strs += "  update_info:变量的更新信息\n";
  strs += "输出参数:\n";
  strs += "  step:取得更优可行解时的步长\n";
  strs += "  cur_obj:微调后无损局部最优解的目标函数值\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "template <typename SpecificCase>\n";
  general_solver +=
      "inline bool GeneralSolver<SpecificCase>::GetHarmlessLoc(const UpdateInfo &update_info, uint64_t &step, double "
      "&cur_obj)\n";
  general_solver += "{\n";
  general_solver += "    double rec_obj;\n";
  general_solver += "    int32_t update_value;\n";
  general_solver += "    TunePriority rec_priority;\n";
  general_solver += "    int32_t idx = update_info.idx;\n";
  general_solver += "    if ((idx < 0) || (idx >= var_info_->var_num)) {\n";
  general_solver +=
      "        OP_LOGW(OP_NAME, \"idx = %d, var_info_->var_num = %d, idx illegal.\", idx, var_info_->var_num);\n";
  general_solver += "        return false;\n";
  general_solver += "    }\n";
  general_solver += "    uint64_t thres = update_info.thres;\n";
  general_solver += "    UpdateDirection update_direction = update_info.update_direction;\n";
  general_solver += "    var_info_->cur_vars[idx] = var_info_->rec_vars[idx];\n";
  general_solver += "    while (step < thres)\n";
  general_solver += "    {\n";
  general_solver += "        step = solver_config_.simple_ver ? (step == 0 ? 1 : (step << 1)) : (step + 1);\n";
  general_solver +=
      "        update_value = Bound(var_info_->upper_bound[idx], var_info_->lower_bound[idx], "
      "var_info_->rec_vars[idx], step, update_direction);\n";
  general_solver += "        UpdateCurVarVal(update_value, idx);\n";
  general_solver += "        rec_priority = GetTunePriority(idx, cur_obj, rec_obj);\n";
  general_solver += "        if (rec_priority != TunePriority::HARMLESS)\n";
  general_solver += "        {\n";
  general_solver += "            step = solver_config_.simple_ver ? (step >> 1) : (step - 1);\n";
  general_solver += "            break;\n";
  general_solver += "        }\n";
  general_solver += "        cur_obj = rec_obj;\n";
  general_solver += "    }\n";
  general_solver += "    return true;\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenGetDilatedLoc() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "函数名:GetDilatedLoc\n";
  strs += "功能描述:\n";
  strs += "  当且仅当存在一个目标函数不变但更接近可行域边界的可行解时称求解器能找到膨胀局部最优解\n";
  strs += "  该函数沿着缓存占用边界更新变量,寻找更新方向上最接近可行域边界的膨胀局部最优解\n";
  strs += "输入参数:\n";
  strs += "  update_info:变量的更新信息\n";
  strs += "输出参数:\n";
  strs += "  step:取得更优可行解时的步长\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "template <typename SpecificCase>\n";
  general_solver +=
      "inline bool GeneralSolver<SpecificCase>::GetDilatedLoc(const UpdateInfo &update_info, uint64_t &step)\n";
  general_solver += "{\n";
  general_solver += "    int32_t idx = update_info.idx;\n";
  general_solver += "    if ((idx < 0) || (idx >= var_info_->var_num)) {\n";
  general_solver +=
      "        OP_LOGW(OP_NAME, \"idx = %d, var_info_->var_num = %d, idx illegal.\", idx, var_info_->var_num);\n";
  general_solver += "        return false;\n";
  general_solver += "    }\n";
  general_solver += "    uint64_t update_value;\n";
  general_solver += "    uint64_t thres = update_info.thres;\n";
  general_solver += "    UpdateDirection update_direction = update_info.update_direction;\n";
  general_solver += "    double cur_obj;\n";
  general_solver += "    double cur_cons;\n";
  general_solver += "    double init_obj = update_info.init_obj;\n";
  general_solver += "    double init_cons = update_info.init_cons;\n";
  general_solver += "    double pre_cons = init_cons;\n";
  general_solver += "    while (step < thres)\n";
  general_solver += "    {\n";
  general_solver += "        step = solver_config_.simple_ver ? (step == 0 ? 1 : (step << 1)) : (step + 1);\n";
  general_solver +=
      "        update_value = Bound(var_info_->upper_bound[idx], var_info_->lower_bound[idx], "
      "var_info_->rec_vars[idx], step, update_direction);\n";
  general_solver += "        UpdateCurVarVal(update_value, idx);\n";
  general_solver += "        cur_obj = static_cast<SpecificCase*>(this)->GetSmoothObj(var_info_->cur_vars);\n";
  general_solver += "        cur_cons = static_cast<SpecificCase*>(this)->GetBuffCost(var_info_->cur_vars);\n";
  general_solver +=
      "        if (!static_cast<SpecificCase*>(this)->CheckLocalValid(cons_info_->leqs, idx) || (!IsEqual(init_obj, "
      "cur_obj)) || (cur_cons > "
      "pre_cons))\n";
  general_solver += "        {\n";
  general_solver += "            step = solver_config_.simple_ver ? (step >> 1) : (step - 1);\n";
  general_solver += "            break;\n";
  general_solver += "        }\n";
  general_solver += "        pre_cons = cur_cons;\n";
  general_solver += "    }\n";
  general_solver += "    return true;\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenUpdatePriorityMomentum() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "函数名:UpdateMomentum\n";
  strs += "功能描述:\n";
  strs += "  是前一个UpdateMomentum的重载\n";
  strs += "  前一个UpdateMomentum函数用于更新定域过程中的动量信息\n";
  strs += "  本函数用于更新微调过程中的动量信息\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "template <typename SpecificCase>\n";
  general_solver +=
      "inline void GeneralSolver<SpecificCase>::UpdateMomentum(int32_t idx, double update_value, TunePriority "
      "cur_priority, TunePriority &best_priority)\n";
  general_solver += "{\n";
  general_solver += "    if (!visited_node_->SearchVars(var_info_->cur_vars, false))\n";
  general_solver += "    {\n";
  general_solver += "        if (cur_priority < best_priority)\n";
  general_solver += "        {\n";
  general_solver += "            ResetMomentum();\n";
  general_solver += "            best_priority = cur_priority;\n";
  general_solver += "        }\n";
  general_solver += "        if (cur_priority == best_priority)\n";
  general_solver += "        {\n";
  general_solver +=
      "            if (update_value > momentum_info_->cur_value[idx] || !momentum_info_->is_valid[idx])\n";
  general_solver += "            {\n";
  general_solver += "                var_info_->target_val[idx] = var_info_->cur_vars[idx];\n";
  general_solver += "                momentum_info_->is_valid[idx] = true;\n";
  general_solver += "                momentum_info_->cur_value[idx] = update_value;\n";
  general_solver += "            }\n";
  general_solver += "        }\n";
  general_solver += "    }\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenTuneLocAnnotation() {
  std::string strs = "";
  strs += "函数名:TuneLoc\n";
  strs += "功能描述:\n";
  strs += "  根据变量的更新信息对某个变量进行进一步的微调\n";
  strs += "  根据输入的微调优先级cur_priority选取微调策略对变量进行更新\n";
  strs += "  若优先级为HARMLESS,则会调用GetHarmlessLoc函数进行无损更新\n";
  strs += "  若优先级为DILATED,则会调用GetDilatedLoc函数进行膨胀更新\n";
  return strs;
}

inline std::string GenTuneLoc() {
  std::string general_solver = "";
  general_solver += AddAnotationBlock(GenTuneLocAnnotation());
  general_solver += "template <typename SpecificCase>\n";
  general_solver +=
      "inline bool GeneralSolver<SpecificCase>::TuneLoc(const UpdateInfo &update_info, double cur_obj, uint64_t &step, "
      "TunePriority &cur_priority, "
      "TunePriority &best_priority)\n";
  general_solver += "{\n";
  general_solver += "    if (cur_priority <= best_priority)\n";
  general_solver += "    {\n";
  general_solver += "        uint64_t update_value;\n";
  general_solver += "        int32_t idx = update_info.idx;\n";
  general_solver += "        if ((idx < 0) || (idx >= var_info_->var_num)) {\n";
  general_solver +=
      "            OP_LOGW(OP_NAME, \"idx = %d, var_info_->var_num = %d, idx illegal.\", idx, var_info_->var_num);\n";
  general_solver += "            return false;\n";
  general_solver += "        }\n";
  general_solver += "        UpdateDirection update_direction = update_info.update_direction;\n";
  general_solver += "        double init_obj = update_info.init_obj;\n";
  general_solver += "        if (cur_priority == TunePriority::HARMLESS)\n";
  general_solver += "        {\n";
  general_solver += "            GetHarmlessLoc(update_info, step, cur_obj);\n";
  general_solver += "        }\n";
  general_solver += "        else if (cur_priority == TunePriority::DILATED)\n";
  general_solver += "        {\n";
  general_solver +=
      "            UpdateDirection cur_direction = GetDescent(var_info_->cur_vars, idx, FuncInfo::BUFFER);\n";
  general_solver += "            if (GetValue(cur_direction) * GetValue(update_direction) >= 0)\n";
  general_solver += "            {\n";
  general_solver += "                GetDilatedLoc(update_info, step);\n";
  general_solver += "            }\n";
  general_solver += "            else\n";
  general_solver += "            {\n";
  general_solver +=
      "                cur_priority = solver_config_.simple_ver ? TunePriority::REFUSE : TunePriority::OTHER;\n";
  general_solver += "            }\n";
  general_solver += "        }\n";
  general_solver +=
      "        update_value = Bound(var_info_->upper_bound[idx], var_info_->lower_bound[idx], "
      "var_info_->rec_vars[idx], step, update_direction);\n";
  general_solver += "        UpdateCurVarVal(update_value, idx);\n";
  general_solver += "        UpdateMomentum(idx, (init_obj - cur_obj), cur_priority, best_priority);\n";
  general_solver += "        return true;\n";
  general_solver += "    }\n";
  general_solver += "    return false;\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenTryTune() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "函数名:TryTune\n";
  strs += "功能描述:\n";
  strs += "  对某个变量进行微调\n";
  strs += "  首先利用SearchLoc函数在领域内判断是否存在更优的可行解\n";
  strs += "  然后根据微调优先级cur_priority选取微调策略对变量进行更新\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "template <typename SpecificCase>\n";
  general_solver +=
      "inline bool GeneralSolver<SpecificCase>::TryTune(int32_t idx, UpdateDirection update_direction, double "
      "init_obj, double init_cons, "
      "TunePriority &best_priority)\n";
  general_solver += "{\n";
  general_solver += "    uint64_t step = 0;\n";
  general_solver += "    uint64_t pos_thres = var_info_->upper_bound[idx] - var_info_->cur_vars[idx];\n";
  general_solver += "    uint64_t neg_thres = var_info_->cur_vars[idx] - var_info_->lower_bound[idx];\n";
  general_solver += "    uint64_t thres = (update_direction == UpdateDirection::POSITIVE) ? pos_thres : neg_thres;\n";
  general_solver += "    double cur_obj = 0;\n";
  general_solver += "    TunePriority cur_priority = (thres > 0) ? best_priority : TunePriority::REFUSE;\n";
  general_solver += "    if (thres > 0)\n";
  general_solver += "    {\n";
  general_solver +=
      "        UpdateInfo update_info = UpdateInfo(idx, thres, update_direction, init_obj, "
      "init_cons);\n";
  general_solver += "        if (SearchLoc(update_info, step, cur_obj, cur_priority))\n";
  general_solver += "        {\n";
  general_solver += "            if (!TuneLoc(update_info, cur_obj, step, cur_priority, best_priority))\n";
  general_solver += "            {\n";
  general_solver += "                return false;\n";
  general_solver += "            }\n";
  general_solver += "            UpdateCurVarVal(var_info_->rec_vars[idx], idx);\n";
  general_solver += "        }\n";
  general_solver += "    }\n";
  general_solver += "    return cur_priority < TunePriority::NORMAL;\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenFineTune() {
  std::string general_solver = "";
  std::string strs = "";
  strs += "函数名:FineTune\n";
  strs += "功能描述:\n";
  strs += "  实现待求解变量的微调操作\n";
  strs += "  首先沿正方向对变量进行更新,若更新方向上存在更优的可行解则进行微调\n";
  strs += "  若正方向上不存在更优的可行解或采用高性能版本进行求解,则尝试沿负方向进行更新\n";
  general_solver += AddAnotationBlock(strs);
  general_solver += "template <typename SpecificCase>\n";
  general_solver += "inline bool GeneralSolver<SpecificCase>::FineTune()\n";
  general_solver += "{\n";
  general_solver += "    OP_LOGD(OP_NAME, \"Feasible solution, start tuning the tiling data.\");\n";
  general_solver += "    double init_obj = static_cast<SpecificCase*>(this)->GetSmoothObj(var_info_->cur_vars);\n";
  general_solver += "    double init_cons = static_cast<SpecificCase*>(this)->GetBuffCost(var_info_->cur_vars);\n";
  general_solver += "    if (!RecordBestVarVal())\n";
  general_solver += "    {\n";
  general_solver += "        OP_LOGW(OP_NAME, \"Failed to add a solution to the result.\");\n";
  general_solver += "        return false;\n";
  general_solver += "    }\n";
  general_solver += "    TunePriority best_priority = TunePriority::TABU;\n";
  general_solver += "    for (int32_t idx = 0; idx < var_info_->var_num; idx++)\n";
  general_solver += "    {\n";
  general_solver +=
      "        if (!TryTune(idx, UpdateDirection::POSITIVE, init_obj, init_cons, best_priority) || "
      "!solver_config_.simple_ver)\n";
  general_solver += "        {\n";
  general_solver += "            TryTune(idx, UpdateDirection::NEGATIVE, init_obj, init_cons, best_priority);\n";
  general_solver += "        }\n";
  general_solver += "    }\n";
  general_solver += "    if (!GetBestChoice())\n";
  general_solver += "    {\n";
  general_solver += "        OP_LOGW(OP_NAME, \"Unable to find a valuable update.\");\n";
  general_solver += "        return false;\n";
  general_solver += "    }\n";
  general_solver += "    UpdateBestVar();\n";
  general_solver += "    OP_LOGD(OP_NAME, \"Tuned the tiling data successfully.\");\n";
  general_solver += "    return true;\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GetFineTune() {
  std::string general_solver = "";
  general_solver += GenGetTunePriority();
  general_solver += GenSearchLoc();
  general_solver += GenGetHarmlessLoc();
  general_solver += GenGetDilatedLoc();
  general_solver += GenUpdatePriorityMomentum();
  general_solver += GenTuneLoc();
  general_solver += GenTryTune();
  general_solver += GenFineTune();
  return general_solver;
}

inline std::string GenRecordBestVarVal() {
  std::string general_solver = "";
  general_solver += "template <typename SpecificCase>\n";
  general_solver += "inline bool GeneralSolver<SpecificCase>::RecordBestVarVal()\n";
  general_solver += "{\n";
  general_solver += "    if (is_feasible_)\n";
  general_solver += "    {\n";
  general_solver += "        double obj = static_cast<SpecificCase*>(this)->GetObj(var_info_->cur_vars);\n";
  general_solver += "        double cons = static_cast<SpecificCase*>(this)->GetBuffCost(var_info_->cur_vars);\n";
  general_solver += "        return result_->AddVarVal(var_info_->cur_vars, obj, cons);\n";
  general_solver += "    }\n";
  general_solver += "    return false;\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string RunAnotation() {
  std::string strs = "";
  strs += "函数名:Run\n";
  strs += "功能描述:\n";
  strs += "  通用求解器求解函数\n";
  strs += "  算法会迭代solver_config_.iterations次\n";
  strs += "  在每轮迭代中根据当前的变量值选取定域或微调策略对变量进行更新\n";
  strs += "输出参数:\n";
  strs += "  solution_num:uint32_t类型的参数,用来输出实际得到的解的个数\n";
  strs += "  solutions:uint64_t类型的数组,指向一块num_var * top_num的内存,求解算法获取到的可行解放入该空间\n";
  return AddAnotationBlock(strs);
}

inline std::string GenDTStrategy() {
  std::string str = "";
  str += "              RecordBestVarVal();\n";
  str += "              break;\n";
  return str;
}

inline std::string GenRun(bool open_dt = false) {
  std::string general_solver;
  general_solver += RunAnotation();
  general_solver += "template <typename SpecificCase>\n";
  general_solver += "inline bool GeneralSolver<SpecificCase>::Run(int32_t &solution_num, uint64_t *solutions)\n";
  general_solver += "{\n";
  general_solver += "    uint64_t iter = 1;\n";
  general_solver += "    has_feasible_ = false;\n";
  general_solver += "    while (iter <= solver_config_.iterations)\n";
  general_solver += "    {\n";
  general_solver += "        Initialize(iter);\n";
  general_solver += "        OP_LOGD(OP_NAME, \"iter : %lu\", iter);\n";
  general_solver += "        static_cast<SpecificCase*>(this)->DisplayVarVal(var_info_->cur_vars);\n";
  general_solver += "        if (!is_feasible_)\n";
  general_solver += "        {\n";
  general_solver += "            if (!LocateRegion())\n";
  general_solver += "            {\n";
  general_solver +=
      "                OP_LOGW(OP_NAME, \"The locating process cannot find more valuable updates, triggering an early "
      "stop.\");\n";
  general_solver += "                break;\n";
  general_solver += "            }\n";
  general_solver += "        }\n";
  general_solver += "        else\n";
  general_solver += "        {\n";
  if (open_dt) {
    general_solver += GenDTStrategy();
  } else {
    general_solver += "            if (visited_node_->SearchVars(var_info_->cur_vars, true))\n";
    general_solver += "            {\n";
    general_solver +=
        "                OP_LOGW(OP_NAME, \"Searched a feasible solution again, triggering an early stop.\");\n";
    general_solver += "                break;\n";
    general_solver += "            }\n";
    general_solver += "            if (!FineTune())\n";
    general_solver += "            {\n";
    general_solver += "                break;\n";
    general_solver += "            }\n";
  }
  general_solver += "        }\n";
  general_solver += "        iter++;\n";
  general_solver += "    }\n";
  general_solver += "    result_->GetResult(solution_num, solutions);\n";
  general_solver += "    return solution_num > 0;\n";
  general_solver += "}\n";
  general_solver += "\n";
  return general_solver;
}

inline std::string GenGetVarNum() {
  std::string general_solver = "";
  general_solver += "template <typename SpecificCase>\n";
  general_solver += "inline int32_t GeneralSolver<SpecificCase>::GetVarNum() const\n";
  general_solver += "{\n";
  general_solver += "    return var_info_->var_num;\n";
  general_solver += "}\n";
  return general_solver;
}

inline std::string GetGeneralSolver(bool open_dt = false) {
  std::string general_solver = "";
  general_solver += GenConstVars();
  general_solver += GenDef();
  general_solver += GenToolFunc();
  general_solver += GenResultFunc();
  general_solver += GenGetFuncVal();
  general_solver += GenGetDescent();
  general_solver += GenInit();
  general_solver += GenUpdateCurVarVal();
  general_solver += GenCmp();
  general_solver += GenSearchVars();
  general_solver += GenCheckValid();
  general_solver += GenResetMomentum();
  general_solver += GenInitialize();
  general_solver += GetLocateRegion();
  general_solver += GetFineTune();
  general_solver += GenRecordBestVarVal();
  general_solver += GenRun(open_dt);
  general_solver += GenGetVarNum();
  return general_solver;
}

inline const std::string GENERAL_SOLVER_CODE = GetGeneralSolver(false);
inline const std::string GENERAL_SOLVER_CODE_DT = GetGeneralSolver(true);
}  // namespace att
#endif
