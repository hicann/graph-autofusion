/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "solver.h"
#define OP_LOG(log)
#define MAX_SOLUTION 5

void VarVal::GetVarInfo(double &obj, double &cons) const {
  obj = obj_;
  cons = cons_;
}

void VarVal::GetVars(uint64_t *vars) {
  for (int32_t i = 0; i < var_num_; i++) {
    vars[i] = vars_[i];
  }
}

/*
函数名:GetVarVal
功能描述:在VisitedNode中检查vars是否曾被搜索,若未被搜索则会在VisitedNode中构建vars对象
输入参数:
  vars:待求解变量所对应的一组解
*/
Node *VisitedNode::GetVarVal(uint64_t *vars) {
  Node *new_node;
  Node *cur_node = head;
  for (uint32_t i = 0; i < depth; i++) {
    if (!cur_node->next_var) {
      new_node = new (std::nothrow) Node(vars[i]);
      if (new_node == nullptr) {
        OP_LOG("Create new_node failed!");
        return nullptr;
      }
      if (new_node != nullptr) {
        cur_node->next_var = new_node;
        tail->next_node = new_node;
        tail = tail->next_node;
      }
    }
    cur_node = cur_node->next_var;
    while (cur_node->next_val != nullptr) {
      if (cur_node->value == vars[i]) {
        break;
      }
      cur_node = cur_node->next_val;
    }
    if (cur_node->value != vars[i]) {
      new_node = new (std::nothrow) Node(vars[i]);
      if (new_node == nullptr) {
        OP_LOG("Create new_node failed!");
        return nullptr;
      }
      if (new_node != nullptr) {
        cur_node->next_val = new_node;
        tail->next_node = new_node;
        tail = tail->next_node;
        cur_node = new_node;
      }
    }
  }
  return cur_node;
}

/*
函数名:AddVarVal
功能描述:将一组可行解vars传入Result
  若这组可行解的质量较差(目标函数值较大或距离约束边界较远),则舍弃
  若这组可行解可以被排进前top_n_,则保留该组可行解
  temp: 最大容量为top_n的备选可行解集
  先将solution_复制到temp中
  然后比较new_vars的目标值与temp中元素的目标值
  自小到大地将可行解填入solution_
输入参数:
  vars:一组可行解
  obj:该可行解所对应的目标函数值
  cons:可行解距约束边界的距离
*/
bool Result::InsertSorted(VarVal **temp, uint64_t rec_num, VarVal *new_vars, double obj, double cons) {
  uint32_t cnt_num = 0;
  uint32_t temp_idx = 0;
  bool has_add = false;
  while ((cnt_num < solution_num_) && (temp_idx < rec_num)) {
    double cur_obj, cur_cons;
    temp[temp_idx]->GetVarInfo(cur_obj, cur_cons);
    if (!has_add && (obj < cur_obj || (IsEqual(obj, cur_obj) && cons < cur_cons))) {
      has_add = true;
      solution_[cnt_num++] = new_vars;
    } else {
      solution_[cnt_num++] = temp[temp_idx++];
    }
  }
  if ((!has_add) && (cnt_num < solution_num_)) {
    solution_[cnt_num++] = new_vars;
    has_add = true;
  }
  if (!has_add) {
    delete new_vars;
  } else if (rec_num == solution_num_) {
    delete temp[temp_idx];
  }
  return has_add;
}

bool Result::AddVarVal(uint64_t *vars, double obj, double cons) {
  uint64_t rec_num = solution_num_;
  if (rec_num > MAX_SOLUTION) {
    OP_LOG("Too many solutions!");
    return false;
  }
  solution_num_ = SMIN(solution_num_ + 1, top_n_);
  VarVal *new_vars = new (std::nothrow) VarVal(var_num_, obj, cons, vars);
  if (new_vars == nullptr) {
    OP_LOG("Create new_vars failed!");
    return false;
  }
  if (rec_num == 0) {
    solution_[0] = new_vars;
    return true;
  }
  VarVal **temp = new (std::nothrow) VarVal *[rec_num];
  if (temp == nullptr) {
    OP_LOG("Create temp failed!");
    return false;
  }
  for (uint64_t i = 0; i < rec_num; i++) {
    temp[i] = solution_[i];
  }
  InsertSorted(temp, rec_num, new_vars, obj, cons);
  for (uint32_t i = 0; i < rec_num; i++) {
    temp[i] = nullptr;
  }
  delete[] temp;
  return solution_num_ > 0;
}

bool Result::GetResult(int32_t &solution_num, uint64_t *solution) {
  for (uint32_t i = 0; i < solution_num_; i++) {
    solution_[i]->GetVars(solution + i * var_num_);
  }
  solution_num = solution_num_;
  return true;
}

double GeneralSolver::GetFuncVal(uint64_t *vars, double *weight, FuncInfo func_info) {
  if (func_info == FuncInfo::BUFFER) {
    return GetBuffDiff(vars, weight);
  } else if (func_info == FuncInfo::LEQ) {
    return GetLeqDiff(vars, weight);
  }
  return 0;
}

/*
函数名:GetDescent
功能描述:获取“缓存占用函数/不等式约束的罚函数”的下降方向
输入参数:
  vars:当前待求解参数的下降方向
  idx:关于某参数下降方向中,某参数的下标
  func_info:用于指明计算下降方向的函数(FuncInfo::BUFFER/FuncInfo::LEQ)
*/
UpdateDirection GeneralSolver::GetDescent(uint64_t *vars, int32_t idx, FuncInfo func_info) {
  if ((idx < 0) || (idx >= var_info_->var_num)) {
    OP_LOG("idx illegal!");
    return UpdateDirection::NONE;
  }
  double weight[cons_info_->leq_num];
  UpdateLeqs(vars, -1, weight);
  double cur_val = GetFuncVal(vars, weight, func_info);
  if (vars[idx] + 1 <= var_info_->upper_bound[idx]) {
    vars[idx] += 1;
    double next_val = GetFuncVal(vars, weight, func_info);
    vars[idx] -= 1;
    if (!IsEqual(cur_val, next_val)) {
      return (cur_val > next_val) ? UpdateDirection::POSITIVE : UpdateDirection::NEGATIVE;
    }
  }
  if (vars[idx] >= var_info_->lower_bound[idx] + 1) {
    vars[idx] -= 1;
    double pre_val = GetFuncVal(vars, weight, func_info);
    vars[idx] += 1;
    if (!IsEqual(cur_val, pre_val)) {
      return (pre_val > cur_val) ? UpdateDirection::POSITIVE : UpdateDirection::NEGATIVE;
    }
  }
  return UpdateDirection::NONE;
}

bool GeneralSolver::SetSolverInput(const SolverInput &input) {
  if (input.var_num <= 0) {
    return false;
  }
  visited_node_ = new (std::nothrow) VisitedNode(input.var_num);
  if (visited_node_ == nullptr) {
    OP_LOG("Create visited_node_ failed!");
    return false;
  }
  var_info_ = new (std::nothrow) VarInfo(input);
  cons_info_ = new (std::nothrow) ConsInfo(input.leq_num);
  momentum_info_ = new (std::nothrow) Momentum(input.var_num);
  if (var_info_ != nullptr && cons_info_ != nullptr && momentum_info_ != nullptr) {
    for (int32_t i = 0; i < var_info_->var_num; i++) {
      var_info_->update_last[i] = input.update_last[i];
    }
    return true;
  }
  return false;
}

/*
函数名:Init
功能描述:初始化通用求解器,导入待求解变量的先验信息,分配求解器所需的空间
*/
bool GeneralSolver::Init(const SolverInput &input) {
  if (!SetSolverInput(input)) {
    return false;
  }
  result_ = new (std::nothrow) Result(solver_config_.top_num, input.var_num);
  if (result_ == nullptr) {
    OP_LOG("Create result_ failed!");
    return false;
  }
  return true;
}

/*
函数名:UpdateCurVarVal
功能描述:更新cur_var中某个待求解变量的值,并同步更新不等式约束的值
输入参数:
  value:待求解变量被更新成为的值
  idx:更新的待求解变量的下标
*/
bool GeneralSolver::UpdateCurVarVal(uint64_t value, int32_t idx) {
  if (idx < 0 || idx >= var_info_->var_num) {
    return false;
  }
  var_info_->cur_vars[idx] = value;
  UpdateLeqs(var_info_->cur_vars, idx, cons_info_->leqs);
  return true;
}

/*
函数名:SearchVars
功能描述:用于判断某组解是否曾被搜索过
*/
bool GeneralSolver::SearchVars(uint64_t *vars) const {
  Node *cur_node = visited_node_->GetVarVal(vars);
  if (cur_node != nullptr) {
    return cur_node->searched;
  }
  return false;
}

/*
函数名:CheckValid
功能描述:用于判断cur_var所对应的解是否为可行解
*/
bool GeneralSolver::CheckValid() const {
  for (int32_t i = 0; i < cons_info_->leq_num; i++) {
    if (cons_info_->leqs[i] > 0) {
      return false;
    }
  }
  return true;
}

void GeneralSolver::ResetMomentum() {
  for (int32_t i = 0; i < var_info_->var_num; i++) {
    momentum_info_->is_valid[i] = false;
  }
}

/*
函数名:Initialize
功能描述:用于在每一轮迭代开始执行前进行初始化操作
  在此过程中会重置var_info_中的部分参数
  并根据当前状态的cur_vars信息更新不等式约束值
输入参数:
  iter:迭代轮次
*/
void GeneralSolver::Initialize(int32_t iter) {
  var_info_->chosen_var_idx = -1;
  UpdateLeqs(var_info_->cur_vars, -1, cons_info_->leqs);
  is_feasible_ = CheckValid();
  has_feasible_ = has_feasible_ || is_feasible_;
  for (int32_t i = 0; i < var_info_->var_num; i++) {
    var_info_->history_vars[i] = (iter == 1) ? (var_info_->cur_vars[i]) : (var_info_->rec_vars[i]);
    var_info_->rec_vars[i] = var_info_->cur_vars[i];
  }
}

/*
函数名:GetLocality
功能描述:用来检测定域操作过程中所选变量的优先级
输入参数:
  idx:变量的下标
  update_direction:变量在当前位置的下降方向
输出参数:
  Locality类型的优先级指标
*/
Locality GeneralSolver::GetLocality(int32_t idx, UpdateDirection update_direction) {
  UpdateDirection cur_direction = GetDescent(var_info_->cur_vars, idx, FuncInfo::LEQ);
  if (CheckValid()) {
    return Locality::GLOBALVALID;
  } else if (CheckLocalValid(cons_info_->leqs, idx)) {
    return Locality::LOCALVALID;
  } else if (GetValue(update_direction) * GetValue(cur_direction) < 0) {
    return (var_info_->cur_vars[idx] != var_info_->history_vars[idx]) ? Locality::CROSSREGION : Locality::REJECT;
  }
  return Locality::INVALID;
}

/*
函数名:GetCoarseLoc
功能描述:
  定域过程中的变量粗调,大致确定变量的落点信息
  该函数会沿不等式约束的下降方向进行二分搜索
  最终会输出一个位于约束边界/可行域边界的候选落点
输入参数:
  update_info:变量的更新信息,包括下标(idx),下降方向(update_direction)等指标
  step:变量的更新步长
  cur_locality:粗调过程中确定的定域优先级
*/
bool GeneralSolver::GetCoarseLoc(const UpdateInfo *update_info, uint64_t &step, Locality &cur_locality) {
  uint64_t update_value;
  int32_t idx = update_info->idx;
  if ((idx < 0) || (idx >= var_info_->var_num)) {
    OP_LOG("idx illegal!");
    return false;
  }
  uint64_t thres = update_info->thres;
  UpdateDirection update_direction = update_info->update_direction;
  do {
    step = (step == 0) ? 1 : (step << 1);
    update_value = Bound(var_info_->upper_bound[idx], var_info_->lower_bound[idx], var_info_->rec_vars[idx], step,
                         update_direction);
    UpdateCurVarVal(update_value, idx);
    cur_locality = GetLocality(idx, update_direction);
    var_info_->cur_vars[idx] = var_info_->rec_vars[idx];
    if (cur_locality <= Locality::CROSSREGION) {
      step = ((cur_locality == Locality::CROSSREGION) && (step != 1)) ? (step >> 1) : step;
      break;
    }
  } while (step < thres);
  update_value =
      Bound(var_info_->upper_bound[idx], var_info_->lower_bound[idx], var_info_->rec_vars[idx], step, update_direction);
  UpdateCurVarVal(update_value, idx);
  return thres != 0;
}

/*
函数名:GetFineLoc
功能描述:
  定域过程中的变量精调,细致地确定变量的落点
  后验知识表明约束边界的解相对更好,因此尝试寻找位于边界的可行解
  该函数会在粗调所得的大致落点附近搜索,寻找不等式约束的边界点
*/
bool GeneralSolver::GetFineLoc(const UpdateInfo *update_info, uint64_t &step, Locality &cur_locality) {
  uint64_t update_value;
  Locality rec_locality;

  int32_t idx = update_info->idx;
  if ((idx < 0) || (idx >= var_info_->var_num)) {
    OP_LOG("idx illegal!");
    return false;
  }
  UpdateDirection update_direction = update_info->update_direction;
  if (GetLocality(idx, update_direction) <= Locality::LOCALVALID) {
    while (step > 1) {
      step >>= 1;
      update_value = var_info_->cur_vars[idx] - GetValue(update_direction) * step;
      UpdateCurVarVal(update_value, idx);
      rec_locality = GetLocality(idx, update_direction);
      if (rec_locality > Locality::CROSSREGION) {
        update_value = var_info_->cur_vars[idx] + GetValue(update_direction) * step;
      } else {
        update_value = var_info_->cur_vars[idx];
      }
      UpdateCurVarVal(update_value, idx);
    }
    cur_locality = GetLocality(idx, update_direction);
  }
  return true;
}

/*
函数名:GetPeerLoc
功能描述:
  在定域过程中搜索某个解的对端解
  对端解:若当前解位于约束边界,则对端解位于可行域另一侧的约束边界
  当某个方向的可行解最优但曾被搜索过,该函数可以跨越可行域寻找另一个可行域边界上的解,跳出局部最优
*/
bool GeneralSolver::GetPeerLoc(const UpdateInfo *update_info, Locality &cur_locality) {
  uint64_t left_value;
  uint64_t right_value;
  uint64_t mid_value;
  Locality rec_locality;
  int32_t idx = update_info->idx;
  if ((idx < 0) || (idx >= var_info_->var_num)) {
    OP_LOG("idx illegal!");
    return false;
  }
  uint64_t rec_value = var_info_->cur_vars[idx];
  UpdateDirection update_direction = update_info->update_direction;
  UpdateCurVarVal(
      (update_direction == UpdateDirection::NEGATIVE) ? var_info_->lower_bound[idx] : var_info_->upper_bound[idx], idx);
  rec_locality = GetLocality(idx, update_direction);
  if (rec_locality <= Locality::LOCALVALID) {
    var_info_->cur_vars[idx] = rec_value;
  } else {
    left_value = (update_direction == UpdateDirection::POSITIVE) ? (rec_value + 1) : 1;
    right_value = (update_direction == UpdateDirection::POSITIVE) ? (var_info_->upper_bound[idx])
                                                                  : (rec_value - var_info_->lower_bound[idx]);
    while (left_value < right_value) {
      mid_value = (left_value + right_value) >> 1;
      UpdateCurVarVal(mid_value, idx);
      rec_locality = GetLocality(idx, update_direction);
      if (rec_locality > Locality::LOCALVALID) {
        left_value = mid_value + 1;
      } else {
        right_value = mid_value;
      }
    }
    var_info_->cur_vars[idx] = left_value;
    cur_locality = Locality::ALTERNATIVE;
  }
  return true;
}

/*
函数名:UpdateMomentum
功能描述:
  更新算法中的动量信息，以帮助算法更快地收敛到最优解
输入参数:
  idx:更新动量信息的变量索引。
  update_value:更新值。
  cur_locality:当前的LOCALITY信息
输出参数:
  best_locality:当前找到的最好的LOCALITY信息
*/
void GeneralSolver::UpdateMomentum(int32_t idx, double update_value, Locality cur_locality, Locality &best_locality) {
  if (!SearchVars(var_info_->cur_vars)) {
    if (cur_locality < best_locality) {
      ResetMomentum();
      best_locality = cur_locality;
    }
    if (cur_locality == best_locality) {
      var_info_->target_val[idx] = var_info_->cur_vars[idx];
      momentum_info_->is_valid[idx] = true;
      momentum_info_->cur_value[idx] = update_value;
    }
  }
}

/*
函数名:GetBestChoice
功能描述:
  根据动量信息选择最佳变量进行更新
  使用idx遍历所有变量,检查动量信息是否有效,并计算动量值
  选取动量值最佳的变量作为输出
输出参数:
  bool类型参数,用于标记是否找到了最佳变量
*/
bool GeneralSolver::GetBestChoice() {
  bool better_choice;
  bool make_sense;
  double cur_value = 0.0;
  bool has_chosen = false;
  for (int32_t idx = 0; idx < var_info_->var_num; idx++) {
    if (momentum_info_->is_valid[idx]) {
      momentum_info_->momentum[idx] *= solver_config_.momentum_factor;
      momentum_info_->momentum[idx] += momentum_info_->cur_value[idx] * (1 - solver_config_.momentum_factor);
      better_choice = !has_chosen || momentum_info_->momentum[idx] > cur_value;
      make_sense = var_info_->cur_vars[idx] != var_info_->target_val[idx];
      if (better_choice && make_sense) {
        var_info_->chosen_var_idx = idx;
        has_chosen = true;
        cur_value = momentum_info_->momentum[idx];
      }
    }
  }
  return var_info_->chosen_var_idx != -1;
}

/*
函数名:UpdateBestVar
功能描述:
  根据chosen_var_idx的值对变量进行更新
  并调整momentum_info_中其他变量的动量信息
*/
bool GeneralSolver::UpdateBestVar() {
  for (int32_t idx = 0; idx < var_info_->var_num; idx++) {
    if (var_info_->chosen_var_idx == idx) {
      var_info_->cur_vars[idx] = var_info_->target_val[idx];
    } else {
      momentum_info_->momentum[idx] = 0;
    }
    momentum_info_->is_valid[idx] = false;
  }
  UpdateLeqs(var_info_->cur_vars, -1, cons_info_->leqs);
  return true;
}

/*
函数名:LocateLoc
功能描述:
  在需要精调变量落点的情况下寻找变量的落点
  该函数会根据cur_locality和best_locality确定是否需要精调
  若需要,则会调用GetFineLoc函数进行精调,并根据精调结果判断是否要取对端解
  最后根据预期落点更新动量信息
*/
bool GeneralSolver::LocateLoc(const UpdateInfo *update_info, uint64_t &step, Locality &cur_locality,
                              Locality &best_locality) {
  int32_t idx = update_info->idx;
  double init_obj = update_info->init_obj;
  if (cur_locality <= best_locality) {
    GetFineLoc(update_info, step, cur_locality);
    if (!solver_config_.simple_ver && SearchVars(var_info_->cur_vars)) {
      GetPeerLoc(update_info, cur_locality);
    }
    double update_value = init_obj - GetSmoothObj(var_info_->cur_vars);
    UpdateMomentum(idx, update_value, cur_locality, best_locality);
    return true;
  }
  return false;
}

/*
函数名:TryLocate
功能描述:
  尝试对特定变量进行定域操作
  若该更新该变量有希望走入可行域,则会使用GetCoarseLoc函数进行粗调
  根据粗调结果判断是否需要精调,若需要则调用LocateLoc函数进行精调
输入参数:
  idx:变量的索引
  init_idx:变量在当前位置的初始目标函数值
  best_locality:当前找到的最好的LOCALITY信息
*/
bool GeneralSolver::TryLocate(int32_t idx, double init_obj, Locality &best_locality) {
  Locality cur_locality;
  uint64_t step = 0;
  UpdateDirection update_direction = GetDescent(var_info_->cur_vars, idx, FuncInfo::LEQ);
  if (update_direction != UpdateDirection::NONE) {
    uint64_t neg_thres = var_info_->cur_vars[idx] - var_info_->lower_bound[idx];
    uint64_t pos_thres = var_info_->upper_bound[idx] - var_info_->cur_vars[idx];
    uint64_t thres = (update_direction == UpdateDirection::POSITIVE) ? pos_thres : neg_thres;
    UpdateInfo *update_info = new (std::nothrow) UpdateInfo(idx, thres, update_direction, init_obj);
    if (update_info == nullptr) {
      OP_LOG("Create update_info failed!");
      return false;
    }
    if (GetCoarseLoc(update_info, step, cur_locality)) {
      if (!LocateLoc(update_info, step, cur_locality, best_locality)) {
        delete update_info;
        return false;
      }
      UpdateCurVarVal(var_info_->rec_vars[idx], idx);
    }
    delete update_info;
  }
  return true;
}

/*
函数名:LocateRegion
功能描述:
  定域操作,用于实现可行域外的变量更新
  当变量位于可行域外时,由不等式约束驱动变量进行调整
  使用TryLocate函数确定变量的落点信息
  优先检测update_last为false的变量,在不存在可行的定域解时检测update_last为true的变量
  寻找目标函数更优的落点
*/
bool GeneralSolver::LocateRegion() {
  Locality best_locality = Locality::REJECT;
  double init_obj = GetSmoothObj(var_info_->cur_vars);
  for (int32_t idx = 0; idx < var_info_->var_num; idx++) {
    if (!var_info_->update_last[idx]) {
      TryLocate(idx, init_obj, best_locality);
    }
  }
  if (has_feasible_ || best_locality == Locality::REJECT) {
    for (int32_t idx = 0; idx < var_info_->var_num; idx++) {
      if (var_info_->update_last[idx]) {
        TryLocate(idx, init_obj, best_locality);
      }
    }
  }
  if (best_locality == Locality::REJECT || !GetBestChoice()) {
    return false;
  }
  UpdateBestVar();
  return true;
}

/*
函数名:GetTunePriority
功能描述:
  确定微调过程中某个待求解变量的优先级
输入参数:
  idx:待求解变量的下标
  rec_obj:本轮迭代前的初始目标函数值
输出参数:
  cur_obj:微调后变量的目标函数值
*/
TunePriority GeneralSolver::GetTunePriority(int32_t idx, double rec_obj, double &cur_obj) {
  cur_obj = GetSmoothObj(var_info_->cur_vars);
  int64_t last_update = var_info_->rec_vars[idx] - var_info_->history_vars[idx];
  int64_t next_update = var_info_->cur_vars[idx] - var_info_->rec_vars[idx];
  if (last_update * next_update < 0) {
    return TunePriority::TABU;
  } else if (cur_obj <= rec_obj) {
    if (CheckLocalValid(cons_info_->leqs, idx)) {
      return (cur_obj < rec_obj) ? TunePriority::HARMLESS : TunePriority::DILATED;
    } else {
      return (cur_obj < rec_obj) ? TunePriority::NORMAL
                                 : (solver_config_.simple_ver ? TunePriority::REFUSE : TunePriority::OTHER);
    }
  }
  return solver_config_.simple_ver ? TunePriority::REFUSE : TunePriority::OTHER;
}

/*
函数名:SearchLoc
功能描述:
  沿着指定的更新方向进行探索,检查是否有机会取到更优的可行解
  该函数会探索至多solver_config_.search_length步,若存在更优的可行解则会进行标记
输入参数:
  update_info:变量的更新信息
输出参数:
  step:取得更优可行解时的步长
  cur_obj:微调后变量的目标函数值
  cur_priority:微调后变量的优先级
*/
bool GeneralSolver::SearchLoc(const UpdateInfo *update_info, uint64_t &step, double &cur_obj,
                              TunePriority &cur_priority) {
  TunePriority rec_priority;
  int32_t idx = update_info->idx;
  if ((idx < 0) || (idx >= var_info_->var_num)) {
    OP_LOG("idx illegal!");
    return false;
  }
  uint64_t thres = update_info->thres;
  UpdateDirection update_direction = update_info->update_direction;
  double init_obj = update_info->init_obj;
  while (step < SMIN(thres, solver_config_.search_length)) {
    step++;
    UpdateCurVarVal(var_info_->rec_vars[idx] + GetValue(update_direction) * step, idx);
    rec_priority = GetTunePriority(idx, init_obj, cur_obj);
    if (rec_priority <= cur_priority) {
      cur_priority = rec_priority;
      break;
    }
  }
  UpdateCurVarVal(var_info_->rec_vars[idx], idx);
  return rec_priority == cur_priority;
}

/*
函数名:GetHarmlessLoc
功能描述:
  当且仅当存在一个目标函数更优的可行解时称求解器能找到无损的局部最优解
  该函数尝试在搜索范围内检查所有的可行解,寻找最优的无损局部最优解
输入参数:
  update_info:变量的更新信息
输出参数:
  step:取得更优可行解时的步长
  cur_obj:微调后无损局部最优解的目标函数值
*/
bool GeneralSolver::GetHarmlessLoc(const UpdateInfo *update_info, uint64_t &step, double &cur_obj) {
  double rec_obj;
  int32_t update_value;
  TunePriority rec_priority;
  int32_t idx = update_info->idx;
  if ((idx < 0) || (idx >= var_info_->var_num)) {
    OP_LOG("idx illegal!");
    return false;
  }
  uint64_t thres = update_info->thres;
  UpdateDirection update_direction = update_info->update_direction;
  var_info_->cur_vars[idx] = var_info_->rec_vars[idx];
  while (step < thres) {
    step = solver_config_.simple_ver ? (step == 0 ? 1 : (step << 1)) : (step + 1);
    update_value = Bound(var_info_->upper_bound[idx], var_info_->lower_bound[idx], var_info_->rec_vars[idx], step,
                         update_direction);
    UpdateCurVarVal(update_value, idx);
    rec_priority = GetTunePriority(idx, cur_obj, rec_obj);
    if (rec_priority != TunePriority::HARMLESS) {
      step = solver_config_.simple_ver ? (step >> 1) : (step - 1);
      break;
    }
    cur_obj = rec_obj;
  }
  return true;
}

/*
函数名:GetDilatedLoc
功能描述:
  当且仅当存在一个目标函数不变但更接近可行域边界的可行解时称求解器能找到膨胀局部最优解
  该函数沿着缓存占用边界更新变量,寻找更新方向上最接近可行域边界的膨胀局部最优解
输入参数:
  update_info:变量的更新信息
输出参数:
  step:取得更优可行解时的步长
*/
bool GeneralSolver::GetDilatedLoc(const UpdateInfo *update_info, uint64_t &step) {
  int32_t idx = update_info->idx;
  if ((idx < 0) || (idx >= var_info_->var_num)) {
    OP_LOG("idx illegal!");
    return false;
  }
  uint64_t update_value;
  uint64_t thres = update_info->thres;
  UpdateDirection update_direction = update_info->update_direction;
  double cur_obj;
  double cur_cons;
  double init_obj = update_info->init_obj;
  double init_cons = update_info->init_cons;
  double pre_cons = init_cons;
  while (step < thres) {
    step = solver_config_.simple_ver ? (step == 0 ? 1 : (step << 1)) : (step + 1);
    update_value = Bound(var_info_->upper_bound[idx], var_info_->lower_bound[idx], var_info_->rec_vars[idx], step,
                         update_direction);
    UpdateCurVarVal(update_value, idx);
    cur_obj = GetSmoothObj(var_info_->cur_vars);
    cur_cons = GetBuffCost(var_info_->cur_vars);
    if (!CheckLocalValid(cons_info_->leqs, idx) || (!IsEqual(init_obj, cur_obj)) || (cur_cons > pre_cons)) {
      step = solver_config_.simple_ver ? (step >> 1) : (step - 1);
      break;
    }
    pre_cons = cur_cons;
  }
  return true;
}

/*
函数名:UpdateMomentum
功能描述:
  是前一个UpdateMomentum的重载
  前一个UpdateMomentum函数用于更新定域过程中的动量信息
  本函数用于更新微调过程中的动量信息
*/
void GeneralSolver::UpdateMomentum(int32_t idx, double update_value, TunePriority cur_priority,
                                   TunePriority &best_priority) {
  if (!SearchVars(var_info_->cur_vars)) {
    if (cur_priority < best_priority) {
      ResetMomentum();
      best_priority = cur_priority;
    }
    if (cur_priority == best_priority) {
      if (update_value > momentum_info_->cur_value[idx] || !momentum_info_->is_valid[idx]) {
        var_info_->target_val[idx] = var_info_->cur_vars[idx];
        momentum_info_->is_valid[idx] = true;
        momentum_info_->cur_value[idx] = update_value;
      }
    }
  }
}

/*
函数名:TuneLoc
功能描述:
  根据变量的更新信息对某个变量进行进一步的微调
  根据输入的微调优先级cur_priority选取微调策略对变量进行更新
  若优先级为HARMLESS,则会调用GetHarmlessLoc函数进行无损更新
  若优先级为DILATED,则会调用GetDilatedLoc函数进行膨胀更新
*/
bool GeneralSolver::TuneLoc(const UpdateInfo *update_info, double cur_obj, uint64_t &step, TunePriority &cur_priority,
                            TunePriority &best_priority) {
  if (cur_priority <= best_priority) {
    uint64_t update_value;
    int32_t idx = update_info->idx;
    if ((idx < 0) || (idx >= var_info_->var_num)) {
      OP_LOG("idx illegal!");
      return false;
    }
    UpdateDirection update_direction = update_info->update_direction;
    double init_obj = update_info->init_obj;
    if (cur_priority == TunePriority::HARMLESS) {
      GetHarmlessLoc(update_info, step, cur_obj);
    } else if (cur_priority == TunePriority::DILATED) {
      UpdateDirection cur_direction = GetDescent(var_info_->cur_vars, idx, FuncInfo::BUFFER);
      if (GetValue(cur_direction) * GetValue(update_direction) >= 0) {
        GetDilatedLoc(update_info, step);
      } else {
        cur_priority = solver_config_.simple_ver ? TunePriority::REFUSE : TunePriority::OTHER;
      }
    }
    update_value = Bound(var_info_->upper_bound[idx], var_info_->lower_bound[idx], var_info_->rec_vars[idx], step,
                         update_direction);
    UpdateCurVarVal(update_value, idx);
    UpdateMomentum(idx, (init_obj - cur_obj), cur_priority, best_priority);
    return true;
  }
  return false;
}

/*
函数名:TryTune
功能描述:
  对某个变量进行微调
  首先利用SearchLoc函数在领域内判断是否存在更优的可行解
  然后根据微调优先级cur_priority选取微调策略对变量进行更新
*/
bool GeneralSolver::TryTune(int32_t idx, UpdateDirection update_direction, double init_obj, double init_cons,
                            TunePriority &best_priority) {
  uint64_t step = 0;
  uint64_t pos_thres = var_info_->upper_bound[idx] - var_info_->cur_vars[idx];
  uint64_t neg_thres = var_info_->cur_vars[idx] - var_info_->lower_bound[idx];
  uint64_t thres = (update_direction == UpdateDirection::POSITIVE) ? pos_thres : neg_thres;
  double cur_obj;
  TunePriority cur_priority = (thres > 0) ? best_priority : TunePriority::REFUSE;
  if (thres > 0) {
    UpdateInfo *update_info = new (std::nothrow) UpdateInfo(idx, thres, update_direction, init_obj, init_cons);
    if (update_info == nullptr) {
      OP_LOG("Create update_info failed!");
      return false;
    }
    if (SearchLoc(update_info, step, cur_obj, cur_priority)) {
      if (!TuneLoc(update_info, cur_obj, step, cur_priority, best_priority)) {
        return false;
      }
      UpdateCurVarVal(var_info_->rec_vars[idx], idx);
    }
    delete update_info;
  }
  return cur_priority >= TunePriority::NORMAL;
}

/*
函数名:FineTune
功能描述:
  实现待求解变量的微调操作
  首先沿正方向对变量进行更新,若更新方向上存在更优的可行解则进行微调
  若正方向上不存在更优的可行解或采用高性能版本进行求解,则尝试沿负方向进行更新
*/
bool GeneralSolver::FineTune() {
  double init_obj = GetSmoothObj(var_info_->cur_vars);
  double init_cons = GetBuffCost(var_info_->cur_vars);
  if (!RecordBestVarVal()) {
    return false;
  }
  TunePriority best_priority = TunePriority::TABU;
  for (int32_t idx = 0; idx < var_info_->var_num; idx++) {
    if (TryTune(idx, UpdateDirection::POSITIVE, init_obj, init_cons, best_priority) || !solver_config_.simple_ver) {
      TryTune(idx, UpdateDirection::NEGATIVE, init_obj, init_cons, best_priority);
    }
  }
  if (!GetBestChoice()) {
    return false;
  }
  UpdateBestVar();
  return true;
}

bool GeneralSolver::RecordBestVarVal() {
  if (is_feasible_) {
    double obj = GetObj(var_info_->cur_vars);
    double cons = GetBuffCost(var_info_->cur_vars);
    return result_->AddVarVal(var_info_->cur_vars, obj, cons);
  }
  return false;
}

/*
函数名:Run
功能描述:
  通用求解器求解函数
  算法会迭代solver_config_.iterations次
  在每轮迭代中根据当前的变量值选取定域或微调策略对变量进行更新
输出参数:
  solution_num:int32_t类型的参数,用来输出实际得到的解的个数
  solutions:uint64_t类型的数组,指向一块num_var * top_num的内存,求解算法获取到的可行解放入该空间
*/
bool GeneralSolver::Run(int32_t &solution_num, uint64_t *solutions) {
  Node *cur_node;
  uint64_t iter = 1;
  has_feasible_ = false;
  while (iter <= solver_config_.iterations) {
    Initialize(iter);
    if (solver_config_.get_log) {
      std::cout << "iter:" << iter << std::endl;
      DisplayVarVal(var_info_->cur_vars);
      std::cout << std::endl;
    }
    if (!is_feasible_) {
      if (!LocateRegion()) {
        break;
      }
    } else {
      if (SearchVars(var_info_->cur_vars)) {
        break;
      }
      cur_node = visited_node_->GetVarVal(var_info_->cur_vars);
      if (cur_node == nullptr) {
        break;
      }
      cur_node->searched = true;
      if (!FineTune()) {
        break;
      }
    }
    iter++;
  }
  result_->GetResult(solution_num, solutions);
  return solution_num > 0;
}

int32_t GeneralSolver::GetVarNum() const {
  return var_info_->var_num;
}
