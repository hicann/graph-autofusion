/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "expr_print_manager.h"

#include <set>
#include <symengine/basic.h>
#include <symengine/constants.h>
#include <symengine/rational.h>
#include <symengine/symengine_casts.h>
#include <symengine/pow.h>

#include "const_values.h"
#include "common/checker.h"

namespace af {
namespace {
const std::string kPrintAdd = " + ";
const std::string kPrintSub = " - ";
const std::string kPrintMul = " * ";
const std::string kPrintDiv = " / ";
const std::string kPrintMod = "Mod";
const std::string kPrintEq = "ExpectEq";
const std::string kPrintNe = "ExpectNe";
const std::string kPrintLe = "ExpectLe";
const std::string kPrintLt = "ExpectLt";
const std::string kPrintPow = "Pow";
const std::string kPrintLog = "Log";
const std::string kPrintMax = "Max";
const std::string kPrintMin = "Min";
const std::string kPrintExp = "Exp";
const std::string kPrintSqrt = "Sqrt";
const std::string kPrintCeil = "Ceiling";
const std::string kPrintFloor = "Floor";
const std::string kPrintAbs = "Abs";
const std::string kPrintLogicalAnd = "LogicAnd";
const std::string kPrintLogicalOr = "LogicOr";
const std::string kPrintDelim = ", ";
const std::string kPrintBracket_L = "(";
const std::string kPrintBracket_R = ")";
const size_t kRelationArgsNum = 2UL;

std::string PrintArgs(const std::vector<SymEngineExprPtr> &args, const std::string &delim, StrType type) {
  std::string res;
  std::vector<std::string> args_str;
  for (size_t i = 0u; i < args.size(); ++i) {
    args_str.emplace_back(ExpressionImpl::SymExprToExpressionImplRef(args[i]).Str(type));
  }
  // 保证序列化反序列化后的顺序
  std::sort(args_str.begin(), args_str.end());
  for (size_t i = 0u; i < args_str.size(); ++i) {
    if (i > 0u) {
      res += delim + args_str[i];
      continue;
    }
    res = args_str[i];
  }
  return res;
}

std::string DefaultCeilPrinter(const std::vector<SymEngineExprPtr> &args, StrType type) {
  return kPrintCeil + kPrintBracket_L + PrintArgs(args, kPrintDelim, type) + kPrintBracket_R;
}
REGISTER_EXPR_DEFAULT_PRINTER(OperationType::kOpCeil, DefaultCeilPrinter);

std::string DefaultFloorPrinter(const std::vector<SymEngineExprPtr> &args, StrType type) {
  return kPrintFloor + kPrintBracket_L + PrintArgs(args, kPrintDelim, type) + kPrintBracket_R;
}
REGISTER_EXPR_DEFAULT_PRINTER(OperationType::kOpFloor, DefaultFloorPrinter);

std::string DefaultAbsPrinter(const std::vector<SymEngineExprPtr> &args, StrType type) {
  return kPrintAbs + kPrintBracket_L + PrintArgs(args, kPrintDelim, type) + kPrintBracket_R;
}
REGISTER_EXPR_DEFAULT_PRINTER(OperationType::kOpAbs, DefaultAbsPrinter);

std::string DefaultLogicalAndPrinter(const std::vector<SymEngineExprPtr> &args, StrType type) {
  return kPrintLogicalAnd + kPrintBracket_L + PrintArgs(args, kPrintDelim, type) + kPrintBracket_R;
}
REGISTER_EXPR_DEFAULT_PRINTER(OperationType::kOpLogicalAnd, DefaultLogicalAndPrinter);

std::string DefaultLogicalOrPrinter(const std::vector<SymEngineExprPtr> &args, StrType type) {
  return kPrintLogicalOr + kPrintBracket_L + PrintArgs(args, kPrintDelim, type) + kPrintBracket_R;
}
REGISTER_EXPR_DEFAULT_PRINTER(OperationType::kOpLogicalOr, DefaultLogicalOrPrinter);

std::string DefaultAddPrinter(const std::vector<SymEngineExprPtr> &args, StrType type) {
  std::vector<SymEngineExprPtr> positive_args;
  std::vector<SymEngineExprPtr> negative_args;
  for (const auto &arg : args) {
    if (SymEngine::is_a<SymEngine::Mul>(*arg) &&
        (SymEngine::down_cast<const SymEngine::Mul &>(*arg)).get_coef()->is_negative()) {
      negative_args.push_back(SymEngine::mul(arg, SymEngine::minus_one));
      continue;
    }
    positive_args.push_back(arg);
  }
  std::string res_str = kPrintBracket_L;
  if (!positive_args.empty()) {
    res_str += PrintArgs(positive_args, kPrintAdd, type);
  }
  if (!negative_args.empty()) {
    res_str += kPrintSub;
    res_str += PrintArgs(negative_args, kPrintSub, type);
  }
  res_str += kPrintBracket_R;
  return res_str;
}
REGISTER_EXPR_DEFAULT_PRINTER(OperationType::kOpAdd, DefaultAddPrinter);

std::string DefaultMulPrinter(const std::vector<SymEngineExprPtr> &args, StrType type) {
  // split mul to num and dens
  std::vector<SymEngineExprPtr> positive_args;
  std::vector<SymEngineExprPtr> negative_args;
  for (const auto &arg : args) {
    if (SymEngine::is_a<SymEngine::Pow>(*arg)) {
      const auto exp = SymEngine::down_cast<const SymEngine::Pow &>(*arg).get_exp();
      if (SymEngine::is_a_Number(*exp) && SymEngine::down_cast<const SymEngine::Number &>(*exp).is_negative()) {
        negative_args.push_back(SymEngine::div(SymEngine::one, arg));
        continue;
      }
    }
    positive_args.push_back(arg);
  }
  std::string res_str = kPrintBracket_L;
  if (!positive_args.empty()) {
    res_str += PrintArgs(positive_args, kPrintMul, type);
  } else {
    res_str += std::to_string(sym::kConstOne);
  }
  if (!negative_args.empty()) {
    res_str += kPrintDiv;
    res_str += kPrintBracket_L + PrintArgs(negative_args, kPrintMul, type) + kPrintBracket_R;
  }
  res_str += kPrintBracket_R;
  return res_str;
}
REGISTER_EXPR_DEFAULT_PRINTER(OperationType::kOpMul, DefaultMulPrinter);

std::string DefaultMaxPrinter(const std::vector<SymEngineExprPtr> &args, StrType type) {
  std::string res_str;
  if (args.size() >= kSizeTwo) {
    res_str = kPrintMax + kPrintBracket_L + ExpressionImpl::SymExprToExpressionImplRef(args[0]).Str(type) +
              kPrintDelim + ExpressionImpl::SymExprToExpressionImplRef(args[1]).Str(type) + kPrintBracket_R;
  }
  for (size_t i = kSizeTwo; i < args.size(); ++i) {
    res_str = kPrintMax + kPrintBracket_L + res_str + kPrintDelim +
              ExpressionImpl::SymExprToExpressionImplRef(args[i]).Str(type) + kPrintBracket_R;
  }
  return res_str;
}
REGISTER_EXPR_DEFAULT_PRINTER(OperationType::kOpMax, DefaultMaxPrinter);

std::string DefaultMinPrinter(const std::vector<SymEngineExprPtr> &args, StrType type) {
  std::string res_str;
  if (args.size() >= kSizeTwo) {
    res_str = kPrintMin + kPrintBracket_L + ExpressionImpl::SymExprToExpressionImplRef(args[0]).Str(type) +
              kPrintDelim + ExpressionImpl::SymExprToExpressionImplRef(args[1]).Str(type) + kPrintBracket_R;
  }
  for (size_t i = kSizeTwo; i < args.size(); ++i) {
    res_str = kPrintMin + kPrintBracket_L + res_str + kPrintDelim +
              ExpressionImpl::SymExprToExpressionImplRef(args[i]).Str(type) + kPrintBracket_R;
  }
  return res_str;
}
REGISTER_EXPR_DEFAULT_PRINTER(OperationType::kOpMin, DefaultMinPrinter);

std::string PrintIntExpPow(const SymEngineExprPtr &base, const uint32_t exp, StrType type) {
  std::string res_str = "(";
  for (uint32_t i = 0u; i < exp; ++i) {
    if (i > 0u) {
      res_str += " * " + ExpressionImpl::SymExprToExpressionImplRef(base).Str(type);
      continue;
    }
    res_str += ExpressionImpl::SymExprToExpressionImplRef(base).Str(type);
  }
  return res_str + ")";
}

std::string GetDefaultPowPrint(const std::vector<SymEngineExprPtr> &base_args, StrType type) {
  const size_t base_idx = 0u;
  const size_t exp_idx = 1u;
  return kPrintPow + "(" + ExpressionImpl::SymExprToExpressionImplRef(base_args[base_idx]).Str(type) + ", " +
         ExpressionImpl::SymExprToExpressionImplRef(base_args[exp_idx]).Str(type) + ")";
}

std::string DefaultPowPrinter(const std::vector<SymEngineExprPtr> &args, StrType type) {
  constexpr const size_t pow_args_num = 2UL;
  GE_ASSERT_TRUE(args.size() == pow_args_num, "Symbol operator Pow args num should be 2 but get: %zu", args.size());
  const size_t base_idx = 0u;
  const size_t exp_idx = 1u;
  if (args[base_idx]->__eq__(*(SymEngine::E))) {
    return kPrintExp + "(" + ExpressionImpl::SymExprToExpressionImplRef(args[exp_idx]).Str(type) + ")";
  }
  if (args[exp_idx]->__eq__(*SymEngine::rational(sym::kNumOne, sym::kNumTwo))) {
    return kPrintSqrt + "(" + ExpressionImpl::SymExprToExpressionImplRef(args[base_idx]).Str(type) + ")";
  }
  if (args[exp_idx]->__eq__(*SymEngine::integer(sym::kNumOne))) {
    return "(" + ExpressionImpl::SymExprToExpressionImplRef(args[base_idx]).Str(type) + ")";
  }
  if (SymEngine::is_a<SymEngine::Integer>(*(args[exp_idx]))) {
    const SymEngine::Integer &exp_arg = SymEngine::down_cast<const SymEngine::Integer &>(*(args[exp_idx]));
    if (exp_arg.is_positive()) {
      return PrintIntExpPow(args[base_idx], exp_arg.as_uint(), type);
    }
  }
  return GetDefaultPowPrint(args, type);
}
REGISTER_EXPR_DEFAULT_PRINTER(OperationType::kOpPow, DefaultPowPrinter);

std::string GetDefaultModPrint(const std::vector<SymEngineExprPtr> &base_args, StrType type) {
  constexpr const size_t mod_args_num = 2UL;
  GE_ASSERT_TRUE(base_args.size() == mod_args_num, "Symbol operator Mod args num should be 2 but get: %zu",
                 base_args.size());
  const size_t dividend_idx = 0u;
  const size_t divisor_idx = 1u;
  return kPrintMod + "(" + ExpressionImpl::SymExprToExpressionImplRef(base_args[dividend_idx]).Str(type) + ", " +
         ExpressionImpl::SymExprToExpressionImplRef(base_args[divisor_idx]).Str(type) + ")";
}
REGISTER_EXPR_DEFAULT_PRINTER(OperationType::kOpMod, GetDefaultModPrint);

std::string DefaultLogPrinter(const std::vector<SymEngineExprPtr> &args, StrType type) {
  return kPrintLog + kPrintBracket_L + PrintArgs(args, kPrintDelim, type) + kPrintBracket_R;
}
REGISTER_EXPR_DEFAULT_PRINTER(OperationType::kOpLog, DefaultLogPrinter);

std::string DefaultEqualPrinter(const std::vector<SymEngineExprPtr> &args, StrType type) {
  GE_ASSERT_TRUE(args.size() == kRelationArgsNum, "Equal operator args size should be 2, but get %zu", args.size());

  return kPrintEq + kPrintBracket_L + PrintArgs(args, kPrintDelim, type) + kPrintBracket_R;
}
REGISTER_EXPR_DEFAULT_PRINTER(OperationType::kOpEq, DefaultEqualPrinter);

std::string DefaultUnEqualPrinter(const std::vector<SymEngineExprPtr> &args, StrType type) {
  GE_ASSERT_TRUE(args.size() == kRelationArgsNum, "Unequal operator args size should be 2, but get %zu", args.size());
  return kPrintNe + kPrintBracket_L + PrintArgs(args, kPrintDelim, type) + kPrintBracket_R;
}
REGISTER_EXPR_DEFAULT_PRINTER(OperationType::kOpNe, DefaultUnEqualPrinter);

std::string DefaultStrictLessThanPrinter(const std::vector<SymEngineExprPtr> &args, StrType type) {
  GE_ASSERT_TRUE(args.size() == kRelationArgsNum, "StrictLessThan operator args size should be 2, but get %zu",
                 args.size());
  return kPrintLt + kPrintBracket_L + ExpressionImpl::SymExprToExpressionImplRef(args[0]).Str(type) + kPrintDelim +
         ExpressionImpl::SymExprToExpressionImplRef(args[1]).Str(type) + kPrintBracket_R;
}
REGISTER_EXPR_DEFAULT_PRINTER(OperationType::kOpLt, DefaultStrictLessThanPrinter);

std::string DefaultLessThanPrinter(const std::vector<SymEngineExprPtr> &args, StrType type) {
  GE_ASSERT_TRUE(args.size() == kRelationArgsNum, "LessThan operator args size should be 2, but get %zu", args.size());
  return kPrintLe + kPrintBracket_L + ExpressionImpl::SymExprToExpressionImplRef(args[0]).Str(type) + kPrintDelim +
         ExpressionImpl::SymExprToExpressionImplRef(args[1]).Str(type) + kPrintBracket_R;
}
REGISTER_EXPR_DEFAULT_PRINTER(OperationType::kOpLe, DefaultLessThanPrinter);
}  // namespace
}  // namespace af
