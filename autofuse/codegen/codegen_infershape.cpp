/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen_infershape.h"
#include "code_printer.h"
#include <set>
#include "graph/symbolizer/symbolic.h"
namespace codegen {
namespace {
std::string GetFileHeaderDefine() {
  std::string file_header_str = R"(
#include <cmath>
#include <type_traits>
#include <unordered_map>
#include "exe_graph/runtime/infer_shape_context.h"

namespace {
#define Max(a, b) ((a) > (b) ? (a) : (b))
#define Min(a, b) ((a) < (b) ? (a) : (b))
#define Pow(a, b) (std::pow(a, b))
#define Exp(a) (std::exp(a))
#define Log(a) (std::log(a))
#define Sqrt(a) (std::sqrt(a))
#define Ceiling(a) (std::ceil(a))
#define Floor(a) (std::floor(a))
#define Abs(a) (std::abs(a))
#define Rational(a, b) (static_cast<double>(a) / static_cast<double>(b))

const double kThreshold = 0.00001;

template <typename Ta, typename Tb>
auto Mod(Ta left, Tb right) -> decltype(left % right) {
  return left % right;
}

// 针对浮点数的特化版本（使用 std::fmod）
template <typename Ta, typename Tb>
auto Mod(Ta left, Tb right) ->
    typename std::enable_if<std::is_floating_point<Ta>::value || std::is_floating_point<Tb>::value,
                            decltype(std::fmod(left, right))>::type {
  return std::fmod(left, right);
}

class InferShapeSymbolEvalContext : public gert::InferShapeContext {
 public:
  const gert::Tensor *GetGraphInputTensor(size_t data_index) const {
    auto *tensor = GetInputPointer<gert::Tensor>(data_index + 1);
    if (tensor == nullptr) {
      return nullptr;
    }
    return tensor;
  }
};
static_assert(std::is_standard_layout<InferShapeSymbolEvalContext>::value,
              "The class InferShapeSymbolEvalContext must be a POD");
} // namespace
)";
  return file_header_str;
}

std::set<std::string> CollectUsedSymbols(const std::vector<std::vector<std::string>> &symbol_shape_str,
                                         const std::map<std::string, std::string> &shape_info) {
  std::set<std::string> used_symbols;
  for (const auto &shape_dims : symbol_shape_str) {
    for (const auto &expr_str : shape_dims) {
      const auto expr = af::Expression::Parse(expr_str.c_str());
      for (const auto &symbol : expr.FreeSymbols()) {
        const auto symbol_name = symbol.Str();
        if (symbol_name != nullptr && shape_info.find(symbol_name.get()) != shape_info.end()) {
          used_symbols.insert(symbol_name.get());
        }
      }
    }
  }
  return used_symbols;
}
}  // namespace

std::string InfershapeGen::GenInferShapeFunc(const std::vector<std::vector<std::string>> &symbol_shape_str,
                                             const std::map<std::string, std::string> &shape_info) const {
  ge::CodePrinter printer;
  const std::string blank_space = "  ";
  printer.AddLine(GetFileHeaderDefine());

  std::string common_get_input_str;
  std::set<std::string> used_symbols = CollectUsedSymbols(symbol_shape_str, shape_info);
  for (const auto &sym_name : used_symbols) {
    common_get_input_str += (blank_space + "auto " + sym_name + " = " + shape_info.at(sym_name) + ";\n");
  }

  printer.DefineFuncBegin("extern \"C\" ge::graphStatus", "InferShape", "InferShapeSymbolEvalContext *context");
  printer.AddLine(common_get_input_str);

  size_t expr_value_num = 0;
  for (size_t i = 0U; i < symbol_shape_str.size(); ++i) {
    printer.AddLine(blank_space + "context->GetOutputShape(" + std::to_string(i) + ")->SetDimNum(0);");
    for (const auto &sym_expr : symbol_shape_str[i]) {
      std::string append_dim_code = blank_space + "context->GetOutputShape(" + std::to_string(i) + ")->AppendDim(";
      // 如果表达式中包含Rational, 结果可能是浮点数, 强转成整形会舍去小数部分导致结果错误, 因此要进行四舍五入处理
      // 然后对损失的小数部分做校验, 小于设定的阈值才认为计算成功
      if (sym_expr.find("Rational") != std::string::npos) {
        std::string expr_value_name = "expr_value_" + std::to_string(expr_value_num);
        std::string round_value_name = "round_value_" + std::to_string(expr_value_num);
        printer.AddLine(
            blank_space +
            "// 表达式中包含Rational, 结果可能是浮点数, 强转成整形会舍去小数部分导致结果错误, 因此要进行四舍五入处理");
        printer.AddLine(blank_space + "double " + expr_value_name + " = " + sym_expr + ";");
        printer.AddLine(blank_space + "int64_t " + round_value_name + " = std::round(" + expr_value_name + ");");
        printer.AddLine(blank_space + "// 对损失的小数部分做校验, 小于设定的阈值才认为计算成功");
        printer.AddLine(blank_space + "if ((fabs(" + expr_value_name + " - static_cast<double>(" + round_value_name +
                        ")) > kThreshold)) {");
        printer.AddLine(blank_space + "  return ge::GRAPH_FAILED;");
        printer.AddLine(blank_space + "}");
        append_dim_code += (round_value_name + ");");
        expr_value_num++;
      } else {
        append_dim_code += (sym_expr + ");");
      }
      printer.AddLine(append_dim_code);
    }
  }

  printer.AddLine(blank_space + "return ge::GRAPH_SUCCESS;");
  printer.DefineFuncEnd();

  return printer.GetOutputStr();
}
}  // namespace codegen
