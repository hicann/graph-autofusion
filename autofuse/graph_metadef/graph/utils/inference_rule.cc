/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "inference_rule.h"

#include <utility>
#include <vector>
#include <string>
#include <unordered_set>
#include <cctype>
#include <regex>
#include <stack>
#include <mutex>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <dlfcn.h>
#include <sys/wait.h>
#include <linux/memfd.h>
#include <sys/syscall.h>
#include <nlohmann/json.hpp>

#include "common/checker.h"
#include "graph/ge_error_codes_af.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"

using Json = nlohmann::json;
namespace af {
namespace {
/**
 * @brief 表达一个符号的来源
 *
 * 用于描述某个符号源自输入的某个维度或某个值。并支持生成对应的C++定义代码片段。
 */
class SymbolDef {
 public:
  explicit SymbolDef(const std::string &name) : name_(name), is_value_(name[0] == 'v') {}

  void RecordSource(size_t input_index, size_t offset) {
    sources_.emplace_back(input_index, offset);
  }

  [[nodiscard]] std::string Codegen() const {
    std::stringstream ss;
    if (!sources_.empty()) {
      const size_t input = sources_.front().first;
      const size_t offset = sources_.front().second;
      if (is_value_) {
        ss << "    GET_SYMBOL_VALUE(" << name_ << ", " << input << ", " << offset << ");";
      } else {
        ss << "    GET_SYMBOL_DIM(" << name_ << ", " << input << ", " << offset << ");";
      }
    }
    return ss.str();
  }

 private:
  std::string name_;
  std::vector<std::pair<size_t, size_t>> sources_;
  bool is_value_;
};

/**
 * @brief 表达一个Shape维度由符号表达的输出Tensor
 *
 * 用于描述输出Shape每个维度的计算表达式，表达式是支持受限的表达式（+，-，*，Div，Floor，Ceil，Mod，Pow），也可以是常量表达式。
 */
class SymbolTensor {
 public:
  explicit SymbolTensor(const size_t output_index) : output_index_(output_index) {}

  void AppendDim(const std::string &dim) {
    dims_.push_back(dim);
  }

  // 生成执行时的Shape设置代码片段
  [[nodiscard]] std::string Codegen() const {
    std::stringstream ss;
    ss << "    SET_OUTPUT_RANK(" << output_index_ << ", " << dims_.size() << ");" << std::endl;
    for (size_t i = 0; i < dims_.size(); i++) {
      ss << "    SET_OUTPUT_DIM(" << output_index_ << ", " << i << ", static_cast<int64_t>(" << dims_[i] << "));"
         << std::endl;
    }
    return ss.str();
  }

  // 生成编译时的Shape设置代码片段
  [[nodiscard]] std::string CodegenCompileTime() const {
    std::stringstream ss;
    ss << "    SET_OUTPUT_RANK(" << output_index_ << ", " << dims_.size() << ");" << std::endl;
    for (size_t i = 0; i < dims_.size(); i++) {
      const bool has_symbol = dims_[i].find('s') != std::string::npos || dims_[i].find('v') != std::string::npos;
      ss << "    SET_OUTPUT_DIM(" << output_index_ << ", " << i << ", " << (has_symbol ? "-1" : dims_[i]) << ");"
         << std::endl;
    }
    return ss.str();
  }

 private:
  size_t output_index_;
  std::vector<std::string> dims_;
};

/**
 * @brief Shape推导规则的JSON解析器
 *
 * 完成推导规则JSON的解析、合法性校验以及到InferShape代码的生成。
 */
class RuleJsonParser {
 public:
  std::string ParseJson(const std::string &json_str) {
    std::stringstream ss;
    Json rule_json;
    try {
      rule_json = Json::parse(json_str);
    } catch (const std::exception &e) {
      ss << "Error parsing json: " << e.what();
      return ss.str();
    }

    if (!rule_json.contains("shape")) {
      ss << "Missing 'shape' field in rule json.";
      return ss.str();
    }

    auto shape_json = rule_json["shape"];
    std::vector<std::vector<std::string>> inputs;
    std::vector<std::vector<std::string>> outputs;

    std::string error_msg = ParseJsonToVecVecString(shape_json["inputs"], inputs);
    if (!error_msg.empty()) {
      ss << "Invalid 'shape.inputs' field: " << shape_json["inputs"] << " " << error_msg;
      return ss.str();
    }
    error_msg = ParseJsonToVecVecString(shape_json["outputs"], outputs);
    if (!error_msg.empty()) {
      ss << "Invalid 'shape.outputs' field: " << shape_json["outputs"] << " " << error_msg;
      return ss.str();
    }
    std::map<std::string, SymbolDef> symbol_defs;
    error_msg = GetInputSymbolDefs(inputs, symbol_defs);
    if (!error_msg.empty()) {
      ss << "Error parsing input symbols: " << error_msg;
      return ss.str();
    }
    error_msg = GetOutputSymbolTensors(outputs, symbol_defs, symbols_, symbol_tensors_);
    if (!error_msg.empty()) {
      ss << "Error parsing output tensors: " << error_msg;
      return ss.str();
    }
    return ss.str();
  }

  void CodegenInferShape(std::stringstream &code_ss) const {
    code_ss << R"(extern "C" {)";
    code_ss << R"(bool infer_shape(Ctx *ctx) {)" << std::endl;

    for (const auto &symbol : symbols_) {
      code_ss << symbol.Codegen() << std::endl;
    }

    code_ss << std::endl;

    for (const auto &tensor : symbol_tensors_) {
      code_ss << tensor.Codegen() << std::endl;
    }

    code_ss << "    return true;\n}" << std::endl;

    code_ss << R"(bool infer_shape_on_compile(Ctx *ctx) {)" << std::endl;
    for (const auto &tensor : symbol_tensors_) {
      code_ss << tensor.CodegenCompileTime() << std::endl;
    }
    code_ss << "    return true;\n}";

    code_ss << "}";
  }

 private:
  std::vector<SymbolDef> symbols_;
  std::vector<SymbolTensor> symbol_tensors_;

  static std::string GetInputSymbolDefs(const std::vector<std::vector<std::string>> &inputs,
                                        std::map<std::string, SymbolDef> &symbol_defs) {
    for (size_t i = 0; i < inputs.size(); i++) {
      const auto &dims = inputs[i];
      for (size_t j = 0; j < dims.size(); j++) {
        const auto &dim = dims[j];
        if (dim.empty() || IsNumber(dim)) {
          continue;
        }
        if (!IsSymbol(dim)) {
          std::stringstream ss;
          ss << "Invalid input[" << i << "].size(" << j << "): " << dim
             << ", symbol dimension must start with 's' or 'v' and follow with a number";
          return ss.str();
        }
        auto it = symbol_defs.find(dim);
        if (it != symbol_defs.end()) {
          // 已经存在，记录来源
          it->second.RecordSource(i, j);
        } else {
          // 新建符号定义
          SymbolDef symbol(dim);
          symbol.RecordSource(i, j);
          symbol_defs.emplace(dim, std::move(symbol));
        }
      }
    }
    return "";
  }

  static std::string GetOutputSymbolTensors(const std::vector<std::vector<std::string>> &outputs,
                                            const std::map<std::string, SymbolDef> &symbol_defs,
                                            std::vector<SymbolDef> &used_symbol_defs,
                                            std::vector<SymbolTensor> &symbol_tensors) {
    std::set<std::string> used_symbols;
    std::stringstream ss;
    for (size_t i = 0; i < outputs.size(); i++) {
      symbol_tensors.emplace_back(i);
      const auto &dims = outputs[i];

      for (size_t j = 0; j < dims.size(); j++) {
        auto &dim = dims[j];
        if (dim.empty()) {
          ss << "Invalid output[" << i << "].size(" << j << "): empty dimension";
          return ss.str();
        }
        std::string error_msg = ValidateDimExpr(dim, used_symbols);
        if (!error_msg.empty()) {
          ss << "Invalid dim expr '" << dim << "': " << error_msg;
          return ss.str();
        }
        symbol_tensors.back().AppendDim(dim);
      }
    }

    for (const auto &symbol : used_symbols) {
      auto it = symbol_defs.find(symbol);
      if (it == symbol_defs.end()) {
        ss << "Symbol '" << symbol << "' used in output but not defined in inputs";
        return ss.str();
      }
      used_symbol_defs.emplace_back(it->second);
    }

    return "";
  }

  static std::string ValidateDimExpr(std::string expr, std::set<std::string> &used_symbols) {
    expr.erase(remove_if(expr.begin(), expr.end(), isspace), expr.end());

    // 2. 定义 token 正则
    //   - 函数/变量名: [A-Za-z0-9_]*
    //   - 运算符: [+*()-,]
    const std::regex token_regex(R"([A-Za-z0-9_]*|\+|\-|\*|\(|\)|,)");
    const auto begin = std::sregex_iterator(expr.begin(), expr.end(), token_regex);
    const auto end = std::sregex_iterator();

    std::vector<std::string> tokens;  // 存储匹配到的 token，应当为操作符、操作数、函数名、括号之一
    for (auto it = begin; it != end; ++it) {
      if (!it->str().empty()) {
        tokens.push_back(it->str());
      }
    }

    // 检查是否所有字符都被匹配（防止非法字符）
    size_t totalLen = 0U;
    for (auto &t : tokens) totalLen += t.size();
    if (totalLen != expr.size()) {
      return "Expression contains invalid characters";
    }

    // 3. 遍历 tokens 检查合法性
    std::stack<std::string> func_stack;
    for (size_t i = 0U; i < tokens.size(); i++) {
      const std::string &token = tokens[i];

      if (std::isalpha(token[0])) {
        if (i + 1U < tokens.size() && tokens[i + 1U] == "(") {
          if (!IsSupportedFunc(token)) {
            return "Invalid function: " + token + ", supported [Div, Floor, Ceil, Pow, Mod]";
          }
        } else {
          used_symbols.insert(token);
        }
      } else if (token == "(") {
        func_stack.emplace("(");
      } else if (token == ")") {
        if (func_stack.empty()) {
          return "Unmatched ')'";
        }
        func_stack.pop();
      } else if (IsSupportedOperator(token) || IsNumber(token)) {
        // 运算符不做额外语法检查，由C++编译器处理
      } else {
        return "Invalid identifier: '" + token + "', expected start with 's' or 'v' and follow with a number";
      }
    }

    if (!func_stack.empty()) {
      return "Unmatched '('";
    }

    return "";
  }

  static std::string ParseJsonToVecVecString(const Json &json, std::vector<std::vector<std::string>> &result) {
    if (json.is_null()) {
      return "";
    }
    if (!json.is_array()) {
      return "field must be an array or null.";
    }

    for (const auto &dims : json) {
      if (dims.is_null()) {
        result.emplace_back();
        continue;
      }
      if (!dims.is_array()) {
        return "element must be an array of dimension expressions.";
      }
      result.emplace_back();
      for (const auto &dim : dims) {
        if (dim.is_null()) {
          result.back().emplace_back();
          continue;
        }
        if (!dim.is_string() && !dim.is_number_integer()) {
          return "dimension expression must be a string or integer.";
        }
        result.back().push_back(dim.is_string() ? dim.get<std::string>() : std::to_string(dim.get<int64_t>()));
      }
    }
    return "";
  }

  static bool IsSymbol(const std::string &token) {
    // 符号必须以 's' 或 'v' 开头，后跟数字
    return token.size() > 1 && (token[0] == 's' || token[0] == 'v') && IsNumber(&token[1]);
  }

  static bool IsSupportedFunc(const std::string &func) {
    static const std::unordered_set<std::string> kAllowedFuncs = {"Div", "Floor", "Ceil", "Pow", "Mod"};
    return kAllowedFuncs.find(func) != kAllowedFuncs.end();
  }

  static bool IsSupportedOperator(const std::string &op) {
    // 支持的运算符
    return op == "+" || op == "-" || op == "*" || op == ",";
  }

  static bool IsNumber(const std::string &s) {
    try {
      size_t idx;
      std::stod(s, &idx);
      return idx == s.size();  // 必须整个字符串都被解析
    } catch (...) {
      return false;
    }
  }
};

/**
 * @brief Cpp JIT编译器
 *
 * 用于将生成的C++代码编译为内存中的.so，并加载以供调用。
 */
class CppJitCompiler {
 public:
  std::string Error() const {
    return err_.str();
  }

  std::vector<uint8_t> Compile(const std::string &source_code) {
    std::vector<uint8_t> so_data;

    const int32_t cpp_fd = CreateMemFd("source.cpp");
    const int32_t so_fd = CreateMemFd("output.so");
    if (cpp_fd == -1 || so_fd == -1) {
      err_ << "mem fd create failed: " << strerror(errno);
      return {};
    }

    ClearCloexec(cpp_fd);
    ClearCloexec(so_fd);

    if (!WriteToFd(cpp_fd, source_code)) {
      err_ << "write source code to mem fd failed: " << strerror(errno);
      return {};
    }

    lseek(cpp_fd, 0, SEEK_SET);
    lseek(so_fd, 0, SEEK_SET);

    if (!CompileToSo(cpp_fd, so_fd)) {
      return {};
    }

    lseek(so_fd, 0, SEEK_SET);

    char buf[4096];
    ssize_t n;
    while ((n = read(so_fd, buf, sizeof(buf))) > 0) {
      so_data.insert(so_data.end(), buf, buf + n);
    }

    close(cpp_fd);
    close(so_fd);
    return so_data;
  }

  void *Load(const uint8_t *so_binary, const size_t so_size) {
    static std::atomic<int64_t> loaded{0};

    char tmp_filename[256] = {};
    // make sure the filename is unique for disable cache for dlopen
    const std::string filename = "/tmp/temp_so" + std::to_string(loaded++) + "XXXXXX";
    if (snprintf_s(tmp_filename, sizeof(tmp_filename), filename.size(), "%s", filename.c_str()) < 0) {
      err_ << "snprintf file name failed: " << strerror(errno);
      return nullptr;
    }

    const int32_t fd = mkstemp(tmp_filename);
    if (fd == -1) {
      err_ << "mkstemp failed: " << strerror(errno);
      return nullptr;
    }

    const ssize_t written = write(fd, so_binary, so_size);
    if (written != static_cast<ssize_t>(so_size)) {
      err_ << "write so binary to temp file failed: " << strerror(errno);
      close(fd);
      unlink(tmp_filename);
      return nullptr;
    }

    close(fd);

    void *handle = dlopen(tmp_filename, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
      err_ << "dlopen failed: " << dlerror();
      unlink(tmp_filename);
      return nullptr;
    }

    unlink(tmp_filename);
    return handle;
  }

 private:
  std::stringstream err_;
  static std::string GetSystemCompiler() {
    if (system("g++ --version > /dev/null 2>&1") == 0) {
      return "g++";
    }
    if (system("gcc --version > /dev/null 2>&1") == 0) {
      return "gcc";
    }
    return "";
  }

  static int32_t CreateMemFd(const std::string &name) {
    return syscall(__NR_memfd_create, name.c_str(), MFD_CLOEXEC);
  }

  static void ClearCloexec(const int32_t fd) {
    const int32_t flags = fcntl(fd, F_GETFD);
    if (flags != -1) {
      fcntl(fd, F_SETFD, flags & ~FD_CLOEXEC);
    }
  }

  static bool WriteToFd(const int32_t fd, const std::string &data) {
    size_t written = 0;
    while (written < data.size()) {
      const ssize_t n = write(fd, data.data() + written, data.size() - written);
      if (n <= 0) {
        return false;
      }
      written += n;
    }
    return true;
  }

  bool CompileToSo(const int32_t input_fd, const int32_t output_fd) {
    const std::string input_path = "/proc/self/fd/" + std::to_string(input_fd);
    const std::string output_path = "/proc/self/fd/" + std::to_string(output_fd);

    const std::string compiler = GetSystemCompiler();
    if (compiler.empty()) {
      err_ << "No C++ compiler found (g++ or gcc) for jit compiling symbol infer";
      return false;
    }

    const std::vector<const char *> args = {
        compiler.c_str(),   "-x",       "c++",  "-shared", "-fPIC", "-o", output_path.c_str(),
        input_path.c_str(), "-lstdc++", nullptr};

    const pid_t pid = fork();
    if (pid == 0) {
      execvp(compiler.c_str(), const_cast<char *const *>(args.data()));
      _exit(1);
    }

    int32_t status = 0;
    waitpid(pid, &status, 0);
    const bool succeed = WIFEXITED(status) && WEXITSTATUS(status) == 0;
    if (!succeed) {
      err_ << "syntax error";
    }
    return succeed;
  }
};

const std::string kHeader = R"(
#include <cmath>
#include <cstdint>

inline double Pow(const double base, const double exp) { return std::pow(base, exp); }
inline double Floor(const double x) { return std::floor(x); }
inline double Div(const double x, const double y) { return x / y; }
inline double Ceil(const double x) { return std::ceil(x); }
inline double Mod(const double a, const double b) {
    double r = std::fmod(a, b);
    if ((r != 0) && ((b < 0 && r > 0) || (b > 0 && r < 0))) {
        r += b;
    }
    return r;
}

extern "C" {
int64_t version() { return 1; }
}

class Ctx {
    public:
        virtual ~Ctx() = default;
        virtual bool GetInputDim(int64_t input, int64_t dim_index, int64_t &dim) = 0;
        virtual bool GetInputValue(int64_t input, int64_t offset, int64_t &value) = 0;
        virtual bool SetOutputDimNum(int64_t output, int64_t dim_num) = 0;
        virtual bool SetOutputDim(int64_t output, int64_t dim_index, int64_t dim) = 0;
        virtual void SetError(const char *) = 0;
};

#define GET_SYMBOL_DIM(S, INPUT, DIM) \
int64_t S##_int; \
if (!ctx->GetInputDim(INPUT, DIM, S##_int)) { \
    ctx->SetError("Failed to get dim sym '" #S "' from input[" #INPUT "], dim: " #DIM); \
    return false; \
} \
const double S = static_cast<double>(S##_int);

#define GET_SYMBOL_VALUE(S, INPUT, DIM) \
int64_t S##_int; \
if (!ctx->GetInputValue(INPUT, DIM, S##_int)) { \
    ctx->SetError("Failed to get value sym '" #S "' from input[" #INPUT "], offset: " #DIM); \
    return false; \
} \
const double S = static_cast<double>(S##_int);

#define SET_OUTPUT_RANK(OUTPUT, RANK) \
if (!ctx->SetOutputDimNum(OUTPUT, RANK)) { \
    ctx->SetError("Failed to set rank " #RANK " for output[" #OUTPUT "]"); \
    return false; \
}

#define SET_OUTPUT_DIM(OUTPUT, INDEX, DIM) \
if (!ctx->SetOutputDim(OUTPUT, INDEX, DIM)) { \
    ctx->SetError("Failed to set dim " #DIM " for output[" #OUTPUT "], dim: " #INDEX); \
    return false; \
}
)";

/**
 * @brief 适用于GertCtx的包装器
 *
 * Jit生成InferShape代码时，设计时保证不使用任何本地头文件参与编译，通过运行时的Ctx封装，隔离本地文件依赖。
 */
class GertContextWrapper final : public ShapeInferenceRule::Ctx {
 public:
  explicit GertContextWrapper(gert::InferShapeContext *ctx) : ctx_(ctx) {}

  bool GetInputDim(int64_t input, int64_t dim_index, int64_t &dim) override {
    const auto shape = ctx_->GetInputShape(input);
    if (shape == nullptr) {
      return false;
    }
    dim = shape->GetDim(dim_index);
    return true;
  }

  bool GetInputValue(int64_t input, int64_t offset, int64_t &value) override {
    auto *tensor = ctx_->GetInputTensor(input);
    if (tensor == nullptr || tensor->GetAddr() == nullptr) {
      return false;
    }
    if (offset < 0 || offset >= tensor->GetShapeSize()) {
      return false;
    }
    if (tensor->GetDataType() == ge::DT_INT64) {
      value = tensor->GetData<int64_t>()[offset];
    } else if (tensor->GetDataType() == ge::DT_INT32) {
      value = tensor->GetData<int32_t>()[offset];
    } else if (tensor->GetDataType() == ge::DT_UINT32) {
      value = tensor->GetData<uint32_t>()[offset];
    } else {
      SetError("Only int32, uint32 and int64 are supported for input value tensors");
      return false;
    }
    return true;
  }

  bool SetOutputDimNum(int64_t output, int64_t dim_num) override {
    const auto shape = ctx_->GetOutputShape(output);
    if (shape == nullptr) {
      return false;
    }
    shape->SetDimNum(dim_num);
    return true;
  }

  bool SetOutputDim(int64_t output, int64_t dim_index, int64_t dim) override {
    const auto shape = ctx_->GetOutputShape(output);
    if (shape == nullptr) {
      return false;
    }
    shape->SetDim(dim_index, dim);
    return true;
  }

  void SetError(const char *msg) override {
    if (msg != nullptr) {
      error_message_ << msg << std::endl;
    }
  }

  std::string Error() const {
    return error_message_.str();
  }

 private:
  gert::InferShapeContext *ctx_ = nullptr;
  std::stringstream error_message_;
};

template <typename T>
class Cache {
 public:
  std::shared_ptr<T> Get(const std::string &key) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      return it->second;
    }
    return nullptr;
  }

  std::shared_ptr<T> GetWithDefault(const std::string &key, const std::shared_ptr<T> &value) {
    std::lock_guard<std::mutex> lock(mtx_);
    return cache_.emplace(key, value).first->second;
  }

 private:
  std::mutex mtx_;
  std::map<std::string, std::shared_ptr<T>> cache_;
};

Cache<ShapeInferenceRule> g_shape_rule_cache;
Cache<DtypeInferenceRule> g_dtype_rule_cache;
}  // namespace

ShapeInferenceRule::~ShapeInferenceRule() {
  if (handle_) {
    dlclose(handle_);
    handle_ = nullptr;
    infer_shape_ = nullptr;
    infer_shape_on_compile_ = nullptr;
  }
}

ge::graphStatus ShapeInferenceRule::InferOnRuntime(Ctx *ctx) const {
  if (!infer_shape_) {
    ctx->SetError("infer_shape function is not set");
    return ge::GRAPH_FAILED;
  }
  if (!infer_shape_(ctx)) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeInferenceRule::InferOnCompile(Ctx *ctx) const {
  if (!infer_shape_on_compile_) {
    ctx->SetError("infer_shape_on_compile function is not set");
    return ge::GRAPH_FAILED;
  }
  if (!infer_shape_on_compile_(ctx)) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ShapeInferenceRule::InferOnRuntime(gert::InferShapeContext *infer_shape_ctx) const {
  GE_ASSERT_NOTNULL(infer_shape_ctx);
  GertContextWrapper ctx(infer_shape_ctx);
  const ge::graphStatus result = InferOnRuntime(&ctx);
  if (result != ge::GRAPH_SUCCESS) {
    GELOGE(af::FAILED, "Failed to infer shape by rule for op %s(%s): %s", infer_shape_ctx->GetNodeName(),
           infer_shape_ctx->GetNodeType(), ctx.Error().c_str());
  }
  return result;
}

ge::graphStatus ShapeInferenceRule::InferOnCompile(gert::InferShapeContext *infer_shape_ctx) const {
  GE_ASSERT_NOTNULL(infer_shape_ctx);
  GertContextWrapper ctx(infer_shape_ctx);
  const ge::graphStatus result = InferOnCompile(&ctx);
  if (result != ge::GRAPH_SUCCESS) {
    GELOGE(af::FAILED, "Failed to infer shape on compile by rule for op %s(%s): %s", infer_shape_ctx->GetNodeName(),
           infer_shape_ctx->GetNodeType(), ctx.Error().c_str());
  }
  return result;
}

std::string InferenceRule::GetInferenceRule(const af::OpDescPtr &op) {
  if (op == nullptr) {
    return "";
  }
  std::string rule_json;
  const std::string *rule_json_str = af::AttrUtils::GetStr(op, af::ATTR_NAME_INFER_RULE);
  if (rule_json_str != nullptr) {
    rule_json = *rule_json_str;
  }
  return rule_json;
}

std::shared_ptr<ShapeInferenceRule> ShapeInferenceRule::FromOpDesc(const af::OpDescPtr &op) {
  const std::string *rule_json = af::AttrUtils::GetStr(op, af::ATTR_NAME_INFER_RULE);
  if (rule_json == nullptr) {
    // Skip log error if op does not with rule
    return nullptr;
  }
  return FromJsonString(*rule_json);
}

std::shared_ptr<ShapeInferenceRule> ShapeInferenceRule::FromJsonString(const std::string &json_str) {
  auto cached = g_shape_rule_cache.Get(json_str);
  if (cached != nullptr) {
    return cached;
  }

  const auto rule = std::make_shared<ShapeInferenceRule>();
  RuleJsonParser parser;
  const std::string error_msg = parser.ParseJson(json_str);
  if (!error_msg.empty()) {
    *rule << error_msg;
    return g_shape_rule_cache.GetWithDefault(json_str, rule);
  }

  std::stringstream gen_code_ss;
  parser.CodegenInferShape(gen_code_ss);

  std::stringstream code_ss;
  code_ss << kHeader << std::endl;
  code_ss << gen_code_ss.str() << std::endl;

  CppJitCompiler compiler;
  const auto binary = compiler.Compile(code_ss.str());
  if (binary.empty()) {
    *rule << "Failed to compile C++ code to shared object:\n" << gen_code_ss.str() << "\nError: " << compiler.Error();
    return g_shape_rule_cache.GetWithDefault(json_str, rule);
  }
  return g_shape_rule_cache.GetWithDefault(json_str, std::make_shared<ShapeInferenceRule>(FromCompiledBinary(binary)));
}

ShapeInferenceRule ShapeInferenceRule::FromCompiledBinary(const uint8_t *binary, const size_t size) {
  ShapeInferenceRule infer_handle;
  CppJitCompiler compiler;
  void *handle = compiler.Load(binary, size);
  if (!handle) {
    infer_handle << "Failed to load compiled shared object from memory: " << compiler.Error();
    return infer_handle;
  }

  infer_handle.handle_ = handle;
  infer_handle.infer_shape_ = (InferShapeFunc)dlsym(handle, "infer_shape");
  if (!infer_handle.infer_shape_) {
    infer_handle << "dlsym infer_shape failed: " << dlerror();
    return infer_handle;
  }
  infer_handle.infer_shape_on_compile_ = (InferShapeFunc)dlsym(handle, "infer_shape_on_compile");
  if (!infer_handle.infer_shape_on_compile_) {
    infer_handle << "dlsym infer_shape_on_compile failed: " << dlerror();
    return infer_handle;
  }
  return infer_handle;
}

ShapeInferenceRule ShapeInferenceRule::FromCompiledBinary(const std::vector<uint8_t> &binary) {
  return FromCompiledBinary(binary.data(), binary.size());
}

ge::graphStatus ShapeInferenceRule::CompileJsonString(const std::string &json_str, std::vector<uint8_t> &binary) {
  RuleJsonParser parser;
  const std::string error_msg = parser.ParseJson(json_str);
  if (!error_msg.empty()) {
    GELOGE(af::FAILED, "%s", error_msg.c_str());
    return ge::GRAPH_FAILED;
  }

  std::stringstream code_ss;
  code_ss << kHeader << std::endl;
  parser.CodegenInferShape(code_ss);

  CppJitCompiler compiler;
  binary = compiler.Compile(code_ss.str());
  if (binary.empty()) {
    GELOGE(af::FAILED, "Failed to compile C++ code to shared object:%s,\nError:%s", code_ss.str().c_str(),
           compiler.Error().c_str());
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DtypeInferenceRule::InferDtype(gert::InferDataTypeContext *infer_dtype_ctx) const {
  GE_ASSERT_NOTNULL(infer_dtype_ctx);
  if (!Error().empty()) {
    GELOGE(af::FAILED, "Failed to infer dtype by rule for op %s(%s): %s", infer_dtype_ctx->GetNodeName(),
           infer_dtype_ctx->GetNodeType(), Error().c_str());
    return ge::GRAPH_FAILED;
  }
  for (size_t i = 0U; i < dtypes_.size(); i++) {
    GE_ASSERT_GRAPH_SUCCESS(infer_dtype_ctx->SetOutputDataType(i, dtypes_[i]));
  }
  return ge::GRAPH_SUCCESS;
}

std::shared_ptr<DtypeInferenceRule> DtypeInferenceRule::FromOpDesc(const af::OpDescPtr &op) {
  const std::string *rule_json = af::AttrUtils::GetStr(op, af::ATTR_NAME_INFER_RULE);
  if (rule_json == nullptr) {
    // Skip log error if op does not with rule
    return nullptr;
  }
  return FromJsonString(*rule_json);
}

std::shared_ptr<DtypeInferenceRule> DtypeInferenceRule::FromJsonString(const std::string &json_str) {
  auto cached = g_dtype_rule_cache.Get(json_str);
  if (cached != nullptr) {
    return cached;
  }

  const auto rule = std::make_shared<DtypeInferenceRule>();
  Json rule_json;
  try {
    rule_json = Json::parse(json_str);
  } catch (const std::exception &e) {
    *rule << "Error parsing json: " << e.what();
    return g_dtype_rule_cache.GetWithDefault(json_str, rule);
  }

  if (!rule_json.contains("dtype")) {
    *rule << "Missing 'dtype' field in rule json.";
    return g_dtype_rule_cache.GetWithDefault(json_str, rule);
  }

  const auto dtype_json = rule_json["dtype"];
  if (dtype_json.is_null()) {
    *rule << "Field 'dtype' must not be null.";
    return g_dtype_rule_cache.GetWithDefault(json_str, rule);
  }

  if (!dtype_json.is_array()) {
    *rule << "Field 'dtype' must be an array.";
    return g_dtype_rule_cache.GetWithDefault(json_str, rule);
  }

  for (const auto &dtype : dtype_json) {
    if (dtype.is_null()) {
      *rule << "Element in 'dtype' field must not be null.";
      return g_dtype_rule_cache.GetWithDefault(json_str, rule);
    }
    if (!dtype.is_number_integer()) {
      *rule << "Element in 'dtype' field must be an integer.";
      return g_dtype_rule_cache.GetWithDefault(json_str, rule);
    }
    const int32_t dtype_value = dtype.get<int32_t>();
    if (dtype_value >= ge::DataType::DT_MAX || dtype_value < 0 || dtype_value == ge::DataType::DT_UNDEFINED) {
      *rule << "Element " << dtype_value << " in 'dtype' field is out of range [0," << ge::DataType::DT_MAX
            << "(DT_MAX)) and cannot be " << ge::DataType::DT_UNDEFINED << "(DT_UNDEFINED).";
      return g_dtype_rule_cache.GetWithDefault(json_str, rule);
    }
    rule->dtypes_.emplace_back(static_cast<ge::DataType>(dtype_value));
  }

  return g_dtype_rule_cache.GetWithDefault(json_str, rule);
}
}  // namespace af
