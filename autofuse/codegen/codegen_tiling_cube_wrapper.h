#include <string>

inline const std::string kCubeKernelTilingWrapperHppValue = R"(
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CUBE_KERNEL_TILING_WRAPPER_H
#define CUBE_KERNEL_TILING_WRAPPER_H

#include <string>
#include <vector>
#include <map>
#include <utility>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <algorithm>
#include "acl/acl.h"
#include "platform/platform_info.h"

namespace ge {
namespace autofuse {

namespace json_internal {

enum class Type {
    null,
    boolean,
    number_integer,
    number_float,
    string,
    array,
    object
};

class Json {
public:
    Json() : type_(Type::null) {}
    Json(bool value) : type_(Type::boolean), bool_value_(value) {}
    Json(int value) : type_(Type::number_integer), int_value_(value) {}
    Json(int64_t value) : type_(Type::number_integer), int_value_(value) {}
    Json(double value) : type_(Type::number_float), float_value_(value) {}
    Json(const char* value) : type_(Type::string), string_value_(new std::string(value)) {}
    Json(const std::string& value) : type_(Type::string), string_value_(new std::string(value)) {}
    Json(const std::vector<int64_t>& value) : type_(Type::array), array_value_(new std::vector<Json>()) {
        for (const auto& v : value) {
            array_value_->push_back(Json(v));
        }
    }
    Json(const std::vector<double>& value) : type_(Type::array), array_value_(new std::vector<Json>()) {
        for (const auto& v : value) {
            array_value_->push_back(Json(v));
        }
    }
    Json(const std::vector<std::string>& value) : type_(Type::array), array_value_(new std::vector<Json>()) {
        for (const auto& v : value) {
            array_value_->push_back(Json(v));
        }
    }

    Json(const Json& other) : type_(other.type_) {
        CopyValue(other);
    }

    Json(Json&& other) noexcept : type_(other.type_) {
        MoveValue(std::move(other));
        other.type_ = Type::null;
    }

    Json& operator=(const Json& other) {
        if (this != &other) {
            Clear();
            type_ = other.type_;
            CopyValue(other);
        }
        return *this;
    }

    Json& operator=(Json&& other) noexcept {
        if (this != &other) {
            Clear();
            type_ = other.type_;
            MoveValue(std::move(other));
            other.type_ = Type::null;
        }
        return *this;
    }

    ~Json() {
        Clear();
    }

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::null; }
    bool is_boolean() const { return type_ == Type::boolean; }
    bool is_number() const { return type_ == Type::number_integer || type_ == Type::number_float; }
    bool is_string() const { return type_ == Type::string; }
    bool is_array() const { return type_ == Type::array; }
    bool is_object() const { return type_ == Type::object; }

    bool get_bool() const {
        if (type_ != Type::boolean) throw std::runtime_error("Json is not a boolean");
        return bool_value_;
    }

    int64_t get_int64() const {
        if (type_ == Type::number_integer) return int_value_;
        if (type_ == Type::number_float) return static_cast<int64_t>(float_value_);
        throw std::runtime_error("Json is not a number");
    }

    int get_int() const {
        return static_cast<int>(get_int64());
    }

    double get_double() const {
        if (type_ == Type::number_float) return float_value_;
        if (type_ == Type::number_integer) return static_cast<double>(int_value_);
        throw std::runtime_error("Json is not a number");
    }

    std::string get_string() const {
        if (type_ != Type::string) throw std::runtime_error("Json is not a string");
        return *string_value_;
    }

    std::vector<int64_t> get_int64_array() const {
        if (type_ != Type::array) throw std::runtime_error("Json is not an array");
        std::vector<int64_t> result;
        for (const auto& item : *array_value_) {
            result.push_back(item.get_int64());
        }
        return result;
    }

    std::vector<double> get_double_array() const {
        if (type_ != Type::array) throw std::runtime_error("Json is not an array");
        std::vector<double> result;
        for (const auto& item : *array_value_) {
            result.push_back(item.get_double());
        }
        return result;
    }

    std::vector<std::string> get_string_array() const {
        if (type_ != Type::array) throw std::runtime_error("Json is not an array");
        std::vector<std::string> result;
        for (const auto& item : *array_value_) {
            result.push_back(item.get_string());
        }
        return result;
    }

    Json& operator[](const std::string& key) {
        if (type_ == Type::null) {
            type_ = Type::object;
            object_value_ = new std::map<std::string, Json>();
        }
        if (type_ != Type::object) throw std::runtime_error("Json is not an object");
        return (*object_value_)[key];
    }

    const Json& operator[](const std::string& key) const {
        if (type_ != Type::object) throw std::runtime_error("Json is not an object");
        static const Json null_json;
        auto it = object_value_->find(key);
        if (it == object_value_->end()) return null_json;
        return it->second;
    }

    Json& operator[](size_t index) {
        if (type_ != Type::array) throw std::runtime_error("Json is not an array");
        if (index >= array_value_->size()) throw std::runtime_error("Array index out of bounds");
        return (*array_value_)[index];
    }

    const Json& operator[](size_t index) const {
        if (type_ != Type::array) throw std::runtime_error("Json is not an array");
        if (index >= array_value_->size()) throw std::runtime_error("Array index out of bounds");
        return (*array_value_)[index];
    }

    bool contains(const std::string& key) const {
        if (type_ != Type::object) return false;
        return object_value_->find(key) != object_value_->end();
    }

    void push_back(const Json& value) {
        if (type_ == Type::null) {
            type_ = Type::array;
            array_value_ = new std::vector<Json>();
        }
        if (type_ != Type::array) throw std::runtime_error("Json is not an array");
        array_value_->push_back(value);
    }

    void push_back(Json&& value) {
        if (type_ == Type::null) {
            type_ = Type::array;
            array_value_ = new std::vector<Json>();
        }
        if (type_ != Type::array) throw std::runtime_error("Json is not an array");
        array_value_->push_back(std::move(value));
    }

    size_t size() const {
        if (type_ == Type::array) return array_value_->size();
        if (type_ == Type::object) return object_value_->size();
        return 0;
    }

    std::string dump(int indent = -1) const {
        std::ostringstream oss;
        Dump(oss, indent, 0);
        return oss.str();
    }

    static Json parse(const std::string& str) {
        Parser parser(str);
        return parser.Parse();
    }

    static Json array() {
        Json j;
        j.type_ = Type::array;
        j.array_value_ = new std::vector<Json>();
        return j;
    }

    static Json object() {
        Json j;
        j.type_ = Type::object;
        j.object_value_ = new std::map<std::string, Json>();
        return j;
    }

private:
    Type type_;

    union {
        bool bool_value_;
        int64_t int_value_;
        double float_value_;
        std::string* string_value_;
        std::vector<Json>* array_value_;
        std::map<std::string, Json>* object_value_;
    };

    void Clear() {
        switch (type_) {
            case Type::string:
                delete string_value_;
                break;
            case Type::array:
                delete array_value_;
                break;
            case Type::object:
                delete object_value_;
                break;
            default:
                break;
        }
    }

    void CopyValue(const Json& other) {
        switch (other.type_) {
            case Type::null:
                break;
            case Type::boolean:
                bool_value_ = other.bool_value_;
                break;
            case Type::number_integer:
                int_value_ = other.int_value_;
                break;
            case Type::number_float:
                float_value_ = other.float_value_;
                break;
            case Type::string:
                string_value_ = new std::string(*other.string_value_);
                break;
            case Type::array:
                array_value_ = new std::vector<Json>(*other.array_value_);
                break;
            case Type::object:
                object_value_ = new std::map<std::string, Json>(*other.object_value_);
                break;
        }
    }

    void MoveValue(Json&& other) {
        switch (other.type_) {
            case Type::null:
                break;
            case Type::boolean:
                bool_value_ = other.bool_value_;
                break;
            case Type::number_integer:
                int_value_ = other.int_value_;
                break;
            case Type::number_float:
                float_value_ = other.float_value_;
                break;
            case Type::string:
                string_value_ = other.string_value_;
                other.string_value_ = nullptr;
                break;
            case Type::array:
                array_value_ = other.array_value_;
                other.array_value_ = nullptr;
                break;
            case Type::object:
                object_value_ = other.object_value_;
                other.object_value_ = nullptr;
                break;
        }
    }

    void Dump(std::ostringstream& oss, int indent, int level) const {
        std::string indent_str;
        if (indent > 0) {
            indent_str = std::string(level * indent, ' ');
        }

        switch (type_) {
            case Type::null:
                oss << "null";
                break;
            case Type::boolean:
                oss << (bool_value_ ? "true" : "false");
                break;
            case Type::number_integer:
                oss << int_value_;
                break;
            case Type::number_float:
                oss << float_value_;
                break;
            case Type::string:
                oss << "\"" << EscapeString(*string_value_) << "\"";
                break;
            case Type::array:
                oss << "[";
                if (indent > 0 && !array_value_->empty()) {
                    oss << "\n";
                }
                for (size_t i = 0; i < array_value_->size(); ++i) {
                    if (indent > 0) {
                        oss << indent_str << std::string(indent, ' ');
                    }
                    (*array_value_)[i].Dump(oss, indent, level + 1);
                    if (i < array_value_->size() - 1) {
                        oss << ",";
                    }
                    if (indent > 0) {
                        oss << "\n";
                    }
                }
                if (indent > 0 && !array_value_->empty()) {
                    oss << indent_str;
                }
                oss << "]";
                break;
            case Type::object:
                oss << "{";
                if (indent > 0 && !object_value_->empty()) {
                    oss << "\n";
                }
                auto it = object_value_->begin();
                for (size_t i = 0; i < object_value_->size(); ++i, ++it) {
                    if (indent > 0) {
                        oss << indent_str << std::string(indent, ' ');
                    }
                    oss << "\"" << it->first << "\":";
                    if (indent > 0) {
                        oss << " ";
                    }
                    it->second.Dump(oss, indent, level + 1);
                    if (i < object_value_->size() - 1) {
                        oss << ",";
                    }
                    if (indent > 0) {
                        oss << "\n";
                    }
                }
                if (indent > 0 && !object_value_->empty()) {
                    oss << indent_str;
                }
                oss << "}";
                break;
        }
    }

    static std::string EscapeString(const std::string& str) {
        std::string result;
        for (char c : str) {
            switch (c) {
                case '"': result += "\\\""; break;
                case '\\': result += "\\\\"; break;
                case '\b': result += "\\b"; break;
                case '\f': result += "\\f"; break;
                case '\n': result += "\\n"; break;
                case '\r': result += "\\r"; break;
                case '\t': result += "\\t"; break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20) {
                        char buf[7];
                        snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                        result += buf;
                    } else {
                        result += c;
                    }
                    break;
            }
        }
        return result;
    }

    class Parser {
    public:
        Parser(const std::string& str) : str_(str), pos_(0) {
            SkipWhitespace();
        }

        Json Parse() {
            if (pos_ >= str_.size()) {
                throw std::runtime_error("Empty JSON string");
            }
            return ParseValue();
        }

    private:
        const std::string& str_;
        size_t pos_;

        void SkipWhitespace() {
            while (pos_ < str_.size() && (str_[pos_] == ' ' || str_[pos_] == '\t' ||
                                          str_[pos_] == '\n' || str_[pos_] == '\r')) {
                ++pos_;
            }
        }

        char Peek() const {
            if (pos_ >= str_.size()) return '\0';
            return str_[pos_];
        }

        char Consume() {
            if (pos_ >= str_.size()) return '\0';
            return str_[pos_++];
        }

        Json ParseValue() {
            SkipWhitespace();
            char c = Peek();

            if (c == 'n') return ParseNull();
            if (c == 't' || c == 'f') return ParseBoolean();
            if (c == '"') return ParseString();
            if (c == '[') return ParseArray();
            if (c == '{') return ParseObject();
            if (c == '-' || (c >= '0' && c <= '9')) return ParseNumber();

            throw std::runtime_error(std::string("Unexpected character: ") + c);
        }

        Json ParseNull() {
            if (str_.substr(pos_, 4) == "null") {
                pos_ += 4;
                return Json();
            }
            throw std::runtime_error("Expected 'null'");
        }

        Json ParseBoolean() {
            if (str_.substr(pos_, 4) == "true") {
                pos_ += 4;
                return Json(true);
            }
            if (str_.substr(pos_, 5) == "false") {
                pos_ += 5;
                return Json(false);
            }
            throw std::runtime_error("Expected 'true' or 'false'");
        }

        Json ParseNumber() {
            size_t start = pos_;
            if (Peek() == '-') Consume();

            while (pos_ < str_.size() && (str_[pos_] >= '0' && str_[pos_] <= '9')) {
                ++pos_;
            }

            bool is_float = false;
            if (pos_ < str_.size() && str_[pos_] == '.') {
                is_float = true;
                ++pos_;
                while (pos_ < str_.size() && (str_[pos_] >= '0' && str_[pos_] <= '9')) {
                    ++pos_;
                }
            }

            if (pos_ < str_.size() && (str_[pos_] == 'e' || str_[pos_] == 'E')) {
                is_float = true;
                ++pos_;
                if (pos_ < str_.size() && (str_[pos_] == '+' || str_[pos_] == '-')) {
                    ++pos_;
                }
                while (pos_ < str_.size() && (str_[pos_] >= '0' && str_[pos_] <= '9')) {
                    ++pos_;
                }
            }

            std::string num_str = str_.substr(start, pos_ - start);
            if (is_float) {
                return Json(std::stod(num_str));
            } else {
                return Json(static_cast<int64_t>(std::stoll(num_str)));
            }
        }

        Json ParseString() {
            if (Consume() != '"') {
                throw std::runtime_error("Expected '\"'");
            }

            std::string result;
            while (pos_ < str_.size() && str_[pos_] != '"') {
                if (str_[pos_] == '\\') {
                    ++pos_;
                    if (pos_ >= str_.size()) {
                        throw std::runtime_error("Unexpected end of string");
                    }
                    switch (str_[pos_]) {
                        case '"': result += '"'; break;
                        case '\\': result += '\\'; break;
                        case '/': result += '/'; break;
                        case 'b': result += '\b'; break;
                        case 'f': result += '\f'; break;
                        case 'n': result += '\n'; break;
                        case 'r': result += '\r'; break;
                        case 't': result += '\t'; break;
                        case 'u': {
                            if (pos_ + 4 >= str_.size()) {
                                throw std::runtime_error("Invalid unicode escape");
                            }
                            std::string hex_str = str_.substr(pos_ + 1, 4);
                            unsigned int codepoint = std::stoul(hex_str, nullptr, 16);
                            if (codepoint < 0x80) {
                                result += static_cast<char>(codepoint);
                            } else if (codepoint < 0x800) {
                                result += static_cast<char>(0xC0 | (codepoint >> 6));
                                result += static_cast<char>(0x80 | (codepoint & 0x3F));
                            } else {
                                result += static_cast<char>(0xE0 | (codepoint >> 12));
                                result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
                                result += static_cast<char>(0x80 | (codepoint & 0x3F));
                            }
                            pos_ += 4;
                            break;
                        }
                        default:
                            throw std::runtime_error("Invalid escape sequence");
                    }
                } else {
                    result += str_[pos_];
                }
                ++pos_;
            }

            if (pos_ >= str_.size() || Consume() != '"') {
                throw std::runtime_error("Unterminated string");
            }

            return Json(result);
        }

        Json ParseArray() {
            if (Consume() != '[') {
                throw std::runtime_error("Expected '['");
            }

            Json result = Json::array();
            SkipWhitespace();

            if (Peek() == ']') {
                Consume();
                return result;
            }

            while (true) {
                result.push_back(ParseValue());
                SkipWhitespace();

                if (Peek() == ']') {
                    Consume();
                    return result;
                }

                if (Peek() == ',') {
                    Consume();
                } else {
                    throw std::runtime_error("Expected ',' or ']' in array");
                }
            }
        }

        Json ParseObject() {
            if (Consume() != '{') {
                throw std::runtime_error("Expected '{'");
            }

            Json result = Json::object();
            SkipWhitespace();

            if (Peek() == '}') {
                Consume();
                return result;
            }

            while (true) {
                SkipWhitespace();
                Json key = ParseString();
                SkipWhitespace();

                if (Consume() != ':') {
                    throw std::runtime_error("Expected ':' after key");
                }

                Json value = ParseValue();
                result[key.get_string()] = std::move(value);
                SkipWhitespace();

                if (Peek() == '}') {
                    Consume();
                    return result;
                }

                if (Peek() == ',') {
                    Consume();
                } else {
                    throw std::runtime_error("Expected ',' or '}' in object");
                }
            }
        }
    };
};

} // namespace json_internal

namespace crypto {

class SHA1 {
public:
    static constexpr size_t DIGEST_LENGTH = 20;

    static std::string Hash(const std::string& input) {
        SHA1 sha1;
        sha1.Update(reinterpret_cast<const uint8_t*>(input.c_str()), input.length());
        uint8_t digest[DIGEST_LENGTH];
        sha1.Final(digest);
        return DigestToHex(digest);
    }

private:
    SHA1() {
        Reset();
    }

    void Reset() {
        m_digest[0] = 0x67452301;
        m_digest[1] = 0xEFCDAB89;
        m_digest[2] = 0x98BADCFE;
        m_digest[3] = 0x10325476;
        m_digest[4] = 0xC3D2E1F0;
        m_block_len = 0;
        m_total_len = 0;
    }

    void Update(const uint8_t* data, size_t len) {
        while (len) {
            size_t copy_len = std::min(len, 64 - m_block_len);
            std::memcpy(m_block + m_block_len, data, copy_len);

            m_block_len += copy_len;
            m_total_len += copy_len;
            data += copy_len;
            len -= copy_len;

            if (m_block_len == 64) {
                ProcessBlock(m_block);
                m_block_len = 0;
            }
        }
    }

    void Final(uint8_t* digest) {
        uint64_t total_bits = m_total_len * 8;

        m_block[m_block_len++] = 0x80;
        if (m_block_len > 56) {
            while (m_block_len < 64) {
                m_block[m_block_len++] = 0;
            }
            ProcessBlock(m_block);
            m_block_len = 0;
        }

        while (m_block_len < 56) {
            m_block[m_block_len++] = 0;
        }

        for (int i = 7; i >= 0; --i) {
            m_block[m_block_len++] = static_cast<uint8_t>((total_bits >> (i * 8)) & 0xFF);
        }

        ProcessBlock(m_block);

        for (int i = 0; i < 5; ++i) {
            digest[i * 4 + 0] = static_cast<uint8_t>((m_digest[i] >> 24) & 0xFF);
            digest[i * 4 + 1] = static_cast<uint8_t>((m_digest[i] >> 16) & 0xFF);
            digest[i * 4 + 2] = static_cast<uint8_t>((m_digest[i] >> 8) & 0xFF);
            digest[i * 4 + 3] = static_cast<uint8_t>(m_digest[i] & 0xFF);
        }
    }

    void ProcessBlock(const uint8_t* block) {
        uint32_t w[80];

        for (int i = 0; i < 16; ++i) {
            w[i] = (block[i * 4 + 0] << 24) | (block[i * 4 + 1] << 16) |
                   (block[i * 4 + 2] << 8) | block[i * 4 + 3];
        }

        for (int i = 16; i < 80; ++i) {
            uint32_t temp = w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16];
            w[i] = ROTL(temp, 1);
        }

        uint32_t a = m_digest[0];
        uint32_t b = m_digest[1];
        uint32_t c = m_digest[2];
        uint32_t d = m_digest[3];
        uint32_t e = m_digest[4];

        for (int i = 0; i < 80; ++i) {
            uint32_t f, k;

            if (i < 20) {
                f = (b & c) | ((~b) & d);
                k = 0x5A827999;
            } else if (i < 40) {
                f = b ^ c ^ d;
                k = 0x6ED9EBA1;
            } else if (i < 60) {
                f = (b & c) | (b & d) | (c & d);
                k = 0x8F1BBCDC;
            } else {
                f = b ^ c ^ d;
                k = 0xCA62C1D6;
            }

            uint32_t temp = ROTL(a, 5) + f + e + k + w[i];
            e = d;
            d = c;
            c = ROTL(b, 30);
            b = a;
            a = temp;
        }

        m_digest[0] += a;
        m_digest[1] += b;
        m_digest[2] += c;
        m_digest[3] += d;
        m_digest[4] += e;
    }

    static uint32_t ROTL(uint32_t x, uint32_t n) {
        return (x << n) | (x >> (32 - n));
    }

    static std::string DigestToHex(const uint8_t* digest) {
        std::ostringstream oss;
        oss << std::hex << std::setfill('0');
        for (size_t i = 0; i < DIGEST_LENGTH; ++i) {
            oss << std::setw(2) << static_cast<int>(digest[i]);
        }
        return oss.str();
    }

    uint32_t m_digest[5];
    uint8_t m_block[64];
    size_t m_block_len;
    uint64_t m_total_len;
};

} // namespace crypto

using Json = json_internal::Json;

#include "arch35/mat_mul_tiling_data.h"

struct TensorInfo {
    std::string param_name;
    std::vector<int64_t> shape;
    std::vector<int64_t> ori_shape;
    std::string dtype;
    std::string format;
    std::string name;
    int64_t range_start = 0;
    int64_t range_end = 0;
};

struct AttrInfo {
    std::string name;
    std::string dtype;
    std::string value_str;
    bool value_bool = false;
    int64_t value_int = 0;
    double value_float = 0.0;
    std::vector<int64_t> value_list_int;
    std::vector<double> value_list_float;
    std::vector<std::string> value_list_str;
    bool is_list = false;
};

struct CompileInfo {
    std::string soc_version;
    std::string core_type;
    std::string device_id;
    std::string op_kernel_lib;
    std::string op_impl_mode;
    int64_t aicore_num = 0;
    int64_t aiv_num = 0;
    std::map<std::string, std::string> extra_info;
};

struct TilingResult {
    std::vector<uint8_t> tiling_data;
    int64_t tiling_key = 0;
    int64_t block_dim = 0;
    int64_t workspace_size = 0;
    uint32_t cube_used_core_num = 1;
    uint32_t cube_base_m = 16;
    uint32_t cube_base_n = 16;
    bool atomic_flag = false;
    std::string error_msg;
    bool success = false;

    BatchMatMulV3BasicTilingData batch_matmul_tiling_data;
    MatMulV3BasicTilingData matmul_basic_tiling_data;
};

class CubeKernelTilingWrapper {
public:
    CubeKernelTilingWrapper();
    ~CubeKernelTilingWrapper();

    TilingResult DoMatMulTiling(const CompileInfo& compile_info,
                                const std::vector<TensorInfo>& inputs,
                                const std::vector<TensorInfo>& outputs,
                                const std::vector<AttrInfo>& attrs,
                                bool is_batch = false);

    static void BuildMatMulArgs(const std::vector<TensorInfo>& args_list,
                               int input_num,
                               bool transpose_a,
                               bool transpose_b,
                               std::vector<TensorInfo>& origin_inputs,
                               std::vector<TensorInfo>& origin_outputs,
                               std::vector<TensorInfo>& inputs);

    static std::string GenerateCompileInfoHash(const std::string& compile_info_info);

    static void ChangeParamNameToName(std::vector<TensorInfo>& inputs);
    static void InputsPreProcess(std::vector<TensorInfo>& inputs);
    static void AttrsPreProcess(std::vector<AttrInfo>& attrs);
    static std::vector<uint8_t> AlignTilingDataTo8Bytes(const std::vector<uint8_t>& tiling_data, const std::string& soc_version);

private:
    static std::string SerializeToJson(const CompileInfo& compile_info);
    static std::string SerializeToJson(const std::vector<TensorInfo>& tensors);
    static std::string SerializeToJson(const std::vector<AttrInfo>& attrs);
    static std::string SerializeToJson(const std::map<std::string, std::string>& extra_params);

    static bool ParseTilingResult(const std::string& json_str, TilingResult& result);

    char* CallDoOpTilingForCompile(const char* op_type,
                                           const char* compile_info,
                                           const char* compile_info_hash,
                                           const char* inputs,
                                           const char* outputs,
                                           const char* attrs,
                                           char* buf,
                                           size_t buf_size,
                                           uint64_t* timer,
                                           const char* extra_params);
};

} // namespace autofuse
} // namespace ge

#endif // CUBE_KERNEL_TILING_WRAPPER_H
)";

inline const std::string kCubeKernelTilingWrapperCppValue = R"(
#include <sstream>
#include <iomanip>
#include <dlfcn.h>
#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <cmath>
#include <limits>

using json = ge::autofuse::Json;
using SHA1 = ge::autofuse::crypto::SHA1;

#ifndef DEFAULT_ASCEND_OPP_PATH
#define DEFAULT_ASCEND_OPP_PATH "/usr/local/Ascend/cann/opp"
#endif

extern "C" const char *DoOpTilingForCompile(const char *optype,
                                            const char *compile_info,
                                            const char *compile_info_hash,
                                            const char *inputs,
                                            const char *outputs,
                                            const char *attrs,
                                            char *run_info_json,
                                            size_t run_info_len,
                                            uint64_t *elapse,
                                            const char *extra_info);

namespace ge {
namespace autofuse {

CubeKernelTilingWrapper::CubeKernelTilingWrapper() {}

CubeKernelTilingWrapper::~CubeKernelTilingWrapper() {}

void CubeKernelTilingWrapper::BuildMatMulArgs(const std::vector<TensorInfo>& args_list,
                                               int input_num,
                                               bool transpose_a,
                                               bool transpose_b,
                                               std::vector<TensorInfo>& origin_inputs,
                                               std::vector<TensorInfo>& origin_outputs,
                                               std::vector<TensorInfo>& inputs) {
    origin_inputs.clear();
    origin_outputs.clear();
    inputs.clear();

    int64_t m = 0;
    int64_t n = 0;
    std::vector<int64_t> write_shape;

    for (int i = 0; i < input_num && i < static_cast<int>(args_list.size()); ++i) {
        TensorInfo input = args_list[i];
        input.param_name = "input" + std::to_string(i);
        input.ori_shape = input.shape;

        origin_inputs.push_back(input);
        inputs.push_back(input);

        if (i == 0) {
            write_shape = input.shape;
            if (transpose_a) {
                m = input.shape[input.shape.size() - 1];
            } else {
                m = input.shape[input.shape.size() - 2];
            }
        } else if (i == 1) {
            if (transpose_b) {
                n = input.shape[input.shape.size() - 2];
            } else {
                n = input.shape[input.shape.size() - 1];
            }
        }
    }

    if (args_list.size() >= 2) {
        TensorInfo output = args_list[args_list.size() - 2];
        output.param_name = "output0";
        if (!write_shape.empty()) {
            write_shape[write_shape.size() - 1] = n;
            write_shape[write_shape.size() - 2] = m;
            output.shape = write_shape;
            output.ori_shape = write_shape;
        }
        if (!inputs.empty()) {
            output.dtype = inputs.back().dtype;
        }
        origin_outputs.push_back(output);
    }
}

std::string CubeKernelTilingWrapper::SerializeToJson(const CompileInfo& compile_info) {
    json j;
    j["soc_version"] = compile_info.soc_version;
    j["core_type"] = compile_info.core_type;
    j["device_id"] = compile_info.device_id;
    j["op_kernel_lib"] = compile_info.op_kernel_lib;
    j["op_impl_mode"] = compile_info.op_impl_mode;
    j["aicore_num"] = compile_info.aicore_num;
    j["aiv_num"] = compile_info.aiv_num;

    if (!compile_info.extra_info.empty()) {
        json extra;
        for (const auto& pair : compile_info.extra_info) {
            extra[pair.first] = pair.second;
        }
        j["extra_info"] = extra;
    }

    return j.dump();
}

std::string CubeKernelTilingWrapper::SerializeToJson(const std::vector<TensorInfo>& tensors) {
    json j = json::array();
    for (const auto& tensor : tensors) {
        json t;
        t["param_name"] = tensor.param_name;
        t["shape"] = tensor.shape;
        t["ori_shape"] = tensor.ori_shape;
        t["dtype"] = tensor.dtype;
        t["format"] = tensor.format;
        t["name"] = tensor.name;
        t["range_start"] = tensor.range_start;
        t["range_end"] = tensor.range_end;
        j.push_back(t);
    }
    return j.dump();
}

std::string CubeKernelTilingWrapper::SerializeToJson(const std::vector<AttrInfo>& attrs) {
    json j = json::array();
    for (const auto& attr : attrs) {
        json a;
        a["name"] = attr.name;
        a["dtype"] = attr.dtype;

        if (attr.is_list) {
            if (attr.dtype == "list_int" || attr.dtype == "list_int32") {
                a["value"] = attr.value_list_int;
            } else if (attr.dtype == "list_float" || attr.dtype == "list_float32") {
                a["value"] = attr.value_list_float;
            } else if (attr.dtype == "list_str") {
                a["value"] = attr.value_list_str;
            }
        } else {
            if (attr.dtype == "bool") {
                a["value"] = attr.value_bool;
            } else if (attr.dtype == "int" || attr.dtype == "int32" || attr.dtype == "int64") {
                a["value"] = attr.value_int;
            } else if (attr.dtype == "float" || attr.dtype == "float32" || attr.dtype == "float64") {
                a["value"] = attr.value_float;
            } else {
                a["value"] = attr.value_str;
            }
        }
        j.push_back(a);
    }
    return j.dump();
}

std::string CubeKernelTilingWrapper::SerializeToJson(const std::map<std::string, std::string>& extra_params) {
    json j;
    for (const auto& pair : extra_params) {
        j[pair.first] = pair.second;
    }
    return j.dump();
}

std::string CubeKernelTilingWrapper::GenerateCompileInfoHash(const std::string& compile_info_json) {
    return SHA1::Hash(compile_info_json);
}

void CubeKernelTilingWrapper::ChangeParamNameToName(std::vector<TensorInfo>& inputs) {
    for (auto& input : inputs) {
        if (input.name.empty() && !input.param_name.empty()) {
            input.name = input.param_name;
        }
    }
}

void CubeKernelTilingWrapper::InputsPreProcess(std::vector<TensorInfo>& inputs) {
    for (auto& input : inputs) {
        if (input.range_start != 0 || input.range_end != 0) {
            if (input.range_start == std::numeric_limits<int64_t>::min() ||
                input.range_start == std::numeric_limits<int64_t>::max()) {
                input.range_start = 0;
            }
            if (input.range_end == std::numeric_limits<int64_t>::min() ||
                input.range_end == std::numeric_limits<int64_t>::max()) {
                input.range_end = 0;
            }
        }
    }
}

void CubeKernelTilingWrapper::AttrsPreProcess(std::vector<AttrInfo>& attrs) {
    for (auto& attr : attrs) {
        if (attr.dtype == "float" || attr.dtype == "float32" || attr.dtype == "float64") {
            if (!attr.is_list) {
                if (std::isinf(attr.value_float)) {
                    if (attr.value_float > 0) {
                        attr.value_str = "float(1.0 / 0.0) ";
                    } else {
                        attr.value_str = "float(-1.0 / 0.0) ";
                    }
                } else if (std::isnan(attr.value_float)) {
                    attr.value_str = "float(0.0 / 0.0) ";
                }
            } else {
                for (auto& val : attr.value_list_float) {
                    if (std::isinf(val)) {
                        if (val > 0) {
                            val = std::numeric_limits<double>::max();
                        } else {
                            val = std::numeric_limits<double>::min();
                        }
                    } else if (std::isnan(val)) {
                        val = 0.0;
                    }
                }
            }
        } else if (attr.dtype == "list_float" || attr.dtype == "list_float32") {
            for (auto& val : attr.value_list_float) {
                if (std::isinf(val)) {
                    if (val > 0) {
                        val = std::numeric_limits<double>::max();
                    } else {
                        val = std::numeric_limits<double>::min();
                    }
                } else if (std::isnan(val)) {
                    val = 0.0;
                }
            }
        }
    }
}

std::vector<uint8_t> CubeKernelTilingWrapper::AlignTilingDataTo8Bytes(const std::vector<uint8_t>& tiling_data, const std::string& soc_version) {
    size_t original_size = tiling_data.size();
    std::vector<uint8_t> aligned_data = tiling_data;

    if (soc_version == "Ascend310P") {
        return aligned_data;
    }

    size_t aligned_size = ((original_size + 7) / 8) * 8;
    size_t padding_size = aligned_size - original_size;

    aligned_data.resize(aligned_size, 0);

    return aligned_data;
}

bool CubeKernelTilingWrapper::ParseTilingResult(const std::string& json_str, TilingResult& result) {
    try {
        json j = json::parse(json_str);

        if (j.contains("ret_code") && j["ret_code"].get_int64() != 0) {
            result.success = false;
            if (j.contains("error_messages") && j["error_messages"].is_array()) {
                for (size_t i = 0; i < j["error_messages"].size(); ++i) {
                    const auto& err = j["error_messages"][i];
                    if (err.contains("errormsg")) {
                        result.error_msg += err["errormsg"].get_string() + "; ";
                    }
                }
            }
            return false;
        }
        result.success = true;
        return true;
    } catch (const std::exception& e) {
        result.success = false;
        result.error_msg = std::string("Parse JSON failed: ") + e.what();
        return false;
    }
}

char* CubeKernelTilingWrapper::CallDoOpTilingForCompile(const char* op_type,
                                                        const char* compile_info,
                                                        const char* compile_info_hash,
                                                        const char* inputs,
                                                        const char* outputs,
                                                        const char* attrs,
                                                        char* buf,
                                                        size_t buf_size,
                                                        uint64_t* timer,
                                                        const char* extra_params) {
    fe::PlatFormInfos platform_infos;
    fe::OptionalInfos q;
    const char* soc_name = aclrtGetSocName();
    std::string soc_ver = soc_name ? soc_name : "";
    q.Init();
    q.SetSocVersion(soc_ver);
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(q); // 这两个SetOptionalCompilationInfo都得有
    fe::PlatformInfoManager::GeInstance().SetOptionalCompilationInfo(q); // 否则获取soc version失败tiling计算出错
    fe::PlatformInfoManager::GeInstance().InitRuntimePlatformInfos(soc_ver);
    int32_t device_id = 0;
    aclrtGetDevice(&device_id);
    if (fe::PlatformInfoManager::GeInstance().GetRuntimePlatformInfosByDevice(device_id, platform_infos) != 0) {
        OP_LOGE(OP_NAME, "GetRuntimePlatformInfosByDevice failed");
        return nullptr;
    }
    OP_LOGI(OP_NAME, "Calling DoOpTilingForCompile...");
    const char* result = DoOpTilingForCompile(op_type, compile_info, compile_info_hash,
                        inputs, outputs, attrs, buf, buf_size, timer, extra_params);
    OP_LOGI(OP_NAME, "DoOpTilingForCompile returned");

    return const_cast<char*>(result);
}

TilingResult CubeKernelTilingWrapper::DoMatMulTiling(const CompileInfo& compile_info,
                                                       const std::vector<TensorInfo>& inputs,
                                                       const std::vector<TensorInfo>& outputs,
                                                       const std::vector<AttrInfo>& attrs,
                                                       bool is_batch) {
    TilingResult result;

    std::vector<TensorInfo> processed_inputs = inputs;
    std::vector<AttrInfo> processed_attrs = attrs;

    ChangeParamNameToName(processed_inputs);
    InputsPreProcess(processed_inputs);
    AttrsPreProcess(processed_attrs);

    std::string compile_info_json = SerializeToJson(compile_info);
    std::string inputs_json = SerializeToJson(processed_inputs);
    std::string outputs_json = SerializeToJson(outputs);
    std::string attrs_json = SerializeToJson(processed_attrs);

    std::string compile_info_hash = GenerateCompileInfoHash(compile_info_json);

    json extra_params;
    extra_params["op_name"] = is_batch ? "BatchMatMulV3" : "MatMulV3";
    extra_params["deterministic"] = false;
    std::string extra_params_json = extra_params.dump();

    std::string op_type = is_batch ? "BatchMatMulV3" : "MatMulV3";

    const size_t buf_size = 1024 * 64;
    std::vector<char> buf(buf_size, 0);

    char* ret = CallDoOpTilingForCompile(op_type.c_str(),
                                           compile_info_json.c_str(),
                                           compile_info_hash.c_str(),
                                           inputs_json.c_str(),
                                           outputs_json.c_str(),
                                           attrs_json.c_str(),
                                           buf.data(),
                                           buf_size,
                                           nullptr,
                                           extra_params_json.c_str());

    if (ret == nullptr) {
        result.success = false;
        result.error_msg = "DoOpTilingForCompile returned nullptr";
        return (result);
    }

    std::string ret_json(ret);
    ParseTilingResult(ret_json, result);

    if (result.success) {
        std::string buf_json(buf.data());
        json j = json::parse(buf_json);
        if (j.contains("tiling_data")) {
            std::string hex_str = j["tiling_data"].get_string();
            result.tiling_data.clear();
            for (size_t i = 0; i < hex_str.length(); i += 2) {
                std::string byte_str = hex_str.substr(i, 2);
                result.tiling_data.push_back(static_cast<uint8_t>(std::stoul(byte_str, nullptr, 16)));
            }

            result.tiling_data = AlignTilingDataTo8Bytes(result.tiling_data, compile_info.soc_version);

            if (result.tiling_data.size() >= sizeof(MatMulV3BasicTilingData)) {
                memcpy(&result.matmul_basic_tiling_data, result.tiling_data.data(), sizeof(MatMulV3BasicTilingData));
                result.cube_used_core_num = std::max(result.matmul_basic_tiling_data.usedCoreNum, 1U);
                result.cube_base_m = std::max(result.matmul_basic_tiling_data.baseM, 1U);
                result.cube_base_n = std::max(result.matmul_basic_tiling_data.baseN, 1U);
            }
            if (result.tiling_data.size() >= sizeof(BatchMatMulV3BasicTilingData)) {
                memcpy(&result.batch_matmul_tiling_data, result.tiling_data.data(), sizeof(BatchMatMulV3BasicTilingData));
                if (is_batch) {
                    result.cube_used_core_num = std::max(result.batch_matmul_tiling_data.matMulTilingData.usedCoreNum, 1U);
                    result.cube_base_m = std::max(result.batch_matmul_tiling_data.matMulTilingData.baseM, 1U);
                    result.cube_base_n = std::max(result.batch_matmul_tiling_data.matMulTilingData.baseN, 1U);
                }
            }
        }
        if (j.contains("tiling_key")) {
            result.tiling_key = j["tiling_key"].get_int64();
        }
        if (j.contains("block_dim")) {
            result.block_dim = j["block_dim"].get_int64();
        }
        if (j.contains("workspaces")) {
            result.workspace_size = j["workspaces"][0].get_int64();
        }
        if (j.contains("clear_atomic")) {
            result.atomic_flag = j["clear_atomic"].get_bool();
        }
    }

    return result;
}

} // namespace autofuse
} // namespace ge

)";
