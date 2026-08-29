/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "select_model/stub_regex.h"
namespace att {
bool CheckValidTilingkey() {
  bool has_visited = false;
  std::string str;
  std::string line;
  std::ifstream file("./info.log");
  std::map<uint64_t, double> myMap;

  if (file.is_open()) {
    while (getline(file, line)) {
      str += line + "\n";
    }
    file.close();
  }
  std::regex pattern("The optimal objective for tilingCaseId (\\d+) is (\\d+).");
  std::sregex_iterator it(str.begin(), str.end(), pattern);
  std::sregex_iterator end;
  while (it != end) {
    uint64_t key = stoi(it->str(1));
    double value = stoi(it->str(2));
    myMap[key] = value;
    it++;
  }

  std::regex pattern2("tiling_key = (\\d+)");
  std::sregex_iterator it2(str.begin(), str.end(), pattern2);
  std::sregex_iterator end2;
  if (it2 == end2) {
    return false;
  }
  uint64_t used_key = stoi(it2->str(1));

  for (auto it = myMap.begin(); it != myMap.end(); ++it) {
    if (it->first == used_key) {
      has_visited = true;
      break;
    }
  }
  return has_visited;
}

bool ExistOutput(const std::string &pattern) {
  std::string str;
  std::string line;
  std::ifstream file("./info.log");

  if (file.is_open()) {
    while (getline(file, line)) {
      str += line + "\n";
    }
    file.close();
  }

  std::regex cur_pattern(pattern);
  std::sregex_iterator it(str.begin(), str.end(), cur_pattern);
  std::sregex_iterator end;
  return it != end;
}

uint64_t ObtainOutput(const std::string &pattern) {
  std::ifstream file("./info.log");
  std::string str;
  std::string line;

  if (file.is_open()) {
    while (getline(file, line)) {
      str += line + "\n";
    }
    file.close();
  }

  std::regex cur_pattern(pattern);
  std::sregex_iterator it(str.begin(), str.end(), cur_pattern);
  std::sregex_iterator end;
  if (it == end) {
    return 0;
  }
  return stoi(it->str(1));
}
}  // namespace att
