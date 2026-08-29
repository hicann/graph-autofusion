/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_GRAPH_UTILS_FILE_UTILS_H_
#define COMMON_GRAPH_UTILS_FILE_UTILS_H_

#include <cstdint>
#include <string>
#include "graph/types_af.h"
#include "graph/ge_error_codes_af.h"
#include "ge_common_af/ge_api_types_af.h"
#include "mmpa/mmpa_api.h"

namespace af {

/**
 * @ingroup domi_common
 * @brief  Absolute path for obtaining files
 * @param [in] path of input file
 * @return string. Absolute path of a file. If the absolute path cannot be
 * obtained, an empty string is returned
 */
std::string RealPath(const char_t *path);

/**
 * @ingroup domi_common
 * @brief  Recursively Creating a Directory
 * @param [in] directory_path  Path, which can be a multi-level directory.
 * @return 0 success, 1- fail.
 */
int32_t CreateDir(const std::string &directory_path);

/**
 * @ingroup domi_common
 * @brief  Recursively Creating a Directory with mode
 * @param [in] directory_path  Path, which can be a multi-level directory.
 * @param [in] mode  dir mode, E.G., 0700
 * @return 0 success, 1- fail.
 */
int32_t CreateDir(const std::string &directory_path, uint32_t mode);

/**
 * @ingroup domi_common
 * @brief  Recursively Creating a Directory, deprecated, use CreateDir instead
 * @param [in] directory_path  Path, which can be a multi-level directory.
 * @return 0 success, 1- fail.
 */
int32_t CreateDirectory(const std::string &directory_path);

std::unique_ptr<char_t[]> GetBinFromFile(std::string &path, uint32_t &data_len);

std::unique_ptr<char_t[]> GetBinDataFromFile(const std::string &path, uint32_t &data_len);

/**
 * @ingroup domi_common
 * @brief  Get binary data from file
 * @param [in] path  file path.
 * @param [in] offset  offset of the file date
 * @param [in] offset  data len to read
 * @return buffer char[] used to store file data
 */
std::unique_ptr<char[]> GetBinFromFile(const std::string &path, size_t offset, size_t data_len);

graphStatus WriteBinToFile(std::string &path, char_t *data, uint32_t &data_len);

/**
 * @ingroup domi_common
 * @brief  Get binary file from file
 * @param [in] name origin name.
 * @return string. name which replace special code as _.
 */
std::string GetRegulatedName(const std::string name);
/**
 * 跟GetRegulatedName相比，不会处理.字符
 * @param input
 * @return
 */
std::string GetSanitizedName(const std::string &input);

/**
 * @ingroup domi_common
 * @brief  Get binary file from file
 * @param [in] path  file path.
 * @param [out] buffer char[] used to store file data
 * @param [out] data_len store read size
 * @return graphStatus GRAPH_SUCCESS: success, OTHERS: fail.
 */
graphStatus GetBinFromFile(const std::string &path, char_t *buffer, size_t &data_len);

/**
 * @ingroup domi_common
 * @brief  Write binary to file
 * @param [in] fd  file description.
 * @param [in] data char[] used to write to file
 * @param [in] data_len store write size
 * @return graphStatus GRAPH_SUCCESS: success, OTHERS: fail.
 */
graphStatus WriteBinToFile(const int32_t fd, const char_t *const data, size_t data_len);

/**
 * @ingroup domi_common
 * @brief  Save data to file
 * @param [in] file_path  file path.
 * @param [in] data char[] used to store file data
 * @param [in] length store read size
 * @return graphStatus GRAPH_SUCCESS: success, OTHERS: fail.
 */
graphStatus SaveBinToFile(const char *const data, size_t length, const std::string &file_path);

/**
 * @ingroup domi_common
 * @brief  split file path to directory path and file name
 * @param [in] file_path  file path.
 * @param [out] dir_path directory path
 * @param [out] file_name file name
 * @return graphStatus GRAPH_SUCCESS: success, OTHERS: fail.
 */
void SplitFilePath(const std::string &file_path, std::string &dir_path, std::string &file_name);

/**
 * @ingroup domi_common
 * @brief  Get ASCEND_WORK_PATH environment variable
 * @param [out] ascend_work_path ASCEND_WORK_PATH's value.
 * @return graphStatus SUCCESS: success, OTHERS: fail.
 */
Status GetAscendWorkPath(std::string &ascend_work_path);

int32_t Scandir(const CHAR *path, mmDirent ***entry_list, mmFilter filter_func, mmSort sort);
}  // namespace af

#endif  // end COMMON_GRAPH_UTILS_FILE_UTILS_H_
