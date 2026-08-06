/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_GEN_TILING_IMPL_H_
#define ATT_GEN_TILING_IMPL_H_
#include <string>
#include <vector>
#include <map>
#include "../common/schedule_result.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"

namespace att {
inline constexpr char kInternalEnableAutofusePgo[] = "_internal_enable_autofuse_pgo";

extern "C" {
/**
 * @brief 生成Tiling函数
 * @param graphs 待生成Tiling的算子图
 * @param options 生成Tiling的选项
 *        "output_file_path": 生成Tiling的输出文件路径, 默认当前执行路径"./"。
 *        "tiling_data_type_name": TilingData的类型名， 默认graph->Name() + "TilingData"
 *        "gen_extra_info": "0"表示不生成额外信息，"1"表示生成额外信息.默认"0"(额外信息包括外轴大小，各个tensorsize,
 * 高阶api tiling等) "dump_debug_info": Value为落盘路径。落盘路径不为空则表示将中间关键信息落盘,
 * 默认空;(debug信息包括tuningspace,modelinfo, tilingfunc等信息)
 *        "gen_tiling_data_def":"0"不生成tilingdata的声明。"1"表示生成tilingdata的声明。默认"1"。
 * @return 是否成功
 */
bool GenTilingImpl(const std::string &op_name, const std::vector<af::AscGraph> &graphs,
                   std::map<std::string, std::string> &options);

/**
 * @brief 生成Tiling函数
 * @param schedule_results 待生成Tiling的ScheduleResult对象
 * @param options 生成Tiling的选项
 * @param tiling_func 生成的Tiling计算逻辑
 *        "tiling_data_type_name": TilingData的类型名， 默认graph->Name() + "TilingData"
 *        "dump_debug_info": Value为落盘路径。落盘路径不为空则表示将中间关键信息落盘,
 * 默认空;(debug信息包括tuningspace,modelinfo, tilingfunc等信息, 该功能暂时不支持设置)
 * @return 是否成功
 */
bool GenTilingImplAutoFuseV3(const std::string &op_name, const ascir::FusedScheduledResult &fused_schedule_result,
                             std::map<std::string, std::string> &options,
                             std::map<std::string, std::string> &tiling_func, bool is_inductor_scene);
}  // extern C
}  // namespace att
#endif
