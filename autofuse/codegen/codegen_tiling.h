/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __CODEGEN_TILING_H__
#define __CODEGEN_TILING_H__
#include <sstream>
#include "ascir.h"
#include "schedule_result.h"
#include "ascgen_log.h"

namespace codegen {
const std::string kTilingHeadIdentify = "TilingHead";
const std::string kTilingDataIdentify = "TilingData";
const std::string kTilingHeadGuard = "__AUTOFUSE_TILING_FUNC_COMMON_H__";
const std::string kTilingHeadInclude = "#include \"autofuse_tiling_func_common.h\"";
const std::string kTilingHeadCceKtTestGuard = "#ifndef __CCE_KT_TEST__";
const std::string kTilingHeadEndGuard = "#endif";
const std::string kTilingHeadTilingContext = "#include \"exe_graph/runtime/tiling_context.h\"";
const std::string kTilingDefAndConstIdentify = "tiling_def_and_tiling_const";
const std::string kCubeTilingHeadInclude = "#include \"autofuse_cube_tiling_data.h\"";
const std::string kCubeKernelTilingWrapperHpp = "ACubeKernelTilingWrapperHpp";
const std::string kCubeKernelTilingWrapperCpp = "BCubeKernelTilingWrapperCpp";
const std::string kCubeKernelTilingWrapperInclude = "#include \"cube_kernel_tiling_wrapper.h\"";

struct MatMulCubeInfo {
  bool transpose_x1 = false;
  bool transpose_x2 = false;
  int32_t offset_x = 0;
  int64_t enable_hf32 = false;
  bool is_batch = false;
  bool has_relu = false;
  uint32_t input_num = 0U;
  uint32_t type_size = 4U;
  ge::AscNodePtr matmul_node = nullptr;
};

struct TensorInfo {
  std::string param_name;
  std::vector<ge::Expression> shape;
  std::vector<ge::Expression> ori_shape;
  std::string dtype;
  std::string format;
  std::string name;
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
  std::string op_kernel_lib;
  std::string op_impl_mode;
  int64_t aicore_num = 0;
  int64_t aiv_num = 0;
  std::map<std::string, std::string> extra_info;
};

using TilingLibCodegenFunc = bool (*)(const std::string &op_name,
                                      const ::ascir::FusedScheduledResult &fused_schedule_result,
                                      std::map<std::string, std::string> &options,
                                      std::map<std::string, std::string> &tiling_file_name_to_content,
                                      bool is_inductor_scene);
struct PgoShapeStringStream {
  std::stringstream shape_dim_def;
  std::stringstream tiling_set_shape_dim;
  std::stringstream shape_dim_use;
};
class TilingLib {
 public:
  TilingLib(const std::string &lib_path, const std::string &codegen_symbol_name);
  std::map<std::string, std::string> Generate(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                              const std::map<std::string, std::string> &shape_info,
                                              const std::string &pgo_dir, const std::string &core_num) const;
  std::map<std::string, std::string> GenerateForInductor(
      const ::ascir::FusedScheduledResult &fused_schedule_result) const;

  std::string GenerateForPgo(const ::ascir::FusedScheduledResult &fused_schedule_result,
                             const std::string &pgo_dir) const;
  std::string GetTilingIncludeHead(bool is_cv = false) const;

 protected:
  std::string TilingFuncDef(const ::ascir::FusedScheduledResult &fused_schedule_result,
                            const ::ascir::FusedScheduledResult &elemwise_schedule_result,
                            const std::map<std::string, std::string> &shape_info, const std::string &pgo_dir,
                            const std::string &core_num) const;
  std::string TilingFuncDefForInductor(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                       const ::ascir::FusedScheduledResult &elemwise_schedule_result) const;
  std::map<std::string, std::string> GetTilingHeaders(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                                      bool is_inductor_scene, bool is_cv = false) const;
  std::string InferShapeDef(const ::ascir::HintGraph &graph) const;
  std::string OpDef(const ::ascir::HintGraph &graph) const;

  std::string OpInputDef(const ::ascir::NodeView &node) const;
  std::string OpOutputDef(const ::ascir::NodeView &node) const;

  std::string ExternFunctionDeclare(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                    const std::string tiling) const;
  std::string PGOTensorArgsDef() const;
  std::string PGOProfilingCallbackDef(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                      const std::string tiling) const;
  std::string PGOSearchFuncInputOutputCallBackDef(const ::ascir::FusedScheduledResult &fused_schedule_result) const;
  std::string PGOSearchFuncInputOutputDef(const ::ascir::FusedScheduledResult &fused_schedule_result) const;
  std::string PGOSearchFuncInputOutputCall(const ::ascir::FusedScheduledResult &fused_schedule_result) const;
  std::string PGOSearchStructInputOutputDef(const ::ascir::FusedScheduledResult &fused_schedule_result) const;
  std::string PGOSearchTensorInputOutputDef(const ::ascir::FusedScheduledResult &fused_schedule_result) const;
  std::string PGOSearchTensorArgsUpdateDef(const ::ascir::FusedScheduledResult &fused_schedule_result) const;
  std::string PGOSearchFuncInputOutputStructAssignDef(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                                      const std::string &struct_var_name) const;
  uint32_t PGOSearchFuncGetInputOutputCount(const ::ascir::FusedScheduledResult &fused_schedule_result) const;
  uint32_t PGOSearchFuncGetOutputCount(const ::ascir::FusedScheduledResult &fused_schedule_result) const;
  std::string CalculateTensorMemorySizeStr(const ::ascir::TensorAttr &tensor) const;
  std::string PGOSearchTensorMallocDef(const ::ascir::FusedScheduledResult &fused_schedule_result) const;
  std::string PGOSearchTensorFreeDef(const ::ascir::FusedScheduledResult &fused_schedule_result) const;
  std::string StubHeadersWithoutCodegenFunc() const;
  std::string GetStubTilingHeaders(const ::ascir::FusedScheduledResult &fused_schedule_result) const;
  std::string GenGetAutoFuseTilingInput(bool is_inductor_scene) const;
  std::string GenGetResLimitStru(void) const;
  bool IsMixKernelTaskType(const ::ascir::FusedScheduledResult &fused_schedule_result) const;

 private:
  // 判断某个 origin_var 是否被特定 schedule_group 使用
  bool IsVarUsedInScheduleGroup(const std::string &var_define, const ::ascir::ScheduleGroup &schedule_group) const;
  std::string GenGetTilingSizeFunc(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                   const std::string graph_name, const std::string tiling,
                                   bool is_inductor = false) const;
  std::string GenTilingFunc(const std::map<std::string, std::string> &shape_info,
                            const ::ascir::FusedScheduledResult &fused_schedule_result, const std::string func,
                            const std::string tiling, const std::string &core_num) const;
  std::string GenTilingFuncForInductor(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                       const ::ascir::FusedScheduledResult &elemwise_schedule_result,
                                       const std::string func, const std::string tiling) const;
  std::string GenGetTopnSolutionsFuncForInductor(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                                 const std::string &tiling) const;
  void GenTopnInitSearchTiling(std::stringstream &ss, const ::ascir::FusedScheduledResult &fused_schedule_result,
                               const std::string &tiling, int symbol_value_count) const;
  void GenTopnGetTilingFunc(std::stringstream &ss, const ::ascir::FusedScheduledResult &fused_schedule_result,
                            const std::string &tiling, int symbol_value_count) const;
  void GenTopnSearchTilingSetup(std::stringstream &ss, const std::string &tiling,
                                const ::ascir::FusedScheduledResult &fused_schedule_result) const;
  void GenTopnCollectCandidates(std::stringstream &ss, const std::string &tiling) const;
  void GenTopnSearchTilingKeyCall(std::stringstream &ss, const ::ascir::FusedScheduledResult &fused_schedule_result,
                                  const std::string &search_cfg) const;
  void GenTopnSetFailureMessage(std::stringstream &ss, const std::string &indent, const std::string &reason) const;
  void GenTopnDefaultTiling(std::stringstream &ss, const std::string &tiling) const;
  void GenTopnSearchAndFinalChecks(std::stringstream &ss, const std::string &tiling,
                                   const ::ascir::FusedScheduledResult &fused_schedule_result) const;
  void GenGenerateTopnSolutionsEntry(std::stringstream &ss, const ::ascir::FusedScheduledResult &fused_schedule_result,
                                     const std::string &tiling,
                                     const codegen::PgoShapeStringStream &pgo_shape_dim) const;
  std::string GenCandidateSolutionProtocolForInductor(const std::string &tiling) const;
  void GenDeduplicateCandidateSolutions(std::stringstream &ss) const;
  std::string GenTopnSelectorHelpersForInductor() const;
  std::string GenSearchConfigProtocolForInductor() const;
  std::string GenBuiltinTfPgoConfigsForInductor() const;
  std::string GenInductorConfigParserForInductor() const;
  std::string GenGetTilingDataReprFuncForInductor(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                                  const std::string &tiling) const;
  std::string GenEvaluateModeledPerfForInductor(const std::string &tiling,
                                                const ::ascir::FusedScheduledResult &fused_schedule_result) const;
  void GenMultiGroupPerfAggregation(std::stringstream &ss,
                                    const ::ascir::FusedScheduledResult &fused_schedule_result) const;
  void GenGroupPerfForScheduleResult(std::stringstream &ss, size_t asc_graph_id, size_t result_id,
                                     const ::ascir::ScheduledResult &sched_result) const;
  std::string GenUpdateCurPerfAndBlockByGroupHelper() const;
  void GenReprScheduleGroupFields(std::stringstream &ss, const ::ascir::ScheduleGroup &sg,
                                  const std::string &field_prefix, const std::string &emit_fn,
                                  const std::string &indent, bool emit_first_arg) const;
  void GenReprApiTilingFields(std::stringstream &ss, const ::ascir::ImplGraph &graph, size_t tiling_case_id,
                              const std::string &field_prefix, bool top_level) const;
  void GenReprSingleGroup(std::stringstream &ss, const ::ascir::FusedScheduledResult &fused_schedule_result) const;
  void GenReprMultiGroup(std::stringstream &ss, const ::ascir::FusedScheduledResult &fused_schedule_result) const;
  std::string GenPgoTilingFunc(const ::ascir::FusedScheduledResult &fused_schedule_result, const std::string &tiling,
                               codegen::PgoShapeStringStream &pgo_shape_dim, bool is_inductor_scene,
                               const std::string &core_num = "0") const;
  std::string GenPgoAutofuseTiling(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                   codegen::PgoShapeStringStream &pgo_shape_dim, const std::string &tiling,
                                   bool is_inductor_scene) const;

  std::string GenPgoTilingSearchPGO(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                    codegen::PgoShapeStringStream &pgo_shape_dim, const std::string &tiling,
                                    bool is_inductor_scene, const std::string &core_num) const;

  std::string GenPgoTilingSearch(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                 codegen::PgoShapeStringStream &pgo_shape_dim, const std::string &tiling) const;
  std::string GenProfilingAllTilingData(std::string tiling_data_list_name, std::string tiling_data_perf_list_name,
                                        const ::ascir::FusedScheduledResult &fused_schedule_result,
                                        bool is_inductor_scene) const;
  std::string GenGetMaxBlockDimFromInput(const std::string &core_num) const;
  std::string GenPgoTilingSearchByCoreNum(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                          codegen::PgoShapeStringStream &pgo_shape_dim, const std::string &tiling,
                                          bool is_inductor_scene, const std::string &core_num) const;
  std::string GenPGOGetTilingKey(const std::string tiling) const;
  std::string GenSavePGOSearchTilingDataFunc(const std::string tiling) const;
  std::string GenSavePGOConfigTilingDataFunc() const;
  void GenPgoSaveTilingKey(std::stringstream &ss) const;
  void GenPgoAppendSearchTilingData(std::stringstream &ss) const;
  void GenPgoKernelLaunchOpArgs(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                std::stringstream &ss) const;
  void GenDynamicLibraryLoaderCode(std::stringstream &ss) const;
  void GenPgoHeaders(std::stringstream &ss) const;
  void GenPgoMain(const ::ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const;
  void GenPgoEnvInit(const ::ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const;
  void GenPgoCardLock(std::stringstream &ss) const;
  void GenPgoMixTilingTable(const ::ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const;
  void GenPgoCheckTilingIsMix(const ::ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const;
  void GenPgoToolFunction(const ::ascir::FusedScheduledResult &fused_schedule_result, const std::string &pgo_dir,
                          std::stringstream &ss) const;
  void GenPgoLaunchKernelInit(std::stringstream &ss) const;
  void GenPgoLaunchParamsInit(const ::ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const;
  void GenPgoLaunchParamsDeInit(std::stringstream &ss) const;
  void GenPgoUpdateLaunchParams(std::stringstream &ss) const;
  void GenPgoLaunchParams(const ::ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const;
  void GenPgoDeinit(const ::ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const;
  void GenPgoWrapperParmCall(const ::ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const;
  void GenPgoWrapperKernelLaunch(std::stringstream &ss) const;
  void GenPgoWrapper(const ::ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const;
  void GenPgoProfilingConstants(std::stringstream &ss) const;
  void GenPgoMsptiStringTable(std::stringstream &ss) const;
  void GenPgoMsptiRequest(std::stringstream &ss) const;
  void GenPgoMsptiComplete(std::stringstream &ss) const;
  void GenPgoMsptiToolFunction(std::stringstream &ss) const;
  void GenPgoMsptiProfiling(std::stringstream &ss) const;
  void GenPgoBatchCallback(std::stringstream &ss) const;
  void GenPgoBatchProcess(std::stringstream &ss) const;
  void GenPgoGetProfilingBatch(const ::ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const;
  void GenPgoProfilingCallback(std::stringstream &ss) const;
  void GenPgoGetProfiling(const ::ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const;
  void GenPgoFunc(const ::ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const;
  void GenPgoStaticFunc(const ::ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const;
  void GenPgoProfiling(const ::ascir::FusedScheduledResult &fused_schedule_result, std::stringstream &ss) const;
  std::string GenExternTilingFunc(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                  const std::map<std::string, std::string> &shape_info, const std::string tiling,
                                  const std::string &pgo_dir, const std::string &core_num) const;
  void TilingSetShapeDim(std::stringstream &tiling_set_shape_dim, const std::string &var_define,
                         const ::ascir::FusedScheduledResult &fused_schedule_result,
                         const std::string &tiling_expr = "tiling->") const;
  std::string GenTilingCacheFunc(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                 const std::map<std::string, std::string> &shape_info) const;
  void TilingMappingSymbolToTiling(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                   std::unordered_map<std::string, std::string> &ori_sym_tiling_map) const;
  void TilingProcessSymbolToTiling(const ::ascir::ImplGraph &graph, size_t graph_num, size_t res_num, size_t group_num,
                                   std::unordered_map<std::string, std::string> &ori_sym_tiling_map) const;
  std::string GenCheckStaticShapeFunc(bool is_static) const;
  std::string GenGetWorkspaceSizeFunc(const std::string &tiling,
                                      const ::ascir::FusedScheduledResult &fused_schedule_result) const;
  std::string GenImplGraphWorkspaceSize(const ::ascir::ImplGraph &graph, const std::string &tiling_data,
                                        uint32_t index) const;
  std::string GenDfxInputSymbolInfo(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                    const std::map<std::string, std::string> &shape_info) const;
  std::string GenFindBestTilingKeyFunc(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                       const std::string &tiling_data_name) const;
  std::string GenGetTilingKeyCount(const ::ascir::FusedScheduledResult &fused_schedule_result) const;
  std::string GenGetTilingKeyForStatic() const;
  std::string GenGetTilingKeyKernelTypeForStatic(const ::ascir::FusedScheduledResult &fused_schedule_result) const;
  std::string GenCVTilingFunc() const;
  std::string GenTilingDataBlockDimAndWss() const;
  void AppendCVFusionHeaders(std::stringstream &ss, bool is_static, bool is_inductor = false) const;
  std::map<std::string, std::string> GenerateCVFusionStatic(
      const ::ascir::FusedScheduledResult &fused_schedule_result,
      const ::ascir::FusedScheduledResult &elemwise_schedule_result,
      const std::map<std::string, std::string> &shape_info, const std::string &pgo_dir,
      const std::string &core_num) const;
  std::map<std::string, std::string> GenerateCVFusionDynamic(
      const ::ascir::FusedScheduledResult &fused_schedule_result,
      const ::ascir::FusedScheduledResult &elemwise_schedule_result,
      const std::map<std::string, std::string> &shape_info, const std::string &pgo_dir,
      const std::string &core_num) const;
  std::map<std::string, std::string> GenerateCVFusion(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                                      const std::map<std::string, std::string> &shape_info,
                                                      const std::string &pgo_dir, const std::string &core_num) const;
  std::string GenCubeFusionTilingBody(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                      const std::string &shape_dim_param) const;
  std::string GenNonCubeFusionTilingBody(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                         const std::string &tiling, const std::string &shape_dim_param) const;
  std::string GenExternTilingFuncBody(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                      const std::map<std::string, std::string> &shape_info, const std::string &tiling,
                                      const std::string &pgo_dir) const;
  Status ExtractMatMulCubeInfoFromImplGraph(const ge::AscGraph &impl_graph, MatMulCubeInfo &cube_info) const;
  Status ExtractMatMulCubeInfoFromFusedResult(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                              MatMulCubeInfo &cube_info) const;
  Status GetInputTensorInfoFromLoadNode(const ge::NodePtr &load_node, TensorInfo &tensor_info) const;
  Status ExtractInputsFromMatMulNode(const ge::AscNodePtr &matmul_node, std::vector<TensorInfo> &inputs) const;
  Status ExtractOutputsFromMatMulNode(const ge::AscNodePtr &matmul_node, std::vector<TensorInfo> &outputs) const;
  std::string GenerateTensorInfoCode(const TensorInfo &tensor, const std::string &var_name) const;
  std::string GenerateAttrInfoCode(const AttrInfo &attr, const std::string &var_name) const;
  void PrepareMatMulAttrs(const MatMulCubeInfo &cube_info, std::vector<AttrInfo> &attrs) const;
  void GenerateTensorListCode(std::stringstream &code_ss, const std::vector<TensorInfo> &inputs,
                              const std::vector<TensorInfo> &outputs) const;
  void GenerateTilingCallCode(std::stringstream &code_ss, bool is_batch) const;
  std::string GenerateMatMulTilingCode(const CompileInfo &compile_info, const std::vector<TensorInfo> &inputs,
                                       const std::vector<TensorInfo> &outputs, const std::vector<AttrInfo> &attrs,
                                       bool is_batch) const;
  std::string ProcessCubeKernelTilingFromFusedResult(const ::ascir::FusedScheduledResult &fused_schedule_result) const;
  std::string GenCubeFusionTilingBodyInductor(const ::ascir::FusedScheduledResult &fused_schedule_result,
                                              const ::ascir::FusedScheduledResult &elemwise_schedule_result,
                                              const std::string &shape_dim_param) const;
  TilingLibCodegenFunc codegen_func_{nullptr};
  bool enable_autofuse_pgo_{false};
};
}  // namespace codegen

#endif
