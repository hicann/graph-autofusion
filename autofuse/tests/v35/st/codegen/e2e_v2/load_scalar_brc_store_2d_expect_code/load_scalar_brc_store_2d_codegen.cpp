#include <fstream>
#include <gtest/gtest.h>

#include "codegen.h"
#include "e2e_common.h"
#include "e2e_load_scalar_abs_brc_store.h"

class LoadScalarAbsBrcStore2DTest : public testing::Test {};

TEST_F(LoadScalarAbsBrcStore2DTest, LoadScalarAbsBrcStore2DCodegen) {
  af::AscGraph test_graph("load_scalar_brc_store_2d");
  LoadScalarBrcStore_BeforeAutofuse2D(test_graph);
  LoadScalarBrcStore_AfterAutofuse2D(test_graph);

  std::vector<af::AscGraph> test_impl_graphs = {af::AscGraph("load_scalar_brc_store_2d_general_0_nil_0_nil")};
  test_impl_graphs[0].CopyFrom(test_graph);
  auto codegen = codegen::Codegen(codegen::CodegenOptions{
      .tiling_lib_path = ATT_SO_NAME, .tiling_lib_codegen_symbol = "CodegenTiling", .using_att_calc_qbt_size = false});
  ascir::ScheduledResult schedule_result;
  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = af::AscendString("load_scalar_brc_store_2d");
  fused_schedule_result.node_idx_to_scheduled_results.push_back({schedule_result});
  InitScheduleResultsByImplGraphs(test_impl_graphs, fused_schedule_result);

  codegen::CodegenResult result;
  std::map<std::string, std::string> shape_info{{"s0", "GetDimValueFromGraphInputData(0, 0);"},
                                                {"s1", "GetDimValueFromGraphInputData(0, 1);"}};
  EXPECT_EQ(codegen.Generate(shape_info, fused_schedule_result, result), 0);

  std::fstream kernel_file("load_scalar_brc_store_2d_kernel.cpp", std::ios::out);
  std::fstream tiling_file("load_scalar_brc_store_2d_tiling.cpp", std::ios::out);
  std::fstream tiling_data_file("autofuse_tiling_data.h", std::ios::out);
  kernel_file << "#define REGISTER_TILING_DEFAULT(tiling)\n"
                 "#define GET_TILING_DATA(t, tiling) AutofuseTilingData t = *(AutofuseTilingData*)tiling;\n"
              << RemoveSubDirInclude(result.kernel);
  tiling_file << result.tiling;
  tiling_data_file << result.tiling_data;
  EXPECT_TRUE(kernel_file.good());
}
