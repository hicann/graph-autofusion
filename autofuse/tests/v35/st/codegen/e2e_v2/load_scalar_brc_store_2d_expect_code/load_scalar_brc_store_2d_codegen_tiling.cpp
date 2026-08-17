#include "ascendc_ir.h"

#include <sstream>

#include "e2e_common.h"

extern "C" bool CodegenTiling(const std::string &op_name, const ascir::FusedScheduledResult &fused_schedule_result,
                              std::map<std::string, std::string> &options,
                              std::map<std::string, std::string> &tiling_func) {
  std::stringstream ss;
  ss << OptilingStub(fused_schedule_result) << std::endl;
  ss << "extern \"C\" void GetTiling(AutofuseTilingData& tiling_data) {" << std::endl;
  ss << "  tiling_data.set_block_dim(1);" << std::endl;
  ss << "}" << std::endl;
  tiling_func["TilingHead"] += ss.str();
  return true;
}
