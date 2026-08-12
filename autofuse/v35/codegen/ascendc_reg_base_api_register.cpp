/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascendc_api_registry.h"

namespace codegen {
namespace {
class Register {
 public:
  Register();
};

static std::string StripBroadcastSelfIncludes(const std::string &src) {
  std::string result;
  size_t pos = 0;
  while (pos < src.size()) {
    size_t end = src.find('\n', pos);
    if (end == std::string::npos) end = src.size();
    std::string line = src.substr(pos, end - pos);
    if (line.find("#include \"broadcast") == std::string::npos) {
      result += line + "\n";
    }
    pos = end + 1;
  }
  return result;
}

Register::Register() {
  const std::string kAscendcCastRegStr = {
#include "cast_reg_base.h"
  };
  const std::string kAscendcCompareRegStr = {
#include "compare_reg_base.h"
  };
  const std::string kAscendcConcatRegBaseStr = {
#include "concat_reg_base.h"
  };
  const std::string kAscendcDatacopyRegBaseStr = {
#include "datacopy_reg_base.h"
  };
  const std::string kAscendcDatacopyNddmaRegBaseStr = {
#include "datacopy_nddma_reg_base.h"
  };
  const std::string kAscendcReduce_initRegBase = {
#include "reduce_init_reg_base.h"
  };
  const std::string kAscendcFloorDivRegBaseStr = {
#include "floor_div_reg_base.h"
  };
  const std::string kAscendcSignRegBaseStr = {
#include "sign_reg_base.h"
  };
  const std::string kAscendcWhereRegBaseStr = {
#include "where_reg_base.h"
  };
  const std::string kAscendcGatherRegBaseStr = {
#include "gather_reg_base.h"
  };
  const std::string kAscendcUtilsRegBaseStr = {
#include "utils_reg_base.h"
  };
  const std::string kAscendcBroadcastRegStr = StripBroadcastSelfIncludes(std::string{
#include "broadcast_extend_impl_reg_base.h"
                                              }) +
                                              StripBroadcastSelfIncludes(std::string{
#include "broadcast_reg_base.h"
                                              });
  const std::string kAscendcLogicalNotStr = {
#include "logical_not_reg_base.h"
  };
  const std::string kAscendcClipByValueRegStr = {
#include "clipbyvalue_reg_base.h"
  };
  const std::string kAscendcLogicalRegBaseStr = {
#include "logical_reg_base.h"
  };
  const std::string kAscendcLogicalXorRegBaseStr = {
#include "logical_xor_reg_base.h"
  };
  const std::string kAscendcPowRegBaseStr = {
#include "pow_reg_base.h"
  };
  const std::string kAscendcExp2RegBaseStr = {
#include "exp2_reg_base.h"
  };
  const std::string kAscendcLog1pRegBaseStr = {
#include "log1p_reg_base.h"
  };
  const std::string kAscendcErfRegBaseStr = {
#include "erf_reg_base.h"
  };
  const std::string kAscendcTanhRegBaseStr = {
#include "tanh_reg_base.h"
  };
  const std::string kAscendcSubRegBaseStr = {
#include "sub_reg_base.h"
  };
  const std::string kAscendcDivRegBaseStr = {
#include "div_reg_base.h"
  };
  const std::string kAscendcSplitRegBaseStr = {
#include "split_reg_base.h"
  };
  const std::string kAscendcAtan2RegBaseStr = {
#include "atan2_reg_base.h"
  };
  const std::string kAscendcCopySignRegBaseStr = {
#include "copy_sign_reg_base.h"
  };
  const std::string kAscendcErfcxRegBaseStr = {
#include "erfcx_reg_base.h"
  };
  const std::string kAscendcExpmRegBaseStr = {
#include "expm_reg_base.h"
  };
  const std::string kAscendcFmodRegBaseStr = {
#include "fmod_reg_base.h"
  };
  const std::string kAscendcTruncDivRegBaseStr = {
#include "trunc_div_reg_base.h"
  };
  const std::string kAscendcRemainderRegBaseStr = {
#include "remainder_reg_base.h"
  };
  const std::string kAscendcSoftmaxAfRegBaseStr = {
#include "softmax_af_reg_base.h"
  };
  const std::string kAscendcNegRegBaseStr = {
#include "neg_reg_base.h"
  };
  const std::string kAscendcSquareRegBaseStr = {
#include "square_reg_base.h"
  };
  const std::string kAscendcTransposeRegBaseStr = {
#include "transpose_reg_base.h"
  };
  const std::string kAscendcTrigonometricFunctionUtilsRegBaseStr = {
#include "trigonometric_function_utils_reg_base.h"
  };
  const std::string kAscendcModifiedBesselUtilsRegBaseStr = {
#include "modified_bessel_utils_reg_base.h"
  };
  const std::string kAscendcModifiedBesselI0RegBaseStr = {
#include "modified_bessel_i0_reg_base.h"
  };
  const std::string kAscendcModifiedBesselI1RegBaseStr = {
#include "modified_bessel_i1_reg_base.h"
  };
  const std::string kAscendcModifiedBesselK0RegBaseStr = {
#include "modified_bessel_k0_reg_base.h"
  };
  const std::string kAscendcModifiedBesselK1RegBaseStr = {
#include "modified_bessel_k1_reg_base.h"
  };
  const std::string kAscendcLaguerrePolynomialLRegBaseStr = {
#include "laguerre_polynomial_l_reg_base.h"
  };
  const std::string kAscendcLegendrePolynomialPRegBaseStr = {
#include "legendre_polynomial_p_reg_base.h"
  };
  const std::string kAscendcAiryAiRegBaseStr = {
#include "airy_ai_reg_base.h"
  };
  const std::string kAscendcErfinvRegBaseStr = {
#include "erfinv_reg_base.h"
  };
  const std::string kAscendcBesselJUtilsRegBaseStr = {
#include "bessel_j_utils_reg_base.h"
  };
  const std::string kAscendcBesselJ0RegBaseStr = {
#include "bessel_j0_reg_base.h"
  };
  const std::string kAscendcBesselJ1RegBaseStr = {
#include "bessel_j1_reg_base.h"
  };
  const std::string kAscendcBesselY0RegBaseStr = {
#include "bessel_y0_reg_base.h"
  };
  const std::string kAscendcBesselY1RegBaseStr = {
#include "bessel_y1_reg_base.h"
  };
  const std::string kAscendcScaledModifiedBesselK0RegBaseStr = {
#include "scaled_modified_bessel_k0_reg_base.h"
  };
  const std::string kAscendcScaledModifiedBesselK1RegBaseStr = {
#include "scaled_modified_bessel_k1_reg_base.h"
  };
  const std::string kAscendcSphericalBesselJ0RegBaseStr = {
#include "spherical_bessel_j0_reg_base.h"
  };
  const std::string kAscendcIgammaRegBaseStr = {
#include "igamma_reg_base.h"
  };
  const std::string kAscendcIgammacRegBaseStr = {
#include "igammac_reg_base.h"
  };
  const std::string kAscendcIgammacHelperSeriesRegBaseStr = {
#include "igammac_helper/series_reg_base.h"
  };
  const std::string kAscendcIgammacHelperContinuedFractionRegBaseStr = {
#include "igammac_helper/continued_fraction_reg_base.h"
  };
  const std::string kAscendcIgammacHelperAsymptoticSeriesRegBaseStr = {
#include "igammac_helper/asymptotic_series_reg_base.h"
  };
  const std::string kAscendcIgammacHelperSeriesComplementRegBaseStr = {
#include "igammac_helper/series_complement_reg_base.h"
  };
  const std::string kAscendcZetaRegBaseStr = {
#include "zeta_reg_base.h"
  };
  const std::string kAscendcNdtrRegBaseStr = {
#include "ndtr_reg_base.h"
  };
  const std::string kAscendcNdtriRegBaseStr = {
#include "ndtri_reg_base.h"
  };
  const std::string kAscendcSignBitRegBaseStr = {
#include "signbit_reg_base.h"
  };
  const std::string kAscendcFrexpRegBaseStr = {
#include "frexp_reg_base.h"
  };
  const std::string kAscendcShiftedChebyshevPolynomialUtilsRegBaseStr = {
#include "shifted_chebyshev_polynomial_utils_reg_base.h"
  };
  const std::string kAscendcShiftedChebyshevPolynomialTRegBaseStr = {
#include "shifted_chebyshev_polynomial_t_reg_base.h"
  };
  const std::string kAscendcShiftedChebyshevPolynomialURegBaseStr = {
#include "shifted_chebyshev_polynomial_u_reg_base.h"
  };
  const std::string kAscendcShiftedChebyshevPolynomialVRegBaseStr = {
#include "shifted_chebyshev_polynomial_v_reg_base.h"
  };
  const std::string kAscendcShiftedChebyshevPolynomialWRegBaseStr = {
#include "shifted_chebyshev_polynomial_w_reg_base.h"
  };
  const std::string kAscendcIndirectLoadSimtRegBaseStr = {
#include "indirect_load_simt_reg_base.h"
  };
  const std::string kAscendcIndirectLoadSimdPolicyRegBaseStr = {
#include "indirect_load_simd_policy_reg_base.h"
  };
  const std::string kAscendcIndirectLoadSimdRegBaseStr = {
#include "indirect_load_simd_reg_base.h"
  };
  const std::string kAscendcIndirectLoadSkRegBaseStr = {
#include "indirect_load_sk_reg_base.h"
  };
  std::unordered_map<std::string, std::string> api_to_file{
      {"cast_reg_base.h", kAscendcCastRegStr},
      {"compare_reg_base.h", kAscendcCompareRegStr},
      {"concat_reg_base.h", kAscendcConcatRegBaseStr},
      {"datacopy_reg_base.h", kAscendcDatacopyRegBaseStr},
      {"datacopy_nddma_reg_base.h", kAscendcDatacopyNddmaRegBaseStr},
      {"pow_reg_base.h", kAscendcPowRegBaseStr},
      {"exp2_reg_base.h", kAscendcExp2RegBaseStr},
      {"log1p_reg_base.h", kAscendcLog1pRegBaseStr},
      {"erf_reg_base.h", kAscendcErfRegBaseStr},
      {"tanh_reg_base.h", kAscendcTanhRegBaseStr},
      {"reduce_init_reg_base.h", kAscendcReduce_initRegBase},
      {"floor_div_reg_base.h", kAscendcFloorDivRegBaseStr},
      {"sign_reg_base.h", kAscendcSignRegBaseStr},
      {"where_reg_base.h", kAscendcWhereRegBaseStr},
      {"gather_reg_base.h", kAscendcGatherRegBaseStr},
      {"broadcast_reg_base.h", kAscendcBroadcastRegStr},
      {"utils_reg_base.h", kAscendcUtilsRegBaseStr},
      {"logical_not_reg_base.h", kAscendcLogicalNotStr},
      {"clipbyvalue_reg_base.h", kAscendcClipByValueRegStr},
      {"logical_reg_base.h", kAscendcLogicalRegBaseStr},
      {"logical_xor_reg_base.h", kAscendcLogicalXorRegBaseStr},
      {"split_reg_base.h", kAscendcSplitRegBaseStr},
      {"sub_reg_base.h", kAscendcSubRegBaseStr},
      {"div_reg_base.h", kAscendcDivRegBaseStr},
      {"atan2_reg_base.h", kAscendcAtan2RegBaseStr},
      {"copy_sign_reg_base.h", kAscendcCopySignRegBaseStr},
      {"erfcx_reg_base.h", kAscendcErfcxRegBaseStr},
      {"expm_reg_base.h", kAscendcExpmRegBaseStr},
      {"fmod_reg_base.h", kAscendcFmodRegBaseStr},
      {"trunc_div_reg_base.h", kAscendcTruncDivRegBaseStr},
      {"remainder_reg_base.h", kAscendcRemainderRegBaseStr},
      {"softmax_af_reg_base.h", kAscendcSoftmaxAfRegBaseStr},
      {"neg_reg_base.h", kAscendcNegRegBaseStr},
      {"square_reg_base.h", kAscendcSquareRegBaseStr},
      {"transpose_reg_base.h", kAscendcTransposeRegBaseStr},
      {"trigonometric_function_utils_reg_base.h", kAscendcTrigonometricFunctionUtilsRegBaseStr},
      {"modified_bessel_utils_reg_base.h", kAscendcModifiedBesselUtilsRegBaseStr},
      {"modified_bessel_i0_reg_base.h", kAscendcModifiedBesselI0RegBaseStr},
      {"modified_bessel_i1_reg_base.h", kAscendcModifiedBesselI1RegBaseStr},
      {"modified_bessel_k0_reg_base.h", kAscendcModifiedBesselK0RegBaseStr},
      {"modified_bessel_k1_reg_base.h", kAscendcModifiedBesselK1RegBaseStr},
      {"laguerre_polynomial_l_reg_base.h", kAscendcLaguerrePolynomialLRegBaseStr},
      {"legendre_polynomial_p_reg_base.h", kAscendcLegendrePolynomialPRegBaseStr},
      {"airy_ai_reg_base.h", kAscendcAiryAiRegBaseStr},
      {"erfinv_reg_base.h", kAscendcErfinvRegBaseStr},
      {"bessel_j_utils_reg_base.h", kAscendcBesselJUtilsRegBaseStr},
      {"bessel_j0_reg_base.h", kAscendcBesselJ0RegBaseStr},
      {"bessel_j1_reg_base.h", kAscendcBesselJ1RegBaseStr},
      {"bessel_y0_reg_base.h", kAscendcBesselY0RegBaseStr},
      {"bessel_y1_reg_base.h", kAscendcBesselY1RegBaseStr},
      {"scaled_modified_bessel_k0_reg_base.h", kAscendcScaledModifiedBesselK0RegBaseStr},
      {"scaled_modified_bessel_k1_reg_base.h", kAscendcScaledModifiedBesselK1RegBaseStr},
      {"spherical_bessel_j0_reg_base.h", kAscendcSphericalBesselJ0RegBaseStr},
      {"igamma_reg_base.h", kAscendcIgammaRegBaseStr},
      {"igammac_reg_base.h", kAscendcIgammacRegBaseStr},
      {"igammac_helper/series_reg_base.h", kAscendcIgammacHelperSeriesRegBaseStr},
      {"igammac_helper/continued_fraction_reg_base.h", kAscendcIgammacHelperContinuedFractionRegBaseStr},
      {"igammac_helper/asymptotic_series_reg_base.h", kAscendcIgammacHelperAsymptoticSeriesRegBaseStr},
      {"igammac_helper/series_complement_reg_base.h", kAscendcIgammacHelperSeriesComplementRegBaseStr},
      {"zeta_reg_base.h", kAscendcZetaRegBaseStr},
      {"ndtr_reg_base.h", kAscendcNdtrRegBaseStr},
      {"ndtri_reg_base.h", kAscendcNdtriRegBaseStr},
      {"signbit_reg_base.h", kAscendcSignBitRegBaseStr},
      {"frexp_reg_base.h", kAscendcFrexpRegBaseStr},
      {"shifted_chebyshev_polynomial_utils_reg_base.h", kAscendcShiftedChebyshevPolynomialUtilsRegBaseStr},
      {"shifted_chebyshev_polynomial_t_reg_base.h", kAscendcShiftedChebyshevPolynomialTRegBaseStr},
      {"shifted_chebyshev_polynomial_u_reg_base.h", kAscendcShiftedChebyshevPolynomialURegBaseStr},
      {"shifted_chebyshev_polynomial_v_reg_base.h", kAscendcShiftedChebyshevPolynomialVRegBaseStr},
      {"shifted_chebyshev_polynomial_w_reg_base.h", kAscendcShiftedChebyshevPolynomialWRegBaseStr},
      {"indirect_load_simt_reg_base.h", kAscendcIndirectLoadSimtRegBaseStr},
      {"indirect_load_simd_policy_reg_base.h", kAscendcIndirectLoadSimdPolicyRegBaseStr},
      {"indirect_load_simd_reg_base.h", kAscendcIndirectLoadSimdRegBaseStr},
      {"indirect_load_sk_reg_base.h", kAscendcIndirectLoadSkRegBaseStr},
  };

  AscendCApiRegistry::GetInstance().RegisterApi(api_to_file);
}

Register __attribute__((unused)) reg_base_api_register;
}  // namespace
}  // namespace codegen
