function(add_indirect_load_e2e_case test_name rank axis input_pre_type use_exp2 tiling_key selected_template expected_templates x0 x1 x2 x3 index0 index1 index2 index3)
    set(case_workdir ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
    file(MAKE_DIRECTORY ${case_workdir})
    do_backend_e2e_st_test(${test_name}
        ${ARGN}
        WORKDIR ${case_workdir}
        CODEGEN indirect_load_store_backend_generator.cpp
        TILING_KEY ${tiling_key}
        KERNEL_SRC
            indirect_load_store_test_kernel.cpp
            indirect_load_store_test_tiling.cpp
            autofuse_tiling_data.h
        TEST_SRC test_e2e_indirect_load_store_kernel.cpp)

    set(case_definitions
        IL_RANK=${rank}
        IL_AXIS=${axis}
        IL_INPUT_PRE_TYPE=${input_pre_type}
        IL_USE_EXP2=${use_exp2}
        IL_TILING_KEY=${tiling_key}
        IL_SELECTED_TEMPLATE=${selected_template}
        IL_EXPECTED_TEMPLATES=${expected_templates}
        IL_X_S0=${x0}
        IL_X_S1=${x1}
        IL_X_S2=${x2}
        IL_X_S3=${x3}
        IL_INDEX_S0=${index0}
        IL_INDEX_S1=${index1}
        IL_INDEX_S2=${index2}
        IL_INDEX_S3=${index3})
    target_compile_definitions(${test_name}_codegen_v2 PRIVATE ${case_definitions})
    target_compile_definitions(${test_name}_e2e_v2 PRIVATE ${case_definitions})
endfunction()

function(mark_indirect_load_static_shape test_name)
    target_compile_definitions(${test_name}_codegen_v2 PRIVATE IL_STATIC_SHAPE)
    target_compile_definitions(${test_name}_e2e_v2 PRIVATE IL_STATIC_SHAPE)
endfunction()

function(mark_indirect_load_sk test_name)
    target_compile_definitions(${test_name}_codegen_v2 PRIVATE IL_EXPECT_SK)
    target_compile_definitions(${test_name}_e2e_v2 PRIVATE IL_EXPECT_SK)
endfunction()

function(mark_indirect_load_wide_types test_name)
    target_compile_definitions(${test_name}_codegen_v2 PRIVATE IL_DATA_FLOAT IL_INDEX_INT64)
    target_compile_definitions(${test_name}_e2e_v2 PRIVATE IL_DATA_FLOAT IL_INDEX_INT64)
endfunction()

function(mark_indirect_load_input_outer_stride test_name stride)
    target_compile_definitions(${test_name}_codegen_v2 PRIVATE IL_INPUT_OUTER_STRIDE=${stride})
    target_compile_definitions(${test_name}_e2e_v2 PRIVATE IL_INPUT_OUTER_STRIDE=${stride})
endfunction()
function(mark_indirect_load_large_simt test_name)
    target_compile_definitions(${test_name}_e2e_v2 PRIVATE IL_EXPECT_SIMT_MULTI_ROUND)
endfunction()

function(mark_indirect_load_mixed_index_pre test_name)
    target_compile_definitions(${test_name}_codegen_v2 PRIVATE IL_MIXED_INDEX_PRE)
    target_compile_definitions(${test_name}_e2e_v2 PRIVATE IL_MIXED_INDEX_PRE)
endfunction()

function(mark_indirect_load_output_post test_name output_post_type expect_simt)
    target_compile_definitions(${test_name}_codegen_v2 PRIVATE
        IL_OUTPUT_POST_TYPE=${output_post_type}
        IL_EXPECT_SIMT=${expect_simt})
    target_compile_definitions(${test_name}_e2e_v2 PRIVATE
        IL_OUTPUT_POST_TYPE=${output_post_type}
        IL_EXPECT_SIMT=${expect_simt})
endfunction()

function(mark_indirect_load_simt_selected test_name)
    target_compile_definitions(${test_name}_codegen_v2 PRIVATE IL_EXPECT_SIMT_SELECTED)
    target_compile_definitions(${test_name}_e2e_v2 PRIVATE IL_EXPECT_SIMT_SELECTED)
endfunction()

function(mark_indirect_load_simt_codegen test_name policy offset_bits)
    target_compile_definitions(${test_name}_codegen_v2 PRIVATE
        IL_EXPECT_SIMT_POLICY=${policy}
        IL_EXPECT_SIMT_OFFSET_BITS=${offset_bits})
endfunction()

function(mark_indirect_load_implementation test_name implementation)
    target_compile_definitions(${test_name}_codegen_v2 PRIVATE
        IL_SELECTED_IMPLEMENTATION=${implementation})
endfunction()

function(add_indirect_load_mixed_e2e_case test_name rank axis is_simt selected_template expected_templates static_shape x0 x1 x2 x3 index0 index1 index2 index3)
    if(is_simt)
        set(tiling_key 2)
    else()
        set(tiling_key 0)
    endif()
    set(case_workdir ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
    file(MAKE_DIRECTORY ${case_workdir})
    do_backend_e2e_st_test(${test_name}
        WORKDIR ${case_workdir}
        CODEGEN indirect_load_store_backend_generator.cpp
        TILING_KEY ${tiling_key}
        KERNEL_SRC
            indirect_load_mixed_elementwise_test_kernel.cpp
            indirect_load_mixed_elementwise_test_tiling.cpp
            autofuse_tiling_data.h
        TEST_SRC test_e2e_indirect_load_store_kernel.cpp)
    set(case_definitions
        IL_MIXED_ELEMENTWISE
        IL_EXPECT_SIMT=${is_simt}
        IL_RANK=${rank}
        IL_AXIS=${axis}
        IL_INPUT_PRE_TYPE=0
        IL_USE_EXP2=1
        IL_TILING_KEY=${tiling_key}
        IL_SELECTED_TEMPLATE=${selected_template}
        IL_EXPECTED_TEMPLATES=${expected_templates}
        IL_X_S0=${x0}
        IL_X_S1=${x1}
        IL_X_S2=${x2}
        IL_X_S3=${x3}
        IL_INDEX_S0=${index0}
        IL_INDEX_S1=${index1}
        IL_INDEX_S2=${index2}
        IL_INDEX_S3=${index3})
    if(static_shape)
        list(APPEND case_definitions IL_STATIC_SHAPE)
    endif()
    target_compile_definitions(${test_name}_codegen_v2 PRIVATE ${case_definitions})
    target_compile_definitions(${test_name}_e2e_v2 PRIVATE ${case_definitions})
endfunction()

function(mark_indirect_load_simt_elementwise_coverage test_name)
    target_compile_definitions(${test_name}_codegen_v2 PRIVATE IL_SIMT_ELEMENTWISE_COVERAGE)
    target_compile_definitions(${test_name}_e2e_v2 PRIVATE IL_SIMT_ELEMENTWISE_COVERAGE)
endfunction()

add_indirect_load_e2e_case(indirect_load_rank2_axis1_simd 2 1 1 1 0 0 3 5 8 1 1 4 6 1 1)
mark_indirect_load_static_shape(indirect_load_rank2_axis1_simd)

add_indirect_load_e2e_case(indirect_load_rank2_axis1_half_int64_simd 2 1 1 1 0 0 3 5 160 1 1 4 137 1 1)
mark_indirect_load_static_shape(indirect_load_rank2_axis1_half_int64_simd)
target_compile_definitions(indirect_load_rank2_axis1_half_int64_simd_codegen_v2 PRIVATE
    IL_INDEX_INT64 IL_FUNCTIONAL_ONLY)
target_compile_definitions(indirect_load_rank2_axis1_half_int64_simd_e2e_v2 PRIVATE
    IL_INDEX_INT64)

add_indirect_load_e2e_case(indirect_load_rank2_axis1_b16_gather 2 1 1 1 1 0 3 5 8 1 1 4 6 1 1)
mark_indirect_load_static_shape(indirect_load_rank2_axis1_b16_gather)
mark_indirect_load_implementation(indirect_load_rank2_axis1_b16_gather 1)

add_indirect_load_e2e_case(indirect_load_rank3_axis1_float_int64_static 3 1 0 0 0 0 7 16 32 10 1 16 32 10 1)
mark_indirect_load_static_shape(indirect_load_rank3_axis1_float_int64_static)
target_compile_definitions(indirect_load_rank3_axis1_float_int64_static_codegen_v2 PRIVATE
    IL_DATA_FLOAT IL_INDEX_INT64 IL_OUTPUT_POST_TYPE=2 IL_RANDOM_INPUT_INDEX IL_DIRECT_INDEX)
target_compile_definitions(indirect_load_rank3_axis1_float_int64_static_e2e_v2 PRIVATE
    IL_DATA_FLOAT IL_INDEX_INT64 IL_OUTPUT_POST_TYPE=2 IL_RANDOM_INPUT_INDEX IL_DIRECT_INDEX)

add_indirect_load_e2e_case(indirect_load_rank3_axis1_pow2_simd 3 1 0 0 0 0 3 4 64 32 1 4 64 32 1)
mark_indirect_load_static_shape(indirect_load_rank3_axis1_pow2_simd)
target_compile_definitions(indirect_load_rank3_axis1_pow2_simd_codegen_v2 PRIVATE
    IL_RANDOM_INPUT_INDEX IL_DIRECT_INDEX IL_EXPECT_MICRO_SIMD)
target_compile_definitions(indirect_load_rank3_axis1_pow2_simd_e2e_v2 PRIVATE
    IL_RANDOM_INPUT_INDEX IL_DIRECT_INDEX)

add_indirect_load_e2e_case(indirect_load_rank3_axis1_pow2_gather 3 1 0 0 1 0 3 4 64 32 1 4 64 32 1)
mark_indirect_load_static_shape(indirect_load_rank3_axis1_pow2_gather)
mark_indirect_load_implementation(indirect_load_rank3_axis1_pow2_gather 1)
target_compile_definitions(indirect_load_rank3_axis1_pow2_gather_codegen_v2 PRIVATE
    IL_RANDOM_INPUT_INDEX IL_DIRECT_INDEX)
target_compile_definitions(indirect_load_rank3_axis1_pow2_gather_e2e_v2 PRIVATE
    IL_RANDOM_INPUT_INDEX IL_DIRECT_INDEX)

add_indirect_load_e2e_case(indirect_load_rank3_axis1_pow2_simt 3 1 0 0 2 1 3 4 64 32 1 4 64 32 1)
mark_indirect_load_static_shape(indirect_load_rank3_axis1_pow2_simt)
mark_indirect_load_simt_selected(indirect_load_rank3_axis1_pow2_simt)
mark_indirect_load_simt_codegen(indirect_load_rank3_axis1_pow2_simt StaticPowerOfTwo 32)
target_compile_definitions(indirect_load_rank3_axis1_pow2_simt_codegen_v2 PRIVATE
    IL_DATA_FLOAT IL_RANDOM_INPUT_INDEX IL_DIRECT_INDEX)
target_compile_definitions(indirect_load_rank3_axis1_pow2_simt_e2e_v2 PRIVATE
    IL_DATA_FLOAT IL_RANDOM_INPUT_INDEX IL_DIRECT_INDEX)

add_indirect_load_e2e_case(indirect_load_rank3_axis1_float_int32_gather 3 1 0 0 1 0 7 16 32 10 1 16 32 10 1)
mark_indirect_load_static_shape(indirect_load_rank3_axis1_float_int32_gather)
mark_indirect_load_implementation(indirect_load_rank3_axis1_float_int32_gather 1)
target_compile_definitions(indirect_load_rank3_axis1_float_int32_gather_codegen_v2 PRIVATE
    IL_DATA_FLOAT IL_OUTPUT_POST_TYPE=2 IL_RANDOM_INPUT_INDEX IL_DIRECT_INDEX)
target_compile_definitions(indirect_load_rank3_axis1_float_int32_gather_e2e_v2 PRIVATE
    IL_DATA_FLOAT IL_OUTPUT_POST_TYPE=2 IL_RANDOM_INPUT_INDEX IL_DIRECT_INDEX)

add_indirect_load_e2e_case(indirect_load_rank3_axis1_float_int64_gather_exact 3 1 0 0 1 0 7 16 32 10 1 16 32 10 1)
mark_indirect_load_static_shape(indirect_load_rank3_axis1_float_int64_gather_exact)
mark_indirect_load_implementation(indirect_load_rank3_axis1_float_int64_gather_exact 1)
target_compile_definitions(indirect_load_rank3_axis1_float_int64_gather_exact_codegen_v2 PRIVATE
    IL_DATA_FLOAT IL_INDEX_INT64 IL_OUTPUT_POST_TYPE=2 IL_RANDOM_INPUT_INDEX IL_DIRECT_INDEX)
target_compile_definitions(indirect_load_rank3_axis1_float_int64_gather_exact_e2e_v2 PRIVATE
    IL_DATA_FLOAT IL_INDEX_INT64 IL_OUTPUT_POST_TYPE=2 IL_RANDOM_INPUT_INDEX IL_DIRECT_INDEX)

add_indirect_load_e2e_case(indirect_load_rank3_axis1_float_int64_gather 3 1 0 0 1 0 7 16 31 10 1 16 31 10 1)
mark_indirect_load_static_shape(indirect_load_rank3_axis1_float_int64_gather)
mark_indirect_load_implementation(indirect_load_rank3_axis1_float_int64_gather 1)
target_compile_definitions(indirect_load_rank3_axis1_float_int64_gather_codegen_v2 PRIVATE
    IL_DATA_FLOAT IL_INDEX_INT64 IL_OUTPUT_POST_TYPE=2 IL_RANDOM_INPUT_INDEX IL_DIRECT_INDEX)
target_compile_definitions(indirect_load_rank3_axis1_float_int64_gather_e2e_v2 PRIVATE
    IL_DATA_FLOAT IL_INDEX_INT64 IL_OUTPUT_POST_TYPE=2 IL_RANDOM_INPUT_INDEX IL_DIRECT_INDEX)

add_indirect_load_e2e_case(indirect_load_rank4_axis1_float_int64_strided_post_simd
    4 1 0 0 0 0 7 3 17 5 11 3 19 3 7)
mark_indirect_load_static_shape(indirect_load_rank4_axis1_float_int64_strided_post_simd)
target_compile_definitions(indirect_load_rank4_axis1_float_int64_strided_post_simd_codegen_v2 PRIVATE
    IL_DATA_FLOAT IL_INDEX_INT64)
target_compile_definitions(indirect_load_rank4_axis1_float_int64_strided_post_simd_e2e_v2 PRIVATE
    IL_DATA_FLOAT IL_INDEX_INT64)

add_indirect_load_e2e_case(indirect_load_rank4_axis1_direct_index_abs_exp2_sum_simd 4 1 0 0 0 0 3 8 13 3 8 8 17 3 8)
mark_indirect_load_static_shape(indirect_load_rank4_axis1_direct_index_abs_exp2_sum_simd)
mark_indirect_load_output_post(indirect_load_rank4_axis1_direct_index_abs_exp2_sum_simd 8 1)
target_compile_definitions(indirect_load_rank4_axis1_direct_index_abs_exp2_sum_simd_codegen_v2 PRIVATE
    IL_DATA_FLOAT IL_POST_REDUCE IL_POST_REDUCE_EXP2 IL_POST_REDUCE_ABS IL_EXPECT_SIMD_SELECTED
    IL_RANDOM_INPUT_INDEX IL_DIRECT_INDEX)
target_compile_definitions(indirect_load_rank4_axis1_direct_index_abs_exp2_sum_simd_e2e_v2 PRIVATE
    IL_DATA_FLOAT IL_POST_REDUCE IL_POST_REDUCE_EXP2 IL_POST_REDUCE_ABS IL_EXPECT_SIMD_SELECTED
    IL_RANDOM_INPUT_INDEX IL_DIRECT_INDEX)

add_indirect_load_e2e_case(indirect_load_rank4_axis2_sk 4 2 1 0 3 2 7 2 2 5 6 2 2 5 6)
mark_indirect_load_static_shape(indirect_load_rank4_axis2_sk)
mark_indirect_load_sk(indirect_load_rank4_axis2_sk)
add_indirect_load_e2e_case(indirect_load_rank2_axisneg2_simd 2 -2 1 1 0 0 3 4 9 1 1 4 8 1 1)
add_indirect_load_e2e_case(indirect_load_rank2_axis1_relu_exp2_pre_simd 2 1 3 0 0 0 3 5 8 1 1 4 6 1 1)
add_indirect_load_e2e_case(indirect_load_rank3_axis1_torch_gather_frontend 3 1 0 0 3 2 7 54 4 8 1 54 4 8 1)
mark_indirect_load_static_shape(indirect_load_rank3_axis1_torch_gather_frontend)
mark_indirect_load_sk(indirect_load_rank3_axis1_torch_gather_frontend)
mark_indirect_load_wide_types(indirect_load_rank3_axis1_torch_gather_frontend)
mark_indirect_load_output_post(indirect_load_rank3_axis1_torch_gather_frontend 2 1)
mark_indirect_load_input_outer_stride(indirect_load_rank3_axis1_torch_gather_frontend 64)

add_indirect_load_e2e_case(indirect_load_rank2_axis1_large_simt 2 1 1 0 2 1 3 2 64 1 1 2 32771 1 1)
mark_indirect_load_large_simt(indirect_load_rank2_axis1_large_simt)
mark_indirect_load_simt_codegen(indirect_load_rank2_axis1_large_simt Recursive 64)

add_indirect_load_e2e_case(indirect_load_rank2_axis1_bf16_int64_simt 2 1 0 0 2 1 3 5 8 1 1 4 6 1 1)
mark_indirect_load_output_post(indirect_load_rank2_axis1_bf16_int64_simt 2 1)
target_compile_definitions(indirect_load_rank2_axis1_bf16_int64_simt_codegen_v2 PRIVATE IL_DATA_BF16 IL_INDEX_INT64)
target_compile_definitions(indirect_load_rank2_axis1_bf16_int64_simt_e2e_v2 PRIVATE IL_DATA_BF16 IL_INDEX_INT64)

add_indirect_load_e2e_case(indirect_load_rank2_axis1_uint32_int32_simd 2 1 0 1 0 0 3 5 8 1 1 4 6 1 1)
mark_indirect_load_output_post(indirect_load_rank2_axis1_uint32_int32_simd 3 1)
target_compile_definitions(indirect_load_rank2_axis1_uint32_int32_simd_codegen_v2 PRIVATE IL_DATA_UINT32)
target_compile_definitions(indirect_load_rank2_axis1_uint32_int32_simd_e2e_v2 PRIVATE IL_DATA_UINT32)

add_indirect_load_e2e_case(indirect_load_rank4_axis1_add_sum_simt 4 1 0 0 0 1 3 8 13 3 8 8 12 3 8)
mark_indirect_load_static_shape(indirect_load_rank4_axis1_add_sum_simt)
mark_indirect_load_output_post(indirect_load_rank4_axis1_add_sum_simt 10 1)
target_compile_definitions(indirect_load_rank4_axis1_add_sum_simt_codegen_v2 PRIVATE IL_DATA_FLOAT IL_POST_REDUCE IL_POST_REDUCE_ADD)
target_compile_definitions(indirect_load_rank4_axis1_add_sum_simt_e2e_v2 PRIVATE IL_DATA_FLOAT IL_POST_REDUCE IL_POST_REDUCE_ADD)
mark_indirect_load_simt_selected(indirect_load_rank4_axis1_add_sum_simt)
mark_indirect_load_simt_codegen(indirect_load_rank4_axis1_add_sum_simt StructuredMagic 32)

add_indirect_load_e2e_case(indirect_load_rank4_axis1_sum_axis2_with_a_fallback_simt 4 1 3 0 0 1 2 8 13 3 8 8 17 3 8)
mark_indirect_load_static_shape(indirect_load_rank4_axis1_sum_axis2_with_a_fallback_simt)
mark_indirect_load_mixed_index_pre(indirect_load_rank4_axis1_sum_axis2_with_a_fallback_simt)
mark_indirect_load_output_post(indirect_load_rank4_axis1_sum_axis2_with_a_fallback_simt 5 1)
target_compile_definitions(indirect_load_rank4_axis1_sum_axis2_with_a_fallback_simt_codegen_v2 PRIVATE
    IL_DATA_FLOAT IL_POST_REDUCE IL_REDUCE_WITH_A IL_EXPECT_ONLY_SIMT)
target_compile_definitions(indirect_load_rank4_axis1_sum_axis2_with_a_fallback_simt_e2e_v2 PRIVATE
    IL_DATA_FLOAT IL_POST_REDUCE IL_REDUCE_WITH_A IL_EXPECT_ONLY_SIMT)

add_indirect_load_e2e_case(indirect_load_rank4_axis1_sum_last_axis_simt 4 1 0 0 0 1 3 8 13 3 8 8 12 3 8)
mark_indirect_load_static_shape(indirect_load_rank4_axis1_sum_last_axis_simt)
mark_indirect_load_output_post(indirect_load_rank4_axis1_sum_last_axis_simt 6 1)
target_compile_definitions(indirect_load_rank4_axis1_sum_last_axis_simt_codegen_v2 PRIVATE IL_DATA_FLOAT IL_POST_REDUCE IL_REDUCE_LAST_AXIS)
target_compile_definitions(indirect_load_rank4_axis1_sum_last_axis_simt_e2e_v2 PRIVATE IL_DATA_FLOAT IL_POST_REDUCE IL_REDUCE_LAST_AXIS)
mark_indirect_load_simt_selected(indirect_load_rank4_axis1_sum_last_axis_simt)

add_indirect_load_e2e_case(indirect_load_rank4_axis1_full_prefix_bessel_k0_sum_simd 4 1 3 0 0 0 1 8 13 3 8 8 17 3 8)
mark_indirect_load_static_shape(indirect_load_rank4_axis1_full_prefix_bessel_k0_sum_simd)
mark_indirect_load_mixed_index_pre(indirect_load_rank4_axis1_full_prefix_bessel_k0_sum_simd)
mark_indirect_load_output_post(indirect_load_rank4_axis1_full_prefix_bessel_k0_sum_simd 11 0)
target_compile_definitions(indirect_load_rank4_axis1_full_prefix_bessel_k0_sum_simd_codegen_v2 PRIVATE
    IL_DATA_FLOAT IL_POST_REDUCE IL_POST_REDUCE_SIMD)
target_compile_definitions(indirect_load_rank4_axis1_full_prefix_bessel_k0_sum_simd_e2e_v2 PRIVATE
    IL_DATA_FLOAT IL_POST_REDUCE IL_POST_REDUCE_SIMD)
target_include_directories(indirect_load_rank4_axis1_full_prefix_bessel_k0_sum_simd_e2e_v2 PRIVATE
    ${CODE_ROOT_DIR}/v35/ascendc/api_regbase)

add_indirect_load_mixed_e2e_case(indirect_load_mixed_rank4_axis2_simd 4 2 0 0 1 FALSE 3 4 16 7 2 3 17 5)
add_indirect_load_mixed_e2e_case(indirect_load_simt_elementwise_coverage 3 1 1 1 3 FALSE 3 16 7 1 2 11 5 1)
mark_indirect_load_simt_elementwise_coverage(indirect_load_simt_elementwise_coverage)
mark_indirect_load_simt_codegen(indirect_load_simt_elementwise_coverage Recursive 64)

include(indirect_load_broadcast_cases.cmake)
include(indirect_load_torch_gather_strided_cases.cmake)
