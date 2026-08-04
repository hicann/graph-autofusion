function(add_indirect_load_e2e_case test_name rank axis input_pre_type use_exp2 tiling_key x0 x1 x2 x3 index0 index1 index2 index3)
    set(case_workdir ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
    file(MAKE_DIRECTORY ${case_workdir})
    do_backend_e2e_st_test(${test_name}
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

function(mark_indirect_load_large_simt test_name)
    target_compile_definitions(${test_name}_e2e_v2 PRIVATE IL_EXPECT_SIMT_MULTI_ROUND)
endfunction()

function(mark_indirect_load_input_pre_vf test_name expected)
    target_compile_definitions(${test_name}_codegen_v2 PRIVATE IL_EXPECT_INPUT_PRE_VF=${expected})
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

add_indirect_load_e2e_case(indirect_load_rank2_axis1_simt 2 1 1 0 1 5 8 1 1 4 6 1 1)
add_indirect_load_e2e_case(indirect_load_rank2_axis1_simd 2 1 1 1 0 5 8 1 1 4 6 1 1)
mark_indirect_load_static_shape(indirect_load_rank2_axis1_simd)
add_indirect_load_e2e_case(indirect_load_rank2_axisneg2_simt 2 -2 1 0 1 4 9 1 1 4 8 1 1)
add_indirect_load_e2e_case(indirect_load_rank2_axisneg2_simd 2 -2 1 1 0 4 9 1 1 4 8 1 1)
add_indirect_load_e2e_case(indirect_load_rank3_axis1_simt 3 1 1 0 1 3 5 8 1 2 7 6 1)
add_indirect_load_e2e_case(indirect_load_rank3_axis1_simd 3 1 1 1 0 3 5 8 1 2 7 6 1)
add_indirect_load_e2e_case(indirect_load_rank4_axis2_simt 4 2 1 0 1 3 3 5 8 2 2 5 6)
add_indirect_load_e2e_case(indirect_load_rank4_axis2_simd 4 2 1 1 0 3 3 5 8 2 2 5 6)
add_indirect_load_e2e_case(indirect_load_rank4_axis2_sk 4 2 1 0 2 2 2 5 6 2 2 5 6)
mark_indirect_load_static_shape(indirect_load_rank4_axis2_sk)
mark_indirect_load_sk(indirect_load_rank4_axis2_sk)
add_indirect_load_e2e_case(indirect_load_rank2_axis1_relu_exp2_pre_simd 2 1 3 0 0 5 8 1 1 4 6 1 1)

# Static output expansion keeps both data and index preprocessing before IndirectLoad SIMD.
add_indirect_load_e2e_case(indirect_load_rank2_axis0_relu_pre_expanded_simd 2 0 1 1 0 4 9 1 1 6 8 1 1)
add_indirect_load_e2e_case(indirect_load_rank2_axis0_relu_relu_pre_expanded_simd 2 0 4 1 0 4 9 1 1 6 8 1 1)
add_indirect_load_e2e_case(indirect_load_rank3_axis2_exp2_pre_expanded_simd 3 2 2 0 0 4 5 6 1 3 4 11 1)
add_indirect_load_e2e_case(indirect_load_rank4_axis1_relu_exp2_pre_expanded_simd 4 1 3 0 0 2 3 4 5 2 7 4 3)
add_indirect_load_e2e_case(indirect_load_rank3_axis2_mixed_vf_pre_expanded_simd 3 2 3 0 0 4 5 6 1 3 4 11 1)
mark_indirect_load_static_shape(indirect_load_rank2_axis0_relu_pre_expanded_simd)
mark_indirect_load_static_shape(indirect_load_rank2_axis0_relu_relu_pre_expanded_simd)
mark_indirect_load_static_shape(indirect_load_rank3_axis2_exp2_pre_expanded_simd)
mark_indirect_load_static_shape(indirect_load_rank4_axis1_relu_exp2_pre_expanded_simd)
mark_indirect_load_static_shape(indirect_load_rank3_axis2_mixed_vf_pre_expanded_simd)
mark_indirect_load_mixed_index_pre(indirect_load_rank3_axis2_mixed_vf_pre_expanded_simd)
target_compile_definitions(indirect_load_rank2_axis0_relu_pre_expanded_simd_codegen_v2 PRIVATE IL_DATA_FLOAT)
target_compile_definitions(indirect_load_rank2_axis0_relu_pre_expanded_simd_e2e_v2 PRIVATE IL_DATA_FLOAT)
target_compile_definitions(indirect_load_rank2_axis0_relu_relu_pre_expanded_simd_codegen_v2 PRIVATE IL_DATA_FLOAT)
target_compile_definitions(indirect_load_rank2_axis0_relu_relu_pre_expanded_simd_e2e_v2 PRIVATE IL_DATA_FLOAT)
mark_indirect_load_input_pre_vf(indirect_load_rank2_axis0_relu_pre_expanded_simd 0)
mark_indirect_load_input_pre_vf(indirect_load_rank2_axis0_relu_relu_pre_expanded_simd 1)

add_indirect_load_e2e_case(indirect_load_rank2_axis1_large_simt 2 1 1 0 1 2 64 1 1 2 32771 1 1)
mark_indirect_load_large_simt(indirect_load_rank2_axis1_large_simt)
add_indirect_load_e2e_case(indirect_load_rank2_axis1_float_int64_simd 2 1 0 1 0 4 8 1 1 3 6 1 1)
mark_indirect_load_wide_types(indirect_load_rank2_axis1_float_int64_simd)

add_indirect_load_e2e_case(indirect_load_rank2_axis1_modified_bessel_k0_simd_only 2 1 0 0 0 5 8 1 1 4 6 1 1)
mark_indirect_load_output_post(indirect_load_rank2_axis1_modified_bessel_k0_simd_only 1 0)
target_compile_definitions(indirect_load_rank2_axis1_modified_bessel_k0_simd_only_codegen_v2 PRIVATE IL_DATA_FLOAT)
target_compile_definitions(indirect_load_rank2_axis1_modified_bessel_k0_simd_only_e2e_v2 PRIVATE IL_DATA_FLOAT)
target_include_directories(indirect_load_rank2_axis1_modified_bessel_k0_simd_only_e2e_v2 PRIVATE
    ${CODE_ROOT_DIR}/v35/ascendc/api_regbase)

add_indirect_load_e2e_case(indirect_load_rank2_axis1_precision_cast_simt 2 1 5 0 1 5 8 1 1 4 6 1 1)
add_indirect_load_e2e_case(indirect_load_rank2_axis1_precision_cast_simd 2 1 5 1 0 5 8 1 1 4 6 1 1)

add_indirect_load_e2e_case(indirect_load_rank2_axis1_bf16_int64_simt 2 1 0 0 1 5 8 1 1 4 6 1 1)
mark_indirect_load_output_post(indirect_load_rank2_axis1_bf16_int64_simt 2 1)
target_compile_definitions(indirect_load_rank2_axis1_bf16_int64_simt_codegen_v2 PRIVATE IL_DATA_BF16 IL_INDEX_INT64)
target_compile_definitions(indirect_load_rank2_axis1_bf16_int64_simt_e2e_v2 PRIVATE IL_DATA_BF16 IL_INDEX_INT64)

add_indirect_load_e2e_case(indirect_load_rank2_axis1_uint32_int32_simd 2 1 0 1 0 5 8 1 1 4 6 1 1)
mark_indirect_load_output_post(indirect_load_rank2_axis1_uint32_int32_simd 3 1)
target_compile_definitions(indirect_load_rank2_axis1_uint32_int32_simd_codegen_v2 PRIVATE IL_DATA_UINT32)
target_compile_definitions(indirect_load_rank2_axis1_uint32_int32_simd_e2e_v2 PRIVATE IL_DATA_UINT32)
