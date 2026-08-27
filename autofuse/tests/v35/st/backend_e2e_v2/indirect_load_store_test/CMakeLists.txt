function(mark_indirect_load_codegen_and_e2e test_name)
    target_compile_definitions(${test_name}_codegen_v2 PRIVATE ${ARGN})
    target_compile_definitions(${test_name}_e2e_v2 PRIVATE ${ARGN})
endfunction()

function(mark_indirect_load_red test_name)
    set_property(TEST ${test_name}_codegen_v2 ${test_name}_e2e_v2 APPEND PROPERTY LABELS red)
endfunction()

function(mark_indirect_load_codegen test_name)
    target_compile_definitions(${test_name}_codegen_v2 PRIVATE ${ARGN})
endfunction()

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
        IL_CASE_STORE
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
    mark_indirect_load_codegen_and_e2e(${test_name} ${case_definitions})
endfunction()

function(mark_indirect_load_static_shape test_name)
    mark_indirect_load_codegen_and_e2e(${test_name} IL_STATIC_SHAPE)
endfunction()

function(mark_indirect_load_sk test_name)
    mark_indirect_load_codegen_and_e2e(${test_name} IL_EXPECT_SK)
endfunction()

function(mark_indirect_load_wide_types test_name)
    mark_indirect_load_codegen_and_e2e(${test_name} IL_DATA_FLOAT IL_INDEX_INT64)
endfunction()

function(mark_indirect_load_input_outer_stride test_name stride)
    mark_indirect_load_codegen_and_e2e(${test_name} IL_INPUT_OUTER_STRIDE=${stride})
endfunction()
function(mark_indirect_load_large_simt test_name)
    target_compile_definitions(${test_name}_e2e_v2 PRIVATE IL_EXPECT_SIMT_MULTI_ROUND)
endfunction()

function(mark_indirect_load_mixed_index_pre test_name)
    mark_indirect_load_codegen_and_e2e(${test_name} IL_MIXED_INDEX_PRE)
endfunction()

function(mark_indirect_load_output_post test_name output_post_type expect_simt)
    mark_indirect_load_codegen_and_e2e(${test_name}
        IL_OUTPUT_POST_TYPE=${output_post_type}
        IL_EXPECT_SIMT=${expect_simt})
endfunction()

function(mark_indirect_load_simt_selected test_name)
    mark_indirect_load_codegen_and_e2e(${test_name} IL_EXPECT_SIMT_SELECTED)
endfunction()

function(mark_indirect_load_simt_codegen test_name policy offset_bits)
    mark_indirect_load_codegen(${test_name}
        IL_EXPECT_SIMT_POLICY=${policy}
        IL_EXPECT_SIMT_OFFSET_BITS=${offset_bits})
endfunction()

function(mark_indirect_load_implementation test_name implementation)
    mark_indirect_load_codegen(${test_name}
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
        IL_CASE_STORE
        IL_CASE_MIXED
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
    mark_indirect_load_codegen_and_e2e(${test_name} ${case_definitions})
endfunction()

function(mark_indirect_load_simt_elementwise_coverage test_name)
    mark_indirect_load_codegen_and_e2e(${test_name} IL_SIMT_ELEMENTWISE_COVERAGE)
endfunction()

add_indirect_load_e2e_case(indirect_load_rank2_axis1_simd 2 1 1 1 0 0 3 5 8 1 1 4 6 1 1)
mark_indirect_load_static_shape(indirect_load_rank2_axis1_simd)

add_indirect_load_e2e_case(indirect_load_rank2_axis1_half_int64_simd 2 1 1 1 0 0 3 5 160 1 1 4 137 1 1)
mark_indirect_load_static_shape(indirect_load_rank2_axis1_half_int64_simd)
mark_indirect_load_codegen_and_e2e(indirect_load_rank2_axis1_half_int64_simd IL_INDEX_INT64)
target_compile_definitions(indirect_load_rank2_axis1_half_int64_simd_codegen_v2 PRIVATE IL_FUNCTIONAL_ONLY)

add_indirect_load_e2e_case(indirect_load_rank2_axis1_b16_gather 2 1 1 1 1 0 3 5 8 1 1 4 6 1 1)
mark_indirect_load_static_shape(indirect_load_rank2_axis1_b16_gather)
mark_indirect_load_implementation(indirect_load_rank2_axis1_b16_gather 1)

add_indirect_load_e2e_case(indirect_load_rank3_axis1_float_int64_static 3 1 0 0 0 0 7 16 32 10 1 16 32 10 1)
mark_indirect_load_static_shape(indirect_load_rank3_axis1_float_int64_static)
mark_indirect_load_codegen_and_e2e(indirect_load_rank3_axis1_float_int64_static
    IL_DATA_FLOAT IL_INDEX_INT64 IL_OUTPUT_POST_TYPE=2 IL_RANDOM_INPUT_INDEX IL_DIRECT_INDEX)

add_indirect_load_e2e_case(indirect_load_rank3_axis1_pow2_simd 3 1 0 0 0 0 3 4 64 32 1 4 64 32 1)
mark_indirect_load_static_shape(indirect_load_rank3_axis1_pow2_simd)
mark_indirect_load_codegen_and_e2e(indirect_load_rank3_axis1_pow2_simd IL_RANDOM_INPUT_INDEX IL_DIRECT_INDEX)
target_compile_definitions(indirect_load_rank3_axis1_pow2_simd_codegen_v2 PRIVATE IL_EXPECT_MICRO_SIMD)

add_indirect_load_e2e_case(indirect_load_rank3_axis1_pow2_gather 3 1 0 0 1 0 3 4 64 32 1 4 64 32 1)
mark_indirect_load_static_shape(indirect_load_rank3_axis1_pow2_gather)
mark_indirect_load_implementation(indirect_load_rank3_axis1_pow2_gather 1)
mark_indirect_load_codegen_and_e2e(indirect_load_rank3_axis1_pow2_gather IL_RANDOM_INPUT_INDEX IL_DIRECT_INDEX)

add_indirect_load_e2e_case(indirect_load_rank3_axis1_pow2_simt 3 1 0 0 2 1 3 4 64 32 1 4 64 32 1)
mark_indirect_load_static_shape(indirect_load_rank3_axis1_pow2_simt)
mark_indirect_load_simt_selected(indirect_load_rank3_axis1_pow2_simt)
mark_indirect_load_simt_codegen(indirect_load_rank3_axis1_pow2_simt StaticPowerOfTwo 32)
mark_indirect_load_codegen_and_e2e(indirect_load_rank3_axis1_pow2_simt IL_DATA_FLOAT IL_RANDOM_INPUT_INDEX IL_DIRECT_INDEX)

add_indirect_load_e2e_case(indirect_load_rank3_axis1_float_int32_gather 3 1 0 0 1 0 7 16 32 10 1 16 32 10 1)
mark_indirect_load_static_shape(indirect_load_rank3_axis1_float_int32_gather)
mark_indirect_load_implementation(indirect_load_rank3_axis1_float_int32_gather 1)
mark_indirect_load_codegen_and_e2e(indirect_load_rank3_axis1_float_int32_gather
    IL_DATA_FLOAT IL_OUTPUT_POST_TYPE=2 IL_RANDOM_INPUT_INDEX IL_DIRECT_INDEX)

add_indirect_load_e2e_case(indirect_load_rank3_axis1_float_int64_gather 3 1 0 0 1 0 7 16 31 10 1 16 31 10 1)
mark_indirect_load_static_shape(indirect_load_rank3_axis1_float_int64_gather)
mark_indirect_load_implementation(indirect_load_rank3_axis1_float_int64_gather 1)
mark_indirect_load_codegen_and_e2e(indirect_load_rank3_axis1_float_int64_gather
    IL_DATA_FLOAT IL_INDEX_INT64 IL_OUTPUT_POST_TYPE=2 IL_RANDOM_INPUT_INDEX IL_DIRECT_INDEX)

add_indirect_load_e2e_case(indirect_load_rank4_axis1_float_int64_strided_post_simd
    4 1 0 0 0 0 7 3 17 5 11 3 19 3 7)
mark_indirect_load_static_shape(indirect_load_rank4_axis1_float_int64_strided_post_simd)
mark_indirect_load_codegen_and_e2e(indirect_load_rank4_axis1_float_int64_strided_post_simd IL_DATA_FLOAT IL_INDEX_INT64)

add_indirect_load_e2e_case(indirect_load_rank4_axis1_direct_index_abs_exp2_sum_simd 4 1 0 0 0 0 3 8 13 3 8 8 17 3 8)
mark_indirect_load_static_shape(indirect_load_rank4_axis1_direct_index_abs_exp2_sum_simd)
mark_indirect_load_output_post(indirect_load_rank4_axis1_direct_index_abs_exp2_sum_simd 8 1)
mark_indirect_load_codegen_and_e2e(indirect_load_rank4_axis1_direct_index_abs_exp2_sum_simd
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
mark_indirect_load_codegen_and_e2e(indirect_load_rank2_axis1_bf16_int64_simt IL_DATA_BF16 IL_INDEX_INT64)

add_indirect_load_e2e_case(indirect_load_rank2_axis1_uint32_int32_simd 2 1 0 1 0 0 3 5 8 1 1 4 6 1 1)
mark_indirect_load_output_post(indirect_load_rank2_axis1_uint32_int32_simd 3 1)
mark_indirect_load_codegen_and_e2e(indirect_load_rank2_axis1_uint32_int32_simd IL_DATA_UINT32)

add_indirect_load_e2e_case(indirect_load_rank4_axis1_add_sum_simt 4 1 0 0 0 1 3 8 13 3 8 8 12 3 8)
mark_indirect_load_static_shape(indirect_load_rank4_axis1_add_sum_simt)
mark_indirect_load_output_post(indirect_load_rank4_axis1_add_sum_simt 10 1)
mark_indirect_load_codegen_and_e2e(indirect_load_rank4_axis1_add_sum_simt IL_DATA_FLOAT IL_POST_REDUCE IL_POST_REDUCE_ADD)
mark_indirect_load_simt_selected(indirect_load_rank4_axis1_add_sum_simt)
mark_indirect_load_simt_codegen(indirect_load_rank4_axis1_add_sum_simt StructuredMagic 32)

add_indirect_load_e2e_case(indirect_load_rank4_axis1_sum_axis2_with_a_fallback_simt 4 1 3 0 0 1 2 8 13 3 8 8 17 3 8)
mark_indirect_load_static_shape(indirect_load_rank4_axis1_sum_axis2_with_a_fallback_simt)
mark_indirect_load_mixed_index_pre(indirect_load_rank4_axis1_sum_axis2_with_a_fallback_simt)
mark_indirect_load_output_post(indirect_load_rank4_axis1_sum_axis2_with_a_fallback_simt 5 1)
mark_indirect_load_codegen_and_e2e(indirect_load_rank4_axis1_sum_axis2_with_a_fallback_simt
    IL_DATA_FLOAT IL_POST_REDUCE IL_REDUCE_WITH_A IL_EXPECT_ONLY_SIMT)

add_indirect_load_e2e_case(indirect_load_rank4_axis1_sum_last_axis_simt 4 1 0 0 0 1 3 8 13 3 8 8 12 3 8)
mark_indirect_load_static_shape(indirect_load_rank4_axis1_sum_last_axis_simt)
mark_indirect_load_output_post(indirect_load_rank4_axis1_sum_last_axis_simt 6 1)
mark_indirect_load_codegen_and_e2e(indirect_load_rank4_axis1_sum_last_axis_simt IL_DATA_FLOAT IL_POST_REDUCE IL_REDUCE_LAST_AXIS)
mark_indirect_load_simt_selected(indirect_load_rank4_axis1_sum_last_axis_simt)

add_indirect_load_e2e_case(indirect_load_rank4_axis1_full_prefix_bessel_k0_sum_simd 4 1 3 0 0 0 1 8 13 3 8 8 17 3 8)
mark_indirect_load_static_shape(indirect_load_rank4_axis1_full_prefix_bessel_k0_sum_simd)
mark_indirect_load_mixed_index_pre(indirect_load_rank4_axis1_full_prefix_bessel_k0_sum_simd)
mark_indirect_load_output_post(indirect_load_rank4_axis1_full_prefix_bessel_k0_sum_simd 11 0)
mark_indirect_load_codegen_and_e2e(indirect_load_rank4_axis1_full_prefix_bessel_k0_sum_simd
    IL_DATA_FLOAT IL_POST_REDUCE IL_POST_REDUCE_SIMD)
target_include_directories(indirect_load_rank4_axis1_full_prefix_bessel_k0_sum_simd_e2e_v2 PRIVATE
    ${CODE_ROOT_DIR}/v35/ascendc/api_regbase)

add_indirect_load_mixed_e2e_case(indirect_load_mixed_rank4_axis2_simd 4 2 0 0 1 FALSE 3 4 16 7 2 3 17 5)
add_indirect_load_mixed_e2e_case(indirect_load_simt_elementwise_coverage 3 1 1 1 3 FALSE 3 16 7 1 2 11 5 1)
mark_indirect_load_simt_elementwise_coverage(indirect_load_simt_elementwise_coverage)
mark_indirect_load_simt_codegen(indirect_load_simt_elementwise_coverage Recursive 64)

function(add_indirect_load_broadcast_test test_name template input_element_count index_element_count broadcast_axes_mask
         output_relu clear_broadcast_source_view)
    if(ARGC GREATER 7)
        set(input_broadcast ${ARGV7})
    else()
        set(input_broadcast 1)
    endif()
    if(ARGC GREATER 8)
        set(index_broadcast ${ARGV8})
    else()
        set(index_broadcast 1)
    endif()
    set(expect_simt 0)
    set(expect_sk 0)
    set(tiling_options)
    if(template STREQUAL "simt")
        set(expect_simt 1)
        set(tiling_options TILING_KEY 1)
    elseif(template STREQUAL "sk")
        set(expect_sk 1)
        set(tiling_options TILING_KEY 2)
    elseif(NOT template STREQUAL "simd")
        message(FATAL_ERROR "Unsupported IndirectLoad template: ${template}")
    endif()

    set(case_workdir ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
    file(MAKE_DIRECTORY ${case_workdir})
    do_backend_e2e_st_test(${test_name}
        WORKDIR ${case_workdir}
        CODEGEN indirect_load_store_backend_generator.cpp
        ${tiling_options}
        KERNEL_SRC
            indirect_load_broadcast_test_kernel.cpp
            indirect_load_broadcast_test_tiling.cpp
            autofuse_tiling_data.h
        TEST_SRC test_e2e_indirect_load_store_kernel.cpp)
    set(case_definitions
        IL_CASE_BROADCAST
        IL_INPUT_BROADCAST=${input_broadcast}
        IL_INDEX_BROADCAST=${index_broadcast}
        IL_HAS_INPUT_ELEMENT=${input_element_count}
        IL_HAS_INDEX_ELEMENT=${index_element_count}
        IL_HAS_OUTPUT_RELU=${output_relu}
        IL_BROADCAST_AXES_MASK=${broadcast_axes_mask}
        IL_CLEAR_BROADCAST_SOURCE_VIEW=${clear_broadcast_source_view}
        IL_EXPECT_SIMT=${expect_simt}
        IL_EXPECT_SK=${expect_sk})
    mark_indirect_load_codegen_and_e2e(${test_name} ${case_definitions})
endfunction()

# Direct Broadcast covers IndirectLoad axis 2, its inner neighbor, and a degenerate source crossing the axis boundary.
add_indirect_load_broadcast_test(indirect_load_broadcast_cross_boundary_simt_fallback_test simt 0 0 14 0 0)
mark_indirect_load_codegen_and_e2e(indirect_load_broadcast_cross_boundary_simt_fallback_test
                                   IL_DEGENERATE_BROADCAST=1 IL_OUTPUT_S0=10 IL_OUTPUT_S1=10 IL_OUTPUT_S2=20 IL_OUTPUT_S3=20)
add_indirect_load_broadcast_test(indirect_load_broadcast_axis_simd_test simd 1 1 4 0 0)
add_indirect_load_broadcast_test(indirect_load_broadcast_inner_adjacent_simd_test simd 0 0 8 0 0)
mark_indirect_load_codegen_and_e2e(indirect_load_broadcast_inner_adjacent_simd_test
                                   IL_INPUT_ABS_BEFORE_BROADCAST=1)
add_indirect_load_broadcast_test(indirect_load_broadcast_continuous_simd_test simd 0 0 12 0 0)
mark_indirect_load_codegen_and_e2e(indirect_load_broadcast_continuous_simd_test IL_CONTINUOUS_BROADCAST=1)
add_indirect_load_broadcast_test(indirect_load_broadcast_continuous_index_simt_test simt 0 0 12 0 0 0 1)
mark_indirect_load_codegen_and_e2e(indirect_load_broadcast_continuous_index_simt_test
                                   IL_CONTINUOUS_INDEX_BROADCAST=1)
add_indirect_load_broadcast_test(indirect_load_broadcast_cross_boundary_simt_test simt 0 0 10 0 0)
add_indirect_load_broadcast_test(indirect_load_broadcast_axis_simt_test simt 1 1 4 0 0)
add_indirect_load_broadcast_test(indirect_load_broadcast_reduce_simt_fallback_test simt 0 0 2 0 0)
mark_indirect_load_codegen_and_e2e(indirect_load_broadcast_reduce_simt_fallback_test IL_BROADCAST_POST_REDUCE=1)

# A scalar Broadcast after an Element cannot use the physical-view inline path. SIMD keeps it as a regular op;
# SIMT emits it inside the fused scalar body.
add_indirect_load_broadcast_test(indirect_load_broadcast_retained_simd_test simd 0 0 10 0 0 0 0)
mark_indirect_load_codegen_and_e2e(indirect_load_broadcast_retained_simd_test IL_RETAIN_BROADCAST=1)
add_indirect_load_broadcast_test(indirect_load_broadcast_retained_simt_test simt 0 0 10 0 0 0 0)
mark_indirect_load_codegen_and_e2e(indirect_load_broadcast_retained_simt_test IL_RETAIN_BROADCAST=1)

# Keep one SK Broadcast regression outside the SIMD/SIMT matrix.
add_indirect_load_broadcast_test(indirect_load_broadcast_elements_sk_test sk 2 2 3 0 0)
add_indirect_load_broadcast_test(indirect_load_broadcast_index_physical_view_simt_test simt 2 2 3 0 0)
mark_indirect_load_codegen_and_e2e(indirect_load_broadcast_index_physical_view_simt_test IL_AIC_REPRO=1)

# Identity Broadcast keeps the index source axis/repeats unchanged while exercising the direct SIMD Broadcast path.
add_indirect_load_broadcast_test(indirect_load_broadcast_identity_index_simd_test simd 0 0 0 0 0 0 1)

# Regression: a unary elementwise op after an index Broadcast must retain the source physical view in SIMT.
add_indirect_load_broadcast_test(indirect_load_broadcast_index_abs_simt_test simt 0 1 2 0 0 0 1)
mark_indirect_load_codegen_and_e2e(indirect_load_broadcast_index_abs_simt_test IL_INDEX_ABS_DENSE_VIEW=1)

# Regression: a three-input Where chain must remain in the SIMT index region before IndirectLoad.
set(indirect_load_broadcast_index_where_simt_test_workdir
    ${CMAKE_CURRENT_BINARY_DIR}/indirect_load_broadcast_index_where_simt_test)
file(MAKE_DIRECTORY ${indirect_load_broadcast_index_where_simt_test_workdir})
do_backend_e2e_st_test(indirect_load_broadcast_index_where_simt_test
    WORKDIR ${indirect_load_broadcast_index_where_simt_test_workdir}
    CODEGEN indirect_load_store_backend_generator.cpp
    TILING_KEY 1
    KERNEL_SRC
        indirect_load_broadcast_where_test_kernel.cpp
        indirect_load_broadcast_where_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_indirect_load_store_kernel.cpp)
mark_indirect_load_codegen_and_e2e(indirect_load_broadcast_index_where_simt_test IL_CASE_BROADCAST_WHERE)

# Strict regression for the user-provided GraphHint: Where + Broadcast + IndirectLoad
# followed by ReduceSum, including the non-overlapping zero-stride/physical-gap table view.
set(indirect_load_graph_hint_reduce_simt_test_workdir
    ${CMAKE_CURRENT_BINARY_DIR}/indirect_load_graph_hint_reduce_simt_test)
file(MAKE_DIRECTORY ${indirect_load_graph_hint_reduce_simt_test_workdir})
do_backend_e2e_st_test(indirect_load_graph_hint_reduce_simt_test
    WORKDIR ${indirect_load_graph_hint_reduce_simt_test_workdir}
    CODEGEN indirect_load_store_backend_generator.cpp
    TILING_KEY 1
    KERNEL_SRC
        indirect_load_graph_hint_reduce_simt_test_kernel.cpp
        indirect_load_graph_hint_reduce_simt_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_indirect_load_store_kernel.cpp)
mark_indirect_load_codegen_and_e2e(indirect_load_graph_hint_reduce_simt_test
                                   IL_CASE_BROADCAST_WHERE IL_GRAPH_HINT_REDUCE)

# Exact 30x3x23 GraphHint reproduction for the SIMD IndirectLoad gather path.
set(indirect_load_graph_hint_simd_repro_workdir
    ${CMAKE_CURRENT_BINARY_DIR}/indirect_load_graph_hint_simd_repro)
file(MAKE_DIRECTORY ${indirect_load_graph_hint_simd_repro_workdir})
do_backend_e2e_st_test(indirect_load_graph_hint_simd_repro
    WORKDIR ${indirect_load_graph_hint_simd_repro_workdir}
    CODEGEN indirect_load_store_backend_generator.cpp
    TILING_KEY 0
    KERNEL_SRC
        indirect_load_graph_hint_simd_repro_kernel.cpp
        indirect_load_graph_hint_simd_repro_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_indirect_load_store_kernel.cpp)
mark_indirect_load_codegen_and_e2e(indirect_load_graph_hint_simd_repro
                                   IL_CASE_BROADCAST_WHERE IL_GRAPH_HINT_SIMD_REPRO)

# User graph: embedding + Sum over the lookup axis.
set(indirect_load_user_embedding_sum_workdir
    ${CMAKE_CURRENT_BINARY_DIR}/indirect_load_user_embedding_sum)
file(MAKE_DIRECTORY ${indirect_load_user_embedding_sum_workdir})
do_backend_e2e_st_test(indirect_load_user_embedding_sum
    WORKDIR ${indirect_load_user_embedding_sum_workdir}
    CODEGEN indirect_load_store_backend_generator.cpp
    TILING_KEY 1
    KERNEL_SRC
        indirect_load_user_embedding_sum_kernel.cpp
        indirect_load_user_embedding_sum_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_indirect_load_store_kernel.cpp)
mark_indirect_load_codegen_and_e2e(indirect_load_user_embedding_sum IL_CASE_BROADCAST_WHERE IL_USER_EMBEDDING_SUM)

# User graph: embedding followed by scalar Mul.
set(indirect_load_user_embedding_mul_workdir
    ${CMAKE_CURRENT_BINARY_DIR}/indirect_load_user_embedding_mul)
file(MAKE_DIRECTORY ${indirect_load_user_embedding_mul_workdir})
do_backend_e2e_st_test(indirect_load_user_embedding_mul
    WORKDIR ${indirect_load_user_embedding_mul_workdir}
    CODEGEN indirect_load_store_backend_generator.cpp
    TILING_KEY 1
    KERNEL_SRC
        indirect_load_user_embedding_mul_kernel.cpp
        indirect_load_user_embedding_mul_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_indirect_load_store_kernel.cpp)
mark_indirect_load_codegen_and_e2e(indirect_load_user_embedding_mul IL_CASE_BROADCAST_WHERE IL_USER_EMBEDDING_MUL)

# User graph: bf16 embedding with mean/rsqrt normalization and two outputs.
set(indirect_load_user_layernorm_workdir
    ${CMAKE_CURRENT_BINARY_DIR}/indirect_load_user_layernorm)
file(MAKE_DIRECTORY ${indirect_load_user_layernorm_workdir})
do_backend_e2e_st_test(indirect_load_user_layernorm
    WORKDIR ${indirect_load_user_layernorm_workdir}
    CODEGEN indirect_load_store_backend_generator.cpp
    TILING_KEY 1
    KERNEL_SRC
        indirect_load_user_layernorm_kernel.cpp
        indirect_load_user_layernorm_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_indirect_load_store_kernel.cpp)
mark_indirect_load_codegen_and_e2e(indirect_load_user_layernorm IL_CASE_BROADCAST_WHERE IL_USER_LAYERNORM)

# Same LayerNorm split topology with a small shape; force the SIMD candidate to
# determine whether the branch itself, rather than UB pressure, is the blocker.
set(indirect_load_user_layernorm_simd_workdir
    ${CMAKE_CURRENT_BINARY_DIR}/indirect_load_user_layernorm_simd)
file(MAKE_DIRECTORY ${indirect_load_user_layernorm_simd_workdir})
do_backend_e2e_st_test(indirect_load_user_layernorm_simd
    WORKDIR ${indirect_load_user_layernorm_simd_workdir}
    CODEGEN indirect_load_store_backend_generator.cpp
    TILING_KEY 1
    KERNEL_SRC
        indirect_load_user_layernorm_simd_kernel.cpp
        indirect_load_user_layernorm_simd_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_indirect_load_store_kernel.cpp)
mark_indirect_load_codegen_and_e2e(indirect_load_user_layernorm_simd
                                   IL_CASE_BROADCAST_WHERE IL_USER_LAYERNORM IL_USER_LAYERNORM_SIMD)

# User graph: one embedding result fans out to Exp and Abs, then merges by Add.
# Run both IL SIMD and IL SIMT templates to compare branch handling.
set(indirect_load_user_embedding_exp_abs_add_simd_workdir
    ${CMAKE_CURRENT_BINARY_DIR}/indirect_load_user_embedding_exp_abs_add_simd)
file(MAKE_DIRECTORY ${indirect_load_user_embedding_exp_abs_add_simd_workdir})
do_backend_e2e_st_test(indirect_load_user_embedding_exp_abs_add_simd
    WORKDIR ${indirect_load_user_embedding_exp_abs_add_simd_workdir}
    CODEGEN indirect_load_store_backend_generator.cpp
    TILING_KEY 1
    KERNEL_SRC
        indirect_load_user_embedding_exp_abs_add_simd_kernel.cpp
        indirect_load_user_embedding_exp_abs_add_simd_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_indirect_load_store_kernel.cpp)
mark_indirect_load_codegen_and_e2e(indirect_load_user_embedding_exp_abs_add_simd
                                   IL_USER_EMBEDDING_EXP_ABS_ADD IL_USER_EMBEDDING_EXP_ABS_ADD_SIMD)

set(indirect_load_user_embedding_exp_abs_add_simt_workdir
    ${CMAKE_CURRENT_BINARY_DIR}/indirect_load_user_embedding_exp_abs_add_simt)
file(MAKE_DIRECTORY ${indirect_load_user_embedding_exp_abs_add_simt_workdir})
do_backend_e2e_st_test(indirect_load_user_embedding_exp_abs_add_simt
    WORKDIR ${indirect_load_user_embedding_exp_abs_add_simt_workdir}
    CODEGEN indirect_load_store_backend_generator.cpp
    TILING_KEY 1
    KERNEL_SRC
        indirect_load_user_embedding_exp_abs_add_simt_kernel.cpp
        indirect_load_user_embedding_exp_abs_add_simt_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_indirect_load_store_kernel.cpp)
mark_indirect_load_codegen_and_e2e(indirect_load_user_embedding_exp_abs_add_simt
                                   IL_USER_EMBEDDING_EXP_ABS_ADD)

# Fan-out matrix: split at IndirectLoad or at a post-IndirectLoad elementwise node,
# with either two ordinary Stores or one ordinary Store plus one Reduce Store.
function(add_indirect_load_user_fanout_case test_name)
    set(workdir ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
    file(MAKE_DIRECTORY ${workdir})
    do_backend_e2e_st_test(${test_name}
        WORKDIR ${workdir}
        CODEGEN indirect_load_store_backend_generator.cpp
        TILING_KEY 1
        KERNEL_SRC
            ${test_name}_kernel.cpp
            ${test_name}_tiling.cpp
            autofuse_tiling_data.h
        TEST_SRC test_e2e_indirect_load_store_kernel.cpp)
    mark_indirect_load_codegen_and_e2e(${test_name} IL_USER_FANOUT ${ARGN})
endfunction()

add_indirect_load_user_fanout_case(indirect_load_user_fanout_direct_stores_simd
                                   IL_USER_FANOUT_SIMD)
add_indirect_load_user_fanout_case(indirect_load_user_fanout_direct_stores_simt)
add_indirect_load_user_fanout_case(indirect_load_user_fanout_direct_reduce_simd
                                   IL_USER_FANOUT_SIMD IL_USER_FANOUT_REDUCE)
add_indirect_load_user_fanout_case(indirect_load_user_fanout_direct_reduce_simt
                                   IL_USER_FANOUT_REDUCE)
add_indirect_load_user_fanout_case(indirect_load_user_fanout_post_stores_simd
                                   IL_USER_FANOUT_SIMD IL_USER_FANOUT_POST)
add_indirect_load_user_fanout_case(indirect_load_user_fanout_post_stores_simt
                                   IL_USER_FANOUT_POST)
add_indirect_load_user_fanout_case(indirect_load_user_fanout_post_reduce_simd
                                   IL_USER_FANOUT_SIMD IL_USER_FANOUT_POST IL_USER_FANOUT_REDUCE)
add_indirect_load_user_fanout_case(indirect_load_user_fanout_post_reduce_simt
                                   IL_USER_FANOUT_POST IL_USER_FANOUT_REDUCE)

# User graph: two IndirectLoads over two tensors followed by Add (dual-IL graph).
set(indirect_load_user_add_gather_workdir
    ${CMAKE_CURRENT_BINARY_DIR}/indirect_load_user_add_gather)
file(MAKE_DIRECTORY ${indirect_load_user_add_gather_workdir})
do_backend_e2e_st_test(indirect_load_user_add_gather
    WORKDIR ${indirect_load_user_add_gather_workdir}
    CODEGEN indirect_load_store_backend_generator.cpp
    TILING_KEY 1
    KERNEL_SRC
        indirect_load_user_add_gather_kernel.cpp
        indirect_load_user_add_gather_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_indirect_load_store_kernel.cpp)
mark_indirect_load_codegen_and_e2e(indirect_load_user_add_gather IL_CASE_BROADCAST_WHERE IL_DUAL_IL_GATHER)

# Regression: SIMT post-reduce Output() evaluator uses full output_index instead of address.index_offset
# to access index-tensor GM, causing OOB reads when output_index exceeds index tensor size.
set(indirect_load_embedding_reduce_simt_test_workdir
    ${CMAKE_CURRENT_BINARY_DIR}/indirect_load_embedding_reduce_simt_test)
file(MAKE_DIRECTORY ${indirect_load_embedding_reduce_simt_test_workdir})
do_backend_e2e_st_test(indirect_load_embedding_reduce_simt_test
    WORKDIR ${indirect_load_embedding_reduce_simt_test_workdir}
    CODEGEN indirect_load_store_backend_generator.cpp
    TILING_KEY 1
    KERNEL_SRC
        indirect_load_embedding_reduce_simt_test_kernel.cpp
        indirect_load_embedding_reduce_simt_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_indirect_load_store_kernel.cpp)
mark_indirect_load_codegen_and_e2e(indirect_load_embedding_reduce_simt_test IL_CASE_BROADCAST_WHERE IL_EMBEDDING_REDUCE)

# Regression: a dual-input Add before IndirectLoad with a post-fused ReduceSum must keep a valid SIMT kernel.
set(indirect_load_add_il_reduce_test_workdir
    ${CMAKE_CURRENT_BINARY_DIR}/indirect_load_add_il_reduce_test)
file(MAKE_DIRECTORY ${indirect_load_add_il_reduce_test_workdir})
do_backend_e2e_st_test(indirect_load_add_il_reduce_test
    WORKDIR ${indirect_load_add_il_reduce_test_workdir}
    CODEGEN indirect_load_store_backend_generator.cpp
    TILING_KEY 1
    KERNEL_SRC
        indirect_load_add_il_reduce_test_kernel.cpp
        indirect_load_add_il_reduce_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_indirect_load_store_kernel.cpp)
mark_indirect_load_codegen_and_e2e(indirect_load_add_il_reduce_test IL_CASE_BROADCAST_WHERE)
mark_indirect_load_codegen_and_e2e(indirect_load_add_il_reduce_test IL_ADD_IL_REDUCE)

# Same-view tensor fan-in without Broadcast: the binary operation is coordinate-preserving.
add_indirect_load_broadcast_test(indirect_load_index_binary_same_view_simd_test simd 0 1 0 0 0 0 0)
mark_indirect_load_codegen_and_e2e(indirect_load_index_binary_same_view_simd_test
                                   IL_INDEX_BINARY_SAME_VIEW=1 IL_BINARY_ELEMENT_KIND=3)

add_indirect_load_broadcast_test(indirect_load_complex_broadcast_simd_test simd 0 0 3 0 0)
mark_indirect_load_codegen_and_e2e(indirect_load_complex_broadcast_simd_test
                                   IL_COMPLEX_BROADCAST=1 IL_BINARY_ELEMENT_KIND=3)

add_indirect_load_broadcast_test(indirect_load_complex_broadcast_simt_test simt 0 0 3 0 0)
mark_indirect_load_codegen_and_e2e(indirect_load_complex_broadcast_simt_test IL_COMPLEX_BROADCAST=1 IL_COMPLEX_SIMT=1)

# Input Broadcast whose source is a multi-input binary element chain: the SIMD candidate keeps the direct Broadcast and
# candidate consumes the computed source through its physical view (source multi-input scenario).
# Note: SIMT rejects multi-input input sources by design (ValidateSimtTemplateRegion requires a Load
# boundary), covered by UT SimdRegionMetadataAndSimtRejectsMultiInputRegion.
add_indirect_load_broadcast_test(indirect_load_complex_input_broadcast_simd_test simd 0 0 12 0 0 1 0)
mark_indirect_load_codegen_and_e2e(indirect_load_complex_input_broadcast_simd_test
                                   IL_COMPLEX_INPUT_BROADCAST=1 IL_BINARY_ELEMENT_KIND=3)

function(add_indirect_load_stride_zero_test test_name template input_zero_stride_mask index_zero_stride_mask
         input_element_count index_element_count)
    set(expect_simt 0)
    set(expect_sk 0)
    set(tiling_options)
    if(template STREQUAL "simt")
        set(expect_simt 1)
        set(tiling_options TILING_KEY 1)
    elseif(template STREQUAL "sk")
        set(expect_sk 1)
        set(tiling_options TILING_KEY 2)
    elseif(NOT template STREQUAL "simd")
        message(FATAL_ERROR "Unsupported IndirectLoad template: ${template}")
    endif()

    set(case_workdir ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
    file(MAKE_DIRECTORY ${case_workdir})
    do_backend_e2e_st_test(${test_name}
        WORKDIR ${case_workdir}
        CODEGEN indirect_load_store_backend_generator.cpp
        ${tiling_options}
        KERNEL_SRC
            indirect_load_stride_zero_test_kernel.cpp
            indirect_load_stride_zero_test_tiling.cpp
            autofuse_tiling_data.h
        TEST_SRC test_e2e_indirect_load_store_kernel.cpp)
    set(case_definitions
        IL_CASE_STRIDE_ZERO
        IL_INPUT_ZERO_STRIDE_MASK=${input_zero_stride_mask}
        IL_INDEX_ZERO_STRIDE_MASK=${index_zero_stride_mask}
        IL_HAS_INPUT_ELEMENT=${input_element_count}
        IL_HAS_INDEX_ELEMENT=${index_element_count}
        IL_EXPECT_SIMT=${expect_simt}
        IL_EXPECT_SK=${expect_sk})
    mark_indirect_load_codegen_and_e2e(${test_name} ${case_definitions})
endfunction()

# Mask bit d denotes that logical axis d has stride 0. The Element case covers both input and index zero-stride paths.
# Direct zero-stride classification is covered by the layout and schedule UT.
add_indirect_load_stride_zero_test(indirect_load_stride_zero_elements_simd_test simd 10 5 1 1)
add_indirect_load_stride_zero_test(indirect_load_stride_zero_elements_simt_test simt 10 5 1 1)
add_indirect_load_stride_zero_test(indirect_load_stride_zero_elements_sk_test sk 10 5 1 1)

function(add_indirect_load_torch_gather_strided_test test_name tiling_key expect_simt expect_sk
         input_stride0 input_stride1 input_stride2 index_stride0 index_stride1 index_stride2)
    set(case_workdir ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
    file(MAKE_DIRECTORY ${case_workdir})
    do_backend_e2e_st_test(${test_name}
        WORKDIR ${case_workdir}
        CODEGEN indirect_load_store_backend_generator.cpp
        TILING_KEY ${tiling_key}
        KERNEL_SRC
            indirect_load_torch_gather_strided_test_kernel.cpp
            indirect_load_torch_gather_strided_test_tiling.cpp
            autofuse_tiling_data.h
        TEST_SRC test_e2e_indirect_load_store_kernel.cpp)
    set(case_definitions
        IL_CASE_TORCH_STRIDED
        IL_EXPECT_TILING_KEY=${tiling_key}
        IL_EXPECT_SIMT=${expect_simt}
        IL_EXPECT_SK=${expect_sk}
        IL_INPUT_STRIDE0=${input_stride0}
        IL_INPUT_STRIDE1=${input_stride1}
        IL_INPUT_STRIDE2=${input_stride2}
        IL_INDEX_STRIDE0=${index_stride0}
        IL_INDEX_STRIDE1=${index_stride1}
        IL_INDEX_STRIDE2=${index_stride2})
    mark_indirect_load_codegen_and_e2e(${test_name} ${case_definitions})
endfunction()

# Cover non-contiguous layouts without a gap-kind/template Cartesian product: the inner-gap case also has an
# outer gap, while the SIMT outer-gap case covers the pure outer-gap policy. Keep one SK runtime representative.
add_indirect_load_torch_gather_strided_test(indirect_load_rank3_axis1_input_index_gap_simd_test
    0 0 0 384 10 1 192 10 1)
add_indirect_load_torch_gather_strided_test(indirect_load_rank3_axis1_input_index_gap_simt_test
    0 1 0 384 10 1 192 10 1)
add_indirect_load_torch_gather_strided_test(indirect_load_rank3_axis1_input_index_gap_sk_test
    0 0 1 384 10 1 192 10 1)
add_indirect_load_torch_gather_strided_test(indirect_load_rank3_axis1_input_index_outer_gap_simt_test
    0 1 0 192 5 1 128 5 1)
set(indirect_load_index_select_defs IL_INDEX_SELECT_CASE=1)
mark_indirect_load_codegen_and_e2e(indirect_load_rank3_axis1_input_index_gap_simt_test
                                   ${indirect_load_index_select_defs})


set(indirect_load_embedding_test_workdir ${CMAKE_CURRENT_BINARY_DIR}/indirect_load_embedding_test)
file(MAKE_DIRECTORY ${indirect_load_embedding_test_workdir})
do_backend_e2e_st_test(indirect_load_embedding_test
    WORKDIR ${indirect_load_embedding_test_workdir}
    CODEGEN indirect_load_store_backend_generator.cpp
    TILING_KEY 0
    KERNEL_SRC
        indirect_load_embedding_test_kernel.cpp
        indirect_load_embedding_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_indirect_load_store_kernel.cpp)
mark_indirect_load_codegen_and_e2e(indirect_load_embedding_test IL_CASE_EMBEDDING)
