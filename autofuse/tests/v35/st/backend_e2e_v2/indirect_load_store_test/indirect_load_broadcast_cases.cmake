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
        CODEGEN indirect_load_broadcast_backend_generator.cpp
        ${tiling_options}
        KERNEL_SRC
            indirect_load_broadcast_test_kernel.cpp
            indirect_load_broadcast_test_tiling.cpp
            autofuse_tiling_data.h
        TEST_SRC test_e2e_indirect_load_broadcast_kernel.cpp)
    set(case_definitions
        IL_INPUT_BROADCAST=${input_broadcast}
        IL_INDEX_BROADCAST=${index_broadcast}
        IL_HAS_INPUT_ELEMENT=${input_element_count}
        IL_HAS_INDEX_ELEMENT=${index_element_count}
        IL_HAS_OUTPUT_RELU=${output_relu}
        IL_BROADCAST_AXES_MASK=${broadcast_axes_mask}
        IL_CLEAR_BROADCAST_SOURCE_VIEW=${clear_broadcast_source_view}
        IL_EXPECT_SIMT=${expect_simt}
        IL_EXPECT_SK=${expect_sk})
    target_compile_definitions(${test_name}_codegen_v2 PRIVATE ${case_definitions})
    target_compile_definitions(${test_name}_e2e_v2 PRIVATE ${case_definitions})
endfunction()

# Direct Broadcast covers IndirectLoad axis 2, its inner neighbor, and a degenerate source crossing the axis boundary.
add_indirect_load_broadcast_test(indirect_load_broadcast_cross_boundary_simt_fallback_test simt 0 0 14 0 0)
target_compile_definitions(indirect_load_broadcast_cross_boundary_simt_fallback_test_codegen_v2 PRIVATE
                           IL_DEGENERATE_BROADCAST=1 IL_OUTPUT_S0=10 IL_OUTPUT_S1=10 IL_OUTPUT_S2=20 IL_OUTPUT_S3=20)
target_compile_definitions(indirect_load_broadcast_cross_boundary_simt_fallback_test_e2e_v2 PRIVATE
                           IL_DEGENERATE_BROADCAST=1 IL_OUTPUT_S0=10 IL_OUTPUT_S1=10 IL_OUTPUT_S2=20 IL_OUTPUT_S3=20)
add_indirect_load_broadcast_test(indirect_load_broadcast_axis_simd_test simd 0 0 4 0 0)
add_indirect_load_broadcast_test(indirect_load_broadcast_inner_adjacent_simd_test simd 0 0 8 0 0)
add_indirect_load_broadcast_test(indirect_load_broadcast_continuous_simd_test simd 0 0 12 0 0)
target_compile_definitions(indirect_load_broadcast_continuous_simd_test_codegen_v2 PRIVATE IL_CONTINUOUS_BROADCAST=1)
target_compile_definitions(indirect_load_broadcast_continuous_simd_test_e2e_v2 PRIVATE IL_CONTINUOUS_BROADCAST=1)
add_indirect_load_broadcast_test(indirect_load_broadcast_continuous_index_simt_test simt 0 0 12 0 0 0 1)
target_compile_definitions(indirect_load_broadcast_continuous_index_simt_test_codegen_v2 PRIVATE
                           IL_CONTINUOUS_INDEX_BROADCAST=1)
target_compile_definitions(indirect_load_broadcast_continuous_index_simt_test_e2e_v2 PRIVATE
                           IL_CONTINUOUS_INDEX_BROADCAST=1)
add_indirect_load_broadcast_test(indirect_load_broadcast_cross_boundary_simt_test simt 0 0 10 0 0)
add_indirect_load_broadcast_test(indirect_load_broadcast_axis_simt_test simt 0 0 4 0 0)
add_indirect_load_broadcast_test(indirect_load_broadcast_reduce_simt_fallback_test simt 0 0 2 0 0)
target_compile_definitions(indirect_load_broadcast_reduce_simt_fallback_test_codegen_v2 PRIVATE IL_BROADCAST_POST_REDUCE=1)
target_compile_definitions(indirect_load_broadcast_reduce_simt_fallback_test_e2e_v2 PRIVATE IL_BROADCAST_POST_REDUCE=1)

# A scalar Broadcast after an Element cannot use the physical-view inline path. SIMD keeps it as a regular op;
# SIMT emits it inside the fused scalar body.
add_indirect_load_broadcast_test(indirect_load_broadcast_retained_simd_test simd 0 0 10 0 0 0 0)
target_compile_definitions(indirect_load_broadcast_retained_simd_test_codegen_v2 PRIVATE IL_RETAIN_BROADCAST=1)
target_compile_definitions(indirect_load_broadcast_retained_simd_test_e2e_v2 PRIVATE IL_RETAIN_BROADCAST=1)
add_indirect_load_broadcast_test(indirect_load_broadcast_retained_simt_test simt 0 0 10 0 0 0 0)
target_compile_definitions(indirect_load_broadcast_retained_simt_test_codegen_v2 PRIVATE IL_RETAIN_BROADCAST=1)
target_compile_definitions(indirect_load_broadcast_retained_simt_test_e2e_v2 PRIVATE IL_RETAIN_BROADCAST=1)

# Keep one SK Broadcast regression outside the SIMD/SIMT matrix.
add_indirect_load_broadcast_test(indirect_load_broadcast_elements_sk_test sk 2 2 3 0 0)
add_indirect_load_broadcast_test(indirect_load_broadcast_index_physical_view_simt_test simt 2 2 3 0 0)
target_compile_definitions(indirect_load_broadcast_index_physical_view_simt_test_codegen_v2 PRIVATE IL_AIC_REPRO=1)
target_compile_definitions(indirect_load_broadcast_index_physical_view_simt_test_e2e_v2 PRIVATE IL_AIC_REPRO=1)

# Identity Broadcast keeps the index source axis/repeats unchanged but still exercises the direct-Broadcast fold.
add_indirect_load_broadcast_test(indirect_load_broadcast_identity_index_simd_test simd 0 0 0 0 0 0 1)

# Regression: a unary elementwise op after an index Broadcast must retain the source physical view in SIMT.
add_indirect_load_broadcast_test(indirect_load_broadcast_index_abs_simt_test simt 0 1 2 0 0 0 1)
target_compile_definitions(indirect_load_broadcast_index_abs_simt_test_codegen_v2 PRIVATE IL_INDEX_ABS_DENSE_VIEW=1)
target_compile_definitions(indirect_load_broadcast_index_abs_simt_test_e2e_v2 PRIVATE IL_INDEX_ABS_DENSE_VIEW=1)

# Regression: a three-input Where chain must remain in the SIMT index region before IndirectLoad.
set(indirect_load_broadcast_index_where_simt_test_workdir
    ${CMAKE_CURRENT_BINARY_DIR}/indirect_load_broadcast_index_where_simt_test)
file(MAKE_DIRECTORY ${indirect_load_broadcast_index_where_simt_test_workdir})
do_backend_e2e_st_test(indirect_load_broadcast_index_where_simt_test
    WORKDIR ${indirect_load_broadcast_index_where_simt_test_workdir}
    CODEGEN indirect_load_broadcast_where_backend_generator.cpp
    TILING_KEY 1
    KERNEL_SRC
        indirect_load_broadcast_where_test_kernel.cpp
        indirect_load_broadcast_where_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_indirect_load_broadcast_where_kernel.cpp)

# Same-view tensor fan-in without Broadcast: the binary operation is coordinate-preserving.
add_indirect_load_broadcast_test(indirect_load_index_binary_same_view_simd_test simd 0 1 0 0 0 0 0)
target_compile_definitions(indirect_load_index_binary_same_view_simd_test_codegen_v2 PRIVATE
                           IL_INDEX_BINARY_SAME_VIEW=1 IL_BINARY_ELEMENT_KIND=3)
target_compile_definitions(indirect_load_index_binary_same_view_simd_test_e2e_v2 PRIVATE
                           IL_INDEX_BINARY_SAME_VIEW=1 IL_BINARY_ELEMENT_KIND=3)

add_indirect_load_broadcast_test(indirect_load_complex_broadcast_simd_test simd 0 0 3 0 0)
target_compile_definitions(indirect_load_complex_broadcast_simd_test_codegen_v2 PRIVATE IL_COMPLEX_BROADCAST=1)
target_compile_definitions(indirect_load_complex_broadcast_simd_test_e2e_v2 PRIVATE IL_COMPLEX_BROADCAST=1)

add_indirect_load_broadcast_test(indirect_load_complex_broadcast_simt_test simt 0 0 3 0 0)
target_compile_definitions(indirect_load_complex_broadcast_simt_test_codegen_v2 PRIVATE IL_COMPLEX_BROADCAST=1
                                                                                       IL_COMPLEX_SIMT=1)
target_compile_definitions(indirect_load_complex_broadcast_simt_test_e2e_v2 PRIVATE IL_COMPLEX_BROADCAST=1
                                                                                   IL_COMPLEX_SIMT=1)

# Input Broadcast whose source is a multi-input Add: the direct Broadcast is inlined and the SIMD
# candidate consumes the computed source through its physical view (source multi-input scenario).
# Note: SIMT rejects multi-input input sources by design (ValidateSimtTemplateRegion requires a Load
# boundary), covered by UT SimdRegionMetadataAndSimtRejectsMultiInputRegion.
add_indirect_load_broadcast_test(indirect_load_complex_input_broadcast_simd_test simd 0 0 12 0 0 1 0)
target_compile_definitions(indirect_load_complex_input_broadcast_simd_test_codegen_v2 PRIVATE
                           IL_COMPLEX_INPUT_BROADCAST=1)
target_compile_definitions(indirect_load_complex_input_broadcast_simd_test_e2e_v2 PRIVATE
                           IL_COMPLEX_INPUT_BROADCAST=1)

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
        CODEGEN indirect_load_stride_zero_backend_generator.cpp
        ${tiling_options}
        KERNEL_SRC
            indirect_load_stride_zero_test_kernel.cpp
            indirect_load_stride_zero_test_tiling.cpp
            autofuse_tiling_data.h
        TEST_SRC test_e2e_indirect_load_stride_zero_kernel.cpp)
    set(case_definitions
        IL_INPUT_ZERO_STRIDE_MASK=${input_zero_stride_mask}
        IL_INDEX_ZERO_STRIDE_MASK=${index_zero_stride_mask}
        IL_HAS_INPUT_ELEMENT=${input_element_count}
        IL_HAS_INDEX_ELEMENT=${index_element_count}
        IL_EXPECT_SIMT=${expect_simt}
        IL_EXPECT_SK=${expect_sk})
    target_compile_definitions(${test_name}_codegen_v2 PRIVATE ${case_definitions})
    target_compile_definitions(${test_name}_e2e_v2 PRIVATE ${case_definitions})
endfunction()

# Mask bit d denotes that logical axis d has stride 0. The Element case covers both input and index zero-stride paths.
# Direct zero-stride classification is covered by the layout and schedule UT.
add_indirect_load_stride_zero_test(indirect_load_stride_zero_elements_simd_test simd 10 5 1 1)
add_indirect_load_stride_zero_test(indirect_load_stride_zero_elements_simt_test simt 10 5 1 1)
add_indirect_load_stride_zero_test(indirect_load_stride_zero_elements_sk_test sk 10 5 1 1)
