function(add_indirect_load_broadcast_test test_name template input_element_count index_element_count broadcast_axes_mask
         output_relu clear_broadcast_source_view)
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
        IL_INPUT_BROADCAST=1
        IL_INDEX_BROADCAST=1
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

# Each template uses one graph containing input/index Broadcast paths and multiple Element nodes on both paths.
add_indirect_load_broadcast_test(indirect_load_broadcast_elements_simd_test simd 2 2 3 0 1)
add_indirect_load_broadcast_test(indirect_load_broadcast_elements_simt_test simt 2 2 3 0 0)
add_indirect_load_broadcast_test(indirect_load_broadcast_elements_sk_test sk 2 2 3 0 0)

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
