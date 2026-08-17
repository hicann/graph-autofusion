function(add_indirect_load_torch_gather_strided_test test_name tiling_key expect_simt expect_sk
         input_stride0 input_stride1 input_stride2 index_stride0 index_stride1 index_stride2)
    set(case_workdir ${CMAKE_CURRENT_BINARY_DIR}/${test_name})
    file(MAKE_DIRECTORY ${case_workdir})
    do_backend_e2e_st_test(${test_name}
        WORKDIR ${case_workdir}
        CODEGEN indirect_load_torch_gather_strided_backend_generator.cpp
        TILING_KEY ${tiling_key}
        KERNEL_SRC
            indirect_load_torch_gather_strided_test_kernel.cpp
            indirect_load_torch_gather_strided_test_tiling.cpp
            autofuse_tiling_data.h
        TEST_SRC test_e2e_indirect_load_torch_gather_strided_kernel.cpp)
    set(case_definitions
        IL_EXPECT_TILING_KEY=${tiling_key}
        IL_EXPECT_SIMT=${expect_simt}
        IL_EXPECT_SK=${expect_sk}
        IL_INPUT_STRIDE0=${input_stride0}
        IL_INPUT_STRIDE1=${input_stride1}
        IL_INPUT_STRIDE2=${input_stride2}
        IL_INDEX_STRIDE0=${index_stride0}
        IL_INDEX_STRIDE1=${index_stride1}
        IL_INDEX_STRIDE2=${index_stride2})
    target_compile_definitions(${test_name}_codegen_v2 PRIVATE ${case_definitions})
    target_compile_definitions(${test_name}_e2e_v2 PRIVATE ${case_definitions})
endfunction()

function(add_indirect_load_strided_template_tests case_name input_stride0 input_stride1 input_stride2
         index_stride0 index_stride1 index_stride2)
    add_indirect_load_torch_gather_strided_test(${case_name}_simd_test 0 0 0
        ${input_stride0} ${input_stride1} ${input_stride2} ${index_stride0} ${index_stride1} ${index_stride2})
    add_indirect_load_torch_gather_strided_test(${case_name}_simt_test 0 1 0
        ${input_stride0} ${input_stride1} ${input_stride2} ${index_stride0} ${index_stride1} ${index_stride2})
    add_indirect_load_torch_gather_strided_test(${case_name}_sk_test 0 0 1
        ${input_stride0} ${input_stride1} ${input_stride2} ${index_stride0} ${index_stride1} ${index_stride2})
endfunction()

# Keep the SIMD/SK inner-gap coverage and use its SIMT variant for the common IndexSelect Broadcast layout.
add_indirect_load_strided_template_tests(indirect_load_rank3_axis1_input_index_gap 384 10 1 192 10 1)
add_indirect_load_strided_template_tests(indirect_load_rank3_axis1_input_index_outer_gap 192 5 1 128 5 1)
set(indirect_load_index_select_defs IL_INDEX_SELECT_CASE=1)
target_compile_definitions(indirect_load_rank3_axis1_input_index_gap_simt_test_codegen_v2 PRIVATE
                           ${indirect_load_index_select_defs})
target_compile_definitions(indirect_load_rank3_axis1_input_index_gap_simt_test_e2e_v2 PRIVATE
                           ${indirect_load_index_select_defs})
