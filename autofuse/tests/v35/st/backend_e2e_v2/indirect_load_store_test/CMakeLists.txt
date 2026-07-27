function(add_indirect_load_e2e_case test_name rank axis use_exp2 tiling_key x0 x1 x2 x3 index0 index1 index2 index3)
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
        IL_HAS_INPUT_PRE=1
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

add_indirect_load_e2e_case(indirect_load_rank2_axis1_simt 2 1 0 1 5 8 1 1 4 6 1 1)
add_indirect_load_e2e_case(indirect_load_rank2_axis1_simd 2 1 1 0 5 8 1 1 4 6 1 1)
add_indirect_load_e2e_case(indirect_load_rank2_axis0_simt 2 0 0 1 4 9 1 1 6 8 1 1)
add_indirect_load_e2e_case(indirect_load_rank2_axis0_simd 2 0 1 0 4 9 1 1 6 8 1 1)
add_indirect_load_e2e_case(indirect_load_rank2_axisneg2_simt 2 -2 0 1 4 9 1 1 4 8 1 1)
add_indirect_load_e2e_case(indirect_load_rank2_axisneg2_simd 2 -2 1 0 4 9 1 1 4 8 1 1)
add_indirect_load_e2e_case(indirect_load_rank3_axis1_simt 3 1 0 1 3 5 8 1 2 7 6 1)
add_indirect_load_e2e_case(indirect_load_rank3_axis1_simd 3 1 1 0 3 5 8 1 2 7 6 1)
add_indirect_load_e2e_case(indirect_load_rank4_axis2_simt 4 2 0 1 3 3 5 8 2 2 5 6)
add_indirect_load_e2e_case(indirect_load_rank4_axis2_simd 4 2 1 0 3 3 5 8 2 2 5 6)
