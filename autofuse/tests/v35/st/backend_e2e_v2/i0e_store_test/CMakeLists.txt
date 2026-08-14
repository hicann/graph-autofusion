backend_e2e_st_test(i0e_store_test
    CODEGEN i0e_store_backend_generator.cpp
    KERNEL_SRC
        i0e_store_test_kernel.cpp
        i0e_store_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_i0e_store_kernel.cpp)
target_include_directories(i0e_store_test_e2e_v2 PRIVATE ${CODE_ROOT_DIR}/v35/ascendc/api_regbase)
