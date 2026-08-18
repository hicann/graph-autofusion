backend_e2e_st_test(randn_store_test
    CODEGEN randn_store_backend_generator.cpp
    KERNEL_SRC
        randn_store_test_kernel.cpp
        randn_store_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_randn_store_kernel.cpp)
target_include_directories(randn_store_test_e2e_v2 PRIVATE ${CODE_ROOT_DIR}/v35/ascendc/api_regbase)
