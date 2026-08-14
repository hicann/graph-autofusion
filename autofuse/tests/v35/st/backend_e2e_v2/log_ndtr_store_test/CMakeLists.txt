backend_e2e_st_test(log_ndtr_store_test
    CODEGEN log_ndtr_store_backend_generator.cpp
    KERNEL_SRC
        log_ndtr_store_test_kernel.cpp
        log_ndtr_store_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_log_ndtr_store_kernel.cpp)
target_include_directories(log_ndtr_store_test_e2e_v2 PRIVATE ${CODE_ROOT_DIR}/v35/ascendc/api_regbase)
