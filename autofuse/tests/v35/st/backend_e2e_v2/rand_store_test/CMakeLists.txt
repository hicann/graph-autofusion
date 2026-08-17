backend_e2e_st_test(rand_store_test
    CODEGEN rand_store_backend_generator.cpp
    KERNEL_SRC
        rand_store_test_kernel.cpp
        rand_store_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_rand_store_kernel.cpp)
target_include_directories(rand_store_test_e2e_v2 PRIVATE ${CODE_ROOT_DIR}/v35/ascendc/api_regbase)
