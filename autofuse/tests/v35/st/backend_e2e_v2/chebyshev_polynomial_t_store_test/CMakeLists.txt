backend_e2e_st_test(chebyshev_polynomial_t_store_test
    CODEGEN chebyshev_polynomial_t_store_backend_generator.cpp
    KERNEL_SRC
        chebyshev_polynomial_t_store_test_kernel.cpp
        chebyshev_polynomial_t_store_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_chebyshev_polynomial_t_store_kernel.cpp)
target_include_directories(chebyshev_polynomial_t_store_test_e2e_v2 PRIVATE ${CODE_ROOT_DIR}/v35/ascendc/api_regbase)
