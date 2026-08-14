backend_e2e_st_test(hermite_polynomial_h_store_test
    CODEGEN hermite_polynomial_h_store_backend_generator.cpp
    KERNEL_SRC
        hermite_polynomial_h_store_test_kernel.cpp
        hermite_polynomial_h_store_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_hermite_polynomial_h_store_kernel.cpp)
target_include_directories(hermite_polynomial_h_store_test_e2e_v2 PRIVATE ${CODE_ROOT_DIR}/v35/ascendc/api_regbase)
