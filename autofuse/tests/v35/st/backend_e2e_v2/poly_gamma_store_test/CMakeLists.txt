backend_e2e_st_test(poly_gamma_store_test
    CODEGEN poly_gamma_store_backend_generator.cpp
    KERNEL_SRC
        poly_gamma_store_test_kernel.cpp
        poly_gamma_store_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_poly_gamma_store_kernel.cpp)
target_include_directories(poly_gamma_store_test_e2e_v2 PRIVATE ${CODE_ROOT_DIR}/v35/ascendc/api_regbase)
