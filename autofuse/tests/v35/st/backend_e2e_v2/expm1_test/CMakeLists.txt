backend_e2e_st_test(expm1_test
    CODEGEN expm1_backend_generator.cpp
    KERNEL_SRC
        expm1_test_kernel.cpp
        expm1_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_expm1_expect_kernel.cpp)
