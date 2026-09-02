backend_e2e_st_test(fused_backend_elewise_test
    CODEGEN fused_backend_elewise_generate.cpp
    KERNEL_SRC
        fused_backend_elewise_test_kernel.cpp
        fused_backend_elewise_test_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_fused_backend_elewise_expect_kernel.cpp)
