add_codegen_e2e_st_test(load_scalar_brc_store_2d_expect_code
    TILING load_scalar_brc_store_2d_codegen_tiling.cpp
    CODEGEN load_scalar_brc_store_2d_codegen.cpp
    KERNEL_SRC
        load_scalar_brc_store_2d_kernel.cpp
        load_scalar_brc_store_2d_tiling.cpp
        autofuse_tiling_data.h
    TEST_SRC test_e2e_load_scalar_brc_store_2d_expect_kernel.cpp)
