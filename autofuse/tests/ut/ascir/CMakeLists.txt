add_library(test_ascir_ut OBJECT
    #test_ascir.cpp
    #test_ascir_ops.cpp
    #test_header_generator.cpp
    #test_ascir_graph.cpp
    #test_cg_util.cpp
    reg_func/test_reg_func_broadcast.cpp
    reg_func/test_reg_func_clipbyvalue.cpp
    reg_func/test_reg_func_compare.cpp
    reg_func/test_reg_func_concat.cpp
    reg_func/test_reg_func_div.cpp
    reg_func/test_reg_func_remainder.cpp
    reg_func/test_reg_func_is_nan.cpp
    reg_func/test_reg_func_is_inf.cpp
    reg_func/test_reg_func_masked_fill.cpp
    reg_func/test_reg_func_logic_and.cpp
    reg_func/test_reg_func_logic_not.cpp
    reg_func/test_reg_func_logic_or.cpp
    reg_func/test_reg_func_pow.cpp
    reg_func/test_reg_func_rsqrt.cpp
    reg_func/test_reg_func_sigmoid.cpp
    reg_func/test_reg_func_sign.cpp
    reg_func/test_reg_func_sub.cpp
    reg_func/test_reg_func_tanh.cpp
    reg_func/test_reg_func_truediv.cpp
    reg_func/test_reg_func_where.cpp
    # reg_func/test_reg_func_zeros_like.cpp
    reg_func/test_reg_func_cast.cpp
    reg_func/test_reg_func_isfinite.cpp
    reg_func/test_reg_func_reduce.cpp
    reg_func/test_reg_func_gather.cpp
    reg_func/test_reg_func_erf.cpp
    reg_func/test_reg_func_axpy.cpp
    reg_func/test_reg_func_welford.cpp
    reg_func/test_reg_func_abs.cpp
    reg_func/test_reg_func_reduce_max.cpp
    code_dumper_unittest.cc
    ascir_utils_unittest.cc
    test_asc_graph_utils.cpp
)
target_include_directories(test_ascir_ut PRIVATE
    ${ASCEND_ROOT}/x86_64-linux/include
    ${ASCEND_ROOT}/opp/built-in/op_proto/inc)
target_link_libraries(test_ascir_ut
    ascir
    ascir_builtin_ops
    #ge_compiler
    GTest::gtest)

add_executable(test_index_expr_arange_schema test_index_expr_arange_schema.cpp)
target_link_libraries(test_index_expr_arange_schema PRIVATE
    ascir
    ascir_builtin_ops
    ascir_ops_headers
    aihac_symbolizer_af
    metadef
    GTest::gtest
    GTest::gtest_main)
gtest_discover_tests(test_index_expr_arange_schema)
