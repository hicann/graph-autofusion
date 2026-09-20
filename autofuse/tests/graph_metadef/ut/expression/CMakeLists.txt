file(GLOB EXPRESSION_UT_SRCS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/testcase/*.cc")
add_executable(ut_expression ${EXPRESSION_UT_SRCS})

target_compile_options(ut_expression PRIVATE
    -g -O0
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal
    -fno-access-control
)

target_link_libraries(ut_expression PRIVATE
    aihac_symbolizer_af
    GTest::gtest
    GTest::gtest_main
    c_sec
    unified_dlog
    -lgcov
    -Wl,--no-as-needed
)

add_test(NAME ut_expression COMMAND ut_expression)
