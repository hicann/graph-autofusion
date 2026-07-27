set(COMMON_ROOT_DIR ${CODE_ROOT_DIR}/common)
file(GLOB_RECURSE COMMON_SRC_FILES CONFIGURE_DEPENDS
        "${COMMON_ROOT_DIR}/*.cpp"
        "${COMMON_ROOT_DIR}/*.cc"
)
list(REMOVE_ITEM COMMON_SRC_FILES
        "${COMMON_ROOT_DIR}/autofuse_backend_spec_api.cpp"
)

file(GLOB_RECURSE COMMON_TEST_SRCS CONFIGURE_DEPENDS "*.cpp" "*.cc")
add_executable(test_common
        test.cpp
        ${COMMON_TEST_SRCS}
        ${COMMON_SRC_FILES}
)

target_include_directories(test_common PRIVATE
        ${CODE_ROOT_DIR}
        ${CODE_ROOT_DIR}/common
)

set(ASCGEN_DT_COMPILE_OPTION
        -O0
        -g
        --coverage -fprofile-arcs -ftest-coverage
)

target_compile_options(test_common PRIVATE ${ASCGEN_DT_COMPILE_OPTION}
    -Werror=format
    -Wno-deprecated-declarations
    -Wall -Wfloat-equal
    -Wno-subobject-linkage
)

target_link_libraries(test_common PRIVATE
    GTest::gtest
    GTest::gmock
    GTest::gtest_main
    -lgcov
    -Wl,--no-as-needed
    asc_slog_stub
    autofuse_runtime_stub
    atrace
    ascgen_common
    json
    error_manager
    metadef
)

add_test(NAME common_ut_aggregate
    COMMAND $<TARGET_FILE:test_common>
)
