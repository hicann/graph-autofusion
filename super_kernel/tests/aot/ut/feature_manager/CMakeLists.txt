# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# Standalone host-only entry for the same tests included in super_kernel_aot_utest.
cmake_minimum_required(VERSION 3.16)
project(static_compile_features_ut LANGUAGES CXX)
enable_testing()
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)
get_filename_component(SK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
add_executable(feature_manager_ut
    ../test_feature_manager.cpp
    ${SK_ROOT}/src/aot/feature_manager.cpp
    ${SK_ROOT}/src/aot/kernel_compile_features.cpp
)
target_compile_features(feature_manager_ut PRIVATE cxx_std_17)
target_compile_definitions(feature_manager_ut PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
target_compile_options(feature_manager_ut PRIVATE -Wall -Wextra -Werror)
target_include_directories(feature_manager_ut PRIVATE
    ${SK_ROOT}/src/aot
    ${SK_ROOT}/include/super_kernel
)
target_include_directories(feature_manager_ut SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../stub)
target_link_libraries(feature_manager_ut PRIVATE GTest::gtest_main Threads::Threads)
include(GoogleTest)
gtest_discover_tests(feature_manager_ut)

# Compile-only contracts: failures must be caused by the public API, not unresolved library symbols.
function(check_query_contract name expected)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    try_compile(contract_compiled "${CMAKE_CURRENT_BINARY_DIR}/contract_${name}"
        "${CMAKE_CURRENT_SOURCE_DIR}/query_contract.cpp"
        CMAKE_FLAGS
            "-DCMAKE_CXX_STANDARD=17"
            "-DINCLUDE_DIRECTORIES=${SK_ROOT}/src/aot;${SK_ROOT}/include/super_kernel;${CMAKE_CURRENT_SOURCE_DIR}/../stub"
        COMPILE_DEFINITIONS "-D${name}" -Werror=unused-result
        OUTPUT_VARIABLE contract_log
    )
    if(NOT "${contract_compiled}" STREQUAL "${expected}")
        message(FATAL_ERROR "Unexpected query contract result for ${name}: ${contract_log}")
    endif()
endfunction()
check_query_contract(VALID_QUERY TRUE)
check_query_contract(QUERY_MODEL_WITH_ENTRY FALSE)
check_query_contract(QUERY_KERNEL_WITHOUT_ENTRY FALSE)
check_query_contract(QUERY_COUNT FALSE)
check_query_contract(IGNORE_INIT FALSE)
