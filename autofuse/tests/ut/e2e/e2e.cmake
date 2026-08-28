function(do_add_e2e_test)
    set(one_value_arg
        WORKDIR # Workdir
        )
    set(mul_value_arg
        TILING # Tiling codegen library source file
        CODEGEN # Codegen library source file
        KERNEL_SRC # Kernel source file that codegen will generate
        TEST_SRC # Test case source file
        )

    set(TEST_NAME ${ARGV0})
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "${one_value_arg}" "${mul_value_arg}")

    foreach(file ${ARG_KERNEL_SRC})
        list(APPEND KERNEL_SRC "${ARG_WORKDIR}/${file}")
    endforeach()

    add_library(${TEST_NAME}_tiling_gen SHARED ${ARG_TILING})
    target_include_directories(${TEST_NAME}_tiling_gen PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../)
    target_link_libraries(${TEST_NAME}_tiling_gen codegen)

    add_executable(${TEST_NAME}_codegen ${ARG_CODEGEN})
    set_target_properties(${TEST_NAME}_codegen PROPERTIES
        BUILD_RPATH "${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/ascendc_ir/generator:${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/ascendc_ir:${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/expression:${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph:${CMAKE_BINARY_DIR}/autofuse/tests:${CMAKE_BINARY_DIR}/autofuse/tests/depends/common:${CMAKE_BINARY_DIR}/autofuse/tests/depends/slog:${CMAKE_BINARY_DIR}/autofuse/tests/depends/trace:${CMAKE_BINARY_DIR}/autofuse/tests/depends/runtime:${CMAKE_BINARY_DIR}/autofuse/ascir/generator:${CMAKE_BINARY_DIR}/autofuse/ascir/meta:${ASCEND_INSTALL_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64"
    )
    target_link_options(${TEST_NAME}_codegen PRIVATE -Wl,--disable-new-dtags)
    target_link_libraries(${TEST_NAME}_codegen ${TEST_NAME}_tiling_gen codegen e2e aihac_symbolizer_af metadef) #aihac_symbolizer_af

    set(E2E_RUNTIME_LD_LIBRARY_PATH
        "${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/ascendc_ir/generator:${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/ascendc_ir:${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/expression:${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph:${CMAKE_BINARY_DIR}/autofuse/tests:${CMAKE_BINARY_DIR}/autofuse/tests/depends/common:${CMAKE_BINARY_DIR}/autofuse/tests/depends/slog:${CMAKE_BINARY_DIR}/autofuse/tests/depends/trace:${CMAKE_BINARY_DIR}/autofuse/tests/depends/runtime:${CMAKE_BINARY_DIR}/autofuse/ascir/generator:${CMAKE_BINARY_DIR}/autofuse/ascir/meta:${ASCEND_INSTALL_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64:$ENV{LD_LIBRARY_PATH}")

    add_custom_command(OUTPUT ${KERNEL_SRC}
                       WORKING_DIRECTORY ${ARG_WORKDIR}
                       COMMAND ${CMAKE_COMMAND} -E env
                               "LD_LIBRARY_PATH=${E2E_RUNTIME_LD_LIBRARY_PATH}"
                               $<TARGET_FILE:${TEST_NAME}_codegen>
                       DEPENDS ${TEST_NAME}_codegen)

    add_executable(${TEST_NAME} ${KERNEL_SRC} ${ARG_TEST_SRC})
    target_include_directories(${TEST_NAME} PRIVATE ${ARG_WORKDIR})
    target_link_libraries(${TEST_NAME} tikicpulib_ascend910B1 metadef GTest::gtest GTest::gtest_main)

    gtest_discover_tests(${TEST_NAME})
endfunction()

macro(add_e2e_test)
    do_add_e2e_test(${ARGV}
        WORKDIR ${CMAKE_CURRENT_BINARY_DIR})
endmacro()
