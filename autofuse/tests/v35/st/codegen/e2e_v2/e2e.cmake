function(do_add_codegen_e2e_st_test)
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

    message(STATUS "ARG_WORKDIR=${ARG_WORKDIR}")

    foreach(file ${ARG_KERNEL_SRC})
        list(APPEND KERNEL_SRC "${ARG_WORKDIR}/${file}")
    endforeach()
    set_source_files_properties(${KERNEL_SRC} PROPERTIES GENERATED TRUE)

    set(ATT_SO_NAME ${TEST_NAME}_gen_tiling_v2) #拼接so的名字
    set(E2E_ST1_GENERATOR_EXE_NAME ${TEST_NAME}_codegen_v2)
    set(E2E_ST2_EXE_KERNEL_EXE_NAME ${TEST_NAME}_e2e_v2)

    add_library(${ATT_SO_NAME} SHARED ${ARG_TILING})
    target_include_directories(${ATT_SO_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../)
    target_link_libraries(${ATT_SO_NAME} codegen)

    add_executable(${E2E_ST1_GENERATOR_EXE_NAME} ${ARG_CODEGEN})
    set_target_properties(${E2E_ST1_GENERATOR_EXE_NAME} PROPERTIES
        BUILD_RPATH "${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph"
    )
    target_link_options(${E2E_ST1_GENERATOR_EXE_NAME} PRIVATE -Wl,--disable-new-dtags)
    target_include_directories(${E2E_ST1_GENERATOR_EXE_NAME} BEFORE PRIVATE
        ${CODE_ROOT_DIR}/tests/v35/st/optimize
        ${CODE_ROOT_DIR}/tests/st/common
        ${CODE_ROOT_DIR}/tests
    )
    target_link_libraries(${E2E_ST1_GENERATOR_EXE_NAME}
                          ${ATT_SO_NAME}
                          codegen
                          e2e_v2_com
                          ascgen_common
                          aihac_symbolizer_af
                          error_manager
                          metadef
                          json
                          GTest::gtest
                          GTest::gtest_main)
    set_target_properties(${E2E_ST1_GENERATOR_EXE_NAME} PROPERTIES
        BUILD_RPATH "${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph"
        LINK_OPTIONS "-Wl,--disable-new-dtags")


    list (JOIN ARG_KERNEL_SRC ":" KERNEL_SRC_LIST)
    message(STATUS "KERNEL_SRC_LIST = ${KERNEL_SRC_LIST}")

    target_compile_definitions(${E2E_ST1_GENERATOR_EXE_NAME} PRIVATE
       ATT_SO_NAME=\"./lib${ATT_SO_NAME}.so\"
       KERNEL_SRC_LIST=\"${KERNEL_SRC_LIST}\"
    )

    set(CODEGEN_RUNTIME_LD_LIBRARY_PATH
        "${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/ascendc_ir/generator:${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/ascendc_ir:${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/expression:${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph:${CMAKE_BINARY_DIR}/autofuse/tests:${CMAKE_BINARY_DIR}/autofuse/tests/depends/common:${CMAKE_BINARY_DIR}/autofuse/tests/depends/slog:${CMAKE_BINARY_DIR}/autofuse/tests/depends/trace:${CMAKE_BINARY_DIR}/autofuse/tests/depends/runtime:${CMAKE_BINARY_DIR}/autofuse/ascir/generator:${CMAKE_BINARY_DIR}/autofuse/ascir/meta:${ASCEND_INSTALL_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64:$ENV{LD_LIBRARY_PATH}")

    add_test(NAME ${E2E_ST1_GENERATOR_EXE_NAME}
             COMMAND ${CMAKE_COMMAND} -E env
                     "LD_LIBRARY_PATH=${CODEGEN_RUNTIME_LD_LIBRARY_PATH}"
                     $<TARGET_FILE:${E2E_ST1_GENERATOR_EXE_NAME}>
                     --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/v35/st/${E2E_ST1_GENERATOR_EXE_NAME}.xml)
    set_tests_properties(${E2E_ST1_GENERATOR_EXE_NAME} PROPERTIES LABELS "st;codegen_e2e_st_test1;${E2E_ST1_GENERATOR_EXE_NAME}")

    add_custom_target(${TEST_NAME}_generated_sources_v2
                      COMMAND ${CMAKE_COMMAND} -E env
                              "LD_LIBRARY_PATH=${CODEGEN_RUNTIME_LD_LIBRARY_PATH}"
                              $<TARGET_FILE:${E2E_ST1_GENERATOR_EXE_NAME}>
                      WORKING_DIRECTORY ${ARG_WORKDIR}
                      BYPRODUCTS ${KERNEL_SRC}
                      DEPENDS ${E2E_ST1_GENERATOR_EXE_NAME})

    add_executable(${E2E_ST2_EXE_KERNEL_EXE_NAME} ${KERNEL_SRC} ${ARG_TEST_SRC})
    add_dependencies(${E2E_ST2_EXE_KERNEL_EXE_NAME} ${TEST_NAME}_generated_sources_v2)
    target_include_directories(${E2E_ST2_EXE_KERNEL_EXE_NAME} PRIVATE ${ARG_WORKDIR})
    target_link_libraries(${E2E_ST2_EXE_KERNEL_EXE_NAME} tikicpulib_ascend950pr_9599 GTest::gtest GTest::gtest_main)
    set_target_properties(${E2E_ST2_EXE_KERNEL_EXE_NAME} PROPERTIES
        BUILD_RPATH "${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph"
        LINK_OPTIONS "-Wl,--disable-new-dtags")
    target_compile_options(${E2E_ST2_EXE_KERNEL_EXE_NAME} PRIVATE -DAUTO_FUSE_DEVICE=1)
    #gtest_discover_tests(${E2E_ST2_EXE_KERNEL_EXE_NAME})
    add_test(NAME ${E2E_ST2_EXE_KERNEL_EXE_NAME} COMMAND $<TARGET_FILE:${E2E_ST2_EXE_KERNEL_EXE_NAME}> --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/v35/st/${E2E_ST2_EXE_KERNEL_EXE_NAME}.xml)
    set_tests_properties(${E2E_ST2_EXE_KERNEL_EXE_NAME} PROPERTIES LABELS "st;codegen_e2e_st_test2;${E2E_ST2_EXE_KERNEL_EXE_NAME}")
endfunction()

macro(add_codegen_e2e_st_test)
    do_add_codegen_e2e_st_test(${ARGV}  WORKDIR ${CMAKE_CURRENT_BINARY_DIR})
endmacro()
