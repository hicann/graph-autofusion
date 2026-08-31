set(TAIL_REDUCE_WORKDIR ${CMAKE_CURRENT_BINARY_DIR})
set(TAIL_REDUCE_TILING_DEF ${TAIL_REDUCE_WORKDIR}/autofuse_tiling_data.h)
set(TAIL_REDUCE_HOST_CODE ${TAIL_REDUCE_WORKDIR}/inductor_tail_brc_tail_reduce_host.cpp)
set(TAIL_REDUCE_DEVICE_CODE ${TAIL_REDUCE_WORKDIR}/inductor_tail_brc_tail_reduce_device.cpp)
set(TAIL_REDUCE_PGO_TILING_DEF ${TAIL_REDUCE_WORKDIR}/autofuse_pgo_tiling_data.h)
set(TAIL_REDUCE_PGO_HOST_CODE ${TAIL_REDUCE_WORKDIR}/inductor_tail_brc_tail_reduce_pgo_host.cpp)
set(TAIL_REDUCE_PGO_DEVICE_CODE ${TAIL_REDUCE_WORKDIR}/inductor_tail_brc_tail_reduce_pgo_device.cpp)
set(TAIL_REDUCE_OUTPUT_DIR ${TAIL_REDUCE_WORKDIR}/split_compile_out)

set(PYAUTOFUSE_DIR ${CMAKE_BINARY_DIR}/autofuse/tests)
set(AUTOFUSE_PYTHON_DIR ${CODE_ROOT_DIR}/compiler/python)
set(AUTOFUSE_TEST_MSPTI_DIR "$ENV{AUTOFUSE_MSPTI_PATH}")
if(AUTOFUSE_TEST_MSPTI_DIR STREQUAL "")
    set(AUTOFUSE_TEST_MSPTI_DIR "${ASCEND_INSTALL_PATH}/tools/mspti")
endif()
set(INDUCTOR_TAIL_REDUCE_RUNTIME_LD_LIBRARY_PATH
    "${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/ascendc_ir/generator:${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/ascendc_ir:${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/expression:${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph:${CMAKE_BINARY_DIR}/autofuse/tests:${CMAKE_BINARY_DIR}/autofuse/tests/depends/common:${CMAKE_BINARY_DIR}/autofuse/tests/depends/slog:${CMAKE_BINARY_DIR}/autofuse/tests/depends/trace:${CMAKE_BINARY_DIR}/autofuse/tests/depends/runtime:${CMAKE_BINARY_DIR}/autofuse/ascir/generator:${CMAKE_BINARY_DIR}/autofuse/ascir/meta:${ASCEND_INSTALL_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64:$ENV{LD_LIBRARY_PATH}")

add_executable(inductor_tail_brc_tail_reduce_test_codegen inductor_tail_brc_tail_reduce_backend_generate.cpp)
target_link_options(inductor_tail_brc_tail_reduce_test_codegen PRIVATE -Wl,--no-as-needed)
target_link_libraries(inductor_tail_brc_tail_reduce_test_codegen
    atrace optimize att codegen common_stub share_graph ascgen_common
    aihac_symbolizer_af asc_slog_stub error_manager metadef json
    GTest::gtest GTest::gtest_main)
target_compile_definitions(inductor_tail_brc_tail_reduce_test_codegen PRIVATE
    ATT_SO_NAME="./libatt.so"
    KERNEL_SRC_LIST="${TAIL_REDUCE_TILING_DEF}:${TAIL_REDUCE_HOST_CODE}:${TAIL_REDUCE_DEVICE_CODE}"
    PGO_KERNEL_SRC_LIST="${TAIL_REDUCE_PGO_TILING_DEF}:${TAIL_REDUCE_PGO_HOST_CODE}:${TAIL_REDUCE_PGO_DEVICE_CODE}"
)
add_test(NAME inductor_tail_brc_tail_reduce_test_codegen
    COMMAND ${CMAKE_COMMAND} -E env
            "LD_LIBRARY_PATH=${INDUCTOR_TAIL_REDUCE_RUNTIME_LD_LIBRARY_PATH}"
            LD_PRELOAD=$<TARGET_FILE:ascgen_common>
            $<TARGET_FILE:inductor_tail_brc_tail_reduce_test_codegen>
            --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/inductor_tail_brc_tail_reduce_test_codegen.xml)
set_tests_properties(inductor_tail_brc_tail_reduce_test_codegen PROPERTIES LABELS "st;build_backend_test1;inductor_tail_brc_tail_reduce_test_codegen")

file(MAKE_DIRECTORY ${TAIL_REDUCE_OUTPUT_DIR})
add_custom_command(
    OUTPUT ${TAIL_REDUCE_TILING_DEF} ${TAIL_REDUCE_HOST_CODE} ${TAIL_REDUCE_DEVICE_CODE}
           ${TAIL_REDUCE_PGO_TILING_DEF} ${TAIL_REDUCE_PGO_HOST_CODE} ${TAIL_REDUCE_PGO_DEVICE_CODE}
    COMMAND ${CMAKE_COMMAND} -E env
            "LD_LIBRARY_PATH=${INDUCTOR_TAIL_REDUCE_RUNTIME_LD_LIBRARY_PATH}"
            LD_PRELOAD=$<TARGET_FILE:ascgen_common>
            $<TARGET_FILE:inductor_tail_brc_tail_reduce_test_codegen>
            --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/inductor_tail_brc_tail_reduce_test_codegen.xml
    DEPENDS inductor_tail_brc_tail_reduce_test_codegen
    VERBATIM)
add_custom_target(inductor_tail_brc_tail_reduce_test_codegen_outputs
    DEPENDS ${TAIL_REDUCE_TILING_DEF} ${TAIL_REDUCE_HOST_CODE} ${TAIL_REDUCE_DEVICE_CODE}
            ${TAIL_REDUCE_PGO_TILING_DEF} ${TAIL_REDUCE_PGO_HOST_CODE} ${TAIL_REDUCE_PGO_DEVICE_CODE})
add_inductor_host_abi1_helper(inductor_tail_brc_tail_reduce_host_abi1_helper
    inductor_tail_brc_tail_reduce_test_codegen_outputs ${TAIL_REDUCE_WORKDIR})
add_executable(inductor_tail_brc_tail_reduce_test_e2e test_e2e_inductor_tail_brc_tail_reduce_split_compile.cpp)
add_dependencies(inductor_tail_brc_tail_reduce_test_e2e inductor_tail_brc_tail_reduce_host_abi1_helper)
add_inductor_split_compile_test_runtime(inductor_tail_brc_tail_reduce_test_e2e)
target_link_libraries(inductor_tail_brc_tail_reduce_test_e2e dl GTest::gtest GTest::gtest_main unified_dlog)
target_include_directories(inductor_tail_brc_tail_reduce_test_e2e PRIVATE ${TAIL_REDUCE_WORKDIR})
target_compile_definitions(inductor_tail_brc_tail_reduce_test_e2e PRIVATE
    TILING_DEF_FILE="${TAIL_REDUCE_TILING_DEF}"
    HOST_CODE_FILE="${TAIL_REDUCE_HOST_CODE}"
    DEVICE_CODE_FILE="${TAIL_REDUCE_DEVICE_CODE}"
    PGO_TILING_DEF_FILE="${TAIL_REDUCE_PGO_TILING_DEF}"
    PGO_HOST_CODE_FILE="${TAIL_REDUCE_PGO_HOST_CODE}"
    PGO_DEVICE_CODE_FILE="${TAIL_REDUCE_PGO_DEVICE_CODE}"
    OUTPUT_DIR="${TAIL_REDUCE_OUTPUT_DIR}"
    PYAUTOFUSE_DIR="${PYAUTOFUSE_DIR}"
    AUTOFUSE_PYTHON_DIR="${AUTOFUSE_PYTHON_DIR}"
    ASCEND_HOME_PATH="${ASCEND_INSTALL_PATH}"
    MSPTI_DIR="${AUTOFUSE_TEST_MSPTI_DIR}"
    HOST_HELPER_BIN="$<TARGET_FILE:inductor_tail_brc_tail_reduce_host_abi1_helper>"
)
add_test(NAME inductor_tail_brc_tail_reduce_test_e2e
    COMMAND inductor_tail_brc_tail_reduce_test_e2e --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/inductor_tail_brc_tail_reduce_test_e2e.xml)
set_tests_properties(inductor_tail_brc_tail_reduce_test_e2e PROPERTIES LABELS "st;build_backend_test2;inductor_tail_brc_tail_reduce_test_e2e")
