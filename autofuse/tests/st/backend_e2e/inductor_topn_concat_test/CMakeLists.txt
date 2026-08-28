set(CONCAT_WORKDIR ${CMAKE_CURRENT_BINARY_DIR})
set(CONCAT_TILING_DEF ${CONCAT_WORKDIR}/autofuse_tiling_data.h)
set(CONCAT_HOST_CODE ${CONCAT_WORKDIR}/inductor_topn_concat_host.cpp)
set(CONCAT_DEVICE_CODE ${CONCAT_WORKDIR}/inductor_topn_concat_device.cpp)
set(CONCAT_OUTPUT_DIR ${CONCAT_WORKDIR}/split_compile_out)

set(PYAUTOFUSE_DIR ${CMAKE_BINARY_DIR}/autofuse/tests)
set(AUTOFUSE_PYTHON_DIR ${CODE_ROOT_DIR}/compiler/python)
set(INDUCTOR_TOPN_CONCAT_RUNTIME_LD_LIBRARY_PATH
    "${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/ascendc_ir/generator:${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/ascendc_ir:${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/expression:${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph:${CMAKE_BINARY_DIR}/autofuse/tests:${CMAKE_BINARY_DIR}/autofuse/tests/depends/common:${CMAKE_BINARY_DIR}/autofuse/tests/depends/slog:${CMAKE_BINARY_DIR}/autofuse/tests/depends/trace:${CMAKE_BINARY_DIR}/autofuse/tests/depends/runtime:${CMAKE_BINARY_DIR}/autofuse/ascir/generator:${CMAKE_BINARY_DIR}/autofuse/ascir/meta:${ASCEND_INSTALL_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64:$ENV{LD_LIBRARY_PATH}")

# codegen target: generates tiling_def + host_code + device_code
add_executable(inductor_topn_concat_test_codegen inductor_topn_concat_backend_generate.cpp)
target_link_options(inductor_topn_concat_test_codegen PRIVATE -Wl,--no-as-needed)
target_link_libraries(inductor_topn_concat_test_codegen
    atrace optimize att codegen common_stub share_graph ascgen_common
    aihac_symbolizer_af asc_slog_stub error_manager metadef json
    GTest::gtest GTest::gtest_main)
target_compile_definitions(inductor_topn_concat_test_codegen PRIVATE
    ATT_SO_NAME="./libatt.so"
    KERNEL_SRC_LIST="${CONCAT_TILING_DEF}:${CONCAT_HOST_CODE}:${CONCAT_DEVICE_CODE}"
)
add_test(NAME inductor_topn_concat_test_codegen
    COMMAND ${CMAKE_COMMAND} -E env
            "LD_LIBRARY_PATH=${INDUCTOR_TOPN_CONCAT_RUNTIME_LD_LIBRARY_PATH}"
            inductor_topn_concat_test_codegen --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/inductor_topn_concat_test_codegen.xml)
set_tests_properties(inductor_topn_concat_test_codegen PROPERTIES LABELS "st;build_backend_test1;inductor_topn_concat_test_codegen")

# e2e target: split compile chain verification
file(MAKE_DIRECTORY ${CONCAT_OUTPUT_DIR})
add_custom_command(
    OUTPUT ${CONCAT_TILING_DEF} ${CONCAT_HOST_CODE} ${CONCAT_DEVICE_CODE}
    COMMAND ${CMAKE_COMMAND} -E env
            "LD_LIBRARY_PATH=${INDUCTOR_TOPN_CONCAT_RUNTIME_LD_LIBRARY_PATH}"
            $<TARGET_FILE:inductor_topn_concat_test_codegen>
            --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/inductor_topn_concat_test_codegen.xml
    DEPENDS inductor_topn_concat_test_codegen
    VERBATIM)
add_custom_target(inductor_topn_concat_test_codegen_outputs
    DEPENDS ${CONCAT_TILING_DEF} ${CONCAT_HOST_CODE} ${CONCAT_DEVICE_CODE})
add_inductor_host_abi1_helper(inductor_topn_concat_host_abi1_helper inductor_topn_concat_test_codegen_outputs
    ${CONCAT_WORKDIR})
add_executable(inductor_topn_concat_test_e2e test_e2e_inductor_topn_concat_split_compile.cpp)
add_dependencies(inductor_topn_concat_test_e2e inductor_topn_concat_host_abi1_helper)
add_inductor_split_compile_test_runtime(inductor_topn_concat_test_e2e)
target_link_libraries(inductor_topn_concat_test_e2e dl GTest::gtest GTest::gtest_main unified_dlog)
target_include_directories(inductor_topn_concat_test_e2e PRIVATE ${CONCAT_WORKDIR})
target_compile_definitions(inductor_topn_concat_test_e2e PRIVATE
    TILING_DEF_FILE="${CONCAT_TILING_DEF}"
    HOST_CODE_FILE="${CONCAT_HOST_CODE}"
    DEVICE_CODE_FILE="${CONCAT_DEVICE_CODE}"
    OUTPUT_DIR="${CONCAT_OUTPUT_DIR}"
    PYAUTOFUSE_DIR="${PYAUTOFUSE_DIR}"
    AUTOFUSE_PYTHON_DIR="${AUTOFUSE_PYTHON_DIR}"
    ASCEND_HOME_PATH="${ASCEND_INSTALL_PATH}"
    HOST_HELPER_BIN="$<TARGET_FILE:inductor_topn_concat_host_abi1_helper>"
    HOST_INPUT_CONFIGS_JSON="[{}]"
    HOST_TOPN=10
)
add_test(NAME inductor_topn_concat_test_e2e
    COMMAND inductor_topn_concat_test_e2e --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/inductor_topn_concat_test_e2e.xml)
set_tests_properties(inductor_topn_concat_test_e2e PROPERTIES LABELS "st;build_backend_test2;inductor_topn_concat_test_e2e")
