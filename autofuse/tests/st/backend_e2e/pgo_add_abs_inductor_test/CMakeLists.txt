set(PGO_ADD_ABS_INDUCTOR_WORKDIR ${CMAKE_CURRENT_BINARY_DIR})
set(PGO_ADD_ABS_INDUCTOR_TILING_DEF ${PGO_ADD_ABS_INDUCTOR_WORKDIR}/autofuse_tiling_data.h)
set(PGO_ADD_ABS_INDUCTOR_HOST_CODE ${PGO_ADD_ABS_INDUCTOR_WORKDIR}/pgo_add_abs_inductor_host.cpp)
set(PGO_ADD_ABS_INDUCTOR_DEVICE_CODE ${PGO_ADD_ABS_INDUCTOR_WORKDIR}/pgo_add_abs_inductor_device.cpp)
set(PGO_ADD_ABS_INDUCTOR_PGO_TILING_DEF ${PGO_ADD_ABS_INDUCTOR_WORKDIR}/autofuse_pgo_tiling_data.h)
set(PGO_ADD_ABS_INDUCTOR_PGO_HOST_CODE ${PGO_ADD_ABS_INDUCTOR_WORKDIR}/pgo_add_abs_inductor_pgo_host.cpp)
set(PGO_ADD_ABS_INDUCTOR_PGO_DEVICE_CODE ${PGO_ADD_ABS_INDUCTOR_WORKDIR}/pgo_add_abs_inductor_pgo_device.cpp)
set(PGO_ADD_ABS_INDUCTOR_OUTPUT_DIR ${PGO_ADD_ABS_INDUCTOR_WORKDIR}/split_compile_out)

set(PYAUTOFUSE_DIR ${CMAKE_BINARY_DIR}/autofuse/tests)
set(AUTOFUSE_PYTHON_DIR ${CODE_ROOT_DIR}/compiler/python)
set(AUTOFUSE_TEST_MSPTI_DIR "$ENV{AUTOFUSE_MSPTI_PATH}")
if(AUTOFUSE_TEST_MSPTI_DIR STREQUAL "")
    set(AUTOFUSE_TEST_MSPTI_DIR "${ASCEND_INSTALL_PATH}/tools/mspti")
endif()
set(PGO_ADD_ABS_INDUCTOR_RUNTIME_LD_LIBRARY_PATH
    "${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/ascendc_ir/generator:${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/ascendc_ir:${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph/expression:${CMAKE_BINARY_DIR}/autofuse/graph_metadef/graph:${CMAKE_BINARY_DIR}/autofuse/tests:${CMAKE_BINARY_DIR}/autofuse/tests/depends/common:${CMAKE_BINARY_DIR}/autofuse/tests/depends/slog:${CMAKE_BINARY_DIR}/autofuse/tests/depends/trace:${CMAKE_BINARY_DIR}/autofuse/tests/depends/runtime:${CMAKE_BINARY_DIR}/autofuse/ascir/generator:${CMAKE_BINARY_DIR}/autofuse/ascir/meta:${ASCEND_INSTALL_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64:$ENV{LD_LIBRARY_PATH}")

add_executable(pgo_add_abs_inductor_test_codegen pgo_add_abs_inductor_backend_generate.cpp)
target_link_options(pgo_add_abs_inductor_test_codegen PRIVATE -Wl,--no-as-needed)
target_link_libraries(pgo_add_abs_inductor_test_codegen
    atrace optimize att codegen common_stub share_graph ascgen_common
    aihac_symbolizer_af asc_slog_stub error_manager metadef json
    GTest::gtest GTest::gtest_main)
target_compile_definitions(pgo_add_abs_inductor_test_codegen PRIVATE
    ATT_SO_NAME="./libatt.so"
    KERNEL_SRC_LIST="${PGO_ADD_ABS_INDUCTOR_TILING_DEF}:${PGO_ADD_ABS_INDUCTOR_HOST_CODE}:${PGO_ADD_ABS_INDUCTOR_DEVICE_CODE}"
    PGO_KERNEL_SRC_LIST="${PGO_ADD_ABS_INDUCTOR_PGO_TILING_DEF}:${PGO_ADD_ABS_INDUCTOR_PGO_HOST_CODE}:${PGO_ADD_ABS_INDUCTOR_PGO_DEVICE_CODE}"
)
add_test(NAME pgo_add_abs_inductor_test_codegen
    COMMAND ${CMAKE_COMMAND} -E env
            "LD_LIBRARY_PATH=${PGO_ADD_ABS_INDUCTOR_RUNTIME_LD_LIBRARY_PATH}"
            LD_PRELOAD=$<TARGET_FILE:ascgen_common>
            $<TARGET_FILE:pgo_add_abs_inductor_test_codegen>
            --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/pgo_add_abs_inductor_test_codegen.xml)
set_tests_properties(pgo_add_abs_inductor_test_codegen PROPERTIES LABELS "st;build_backend_test1;pgo_add_abs_inductor_test_codegen")

file(MAKE_DIRECTORY ${PGO_ADD_ABS_INDUCTOR_OUTPUT_DIR})
add_custom_command(
    OUTPUT ${PGO_ADD_ABS_INDUCTOR_TILING_DEF} ${PGO_ADD_ABS_INDUCTOR_HOST_CODE} ${PGO_ADD_ABS_INDUCTOR_DEVICE_CODE}
           ${PGO_ADD_ABS_INDUCTOR_PGO_TILING_DEF} ${PGO_ADD_ABS_INDUCTOR_PGO_HOST_CODE}
           ${PGO_ADD_ABS_INDUCTOR_PGO_DEVICE_CODE}
    # common_stub fixes CanUseTilingKey to false; preload the production implementation for the PGO codegen path.
    COMMAND ${CMAKE_COMMAND} -E env
            "LD_LIBRARY_PATH=${PGO_ADD_ABS_INDUCTOR_RUNTIME_LD_LIBRARY_PATH}"
            LD_PRELOAD=$<TARGET_FILE:ascgen_common>
            $<TARGET_FILE:pgo_add_abs_inductor_test_codegen>
            --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/pgo_add_abs_inductor_test_codegen.xml
    DEPENDS pgo_add_abs_inductor_test_codegen
    VERBATIM)
add_custom_target(pgo_add_abs_inductor_test_codegen_outputs
    DEPENDS ${PGO_ADD_ABS_INDUCTOR_TILING_DEF} ${PGO_ADD_ABS_INDUCTOR_HOST_CODE} ${PGO_ADD_ABS_INDUCTOR_DEVICE_CODE}
            ${PGO_ADD_ABS_INDUCTOR_PGO_TILING_DEF} ${PGO_ADD_ABS_INDUCTOR_PGO_HOST_CODE}
            ${PGO_ADD_ABS_INDUCTOR_PGO_DEVICE_CODE})
add_inductor_host_abi1_helper(pgo_add_abs_inductor_host_abi1_helper pgo_add_abs_inductor_test_codegen_outputs
    ${PGO_ADD_ABS_INDUCTOR_WORKDIR} DYNAMIC_ARG_COUNT 3)
add_executable(pgo_add_abs_inductor_test_e2e test_e2e_pgo_add_abs_inductor_split_compile.cpp)
add_dependencies(pgo_add_abs_inductor_test_e2e pgo_add_abs_inductor_host_abi1_helper)
add_inductor_split_compile_test_runtime(pgo_add_abs_inductor_test_e2e)
target_link_libraries(pgo_add_abs_inductor_test_e2e dl GTest::gtest GTest::gtest_main unified_dlog)
target_include_directories(pgo_add_abs_inductor_test_e2e PRIVATE ${PGO_ADD_ABS_INDUCTOR_WORKDIR})
target_compile_definitions(pgo_add_abs_inductor_test_e2e PRIVATE
    TILING_DEF_FILE="${PGO_ADD_ABS_INDUCTOR_TILING_DEF}"
    HOST_CODE_FILE="${PGO_ADD_ABS_INDUCTOR_HOST_CODE}"
    DEVICE_CODE_FILE="${PGO_ADD_ABS_INDUCTOR_DEVICE_CODE}"
    PGO_TILING_DEF_FILE="${PGO_ADD_ABS_INDUCTOR_PGO_TILING_DEF}"
    PGO_HOST_CODE_FILE="${PGO_ADD_ABS_INDUCTOR_PGO_HOST_CODE}"
    PGO_DEVICE_CODE_FILE="${PGO_ADD_ABS_INDUCTOR_PGO_DEVICE_CODE}"
    PGO_FAKE_CALLBACK_SRC="${CMAKE_CURRENT_SOURCE_DIR}/pgo_measured_topn_fake_callback_main.cpp"
    OUTPUT_DIR="${PGO_ADD_ABS_INDUCTOR_OUTPUT_DIR}"
    PYAUTOFUSE_DIR="${PYAUTOFUSE_DIR}"
    AUTOFUSE_PYTHON_DIR="${AUTOFUSE_PYTHON_DIR}"
    ASCEND_HOME_PATH="${ASCEND_INSTALL_PATH}"
    MSPTI_DIR="${AUTOFUSE_TEST_MSPTI_DIR}"
    HOST_HELPER_BIN="$<TARGET_FILE:pgo_add_abs_inductor_host_abi1_helper>"
    HOST_DYNAMIC_SHAPE_ARGS="32,16,16"
)
add_test(NAME pgo_add_abs_inductor_test_e2e
    COMMAND pgo_add_abs_inductor_test_e2e --gtest_output=xml:${CMAKE_INSTALL_PREFIX}/report/st/pgo_add_abs_inductor_test_e2e.xml)
set_tests_properties(pgo_add_abs_inductor_test_e2e PROPERTIES LABELS "st;build_backend_test2;pgo_add_abs_inductor_test_e2e")
