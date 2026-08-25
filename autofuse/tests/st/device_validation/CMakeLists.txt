add_library(device_validation_flat_output STATIC runner/flat_output.cpp)
target_include_directories(device_validation_flat_output PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/runner)

add_custom_target(device_validation)

if (ENABLE_DEVICE_VALIDATION_RUNNER OR ENABLE_DEVICE_VALIDATION_HARDWARE)
    add_library(device_validation_device_resources STATIC
        backend/acl_runtime.cpp
        backend/acl_runtime_cann.cpp
        backend/inprocess_profiler.cpp
        backend/tensor_buffer.cpp
        backend/host_clock.cpp
        backend/kernel_module.cpp
        backend/aclnn_executor.cpp
    )
    target_include_directories(device_validation_device_resources PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/backend)
    find_library(CANN_OPAPI_LIB NAMES opapi
        PATHS $ENV{ASCEND_HOME_PATH}
        PATH_SUFFIXES lib64 lib)
    find_library(CANN_NNOPBASE_LIB NAMES nnopbase
        PATHS $ENV{ASCEND_HOME_PATH}
        PATH_SUFFIXES lib64 lib)
    target_link_libraries(device_validation_device_resources PUBLIC acl_rt dl msprofiler ${CANN_OPAPI_LIB} ${CANN_NNOPBASE_LIB})

    add_library(device_validation_ascendc_backend STATIC backend/ascendc_backend.cpp)
    target_include_directories(device_validation_ascendc_backend PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/backend
        ${CODE_ROOT_DIR}/tests/framework/device_validation
    )
    target_link_libraries(device_validation_ascendc_backend PUBLIC
        device_validation_contract
        device_validation_device_resources
    )

    add_executable(device_validation_runner runner/device_validation_runner.cpp runner/flat_request_validation.cpp)
    target_include_directories(device_validation_runner PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/runner
        ${CMAKE_CURRENT_SOURCE_DIR}/backend
        ${CODE_ROOT_DIR}/tests/framework/device_validation
    )
    target_link_libraries(device_validation_runner PRIVATE
        device_validation_ascendc_backend
        device_validation_contract
        device_validation_flat_output
    )
    target_link_options(device_validation_runner PRIVATE
        "-Wl,--whole-archive"
        "$<TARGET_FILE:device_validation_device_resources>"
        "-Wl,--no-whole-archive"
    )
    add_custom_target(device_validation_st DEPENDS device_validation_runner)
endif()

if (ENABLE_DEVICE_VALIDATION_RUNNER)
    add_executable(device_validation_flat_step_contract_test runner/flat_step_contract_test.cpp)
    target_link_libraries(device_validation_flat_step_contract_test PRIVATE device_validation_contract)
    add_dependencies(device_validation_runner device_validation_flat_step_contract_test)
    add_test(NAME device_validation_flat_step_contract
        COMMAND device_validation_flat_step_contract_test $<TARGET_FILE:device_validation_runner>)
    set_tests_properties(device_validation_flat_step_contract PROPERTIES
        LABELS "st;device_validation_runner;contract"
        DEPENDS device_validation_runner_build
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    add_test(NAME device_validation_ctest_label_contract
        COMMAND ${CMAKE_COMMAND}
            -DCTEST_BINARY_DIR=${CMAKE_BINARY_DIR}/autofuse/tests/st
            -P ${CMAKE_CURRENT_SOURCE_DIR}/check_ctest_labels.cmake)
    set_tests_properties(device_validation_ctest_label_contract PROPERTIES
        LABELS "st;device_validation_runner;contract"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()

if (ENABLE_DEVICE_VALIDATION)
    add_test(NAME device_validation_cli_validation
        COMMAND ${Python3_EXECUTABLE} -m pytest
            ${CMAKE_CURRENT_SOURCE_DIR}/python/test_device_validation.py
            ${CMAKE_CURRENT_SOURCE_DIR}/python/test_comparison.py
            ${CMAKE_CURRENT_SOURCE_DIR}/python/test_helpers.py
            ${CMAKE_CURRENT_SOURCE_DIR}/python/test_jit_adapter.py
            ${CMAKE_CURRENT_SOURCE_DIR}/python/test_profiler.py
            ${CMAKE_CURRENT_SOURCE_DIR}/python/test_real_codegen_runner.py
            ${CMAKE_CURRENT_SOURCE_DIR}/python/test_step_output_validation.py -q -m "not real_codegen")
endif ()
if (ENABLE_DEVICE_VALIDATION_REAL_CODEGEN)
    add_test(NAME device_validation_real_codegen_preflight
        COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CMAKE_BINARY_DIR}/autofuse/compiler/py_module:${CMAKE_SOURCE_DIR}/autofuse/compiler/python:${CMAKE_SOURCE_DIR}/autofuse/tests/st:$ENV{PYTHONPATH}
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/real_codegen_preflight.py)
    set_tests_properties(device_validation_real_codegen_preflight PROPERTIES
        LABELS "st;real_codegen"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    add_test(NAME device_validation_real_codegen
        COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CMAKE_BINARY_DIR}/autofuse/compiler/py_module:${CMAKE_SOURCE_DIR}/autofuse/compiler/python:${CMAKE_SOURCE_DIR}/autofuse/tests/st:$ENV{PYTHONPATH}
            DEVICE_VALIDATION_REAL_CODEGEN_REQUIRED=1
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/real_codegen_runner.py)
    set_tests_properties(device_validation_real_codegen PROPERTIES
        LABELS "st;real_codegen"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS device_validation_real_codegen_preflight)
endif()
if (ENABLE_DEVICE_VALIDATION_RUNNER OR ENABLE_DEVICE_VALIDATION_HARDWARE)
    add_test(NAME device_validation_runner_build
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target device_validation_runner -- -j8)
    set_tests_properties(device_validation_runner_build PROPERTIES
        LABELS "st;device_validation_runner;build")
endif()
set_tests_properties(device_validation_cli_validation PROPERTIES
    LABELS "st;device_validation;host"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/autofuse/compiler/py_module:${CMAKE_SOURCE_DIR}/autofuse/compiler/python:${CMAKE_SOURCE_DIR}/autofuse/tests/st:$ENV{PYTHONPATH}")

if (ENABLE_DEVICE_VALIDATION_HARDWARE)
    set(DEVICE_VALIDATION_HARDWARE_CASE "" CACHE PATH
        "Device validation hardware case directory")
    set(DEVICE_VALIDATION_HARDWARE_PROFILE "" CACHE FILEPATH
        "Device validation hardware profile file")
    set(DEVICE_VALIDATION_HARDWARE_SOC "" CACHE STRING
        "Device validation hardware SoC profile")
    set(DEVICE_VALIDATION_HARDWARE_DEVICE "" CACHE STRING
        "Device validation hardware device id")
endif()

if (ENABLE_DEVICE_VALIDATION_HARDWARE AND
    DEVICE_VALIDATION_HARDWARE_CASE AND
    DEVICE_VALIDATION_HARDWARE_PROFILE AND
    DEVICE_VALIDATION_HARDWARE_SOC AND
    DEVICE_VALIDATION_HARDWARE_DEVICE)
    add_test(NAME device_validation_hardware
    COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CMAKE_SOURCE_DIR}/autofuse/tests/st
            DEVICE_VALIDATION_RUNNER=$<TARGET_FILE:device_validation_runner>
            DEVICE_VALIDATION_DEVICE_ID=${DEVICE_VALIDATION_HARDWARE_DEVICE}
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/hardware_preflight_wrapper.py
            --case ${DEVICE_VALIDATION_HARDWARE_CASE}
            --mode run --variant all --soc-profile ${DEVICE_VALIDATION_HARDWARE_SOC}
            --profile ${DEVICE_VALIDATION_HARDWARE_PROFILE}
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/hardware-artifacts
            )
    set_tests_properties(device_validation_hardware PROPERTIES
        LABELS "st;hardware"
        DEPENDS device_validation_runner_build)
endif()
