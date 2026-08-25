add_executable(device_validation_ut
    test_case_contract.cpp
    test_support_matrix.cpp
    test_device_resources.cpp
    test_inprocess_profiler.cpp
    test_host_clock.cpp
    test_kernel_module_timing.cpp
    test_ascendc_backend.cpp
    test_flat_request_validation.cpp
    test_aclnn_dispatch.cpp
    ../../st/device_validation/backend/acl_runtime.cpp
    ../../st/device_validation/backend/tensor_buffer.cpp
    ../../st/device_validation/backend/kernel_module.cpp
    ../../st/device_validation/backend/host_clock.cpp
    ../../st/device_validation/backend/ascendc_backend.cpp
    ../../st/device_validation/runner/flat_output.cpp
    ../../st/device_validation/runner/flat_request_validation.cpp
)
target_include_directories(device_validation_ut PRIVATE
    ${CODE_ROOT_DIR}/tests/framework/device_validation
    ${CODE_ROOT_DIR}/tests/st/device_validation/backend
    ${CODE_ROOT_DIR}/tests/st/device_validation/runner
)
target_link_libraries(device_validation_ut PRIVATE
    device_validation_contract
    GTest::gtest
    GTest::gtest_main
    dl
)
add_test(NAME device_validation_ut COMMAND $<TARGET_FILE:device_validation_ut>)
set_tests_properties(device_validation_ut PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
