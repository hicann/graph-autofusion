execute_process(
    COMMAND ctest --test-dir "${CTEST_BINARY_DIR}" -N -L real_codegen
    OUTPUT_VARIABLE REAL_CODEGEN_OUTPUT
    RESULT_VARIABLE REAL_CODEGEN_RESULT)
if(NOT REAL_CODEGEN_RESULT EQUAL 0)
    message(FATAL_ERROR "failed to inspect real_codegen discovery")
endif()
if(REAL_CODEGEN_OUTPUT MATCHES "device_validation_flat_step_contract")
    message(FATAL_ERROR "runner contract test leaked into real_codegen labels")
endif()
execute_process(
    COMMAND ctest --test-dir "${CTEST_BINARY_DIR}" -N -L device_validation_runner
    OUTPUT_VARIABLE RUNNER_OUTPUT
    RESULT_VARIABLE RUNNER_RESULT)
if(NOT RUNNER_RESULT EQUAL 0 OR NOT RUNNER_OUTPUT MATCHES "device_validation_flat_step_contract")
    message(FATAL_ERROR "runner contract test was not discovered")
endif()
