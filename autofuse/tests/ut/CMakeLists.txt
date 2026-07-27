set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
add_subdirectory(att)
add_subdirectory(ascendc)
add_subdirectory(codegen)
add_subdirectory(py_module)
add_subdirectory(e2e)
add_subdirectory(common)
add_subdirectory(ascir)
find_program(PYTEST_EXECUTABLE pytest)

if(PYTEST_EXECUTABLE)
  message(STATUS "Found pytest: ${PYTEST_EXECUTABLE}")
  add_subdirectory(python)
else()
  message(STATUS "pytest not found")
endif()
add_subdirectory(optimize)

add_executable(test_main test_main.cpp)

set(PEM_DAVINCI_PATH "${ASCEND_INSTALL_PATH}/x86_64-linux/simulator/dav_2201/lib/libpem_davinci.so")
if(EXISTS "${PEM_DAVINCI_PATH}")
    target_link_libraries(test_main PRIVATE ${PEM_DAVINCI_PATH})
endif()

target_link_libraries(test_main PRIVATE
    atrace_stub
    error_manager
    aihac_symbolizer_af
    test_ascir_ut
    test_codegen
    test_py_module
    test_e2e
    autofuse_runtime_stub
    metadef
    GTest::gtest
)
add_dependencies(test_main ascir_ops_headers)
gtest_discover_tests(test_main)

add_test(NAME test_main_ut_aggregate
    COMMAND $<TARGET_FILE:test_main>
)
