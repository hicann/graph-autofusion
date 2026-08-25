add_library(device_validation_contract STATIC
    case_contract.cpp
    support_matrix.cpp
    profile.cpp
    report.cpp
)
target_include_directories(device_validation_contract PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CODE_ROOT_DIR}
)
target_link_libraries(device_validation_contract PUBLIC json)
