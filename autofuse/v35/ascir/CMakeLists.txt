file(GLOB_RECURSE SOURCES "*.cpp")

target_sources(aihac_codegen PRIVATE
        ${SOURCES}
)

target_include_directories(ascir_builtin_ops PRIVATE
        ${CODE_ROOT_DIR}
        ${CODE_ROOT_DIR}/ascir
        ${CODE_ROOT_DIR}/v35/ascir
)

target_include_directories(aihac_codegen PRIVATE
        ${CODE_ROOT_DIR}/ascir
        ${CODE_ROOT_DIR}/v35/ascir
)

target_sources(ascir_builtin_ops PRIVATE
        ${SOURCES}
)
