set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${ASCEND_ROOT}/tools/tikicpulib/lib)
find_package(tikicpulib QUIET)

# 要修复的目标列表
set(TIKICPULIB_TARGETS
        tikicpulib_ascend910B1
        tikicpulib_ascend950pr_9599
        # 添加其他可能出错的 target
)

foreach(tgt IN LISTS TIKICPULIB_TARGETS)
    if(TARGET ${tgt})
        # 修复 INTERFACE_INCLUDE_DIRECTORIES
        get_target_property(include_dirs ${tgt} INTERFACE_INCLUDE_DIRECTORIES)
        if(NOT include_dirs)
            set(include_dirs "")
        endif()

        set(valid_includes "")
        foreach(dir IN LISTS include_dirs)
            # 跳过明显错误的嵌套路径（可选）
            if(dir MATCHES "tools/tikicpulib/tools/tikicpulib")
                message(VERBOSE "Skipping malformed include dir: ${dir}")
                continue()
            endif()

            # 只保留真实存在的目录
            if(IS_DIRECTORY "${dir}")
                list(APPEND valid_includes "${dir}")
            else()
                message(VERBOSE "Skipping non-existent include dir: ${dir}")
            endif()
        endforeach()

        set_target_properties(${tgt}
                PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${valid_includes}"
        )

        # 同样修复 INTERFACE_LINK_DIRECTORIES（如有需要）
        get_target_property(link_dirs ${tgt} INTERFACE_LINK_DIRECTORIES)
        if(link_dirs)
            set(valid_links "")
            foreach(dir IN LISTS link_dirs)
                if(IS_DIRECTORY "${dir}")
                    list(APPEND valid_links "${dir}")
                endif()
            endforeach()
            set_target_properties(${tgt}
                    PROPERTIES INTERFACE_LINK_DIRECTORIES "${valid_links}"
            )
        endif()
    endif()
endforeach()
include(e2e.cmake)

add_library(e2e_v2_com SHARED
    ./e2e_load_abs_store.cpp
    ./e2e_load_scalar_abs_brc_store.cpp
    ./e2e_load_scalar_brc_store_2d.cpp
    ./e2e_load_scalar_clip_store.cpp
    ./e2e_load_sub_store.cpp
    ./e2e_load_div_store.cpp
    ./e2e_load_nan_out_for_store.cpp
    ./e2e_load_scalar_sub_store.cpp
    ./e2e_load_scalar_div_store.cpp
    ./e2e_load_switch_scalar_sub_store.cpp
    )
target_link_libraries(e2e_v2_com PUBLIC
    -Wl,--no-as-needed
    ascir
    optimize
    codegen
    aihac_ir
    aihac_ir_register
    graph_af
    )
target_include_directories(e2e_v2_com PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ASCEND_INSTALL_PATH}/x86_64-linux/include/acl/error_codes
    )

add_subdirectory(load_abs_store_expect_code)
add_subdirectory(load_scalar_abs_brc_store_expect_code)
add_subdirectory(load_scalar_brc_store_2d_expect_code)
add_subdirectory(load_scalar_clip_store_expect_code)
# add_subdirectory(load_sub_store_expect_code)
add_subdirectory(load_div_store_expect_code)
add_subdirectory(load_nan_out_for_store_expect_code)
add_subdirectory(load_scalar_sub_store_expect_code)
add_subdirectory(load_scalar_div_store_expect_code)
add_subdirectory(load_switch_scalar_sub_store_expect_code)
