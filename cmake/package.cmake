# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

#### CPACK to package run #####
message(STATUS "System processor: ${CMAKE_SYSTEM_PROCESSOR}")
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    message(STATUS "Detected architecture: x86_64")
    set(ARCH x86_64)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
    message(STATUS "Detected architecture: ARM64")
    set(ARCH aarch64)
else ()
    message(WARNING "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    set(ARCH ${CMAKE_SYSTEM_PROCESSOR})
endif ()
# 打印路径
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
message(STATUS "CMAKE_SOURCE_DIR = ${CMAKE_SOURCE_DIR}")
message(STATUS "CANN_CMAKE_DIR = ${CANN_CMAKE_DIR}")
message(STATUS "CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}")
set(ARCH_LINUX_PATH "${ARCH}-linux")

set(script_prefix ${CMAKE_CURRENT_SOURCE_DIR}/scripts/package/graph_autofusion/scripts)
install(DIRECTORY ${script_prefix}/
    DESTINATION share/info/graph_autofusion/script
    FILE_PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE  # 文件权限
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
    DIRECTORY_PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE  # 目录权限
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
    COMPONENT graph-autofusion
)
set(SCRIPTS_FILES
    ${CANN_CMAKE_DIR}/scripts/install/check_version_required.awk
    ${CANN_CMAKE_DIR}/scripts/install/common_func.inc
    ${CANN_CMAKE_DIR}/scripts/install/common_interface.sh
    ${CANN_CMAKE_DIR}/scripts/install/common_interface.csh
    ${CANN_CMAKE_DIR}/scripts/install/common_interface.fish
    ${CANN_CMAKE_DIR}/scripts/install/version_compatiable.inc
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/package/graph_autofusion/scripts/cleanup.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/package/graph_autofusion/scripts/help.info
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/package/graph_autofusion/scripts/install.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/package/graph_autofusion/scripts/run_graph_autofusion_install.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/package/graph_autofusion/scripts/run_graph_autofusion_uninstall.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/package/graph_autofusion/scripts/run_graph_autofusion_upgrade.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/package/graph_autofusion/scripts/uninstall.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/package/graph_autofusion/scripts/ver_check.sh
)

install(FILES ${SCRIPTS_FILES}
    DESTINATION share/info/graph_autofusion/script
    PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE  # 文件权限
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
    COMPONENT graph-autofusion
)
set(COMMON_FILES
    ${CANN_CMAKE_DIR}/scripts/install/install_common_parser.sh
    ${CANN_CMAKE_DIR}/scripts/install/common_func_v2.inc
    ${CANN_CMAKE_DIR}/scripts/install/common_installer.inc
    ${CANN_CMAKE_DIR}/scripts/install/script_operator.inc
    ${CANN_CMAKE_DIR}/scripts/install/version_cfg.inc
)

set(PACKAGE_FILES
    ${COMMON_FILES}
    ${CANN_CMAKE_DIR}/scripts/install/multi_version.inc
)
install(FILES ${CMAKE_BINARY_DIR}/version.graph-autofusion.info
    DESTINATION share/info/graph_autofusion
    RENAME version.info
    COMPONENT graph-autofusion
)
install(FILES ${PACKAGE_FILES}
    DESTINATION share/info/graph_autofusion/script
    COMPONENT graph-autofusion
)

install(TARGETS ascendsk
    DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
    COMPONENT graph-autofusion
)
install(FILES super_kernel/include/super_kernel/super_kernel.h
    DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/include/super_kernel
    COMPONENT graph-autofusion
)

set_cann_cpack_config(graph-autofusion ENABLE_DEVICE ${ENABLE_DEVICE} SHARE_INFO_NAME graph_autofusion PACKAGE_TYPE ${PACKAGE_TYPE})
