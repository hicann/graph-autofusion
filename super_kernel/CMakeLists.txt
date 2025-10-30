# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

file(GLOB_RECURSE SUPER_KERNEL_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.py")

add_custom_target(superkernel_whl ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/superkernel-0.1.0-py3-none-any.whl)
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/superkernel-0.1.0-py3-none-any.whl
        COMMAND echo "Start to build superkernel wheel package"
        && cd ${CMAKE_CURRENT_SOURCE_DIR}
        && python3 -m build --wheel --no-isolation --outdir ${CMAKE_CURRENT_BINARY_DIR}
        && rm -rf build
        && echo "Build superkernel wheel package end"
        DEPENDS ${SUPER_KERNEL_SRC} pyproject.toml
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/superkernel-0.1.0-py3-none-any.whl OPTIONAL
        DESTINATION ${INSTALL_LIBRARY_DIR}
)
