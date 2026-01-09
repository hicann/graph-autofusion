# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

cmake_minimum_required(VERSION 3.16.0)
project(graph-autofusion)

set(CMAKE_C_FLAGS_RELEASE "")
set(CMAKE_CXX_FLAGS_RELEASE "")
SET(CMAKE_C_COMPILER_WORKS TRUE)
SET(CMAKE_CXX_COMPILER_WORKS TRUE)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    message(STATUS "Build graph_autofusion with dependency cann pkgs installed")
    option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG "Build graph_autofusion with dependency cann pkgs installed" TRUE)
else()
    message(STATUS "Build graph_autofusion with cann source exist")
    option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG "Build graph_autofusion need cann source code" FALSE)
endif()

set(OPEN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../open_source")
if(NOT EXISTS ${OPEN_SOURCE_DIR})
    message(STATUS "Build graph_autofusion with dependency communtiy cann pkgs installed")
    option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG_COMMUNITY "Build graph_autofusion with dependency communtiy cann pkgs installed" TRUE)
else()
    message(STATUS "Build graph_autofusion with dependency communtiy cann pkgs not installed")
    option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG_COMMUNITY "Build graph_autofusion with dependency communtiy cann pkgs not installed" FALSE)
endif()

set(INSTALL_LIBRARY_DIR graph_autofusion/lib)

add_subdirectory(super_kernel)

include(cmake/package.cmake)
