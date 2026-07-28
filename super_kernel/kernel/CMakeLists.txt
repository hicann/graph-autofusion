# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set(SK_KERNEL_ARCHS dav-2201 dav-3510)
set(SK_ENTRY_STUB_INPUTS)
set(SK_ENTRY_OBJECTS)
set(SK_ENTRY_TARGETS)

foreach(SK_KERNEL_ARCH IN LISTS SK_KERNEL_ARCHS)
   string(REPLACE "-" "_" SK_KERNEL_ARCH_SUFFIX ${SK_KERNEL_ARCH})
   set(SK_ENTRY_TARGET sk_entry_${SK_KERNEL_ARCH_SUFFIX})

   ascendc_fatbin_library(${SK_ENTRY_TARGET} sk_entry.asc)
   target_compile_options(${SK_ENTRY_TARGET} PRIVATE
      $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${SK_KERNEL_ARCH}>
      $<$<COMPILE_LANGUAGE:ASC>:-D__ASCENDC_SUPERKERNEL_EARLY_START_V2>
   )

   target_include_directories(${SK_ENTRY_TARGET} PRIVATE
      $<$<COMPILE_LANGUAGE:ASC>:${CMAKE_CURRENT_SOURCE_DIR}/../src/aot/>
   )

   target_link_libraries(${SK_ENTRY_TARGET} PRIVATE
      $<BUILD_INTERFACE:runtime_headers>
      $<BUILD_INTERFACE:asc_kernel_headers>
      $<BUILD_INTERFACE:kernel_tiling_headers>
   )
   set_property(TARGET ${SK_ENTRY_TARGET} PROPERTY POSITION_INDEPENDENT_CODE ON)
   set(SK_ENTRY_OBJECT ${CMAKE_CURRENT_BINARY_DIR}/${SK_ENTRY_TARGET}.o)
   list(APPEND SK_ENTRY_STUB_INPUTS ${SK_KERNEL_ARCH}=${SK_ENTRY_OBJECT})
   list(APPEND SK_ENTRY_OBJECTS ${SK_ENTRY_OBJECT})
   list(APPEND SK_ENTRY_TARGETS ${SK_ENTRY_TARGET})
endforeach()

add_custom_command(
   OUTPUT ${GENERATED_SK_ENTRY_STUB_CPP}
   COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/gen_sk_entry_stub.py
   ${GENERATED_SK_ENTRY_STUB_CPP}
   ${SK_ENTRY_STUB_INPUTS}
   DEPENDS
      ${SK_ENTRY_TARGETS}
      ${SK_ENTRY_OBJECTS}
      ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/gen_sk_entry_stub.py
   COMMENT "Generating sk_entry stub cpp from sk_entry objects"
   VERBATIM
)

add_custom_target(generate_entry_stub
   DEPENDS ${GENERATED_SK_ENTRY_STUB_CPP}
)

foreach(SK_KERNEL_ARCH IN LISTS SK_KERNEL_ARCHS)
   string(REPLACE "-" "_" SK_KERNEL_ARCH_SUFFIX ${SK_KERNEL_ARCH})
   set(SK_SCOPE_TARGET sk_scope_${SK_KERNEL_ARCH_SUFFIX})

   add_library(${SK_SCOPE_TARGET} STATIC
      sk_scope_kernel.asc
   )

   target_compile_options(${SK_SCOPE_TARGET} PRIVATE
      $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${SK_KERNEL_ARCH}>
   )

   target_compile_definitions(${SK_SCOPE_TARGET} PRIVATE
      $<$<COMPILE_LANGUAGE:ASC>:SK_SCOPE_SYMBOL_SUFFIX=_${SK_KERNEL_ARCH_SUFFIX}>
   )

   target_include_directories(${SK_SCOPE_TARGET} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../src/aot/
   )
   target_link_libraries(${SK_SCOPE_TARGET} PRIVATE
      $<BUILD_INTERFACE:asc_kernel_headers>
      $<BUILD_INTERFACE:kernel_tiling_headers>
   )
   set_property(TARGET ${SK_SCOPE_TARGET} PROPERTY POSITION_INDEPENDENT_CODE ON)
endforeach()
