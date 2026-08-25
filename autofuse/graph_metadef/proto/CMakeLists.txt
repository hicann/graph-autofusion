# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(METADEF_PROTO_LIST
        "${METADEF_PROTO_DIR}/af_ir.proto"
        "${METADEF_PROTO_DIR}/onnx/af_onnx.proto"
        "${METADEF_PROTO_DIR}/ascendc_ir.proto"
        )

set(METADEF_ATTR_GROUP_PROTO_LIST
        "${METADEF_PROTO_DIR}/af_attr_group_base.proto"
        )

generate_cann_protobuf(metadef_protos_af METADEF_PROTO_SRCS METADEF_PROTO_HDRS ${METADEF_PROTO_LIST} ${METADEF_ATTR_GROUP_PROTO_LIST} TARGET)

set(METADEF_GRAPH_PROTO_SRCS
        "${CMAKE_BINARY_DIR}/proto/metadef_protos_af/proto/af_ir.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos_af/proto/ascendc_ir.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos_af/proto/onnx/af_onnx.pb.cc"
        )

set(METADEF_TENSORFLOW_PROTO_SRCS)

set(METADEF_AICPU_PROTO_SRCS)

set(METADEF_ATTR_GROUP_PROTO_SRCS
        "${CMAKE_BINARY_DIR}/proto/metadef_protos_af/proto/af_attr_group_base.pb.cc"
        )

add_library(metadef_graph_protos_obj_af OBJECT ${METADEF_GRAPH_PROTO_SRCS})
add_dependencies(metadef_graph_protos_obj_af metadef_protos_af)
target_compile_definitions(metadef_graph_protos_obj_af PRIVATE
        google=ascend_private
        )
target_link_libraries(metadef_graph_protos_obj_af PRIVATE ascend_protobuf af_intf_pub autofuse_shared_visibility)
target_compile_options(metadef_graph_protos_obj_af PRIVATE
        $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-O2 -fPIC -Wextra -Wfloat-equal>
        $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
        $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -fno-common -Wextra -Wfloat-equal>
        $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
        $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
        )


add_library(metadef_attr_group_protos_obj_af OBJECT ${METADEF_ATTR_GROUP_PROTO_SRCS})
add_dependencies(metadef_attr_group_protos_obj_af metadef_protos_af)
target_compile_definitions(metadef_attr_group_protos_obj_af PRIVATE
        google=ascend_private
        )
target_link_libraries(metadef_attr_group_protos_obj_af PRIVATE ascend_protobuf af_intf_pub autofuse_shared_visibility)
target_compile_options(metadef_attr_group_protos_obj_af PRIVATE
        $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-O2 -fPIC -Wextra -Wfloat-equal>
        $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
        $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -fno-common -Wextra -Wfloat-equal>
        $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
        $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
        )
