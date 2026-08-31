# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

######### for base #############
set(GRAPH_BASE_SOURCE_LIST
    "type/types.cc"
    "normal_graph/anchor.cc"
    "attr/ge_attr_value.cc"
    "utils/args_format_desc.cc"
    "normal_graph/operator.cc"
    "normal_graph/operator_impl.cc"
    "attr/attr_store.cc"
    "attr/attr_group_serialize.cc"
    "attr/attr_group_base.cc"
    "attr/attr_group_serializer_registry.cc"
    "buffer/buffer.cc"
    "normal_graph/compute_graph.cc"
    "normal_graph/model.cc"
    "serialization/model_serialize.cc"
    "normal_graph/node.cc"
    "normal_graph/op_desc.cc"
    "ir/ir_meta.cc"
    "ir/ir_data_type_symbol_store.cc"
    "type/sym_dtype.cc"
    "attr/ge_attr_define.cc"
    "normal_graph/ge_tensor.cc"
    "buffer/graph_buffer.cc"
    "hcom/hcom_topo_info.cc"
    "common/large_bm.cc"
    "common/hyper_status.cc"
    "detail/attributes_holder.cc"
    "utils/anchor_utils.cc"
    "utils/enum_attr_utils.cc"
    "utils/graph_utils.cc"
    "utils/dumper/ge_graph_dumper.cc"
    "utils/trace/trace_manager.cc"
    "utils/ge_ir_utils.cc"
    "utils/node_utils.cc"
    "utils/type_utils.cc"
    "utils/tensor_utils.cc"
    "utils/constant_utils.cc"
    "utils/connection_matrix.cc"
    "utils/cycle_detector.cc"
    "utils/op_type_utils.cc"
    "utils/fast_node_utils.cc"
    "utils/execute_graph_utils.cc"
    "utils/execute_graph_adapter.cc"
    "utils/args_format_desc_utils.cc"
    "utils/inference_rule.cc"
    "option/ge_context.cc"
    "option/ge_local_context.cc"
    "option/optimization_option.cc"
    "option/optimization_option_info.cc"
    "serialization/attr_serializer.cc"
    "serialization/string_serializer.cc"
    "serialization/data_type_serializer.cc"
    "serialization/named_attrs_serializer.cc"
    "serialization/bool_serializer.cc"
    "serialization/buffer_serializer.cc"
    "serialization/float_serializer.cc"
    "serialization/int_serializer.cc"
    "serialization/tensor_serializer.cc"
    "serialization/tensor_desc_serializer.cc"
    "serialization/graph_serializer.cc"
    "serialization/list_value_serializer.cc"
    "serialization/list_list_int_serializer.cc"
    "serialization/list_list_float_serializer.cc"
    "serialization/attr_serializer_registry.cc"
    "serialization/utils/serialization_util.cc"
    "cache_policy/cache_policy.cc"
    "cache_policy/compile_cache_desc.cc"
    "cache_policy/policy_register.cc"
    "cache_policy/cache_state.cc"
    "cache_policy/policy_management/match_policy/match_policy_exact_only.cc"
    "cache_policy/policy_management/match_policy/match_policy_for_exactly_the_same.cc"
    "cache_policy/policy_management/aging_policy/aging_policy_lru.cc"
    "cache_policy/policy_management/aging_policy/aging_policy_lru_k.cc"
    "utils/profiler.cc"
    "normal_graph/tensor.cc"
    "utils/readable_dump.cc"
    "utils/tensor_value_utils.cc"
    "${GE_METADEF_DIR}/base/utils/type_utils_impl.cc"
    "${GE_METADEF_DIR}/base/utils/type_utils_inner.cc"
    "${GE_METADEF_DIR}/base/utils/file_utils.cc"
    "${GE_METADEF_DIR}/base/utils/aligned_ptr.cc"
    "${GE_METADEF_DIR}/base/common/plugin/plugin_manager.cc"
    "${GE_METADEF_DIR}/base/type/ascend_string_impl.cc"
)

SET(GRAPH_SOURCE_LIST
    "type/ascend_string.cc"
    "attr/attr_value.cc"
    "type/axis_type_info.cc"
    "normal_graph/custom_op_factory.cc"
    "normal_graph/custom_op_factory_impl.cc"
    "normal_graph/operator_factory.cc"
    "normal_graph/operator_factory_impl.cc"
    "normal_graph/graph.cc"
    "normal_graph/gnode.cc"
    "args_format/args_format_serializer.cc"
    "args_format/arg_desc_info.cc"
    "args_format/arg_desc_info_impl.cc"
    "refiner/format_refiner.cc"
    "context/inference_context.cc"
    "refiner/ref_relation.cc"
    "context/resource_context_mgr.cc"
    "context/runtime_inference_context.cc"
    "refiner/shape_refiner.cc"
    "ir/ir_definitions_recover.cc"
    "opsproto/opsproto_manager.cc"
    "utils/op_desc_utils.cc"
    "utils/tuning_utils.cc"
    "utils/ffts_graph_utils.cc"
    "utils/transformer_utils.cc"
    "utils/graph_utils_ex.cc"
    "utils/node_utils_ex.cc"
    "utils/op_desc_utils_ex.cc"
    "utils/graph_thread_pool.cc"
    "utils/multi_thread_graph_builder.cc"
    "utils/type_utils_ex.cc"
    "utils/screen_printer.cc"
    "${GE_METADEF_DIR}/third_party/transformer/src/axis_util.cc"
    "${GE_METADEF_DIR}/third_party/transformer/src/transfer_shape_according_to_format.cc"
    "${GE_METADEF_DIR}/third_party/transformer/src/expand_dimension.cc"
    "${GE_METADEF_DIR}/third_party/transformer/src/transfer_range_according_to_format.cc"
    "${GE_METADEF_DIR}/third_party/transformer/src/transfer_shape_utils.cc"
)

SET(FAST_GRAPH_SOURCE_LIST
    "fast_graph/fast_node.cc"
    "fast_graph/execute_graph.cc"
)

######### libgraph_base.so #############
add_library(graph_base_af SHARED
    ${GRAPH_BASE_SOURCE_LIST}
    ${FAST_GRAPH_SOURCE_LIST}
    $<TARGET_OBJECTS:metadef_graph_protos_obj_af>
)

target_compile_options(graph_base_af PRIVATE
    -include${CMAKE_CURRENT_LIST_DIR}/../../inc/graph_metadef/graph/tensor_af.h
    $<$<BOOL:${ENABLE_ASAN}>:-Werror=maybe-uninitialized>
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -fno-common -Wextra -Wfloat-equal>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>)

target_compile_definitions(graph_base_af PRIVATE
    FMK_SUPPORT_DUMP
    google=ascend_private
    INC_EXTERNAL_GRAPH_TENSOR_H_
    INC_EXTERNAL_GRAPH_ASCEND_STRING_H_
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
)

target_include_directories(graph_base_af PRIVATE
    ${AIR_CODE_DIR}/inc/graph_metadef
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/metadef_protos_af
    ${TOP_DIR}/runtime/include/external
    ${TOP_DIR}/runtime/include/external/acl
)

target_link_options(graph_base_af PRIVATE
    -Wl,-Bsymbolic
    -Wl,--no-undefined
)

target_link_libraries(graph_base_af
    PRIVATE
        af_intf_pub
        mmpa
        -Wl,--no-as-needed
        c_sec
        unified_dlog
        json
        metadef
        -Wl,--as-needed
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
        -pthread
    PUBLIC
        metadef_headers_af
)

target_link_libraries(graph_base_af PRIVATE ascend_protobuf error_manager metadef_headers)
target_compile_options(graph_base_af PRIVATE ${OPTIMIZE_OPTION} -DNO_METADEF_ABI_COMPATIABLE)
set_target_properties(graph_base_af PROPERTIES
    OUTPUT_NAME graph_base_af
)
autofuse_apply_shared_visibility(graph_base_af)

######### libgraph_base.a #############
if (NOT ENABLE_OPEN_SRC)
    add_library(graph_base_static_af STATIC
        ${GRAPH_BASE_SOURCE_LIST}
        ${FAST_GRAPH_SOURCE_LIST}
        $<TARGET_OBJECTS:metadef_graph_protos_obj_af>
    )

    target_compile_options(graph_base_static_af PRIVATE
        -include${CMAKE_CURRENT_LIST_DIR}/../../inc/graph_metadef/graph/tensor_af.h
        $<$<BOOL:${ENABLE_ASAN}>:-Werror=maybe-uninitialized>
        $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
        $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -fno-common -Wextra -Wfloat-equal>
        $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
        $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
        -Os
        -fvisibility=hidden
        -fvisibility-inlines-hidden
        -ffunction-sections
        -fdata-sections
        ${OPTIMIZE_OPTION}
        -DNO_METADEF_ABI_COMPATIABLE
    )

    target_compile_definitions(graph_base_static_af PRIVATE
        FMK_SUPPORT_DUMP
        google=ascend_private
        INC_EXTERNAL_GRAPH_TENSOR_H_
        INC_EXTERNAL_GRAPH_ASCEND_STRING_H_
        $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
        $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
    )

    target_include_directories(graph_base_static_af PRIVATE
        ${AIR_CODE_DIR}/inc/graph_metadef
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_BINARY_DIR}
        ${CMAKE_BINARY_DIR}/proto/metadef_protos_af
        ${TOP_DIR}/runtime/include/external
        ${TOP_DIR}/runtime/include/external/acl
    )

    target_link_options(graph_base_static_af PRIVATE
        -Wl,-Bsymbolic
        -Wl,--no-undefined
    )

    target_link_libraries(graph_base_static_af
        PRIVATE
            af_intf_pub
            mmpa
            -Wl,--no-as-needed
            c_sec
            unified_dlog
            json
            metadef
            -Wl,--as-needed
            $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
            -ldl
            -pthread
            ascend_protobuf
            error_manager
            metadef_headers
        PUBLIC
            metadef_headers_af
    )

    set_target_properties(graph_base_static_af PROPERTIES
        OUTPUT_NAME graph_base
    )
endif ()

######### libgraph.so #############
add_library(graph_af SHARED
    ${GRAPH_SOURCE_LIST}
    $<TARGET_OBJECTS:metadef_attr_group_protos_obj_af>
)
target_compile_options(graph_af PRIVATE -DNO_METADEF_ABI_COMPATIABLE)

target_compile_options(graph_af PRIVATE
    -include${CMAKE_CURRENT_LIST_DIR}/../../inc/graph_metadef/graph/tensor_af.h
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -fno-common -Wextra -Wfloat-equal>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>)

target_compile_definitions(graph_af PRIVATE
    FMK_SUPPORT_DUMP
    google=ascend_private
    INC_EXTERNAL_GRAPH_TENSOR_H_
    INC_EXTERNAL_GRAPH_ASCEND_STRING_H_
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
)

target_include_directories(graph_af PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/metadef_protos_af
    ${AIR_CODE_DIR}/inc/graph_metadef
    ${TOP_DIR}/runtime/include/external
    ${TOP_DIR}/runtime/include/external/acl
)

target_link_options(graph_af PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(graph_af
    PRIVATE
        af_intf_pub
        mmpa
        -Wl,--no-as-needed
        c_sec
        unified_dlog
        json
        platform
        metadef
        -Wl,--as-needed
        ascend_protobuf
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
        -pthread
    PUBLIC
        metadef_headers_af
)

target_link_libraries(graph_af PRIVATE graph_base_af error_manager metadef_headers)
target_compile_options(graph_af PRIVATE ${OPTIMIZE_OPTION})
set_target_properties(graph_af PROPERTIES
    OUTPUT_NAME graph_af
)
autofuse_apply_shared_visibility(graph_af)


add_subdirectory(ascendc_ir)
add_subdirectory(expression)
