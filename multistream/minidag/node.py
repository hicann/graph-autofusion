# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""DAG node and edge objects.

Edges are deliberately private.  Consumers construct them only through
``DAGGraph.add_edge``; the private accessors are used by the allocator
without widening the public API.
"""

from __future__ import annotations

from typing import List

from .types import CoreType, NodeCost


def _copy_cost(cost: NodeCost) -> NodeCost:
    """Return a fresh cost carrying every field of *cost*.

    Copying field by field keeps the copy explicit: a new field added to
    ``NodeCost`` has to be listed here as well, so it cannot be dropped
    silently on the way into or out of a node.
    """

    return NodeCost(
        execution_time=cost.execution_time,
        memory_usage=cost.memory_usage,
        bandwidth_usage=cost.bandwidth_usage,
        cube_block_num=cost.cube_block_num,
        vec_block_num=cost.vec_block_num,
        cube_core_num_limit=cost.cube_core_num_limit,
        vec_core_num_limit=cost.vec_core_num_limit,
    )


class _Edge:
    __slots__ = ("_src_node", "_dst_node", "_src_port", "_dst_port")

    def __init__(
        self, src_node: "DAGNode", src_port: int, dst_node: "DAGNode", dst_port: int
    ) -> None:
        self._src_node = src_node
        self._dst_node = dst_node
        self._src_port = int(src_port)
        self._dst_port = int(dst_port)


class DAGNode:
    """A graph node created by ``DAGGraph.add_node``.

    Every writable attribute has a matching getter, so a caller can print a
    node or check what it passed in.  Name, type and index are fixed at
    creation and only have getters.
    """

    __slots__ = (
        "_name",
        "_type",
        "_index",
        "_stream_label",
        "_node_cost",
        "_core_type",
        "_in_edges",
        "_out_edges",
    )

    def __init__(self, name: str, node_type: str, index: int = -1) -> None:
        self._name = name
        self._type = node_type
        self._index = int(index)
        self._stream_label = ""
        self._node_cost = NodeCost()
        self._core_type = CoreType.UNKNOWN
        self._in_edges: List[_Edge] = []
        self._out_edges: List[_Edge] = []

    def get_name(self) -> str:
        return self._name

    def get_type(self) -> str:
        return self._type

    def get_index(self) -> int:
        """Return the dense insertion index, which is also the topo id."""

        return self._index

    def set_stream_label(self, label: str) -> None:
        if not isinstance(label, str):
            raise TypeError("stream label must be a string")
        self._stream_label = label

    def get_stream_label(self) -> str:
        return self._stream_label

    def set_node_cost(self, cost: NodeCost) -> None:
        if not isinstance(cost, NodeCost):
            raise TypeError("node cost must be a NodeCost")
        # Store a fresh value even though NodeCost itself is immutable.  This
        # keeps the getter's copy semantics explicit and prevents a subclass
        # from smuggling mutable state into a node.
        self._node_cost = _copy_cost(cost)

    def get_node_cost(self) -> NodeCost:
        return _copy_cost(self._node_cost)

    def set_core_type(self, core_type: CoreType) -> None:
        if not isinstance(core_type, CoreType):
            raise TypeError("core type must be a CoreType member")
        self._core_type = core_type

    def get_core_type(self) -> CoreType:
        return self._core_type

    # Edge access stays private: ``_Edge`` is an internal type and the public
    # surface is limited to the methods documented in the design.
    def _add_input_edge(self, edge: _Edge) -> None:
        self._in_edges.append(edge)

    def _add_output_edge(self, edge: _Edge) -> None:
        self._out_edges.append(edge)

    def _get_input_edges(self) -> List[_Edge]:
        return list(self._in_edges)

    def _get_output_edges(self) -> List[_Edge]:
        return list(self._out_edges)

    def _get_input_nodes(self) -> List["DAGNode"]:
        return [edge._src_node for edge in self._in_edges]


__all__ = ["DAGNode"]
