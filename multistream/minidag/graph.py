# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Graph container for the pure Python MiniDAG implementation."""

from __future__ import annotations

from typing import Dict, List, Optional

from .node import DAGNode, _Edge


class DAGGraph:
    __slots__ = ("_name", "_nodes", "_name_to_index", "_edges")

    def __init__(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("graph name must be a string")
        self._name = name
        self._nodes: List[DAGNode] = []
        self._name_to_index: Dict[str, int] = {}
        self._edges: List[_Edge] = []

    def add_node(self, name: str, type: str) -> Optional[DAGNode]:  # noqa: A002 - public API uses ``type``
        if not isinstance(name, str) or not isinstance(type, str):
            raise TypeError("node name and type must be strings")
        if name in self._name_to_index:
            # The original C++ graph returns nullptr for a duplicate.  Keep
            # that behavior; callers can turn it into a ValueError at their
            # boundary if desired, while old users remain source-compatible.
            return None
        node = DAGNode(name, type, len(self._nodes))
        self._name_to_index[name] = node._index
        self._nodes.append(node)
        return node

    def find_node(self, name: str) -> Optional[DAGNode]:
        if not isinstance(name, str):
            raise TypeError("node name must be a string")
        index = self._name_to_index.get(name)
        return self._nodes[index] if index is not None else None

    def get_name(self) -> str:
        return self._name

    def get_nodes(self) -> List[DAGNode]:
        """Return the nodes in insertion order.

        The list is a shallow copy: changing it does not change the graph.
        """

        return list(self._nodes)

    def get_node_count(self) -> int:
        return len(self._nodes)

    # ``_Edge`` is not part of the public API, so the edge list stays private.
    def _get_edges(self) -> List[_Edge]:
        return list(self._edges)

    def _owns_node(self, node: object) -> bool:
        if not isinstance(node, DAGNode):
            return False
        index = node._index
        return 0 <= index < len(self._nodes) and self._nodes[index] is node

    def add_edge(
        self, src: DAGNode, src_port: int, dst: DAGNode, dst_port: int
    ) -> None:
        if not isinstance(src, DAGNode) or not isinstance(dst, DAGNode):
            raise TypeError("edge endpoints must be DAGNode objects")
        if not self._owns_node(src) or not self._owns_node(dst):
            raise ValueError("edge endpoint does not belong to this graph")
        for port in (src_port, dst_port):
            if isinstance(port, bool) or not isinstance(port, int):
                raise TypeError("edge ports must be integers")
            if port < -1:
                raise ValueError("edge ports must be -1 or non-negative")
        edge = _Edge(src, src_port, dst, dst_port)
        self._edges.append(edge)
        src._add_output_edge(edge)
        dst._add_input_edge(edge)


__all__ = ["DAGGraph"]
