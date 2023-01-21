from collections import defaultdict
from typing import Iterable

from exceptions import InvalidGraphException, InvalidNodeException


class Graph:
    def __init__(self,
                 num_vertices: int,
                 edges: Iterable[tuple[int, int]] = None
                 ) -> None:
        if num_vertices <= 0:
            raise InvalidGraphException(
                f'num_vertices must be greater than zero,'
                f' got \'{num_vertices}\' instead.'
            )

        self._num_vertices = num_vertices
        self._adj_list = defaultdict(lambda: set())

        if edges is not None:
            for node_from, node_to in edges:
                self.add_edge(node_from, node_to)

    def __repr__(self):
        if self.is_empty():
            return 'None'

        edge_list = (f'{node}: {neighbors}'
                     for node, neighbors in self._adj_list.items())

        return f"[Graph: [{', '.join(edge_list)}]]"

    def is_empty(self):
        return len(self._adj_list) == 0

    def add_edge(self, node_from: int, node_to: int) -> None:
        if node_from not in range(1, self._num_vertices + 1) or \
                node_to not in range(1, self._num_vertices + 1):
            raise InvalidNodeException(
                f'node_from and node_to must be valid nodes for the graph,'
                f' got \'{node_from}\' and \'{node_to}\' instead.'
            )

        self._adj_list[node_from].add(node_to)

    def degree(self, node: int) -> int:
        if node not in range(self._num_vertices):
            raise InvalidNodeException(
                f'node must be a valid node for the graph,'
                f' got {node} instead.'
            )

        return len(self._adj_list[node])

    def is_euclidean(self) -> bool:
        if self.is_empty():
            return False

        num_odd_nodes = 0

        for node in range(self._num_vertices):
            if self.degree(node) % 2 == 1:
                num_odd_nodes += 1

        return num_odd_nodes in (0, 2)


class UndirectedGraph(Graph):
    def __init__(self,
                 num_vertices: int,
                 edges: Iterable[tuple[int, int]] = None
                 ) -> None:
        super().__init__(num_vertices, edges)

    def add_edge(self, node_from: int, node_to: int) -> None:
        super().add_edge(node_from, node_to)
        super().add_edge(node_to, node_from)
