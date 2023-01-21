import random
from functools import partial

from graphs import UndirectedGraph

NUM_CANDIDATES = 10000
MAX_VERTICES = 50


def generate() -> list[UndirectedGraph]:
    graphs = []

    for _ in range(NUM_CANDIDATES):
        num_vertices = random.randint(1, MAX_VERTICES)
        num_edges = int(random.gauss(mu=3*num_vertices, sigma=num_vertices))
        num_edges = max(num_edges, 0)

        random_node = partial(random.randint, 1, num_vertices)
        edges = [(random_node(), random_node()) for _ in range(num_edges)]

        graph = UndirectedGraph(num_vertices, edges)
        if not graph.is_euclidean() and random.randint(1, 10) <= 9:
            continue

        graphs.append(graph)

    return graphs
