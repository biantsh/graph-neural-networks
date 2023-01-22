"""Generate random graphs and write them to disk as serialized TFRecords.

The graphs are split into train, validation and test sets, and additionally
into Eulerian and non-Eulerian graphs.

Example usage:
    python3 dataset_generator.py  \
      --output_dir generated_dataset/  \
      --num_graphs 10000  \
      --max_vertices 50
"""

import argparse
import os
import random
from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn

from graphs import UndirectedGraph
from utils.progress_bar import ProgressBar


def random_graph(max_vertices: int,
                 keep_probability: float = 0.1
                 ) -> UndirectedGraph:
    num_vertices = random.randint(1, max_vertices)
    num_edges = int(random.gauss(mu=3 * num_vertices, sigma=num_vertices))
    num_edges = max(num_edges, 0)

    random_node = partial(random.randint, 1, num_vertices)
    edges = [(random_node(), random_node()) for _ in range(num_edges)]

    graph = UndirectedGraph(num_vertices, edges)

    # Only include fully connected graphs in training data.
    if not graph.is_connected():
        return random_graph(max_vertices, keep_probability)

    # Balance dataset by removing some non-Eulerian graphs.
    if not graph.is_eulerian() and random.uniform(0, 1) > keep_probability:
        return random_graph(max_vertices, keep_probability)

    return graph


def generate_graphs(num_graphs: int,
                    max_vertices: int,
                    keep_probability: float = 0.1
                    ) -> list[UndirectedGraph]:
    """Generate a `num_graphs` number of random graphs.

    Because random graphs are more likely to not be Eulerian, a
    `keep_probability` argument is provided which represents the probability
    that a non-Eulerian graph is kept.
    """
    graphs = []

    progress_bar = ProgressBar('Generating graph: %current%/%target% '
                               '(%progress%)', target=num_graphs)

    for idx in range(1, num_graphs + 1):
        progress_bar.update(idx)

        graph = random_graph(max_vertices, keep_probability)
        graphs.append(graph)

    progress_bar.close()

    return graphs


def graph_to_tensor(graph: UndirectedGraph) -> tfgnn.GraphTensor:
    """Convert an undirected graph into a tfgnn.GraphTensor."""
    degrees = [graph.degree(node) for node in graph.nodes]
    degrees = tf.constant(degrees, dtype=tf.int64)
    degrees = tf.reshape(degrees, (len(degrees), 1))

    source_nodes, target_nodes = [], []
    if len(graph.edges) > 0:
        source_nodes, target_nodes = zip(*graph.edges)

    source_nodes = tf.constant(source_nodes, dtype=tf.int32)
    target_nodes = tf.constant(target_nodes, dtype=tf.int32)

    num_nodes = tf.shape(graph.nodes)
    num_edges = tf.shape(source_nodes)

    is_eulerian = tf.constant([graph.is_eulerian()], dtype=tf.int64)

    graph_tensor = tfgnn.GraphTensor.from_pieces(
        node_sets={
            'node': tfgnn.NodeSet.from_fields(
                sizes=num_nodes,
                features={
                    'degree': degrees
                }
            )
        },
        edge_sets={
            'edge': tfgnn.EdgeSet.from_fields(
                sizes=num_edges,
                adjacency=tfgnn.Adjacency.from_indices(
                    source=('node', source_nodes),
                    target=('node', target_nodes)
                )
            )
        },
        context=tfgnn.Context.from_fields(
            features={
                'is_eulerian': is_eulerian
            }
        )
    )

    return graph_tensor


def holdout_split(data: np.ndarray) -> tuple[list, list, list]:
    """Split a list of examples into train, validation and test."""
    np.random.shuffle(data)

    num_examples = len(data)
    train_split = int(.6 * num_examples)  # 60% training split
    val_split = int(.8 * num_examples)  # 20% validation split

    train, validation, test = np.split(data, [train_split, val_split])

    return train, validation, test


def main(output_dir: str, num_graphs: int, max_vertices: int) -> None:
    os.makedirs(output_dir, exist_ok=False)
    split_names = ['train', 'validation', 'test']

    graphs = generate_graphs(num_graphs,
                             max_vertices,
                             keep_probability=0.1)

    split_data = holdout_split(np.array(graphs))

    for split, graphs in zip(split_names, split_data):
        output_path = os.path.join(output_dir, f'{split}.tfrecord')
        num_graphs = len(graphs)

        progress_bar = ProgressBar(f'Exporting graph: %current%/%target% '
                                   f'(%progress%) on split {split.title()}',
                                   target=num_graphs)

        with tf.io.TFRecordWriter(output_path) as writer:
            for idx, graph in enumerate(graphs, 1):
                progress_bar.update(idx)

                graph_tensor = graph_to_tensor(graph)

                example = tfgnn.write_example(graph_tensor)
                writer.write(example.SerializeToString())

        progress_bar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--num_graphs', type=int, default=10000)
    parser.add_argument('--max_vertices', type=int, default=50)

    args = parser.parse_args()
    main(args.output_dir, args.num_graphs, args.max_vertices)
