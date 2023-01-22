"""Use a trained model to get predictions on randomly generated graphs.

Example usage:
    python3 model_inference.py  \
      --model_path model/  \
      --num_examples 5
"""

import argparse

import matplotlib.pyplot as plt
import networkx as nx
import tensorflow as tf
from tensorflow import keras

from dataset_generator import graph_to_tensor, random_graph
from graphs import UndirectedGraph


def predict(model: tf.keras.Model, graph: UndirectedGraph) -> bool:
    """Get prediction for a single undirected graph."""
    graph_tensor = graph_to_tensor(graph)
    dataset = tf.data.Dataset.from_tensors(
        graph_tensor
    )
    dataset = dataset.batch(batch_size=1)

    prediction = model.predict(dataset)[0][0]

    return prediction >= 0


def main(model_path: str, num_examples: int) -> None:
    model = keras.models.load_model(model_path)

    for _ in range(num_examples):
        graph = random_graph(max_vertices=50,
                             keep_probability=0.1)
        prediction = predict(model, graph)

        graph_plot = nx.Graph()
        for edge in graph.edges:
            graph_plot.add_edge(*edge)

        plt.title(f'Actual: {graph.is_eulerian()}, '
                  f'Prediction: {prediction}')
        nx.draw(graph_plot)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--num_examples', type=int, default=5)

    args = parser.parse_args()
    main(args.model_path, args.num_examples)
