"""Create and train a Graph Neural Network for binary node classification.

This code was borrowed and adapted from an official Tensorflow GNN example:
https://github.com/tensorflow/gnn/blob/main/examples/notebooks/intro_mutag_example.ipynb
"""

import argparse
import os
from functools import partial
from typing import Iterable

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_gnn as tfgnn


def decode_example(record_bytes: tf.Tensor,
                   graph_tensor_spec: tfgnn.GraphTensorSpec
                   ) -> tuple[tfgnn.GraphTensor, tf.Tensor]:
    """Decode a single training example from the TFRecord file."""
    graph = tfgnn.parse_single_example(
        graph_tensor_spec, record_bytes, validate=True)

    # Extract label from context and remove it from input graph
    context_features = graph.context.get_features_dict()
    label = context_features.pop('is_eulerian')

    new_graph = graph.replace_features(context=context_features)

    return new_graph, label


def build_model(graph_tensor_spec: tfgnn.GraphTensorSpec,
                node_dim: int = 16,
                message_dim: int = 32,
                next_state_dim: int = 64,
                num_message_passing: int = 3,
                l2_regularization: float = 5e-4,
                dropout_rate: float = 0.1,
                ) -> tf.keras.Model:
    """Build a graph neural network for binary graph classification."""
    input_graph = tf.keras.layers.Input(type_spec=graph_tensor_spec)
    graph = input_graph.merge_batch_to_components()

    def set_initial_node_state(node_set, *, node_set_name):
        return tf.keras.layers.Dense(node_dim)(node_set['degree'])

    def set_initial_edge_state(edge_set, *, edge_set_name):
        return None

    graph = tfgnn.keras.layers.MapFeatures(
        node_sets_fn=set_initial_node_state,
        edge_sets_fn=set_initial_edge_state)(
        graph)

    def dense(units, activation='relu'):
        """A Dense layer with regularization (L2 and Dropout)."""
        regularizer = tf.keras.regularizers.l2(l2_regularization)
        return tf.keras.Sequential([
            tf.keras.layers.Dense(
                units,
                activation=activation,
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer),
            tf.keras.layers.Dropout(dropout_rate)
        ])

    for i in range(num_message_passing):
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets={
                'node': tfgnn.keras.layers.NodeSetUpdate(
                    {'edge': tfgnn.keras.layers.SimpleConv(
                        message_fn=dense(message_dim),
                        reduce_type='sum',
                        receiver_tag=tfgnn.TARGET)},
                    tfgnn.keras.layers.NextStateFromConcat(
                        dense(next_state_dim)))}
        )(graph)

    readout_features = tfgnn.keras.layers.Pool(
        tfgnn.CONTEXT, 'mean', node_set_name='node')(graph)

    logits = tf.keras.layers.Dense(1)(readout_features)

    return tf.keras.Model(inputs=[input_graph], outputs=[logits])


def plot_losses(train_loss: Iterable[float],
                val_loss: Iterable[float],
                output_path: str
                ) -> None:
    """Plot the model's loss history after training."""
    plt.clf()
    plt.grid()

    plt.title('Training and Validation Loss', fontsize=16)

    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)

    plt.plot(train_loss, label='Training loss', color='#ff7f0e')
    plt.plot(val_loss, label='Validation loss', color='#1f77b4')

    plt.legend()
    plt.savefig(output_path)


def plot_accuracy(train_acc: Iterable[float],
                  val_acc: Iterable[float],
                  output_path: str
                  ):
    """Plot the model's accuracy history after training."""
    plt.clf()
    plt.grid()

    plt.title('Training and Validation Accuracy', fontsize=16)

    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)

    plt.plot(train_acc, label='Training accuracy', color='#ff7f0e')
    plt.plot(val_acc, label='Validation accuracy', color='#1f77b4')

    plt.legend()
    plt.savefig(output_path)


def main(graph_schema_path: str,
         dataset_dir: str,
         output_dir: str,
         batch_size: int,
         num_epochs: int
         ) -> None:
    os.makedirs(output_dir, exist_ok=False)

    graph_schema = tfgnn.read_schema(graph_schema_path)
    graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

    train_data_path = os.path.join(dataset_dir, 'train.tfrecord')
    val_data_path = os.path.join(dataset_dir, 'validation.tfrecord')

    train_ds = tf.data.TFRecordDataset(train_data_path)
    val_ds = tf.data.TFRecordDataset(val_data_path)

    decode_fn = partial(decode_example, graph_tensor_spec=graph_tensor_spec)
    train_ds = train_ds.map(decode_fn).batch(batch_size=batch_size)
    val_ds = val_ds.map(decode_fn).batch(batch_size=batch_size)

    model_input_graph_spec, _ = train_ds.element_spec
    model = build_model(model_input_graph_spec)

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.BinaryAccuracy(threshold=0.5)]

    model.compile(tf.keras.optimizers.Adam(),
                  loss=loss,
                  metrics=metrics)

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        batch_size=batch_size,
                        epochs=num_epochs)

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    loss_plot_path = os.path.join(output_dir, 'plot_loss.jpg')
    plot_losses(train_loss, val_loss, loss_plot_path)

    train_acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']

    acc_plot_path = os.path.join(output_dir, 'plot_accuracy.jpg')
    plot_accuracy(train_acc, val_acc, acc_plot_path)

    model.save(output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--output_dir', type=str)

    parser.add_argument('--graph_schema_path', type=str,
                        default='data/graph_schema.pbtxt')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)

    args = parser.parse_args()
    main(args.graph_schema_path,
         args.dataset_dir,
         args.output_dir,
         args.batch_size,
         args.num_epochs)
