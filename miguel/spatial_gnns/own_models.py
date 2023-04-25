import tensorflow as tf
from tensorflow import keras
from keras import layers
import deeptrack
from deeptrack.models.utils import KerasModel, GELU
from deeptrack.models.layers import as_block, DenseBlock
from deeptrack.models.embeddings import LearnableDistanceEmbedding

class OneLayerFGNN(KerasModel):
    """
    Message-passing Graph Neural Network.

    Parameters:
    -----------
    dense_layer_dimensions: list of ints
        List of the number of units in each dense layer of the encoder and decoder. The
        number of layers is inferred from the length of this list.
    base_layer_dimensions: list of ints
        List of the latent dimensions of the graph blocks. The number of layers is
        inferred from the length of this list.
    number_of_node_outputs: int
        Number of output node features.
    number_of_edge_outputs: int
        Number of output edge features.
    number_of_global_outputs: int
        Number of output global features.
    node_output_activation: str or activation function or layer
        Activation function for the output node layer. See keras docs for accepted strings.
    edge_output_activation: str or activation function or layer
        Activation function for the output edge layer. See keras docs for accepted strings.
    cls_layer_dimension: int
        Number of units in the decoder layer for global features.
    global_output_activation: str or activation function or layer
        Activation function for the output global layer. See keras docs for accepted strings.
    dense_block: str, keras.layers.Layer, or callable
        The dense block to use for the encoder and decoder.
    graph_block: str, keras.layers.Layer, or callable
        The graph block to use for the graph computation. See gnns.layers for available
        graph blocks.
    readout_block: str, keras.layers.Layer, or callable
        The readout block used to compute global features.
    output_type: str
        Type of output. Either "nodes", "edges", "global" or
        "graph". If 'key' is not a supported output type, then
        the model output will be the concatenation of the node,
        edge, and global predictions.
    kwargs: dict
        Keyword arguments for the dense block.
    Returns:
    --------
    tf.keras.Model
        Keras model for the graph neural network.
    """

    def __init__(
        self,
        dense_layer_dimensions=(32, 64, 96),
        base_layer_dimensions=(96, 96),
        number_of_node_features=3,
        number_of_edge_features=1,
        number_of_node_outputs=1,
        number_of_edge_outputs=1,
        number_of_global_outputs=1,
        node_output_activation=None,
        edge_output_activation=None,
        global_layer_dimension=64,
        global_output_activation=None,
        dense_block=DenseBlock(
            activation=GELU,
            normalization="LayerNormalization",
        ),
        graph_block="FGNN",
        readout_block=layers.Lambda(
            lambda x: tf.math.reduce_sum(x, axis=1), name="global_readout"
        ),
        output_type="graph",
        **kwargs
    ):
        dense_block = as_block(dense_block)
        graph_block = as_block(graph_block)

        node_features, edge_features, edges, edge_dropout = (
            tf.keras.Input(shape=(None, number_of_node_features)),
            tf.keras.Input(shape=(None, number_of_edge_features)),
            tf.keras.Input(shape=(None, 2), dtype=tf.int32),
            tf.keras.Input(shape=(None, 2)),
        )

        node_layer = node_features
        edge_layer = edge_features

        # Encoder for node and edge features
        for dense_layer_number, dense_layer_dimension in zip(
            range(len(dense_layer_dimensions)), dense_layer_dimensions
        ):
            node_layer = dense_block(
                dense_layer_dimension,
                name="node_ide" + str(dense_layer_number + 1),
                **kwargs
            )(node_layer)

            edge_layer = dense_block(
                dense_layer_dimension,
                name="edge_ide" + str(dense_layer_number + 1),
                **kwargs
            )(edge_layer)

        # Bottleneck path, graph blocks
        layer = (
            node_layer,
            edge_layer,
            edges,
            tf.ones_like(edge_features[..., 0:1]),
            edge_dropout,
        )
        gb = graph_block(
            base_layer_dimensions[0],
            name="1",
        )

        layer = gb(layer)

        # Split nodes and edges
        node_layer, edge_layer, *_ = layer

        # Compute global features
        global_layer = readout_block(node_layer)
        global_layer = dense_block(
            global_layer_dimension, name="cls_mlp", **kwargs
        )(global_layer)

        # Output layers
        node_output = layers.Dense(
            number_of_node_outputs,
            activation=node_output_activation,
            name="node_prediction",
        )(node_layer)

        edge_output = layers.Dense(
            number_of_edge_outputs,
            activation=edge_output_activation,
            name="edge_prediction",
        )(edge_layer)

        global_output = layers.Dense(
            number_of_global_outputs,
            activation=global_output_activation,
            name="global_prediction",
        )(global_layer)

        output_dict = {
            "nodes": node_output,
            "edges": edge_output,
            "global": global_output,
            "graph": [node_output, edge_output, global_output],
        }
        try:
            outputs = output_dict[output_type]
        except KeyError:
            outputs = output_dict["graph"]

        model = tf.keras.models.Model(
            [node_features, edge_features, edges, edge_dropout],
            outputs,
        )

        super().__init__(model, **kwargs)



class OneMessagePassingLayerMPNGNN(KerasModel):
    """
    Message-passing Graph Neural Network.

    Parameters:
    -----------
    dense_layer_dimensions: list of ints
        List of the number of units in each dense layer of the encoder and decoder. The
        number of layers is inferred from the length of this list.
    base_layer_dimensions: list of ints
        List of the latent dimensions of the graph blocks. The number of layers is
        inferred from the length of this list.
    number_of_node_outputs: int
        Number of output node features.
    number_of_edge_outputs: int
        Number of output edge features.
    number_of_global_outputs: int
        Number of output global features.
    node_output_activation: str or activation function or layer
        Activation function for the output node layer. See keras docs for accepted strings.
    edge_output_activation: str or activation function or layer
        Activation function for the output edge layer. See keras docs for accepted strings.
    cls_layer_dimension: int
        Number of units in the decoder layer for global features.
    global_output_activation: str or activation function or layer
        Activation function for the output global layer. See keras docs for accepted strings.
    dense_block: str, keras.layers.Layer, or callable
        The dense block to use for the encoder and decoder.
    graph_block: str, keras.layers.Layer, or callable
        The graph block to use for the graph computation. See gnns.layers for available
        graph blocks.
    readout_block: str, keras.layers.Layer, or callable
        The readout block used to compute global features.
    output_type: str
        Type of output. Either "nodes", "edges", "global" or
        "graph". If 'key' is not a supported output type, then
        the model output will be the concatenation of the node,
        edge, and global predictions.
    kwargs: dict
        Keyword arguments for the dense block.
    Returns:
    --------
    tf.keras.Model
        Keras model for the graph neural network.
    """

    def __init__(
        self,
        dense_layer_dimensions=(32, 64, 96),
        base_layer_dimensions=(96, 96),
        number_of_node_features=3,
        number_of_edge_features=1,
        number_of_node_outputs=1,
        number_of_edge_outputs=1,
        number_of_global_outputs=1,
        node_output_activation=None,
        edge_output_activation=None,
        global_layer_dimension=64,
        global_output_activation=None,
        dense_block=DenseBlock(
            activation=GELU,
            normalization="LayerNormalization",
        ),
        graph_block="MPN",
        readout_block=layers.Lambda(
            lambda x: tf.math.reduce_sum(x, axis=1), name="global_readout"
        ),
        output_type="graph",
        **kwargs
    ):
        dense_block = as_block(dense_block)
        graph_block = as_block(graph_block)

        node_features, edge_features, edges, edge_dropout = (
            tf.keras.Input(shape=(None, number_of_node_features)),
            tf.keras.Input(shape=(None, number_of_edge_features)),
            tf.keras.Input(shape=(None, 2), dtype=tf.int32),
            tf.keras.Input(shape=(None, 2)),
        )

        node_layer = node_features
        edge_layer = edge_features

        # Encoder for node and edge features
        for dense_layer_number, dense_layer_dimension in zip(
            range(len(dense_layer_dimensions)), dense_layer_dimensions
        ):
            node_layer = dense_block(
                dense_layer_dimension,
                name="node_ide" + str(dense_layer_number + 1),
                **kwargs
            )(node_layer)

            edge_layer = dense_block(
                dense_layer_dimension,
                name="edge_ide" + str(dense_layer_number + 1),
                **kwargs
            )(edge_layer)

        # Bottleneck path, graph blocks
        layer = (
            node_layer,
            edge_layer,
            edges,
            tf.ones_like(edge_features[..., 0:1]),
            edge_dropout,
        )
        gb = graph_block(
            base_layer_dimensions[0],
            name="One and only graph block",
        )

        layer = gb(layer)

        # Split nodes and edges
        node_layer, edge_layer, *_ = layer

        # Compute global features
        global_layer = readout_block(node_layer)
        global_layer = dense_block(
            global_layer_dimension, name="cls_mlp", **kwargs
        )(global_layer)

        # Output layers
        node_output = layers.Dense(
            number_of_node_outputs,
            activation=node_output_activation,
            name="node_prediction",
        )(node_layer)

        edge_output = layers.Dense(
            number_of_edge_outputs,
            activation=edge_output_activation,
            name="edge_prediction",
        )(edge_layer)

        global_output = layers.Dense(
            number_of_global_outputs,
            activation=global_output_activation,
            name="global_prediction",
        )(global_layer)

        output_dict = {
            "nodes": node_output,
            "edges": edge_output,
            "global": global_output,
            "graph": [node_output, edge_output, global_output],
        }
        try:
            outputs = output_dict[output_type]
        except KeyError:
            outputs = output_dict["graph"]

        model = tf.keras.models.Model(
            [node_features, edge_features, edges, edge_dropout],
            outputs,
        )

        super().__init__(model, **kwargs)


class MPNGNN(KerasModel):
    """
    Message-passing Graph Neural Network.

    Parameters:
    -----------
    dense_layer_dimensions: list of ints
        List of the number of units in each dense layer of the encoder and decoder. The
        number of layers is inferred from the length of this list.
    base_layer_dimensions: list of ints
        List of the latent dimensions of the graph blocks. The number of layers is
        inferred from the length of this list.
    number_of_node_outputs: int
        Number of output node features.
    number_of_edge_outputs: int
        Number of output edge features.
    number_of_global_outputs: int
        Number of output global features.
    node_output_activation: str or activation function or layer
        Activation function for the output node layer. See keras docs for accepted strings.
    edge_output_activation: str or activation function or layer
        Activation function for the output edge layer. See keras docs for accepted strings.
    cls_layer_dimension: int
        Number of units in the decoder layer for global features.
    global_output_activation: str or activation function or layer
        Activation function for the output global layer. See keras docs for accepted strings.
    dense_block: str, keras.layers.Layer, or callable
        The dense block to use for the encoder and decoder.
    graph_block: str, keras.layers.Layer, or callable
        The graph block to use for the graph computation. See gnns.layers for available
        graph blocks.
    readout_block: str, keras.layers.Layer, or callable
        The readout block used to compute global features.
    output_type: str
        Type of output. Either "nodes", "edges", "global" or
        "graph". If 'key' is not a supported output type, then
        the model output will be the concatenation of the node,
        edge, and global predictions.
    kwargs: dict
        Keyword arguments for the dense block.
    Returns:
    --------
    tf.keras.Model
        Keras model for the graph neural network.
    """

    def __init__(
        self,
        dense_layer_dimensions=(32, 64, 96),
        base_layer_dimensions=(96, 96),
        number_of_node_features=3,
        number_of_edge_features=1,
        number_of_node_outputs=1,
        number_of_edge_outputs=1,
        number_of_global_outputs=1,
        node_output_activation=None,
        edge_output_activation=None,
        global_layer_dimension=64,
        global_output_activation=None,
        dense_block=DenseBlock(
            activation=GELU,
            normalization="LayerNormalization",
        ),
        graph_block="MPN",
        readout_block=layers.Lambda(
            lambda x: tf.math.reduce_sum(x, axis=1), name="global_readout"
        ),
        output_type="graph",
        **kwargs
    ):
        dense_block = as_block(dense_block)
        graph_block = as_block(graph_block)

        node_features, edge_features, edges, edge_dropout = (
            tf.keras.Input(shape=(None, number_of_node_features)),
            tf.keras.Input(shape=(None, number_of_edge_features)),
            tf.keras.Input(shape=(None, 2), dtype=tf.int32),
            tf.keras.Input(shape=(None, 2)),
        )

        node_layer = node_features
        edge_layer = edge_features

        # Encoder for node and edge features
        for dense_layer_number, dense_layer_dimension in zip(
            range(len(dense_layer_dimensions)), dense_layer_dimensions
        ):
            node_layer = dense_block(
                dense_layer_dimension,
                name="node_ide" + str(dense_layer_number + 1),
                **kwargs
            )(node_layer)

            edge_layer = dense_block(
                dense_layer_dimension,
                name="edge_ide" + str(dense_layer_number + 1),
                **kwargs
            )(edge_layer)

        # Bottleneck path, graph blocks
        layer = (
            node_layer,
            edge_layer,
            edges,
            tf.ones_like(edge_features[..., 0:1]),
            edge_dropout,
        )

        for base_layer_number, base_layer_dimension in zip(
            range(len(base_layer_dimensions)), base_layer_dimensions
        ):
            layer = graph_block(
                base_layer_dimension,
                name="graph_block_" + str(base_layer_number),
            )(layer)

        # Split nodes and edges
        node_layer, edge_layer, *_ = layer

        # Compute global features
        global_layer = readout_block(node_layer)
        global_layer = dense_block(
            global_layer_dimension, name="cls_mlp", **kwargs
        )(global_layer)

        # Output layers
        node_output = layers.Dense(
            number_of_node_outputs,
            activation=node_output_activation,
            name="node_prediction",
        )(node_layer)

        edge_output = layers.Dense(
            number_of_edge_outputs,
            activation=edge_output_activation,
            name="edge_prediction",
        )(edge_layer)

        global_output = layers.Dense(
            number_of_global_outputs,
            activation=global_output_activation,
            name="global_prediction",
        )(global_layer)

        output_dict = {
            "nodes": node_output,
            "edges": edge_output,
            "global": global_output,
            "graph": [node_output, edge_output, global_output],
        }
        try:
            outputs = output_dict[output_type]
        except KeyError:
            outputs = output_dict["graph"]

        model = tf.keras.models.Model(
            [node_features, edge_features, edges, edge_dropout],
            outputs,
        )

        super().__init__(model, **kwargs)
