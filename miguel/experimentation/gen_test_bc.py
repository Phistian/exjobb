import sys
from pathlib import Path
cur_path = Path(__file__).parent.resolve()

parent_path = cur_path.parent.resolve()
exjobb_address = str(parent_path) + "\\.."
spatial_address = str(parent_path) + '\\spatial_gnns'
datasets_address = str(parent_path) + '\\datasets'
histories_address = str(parent_path) + '\\training_results/saved_histories'
models_address = str(parent_path) + '\\training_results\\saved_models'
sys.path.append(spatial_address)

import deeptrack as dt
from deeptrack.models.gnns.generators import GraphGenerator
from deeptrack.models.gnns.graphs import GraphExtractor
from deeptrack.models.gnns.generators import GraphGenerator
import own_graphs
import own_generators
import own_models

import tensorflow as tf

import pandas as pd
import numpy as np

from deeptrack.extras import datasets

import logging
logging.disable(logging.WARNING)


import random
import matplotlib.pyplot as plt
from copy import deepcopy

#datasets.load("BFC2Cells")
nodesdf = pd.read_csv("datasets/BFC2DLMuSCTra/nodesdf.csv")

# normalize centroids between 0 and 1
nodesdf.loc[:, nodesdf.columns.str.contains("centroid")] = (
    nodesdf.loc[:, nodesdf.columns.str.contains("centroid")]
    / np.array([1000.0, 1000.0])
)

modelname = "testingtesting"


_OUTPUT_TYPE = "nodes"

# Seach radius for the graph edges
radius = 0.5

variables = dt.DummyFeature(
    radius=radius,
    output_type=_OUTPUT_TYPE,
    nofframes=3, # time window to associate nodes (in frames)
)

model = own_models.OneMessagePassingLayerMPNGNN(
    dense_layer_dimensions=(64, 96,),      # number of features in each dense encoder layer
    base_layer_dimensions=(96, 96, 96),    # Latent dimension throughout the message passing layers
    number_of_node_features=2,             # Number of node features in the graphs
    number_of_edge_features=1,             # Number of edge features in the graphs
    number_of_edge_outputs=1,              # Number of predicted features
    number_of_node_outputs=1,
    edge_output_activation="linear",      # Activation function for the output layer
    output_type=_OUTPUT_TYPE,              # Output type. Either "edges", "nodes", or "graph"
)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss = 'mae',
    metrics=['accuracy'],
)

model.summary()




'''
generator = GraphExtractor(
    nodesdf=nodesdf,
    properties=["centroid"],
    min_data_size=511,
    max_data_size=512,
    **variables.properties()
)
generator = own_graphs.GraphExtractor(
    nodesdf=nodesdf,
    properties=["centroid"],
    **variables.properties(),
    box_len=1
)
'''

'''
generator = GraphGenerator(
    nodesdf=nodesdf,
    properties=["centroid"],
    min_data_size=95,
    max_data_size=96,
    **variables.properties()
)
'''

generator = own_generators.GraphGenerator(
    nodesdf=nodesdf,
    properties=["centroid"],
    **variables.properties(),
    min_data_size=95,
    max_data_size=96,
    box_len=1
)

with generator:
    history = model.fit(generator, epochs=100)


model.save(f"saved_models/{modelname}")
