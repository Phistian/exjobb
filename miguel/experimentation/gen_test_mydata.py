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


def make_sets_equal_to_frames(input_df):
    df = input_df.copy()
    new_sets = df["frame"].to_numpy().copy()
    df["set"] = new_sets
    return df


def scale_solution(inputdf, multipliers=[1, 1, 1, 1], solution_dim=4, only_passive=False):
    # This function normalizes all separate dimensions of the solution, and optionally adds a scaling factor to the active and/or passive force
    if len(multipliers) != solution_dim:
        raise Exception("Multipliers must be of same dimension as node labels.")

    df = inputdf.copy()
    n = df.shape[0]
    maxima = np.zeros(solution_dim)
    # Finding max
    for i in df.index:
        for j in range(solution_dim):
            if maxima[j] < np.abs(df.at[i, "solution"][j]):
                maxima[j] = abs(df.at[i, "solution"][j])
                print(maxima[j], end=' ')

    # Share maximum across axes
    maxima[0:2] = np.max(maxima[0:2])
    if not only_passive:
        maxima[2:4] = np.max(maxima[2:4])

    # Apply scaling
    scalings = np.array(multipliers) / maxima
    for i in df.index:
        for j in range(solution_dim):
            df.at[i, "solution"][j] = df.at[i, "solution"][j] * scalings[j]

    if only_passive:
        scalings_dict = {"passive": scalings[0]}
    else:
        scalings_dict = {"active": scalings[0], "passive": scalings[2]}
    return df, scalings_dict, maxima


def set_real_labels(nodesdf):
    last_frame = 0
    label_val = 0
    for i in list(nodesdf.index):
        current_frame = nodesdf.at[i, "frame"]
        if current_frame > last_frame:
            label_val = 0
        nodesdf.at[i, "label"] = label_val
        label_val += 1
        last_frame = current_frame


def shuffle_frames(df):
    dfcpy = df.copy()
    previous_set = df["set"].copy()
    dfgrouping = dfcpy.groupby(["frame"])
    dfs = [_df for _, _df in dfgrouping]
    random.shuffle(dfs)
    catdf = pd.concat(dfs)
    n_rows = catdf.shape[0]
    new_indices = np.arange(n_rows)
    catdf = catdf.set_index(new_indices)
    # catdf["set"] = previous_set.to_numpy()
    return catdf


def setstoframe(df):
    dfcpy = df.copy()
    framecol = deepcopy(dfcpy["frame"])
    dfcpy["set"] = framecol
    return dfcpy


def make_frames_start_at_zero(df):
    dfcpy = df.copy()
    framecol = deepcopy(dfcpy["frame"]).to_numpy()
    minframe = framecol.min()
    new_framecol = framecol - minframe
    dfcpy["frame"] = new_framecol
    return dfcpy


def subset_train_and_val(input_df, val_ratio):
    df = input_df.copy()
    n_particles = int((df.index.max() + 1) / (df.loc[:, "frame"].max() + 1))
    tmp_val_rows = int(val_ratio * df.shape[0])
    i = tmp_val_rows
    while np.mod(i, n_particles) != 0:
        i += 1
    cutoff_index = i
    val_df = df.loc[:cutoff_index - 1, :]
    train_df = df.loc[cutoff_index:, :]
    train_df = make_frames_start_at_zero(train_df)
    train_df = train_df.reset_index(drop=True)
    val_rows = cutoff_index + 1
    return train_df, val_df, val_rows, n_particles


modelname = "onPCtest2"
data_dict = np.load(datasets_address + "\\tslj\\N5 samples1000 F_P60.npy", allow_pickle=True).item() # load data
## Extract some variables and leave only the dictionary which will be input to the graph extractor
node_labels_dim = len(data_dict["solution"][0])
box_len = data_dict['box_len']
del data_dict['box_len']
interaction_radius = data_dict['interaction_radius']
del data_dict["interaction_radius"]
potential_type = data_dict['potential_type']
del data_dict["potential_type"]


## A pandas dataframe is needed as input to the graph extractor.
nodesdf = pd.DataFrame.from_dict(data_dict)


## Make centroids positive only, with zero in bottom left corner of box
nodesdf.loc[:, "centroid-0"] = nodesdf.loc[:, "centroid-0"] + box_len/2
nodesdf.loc[:, "centroid-1"] = nodesdf.loc[:, "centroid-1"] + box_len/2


## Normalize node centroids and orientations so that max is 1
max_vals = {"centroid-0" : box_len, "centroid-1" : box_len, "orientation" : np.pi*2, 'frame': nodesdf["frame"].max(), "solution0": nodesdf["solution"]}
for key in ["centroid-0", "centroid-1", "orientation"]:
  nodesdf.loc[:, key] = nodesdf.loc[:, key] / max_vals[key]


## Normalize each column of the solution and ev. add extra scaling to a force type
a_scale = 1
p_scale = 1
nodesdf, scales, sol_maxima = scale_solution(nodesdf, multipliers=[a_scale, a_scale, p_scale, p_scale], solution_dim=node_labels_dim, only_passive=False)


## Set the labels so that each particle always has one unique label index
nodesdf.loc[:, "label"] = 0
set_real_labels(nodesdf)


## Cut out a validation set, the rest is the training set.
val_ratio = 0.1
train_nodesdf, val_nodesdf, val_rows, n_particles = subset_train_and_val(nodesdf, val_ratio)


## Set the sets so that each frame is seen as one video
#train_nodesdf = setstoframe(train_nodesdf)


## Shuffle the frames in the training set, and re-index
#train_nodesdf = shuffle_frames(train_nodesdf)


## Set the frames so that the first one is 0 (After the validation split, the frames of the training data might not start at 0)
train_nodesdf = make_frames_start_at_zero(train_nodesdf)
train_nodesdf = make_sets_equal_to_frames(train_nodesdf)
val_nodesdf = make_sets_equal_to_frames(val_nodesdf)

## Scale the box length and interaction length as much as the centroids
scaled_interaction_radius = interaction_radius/max_vals["centroid-0"]  # The length at which the potential has come close to 0 (this length is 3 in the simulations).
scaled_box_len = box_len/max_vals["centroid-0"]
scaled_lengths_dict = {"length_scale": 1/box_len, "box_len": scaled_box_len, "interaction_radius": scaled_interaction_radius, "max_x": box_len, "max_y": box_len, "max_orientation": 1}

## Set search radius to be used in graph generators to the radius where particle interaction stops

global_search_radius = scaled_interaction_radius

_OUTPUT_TYPE = "nodes"

variables = dt.DummyFeature(
    radius=global_search_radius,
    output_type=_OUTPUT_TYPE,
    nofframes=3, # time window to associate nodes (in frames)
)

model = dt.models.gnns.MPNGNN(
    dense_layer_dimensions=(64, 96,),      # number of features in each dense encoder layer
    base_layer_dimensions=(96,),    # Latent dimension throughout the message passing layers
    number_of_node_features=3,             # Number of node features in the graphs
    number_of_edge_features=1,             # Number of edge features in the graphs
    number_of_edge_outputs=1,              # Number of predicted features
    number_of_node_outputs=4,
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
graph = own_graphs.GraphExtractor(
    nodesdf=train_nodesdf,
    properties=["centroid", "orientation"],
    box_len=1,
    radius=global_search_radius,
)
'''
generator = own_generators.GraphGenerator(
    nodesdf=nodesdf,
    properties=["centroid", "orientation"],
    batch_size=32,
    **variables.properties(),
    box_len=1
)

with generator:
    history = model.fit(generator, epochs=4)

model.save(f"saved_models/{modelname}")
