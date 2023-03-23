import deeptrack as dt
from deeptrack.models.gnns.generators import GraphGenerator
from deeptrack.models.gnns.graphs import GraphExtractor
from deeptrack.models.gnns.generators import GraphGenerator
from spatial_graphs import own_graphs
from spatial_graphs import own_generators

import tensorflow as tf

import pandas as pd
import numpy as np

from deeptrack.extras import datasets

import logging
logging.disable(logging.WARNING)


import random
import matplotlib.pyplot as plt
from copy import deepcopy


def scale_solution(inputdf, a_scale=1, p_scale=1):
    # This function normalizes all separate dimensions of the solution, and optionally adds a scaling factor to the active and/or passive force
    df = inputdf.copy()
    n = df.shape[0]
    maxima = np.zeros(4)
    # finding max
    for i in df.index:
        for j in range(4):
            if maxima[j] < np.abs(df.at[i, "solution"][j]):
                maxima[j] = abs(df.at[i, "solution"][j])
    # scaling
    scales = np.array([a_scale, a_scale, p_scale, p_scale]) / maxima
    for i in df.index:
        for j in range(4):
            df.at[i, "solution"][j] = df.at[i, "solution"][j] * scales[j]
    return df, scales


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
    return train_df, val_df

modelname = "testingtesting"
data_dict = np.load("datasets/N14 samples10 F_P60COLAB.npy", allow_pickle=True).item() # load data
## Extract some variables and leave only the dictionary which will be input to the graph extractor
box_len = data_dict['box_len']
del data_dict['box_len']
interaction_radius = data_dict['interaction_radius']
del data_dict["interaction_radius"]
potential_type = data_dict['potential_type']
del data_dict["potential_type"]

## A pandas dataframe is needed as input to the graph extractor.
nodesdf = pd.DataFrame.from_dict(data_dict)
n_detections = nodesdf.shape[0]

## Normalize node centroids and orientations
max_vals = {"centroid-0" : box_len/2, "centroid-1" : box_len/2, "orientation" : np.pi*2, 'frame': nodesdf["frame"].max(), "solution0": nodesdf["solution"]}
for key in ["centroid-0", "centroid-1", "orientation"]:
  nodesdf.loc[:, key] = nodesdf.loc[:, key] / max_vals[key]

## Scale the box length and interaction length as much as the centroids
scaled_interaction_radius = interaction_radius/max_vals["centroid-0"]
scaled_box_len = box_len/max_vals["centroid-0"]

## Normalize the solution vector elements and ev. add extra scaling
p_scale = 1
a_scale = 0
nodesdf, scales = scale_solution(nodesdf, p_scale=p_scale, a_scale=a_scale)

## Set the labels of the particles
#nodesdf["label"] = np.arange(0, nodesdf.shape[0])
#nodesdf.loc[250:, "label"] = 1
nodesdf.loc[:, "label"] = 0
set_real_labels(nodesdf)

## Cut out a validation set, the rest is the training set.
val_ratio = 0.2
train_nodesdf, val_nodesdf = subset_train_and_val(nodesdf, val_ratio)


# Remember that from now on, no shuffling before passing into graph extractor
## Set the sets so that each frame is seen as one video
#train_nodesdf = setstoframe(train_nodesdf)

## Shuffle the frames in the training set, and re-index
#train_nodesdf = shuffle_frames(train_nodesdf)
train_nodesdf = make_frames_start_at_zero(train_nodesdf)

radius = scaled_interaction_radius
print(f"Scaled interaction radius becomes {scaled_interaction_radius} length units, with box length {scaled_box_len}")


_OUTPUT_TYPE = "nodes"

# Seach radius for the graph edges
radius = 0.5

variables = dt.DummyFeature(
    radius=radius,
    output_type=_OUTPUT_TYPE,
    nofframes=3, # time window to associate nodes (in frames)
)

model = dt.models.gnns.MPNGNN(
    dense_layer_dimensions=(64, 96,),      # number of features in each dense encoder layer
    base_layer_dimensions=(96, 96, 96),    # Latent dimension throughout the message passing layers
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

print(train_nodesdf)
generator = GraphGenerator(
    nodesdf=train_nodesdf,
    properties=["centroid", "orientation"],
    min_data_size=95,
    max_data_size=96,
    **variables.properties()
)
'''
generator = own_graphs.GraphGenerator(
    nodesdf=nodesdf,
    properties=["centroid"],
    **variables.properties(),
    box_len=1
)
'''
with generator:
    history = model.fit(generator, epochs=100)


model.save(f"saved_models/{modelname}")