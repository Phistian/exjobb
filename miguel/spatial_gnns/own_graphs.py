import numpy as np
import pandas as pd
import itertools

import tqdm
from scipy.spatial.distance import pdist, squareform

import more_itertools as mit
from operator import is_not
from functools import partial

def get_dist_matrix_one_frame_per_video(input_df, additions):
    df_i = input_df.copy()
    locs = df_i[['centroid-0', 'centroid-1']].values
    n_particles = len(locs)
    full_dist_mat = np.zeros((len(additions), n_particles, n_particles))
    for i, addition in enumerate(additions):
        df_i['centroid-0'] = input_df['centroid-0'] + addition[0]
        df_i['centroid-1'] = input_df['centroid-1'] + addition[1]
        distance_matrix = np.expand_dims(squareform(pdist(df_i[['centroid-0', 'centroid-1']])), axis=0)
        full_dist_mat[i, :, :] = distance_matrix
    dist_matrix_with_periodic_boundary = full_dist_mat.max(axis=0)


    return dist_matrix_with_periodic_boundary

def get_min_dist_vectors_mtx(df, box_len):
    dx = np.subtract.outer(df.loc[:,"centroid-0"].values, df.loc[:,"centroid-0"].values)
    dx = np.where(dx > 0.5 * box_len, dx - box_len, np.where(dx < -0.5 * box_len, dx + box_len, dx))
    dy = np.subtract.outer(df.loc[:, "centroid-1"].values, df.loc[:, "centroid-1"].values)
    dy = np.where(dy > 0.5 * box_len, dy - box_len, np.where(dy < -0.5 * box_len, dy + box_len, dy))
    dist_min = np.stack((dx, dy), axis=0)
    return dist_min

def v_mtx_to_d_mtx(v_mtx):
    sq_v = np.square(v_mtx)
    sq_d = np.sum(sq_v, axis=0)
    d = np.sqrt(sq_d)
    return d

def get_distance_matrix(df, box_len):
    d = v_mtx_to_d_mtx(get_min_dist_vectors_mtx(df, box_len))
    return d

def GetEdge(
    df: pd.DataFrame,
    start: int,
    end: int,
    radius: int,
    columns = [],
    **kwargs,
):
    """
    Extracts the edges from a windowed sequence of frames
    Parameters
    ----------
    df: pd.DataFrame
        A dataframe containing the extracted node properties.
    start: int
        Start frame of the edge.
    end: int
        End frame of the edge.
    radius: int
        Search radius for the edge (pixel units).
    parenthood: list
        A list of parent-child relationships
        between nodes.
    Returns
    -------
    edges: pd.DataFrame
        A dataframe containing the extracted
        properties of the edges.
    """



    box_len = kwargs["box_len"]
    additions = [np.array([box_len, 0]), np.array([-box_len, 0]), np.array([0, box_len]), np.array([0, -box_len]),
                 np.array([box_len, box_len]), np.array([-box_len, box_len]), np.array([box_len, -box_len]),
                 np.array([-box_len, -box_len])]


    parenthood = np.ones((1, 2)) * -1
    # Add a column for the indexes of the nodes
    df.loc[:, "index"] = df.index
    

    # Filter the dataframe to only include the the
    # frames, centroids, labels, and indexes
    df = df.loc[(df["frame"] >= start) & (df["frame"] <= end)].filter(
        regex="(frame|centroid|label|index)"
    )

    # Merge columns contaning the centroids into a single column of
    # numpy arrays, i.e., centroid = [centroid_x, centroid_y,...]
    df.loc[:, "centroid"] = df.filter(like="centroid").apply(np.array, axis=1)

    # Add key column to the dataframe
    df["key"] = 1

    # Group the dataframe by frame
    framev = df["frame"].to_numpy()
    if framev[0] != framev[-1]:
        df = df.groupby(["frame"])
        dfs = [_df for _, _df in df]
    else:
        dfs = []
        dfs.append(df)
    #new_indices = np.arange(n_particles)
    for idf in dfs:        
        idf.reset_index(drop=True, inplace=True)


    edges = []
    for dfi in dfs:

        combdf = pd.merge(dfi, dfi, on='key').drop("key", axis=1)

        # Compute distances between centroids
        D = get_distance_matrix(dfi, box_len)
        d = D.reshape(D.size)
        combdf.loc[:, "feature-dist"] = d
        combdf = combdf[combdf["feature-dist"] != 0.0]

        # Filter out edges with a feature-distance less than scale * radius
        combdf = combdf[combdf["feature-dist"] < radius].filter(
            regex=("frame|label|index|feature")
        )
        edges.append(combdf)
        
    # Concatenate the dataframes in a single
    # dataframe for the whole set of edges
    edgedf = pd.concat(edges) if len(edges) > 0 else pd.DataFrame(columns=columns)

    # Merge columns contaning the labels into a single column
    # of numpy arrays, i.e., label = [label_x, label_y]
    edgedf.loc[:, "label"] = edgedf.filter(like="label").apply(
        np.array, axis=1
    )

    # Returns a solution for each edge. If label is the parenthood
    # array or if label_x == label_y, the solution is 1, otherwise
    # it is 0 representing edges are not connected.
    def GetSolution(x):
        if np.any(np.all(x["label"][::-1] == parenthood, axis=1)):
            solution = 1.0
        elif x["label_x"] == x["label_y"]:
            solution = 1.0
        else:
            solution = 0.0

        return solution

    # Initialize solution column
    edgedf["solution"] = 0.0

    return AppendSolution(edgedf, GetSolution)


def EdgeExtractor(nodesdf, **kwargs):
    """
    Extracts edges from a sequence of frames
    Parameters
    ----------
    nodesdf: pd.DataFrame
        A dataframe containing the extracted node properties.
    noframes: int
        Number of frames to be used for
        the edge extraction.
    """
    # Create a copy of the dataframe to avoid overwriting
    df = nodesdf.copy()
    columns = df.columns
    edgedfs = []
    sets = np.unique(df["set"])
    for setid in tqdm.tqdm(sets):
        df_set = df.loc[df["set"] == setid].copy()
        max_frame = df_set['frame'].max()
        #windowing portion removed
        edgedf = GetEdge(df_set, columns=columns, start=0, end=max_frame, **kwargs)
        edgedf["set"] = setid
        edgedfs.append(edgedf)
        

    # Concatenate the dataframes in a single
    # dataframe for the whole set of edges
    edgesdfs = pd.concat(edgedfs)

    return edgesdfs


def AppendSolution(df, func, **kwargs):
    """
    Appends a solution to the dataframe
    Parameters
    ----------
    df: pd.DataFrame
        A dataframe containing the extracted
        properties for the edges/nodes.
    func: function
        A function that takes a dataframe
        and returns a solution.
    Returns
    -------
    df: pd.DataFrame
        A dataframe containing the extracted
        properties for the edges/nodes with
        a solution.
    """
    # Get solution
    df.loc[:, "solution"] = df.apply(lambda x: func(x), axis=1, **kwargs)

    return df


def DataframeSplitter(df, props: tuple, to_array=True, **kwargs):
    """
    Splits a dataframe into features and labels
    Parameters
    ----------
    dt: pd.DataFrame
        A dataframe containing the extracted
        properties for the edges/nodes.
    atts: list
        A list of attributes to be used as features.
    to_array: bool
        If True, the features are converted to numpy arrays.
    Returns
    -------
    X: np.ndarray or pd.DataFrame
        Features.
    """
    # Extract features from the dataframe
    if len(props) == 1:
        features = df.filter(like=props[0])
    else:
        regex = ""
        for prop in props[0:]:
            regex += prop + "|"
        regex = regex[:-1]
        features = df.filter(regex=regex)

    # Extract labels from the dataframe
    label = df["solution"]

    if "index_x" in df:
        outputs = [features, df.filter(like="index"), label]
    else:
        outputs = [features, label]

    # Convert features to numpy arrays if needed
    if to_array:
        outputs = list(
            map(
                lambda x: np.stack(x.apply(np.array, axis=1).values),
                outputs[:-1],
            )
        ) + [
            np.stack(outputs[-1].values),
        ]

    return outputs


def GraphExtractor(
    nodesdf: pd.DataFrame = None,
    properties: list = None,
    parenthood: np.array = np.ones((1, 2)) * -1,
    validation: bool = False,
    global_property: np.array = None,
    **kwargs,
):
    """
    Extracts the graph from a sequence of frames
    
    Parameters
    ----------
    sequence: dt.Feature
        A sequence of frames.
    """

    # Extract edges and edge features from nodes
    print("Creating graph edges...")
    edgesdf = EdgeExtractor(nodesdf, parenthood=parenthood, **kwargs)

    # Split the nodes dataframe into features and labels
    nodefeatures, nfsolution = DataframeSplitter(
        nodesdf, props=properties, **kwargs
    )

    # Split the edges dataframe into features, sparse adjacency
    # matrix, and labels
    edgefeatures, sparseadjmtx, efsolution = DataframeSplitter(
        edgesdf, props=("feature",), **kwargs
    )

    if validation:
        # Add frames to the adjacency matrix
        frames = edgesdf.filter(like="frame").to_numpy()
        sparseadjmtx = np.concatenate((frames, sparseadjmtx), axis=-1)

        # Add frames to the node features matrix
        frames = nodesdf.filter(like="frame").to_numpy()
        nodefeatures = np.concatenate((frames, nodefeatures), axis=-1)

    # Create edge weights matrix
    edgeweights = np.ones(sparseadjmtx.shape[0])
    edgeweights = np.stack(
        (np.arange(0, edgeweights.shape[0]), edgeweights), axis=1
    )

    # Extract set ids
    nodesets = nodesdf[["set", "label"]].to_numpy()
    edgesets = edgesdf[["set", "label_x", "label_y"]].to_numpy()

    if global_property is None:
        global_property = np.zeros(np.unique(nodesdf["set"]).shape[0])

    return (
        (nodefeatures, edgefeatures, sparseadjmtx, edgeweights),
        (nfsolution, efsolution, global_property),
        (nodesets, edgesets),
    )
