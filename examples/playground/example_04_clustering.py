import os
import sys

import numpy as np
import cupy as cp
import pandas as pd
import cudf
import pickle
import plotly.graph_objects as go
import plotly.io as pio

from mygraph import save_graph_to_pkl_gpu, load_graph_from_pkl_gpu
from myclustering import reduce_pca, leiden_clustering_gpu, kmeans_clustering_gpu

def main():
# load .pkl file to DataFrame
    df = pd.read_pickle("../data/features/features.pkl")
# load DataFrame to GPU and cast to cuda array
    features = cudf.DataFrame.from_pandas(df).to_cupy()
# PCA
    features_pca_20 = reduce_pca(features, 20)
# leiden clustering
    labels = leiden_clustering_gpu(features_pca_20, 0.7, 0.3)
# kmeans clustering
    #labels = kmeans_clustering_gpu(features, 15, 42)
# check I
    print(len(labels))
    #print(labels[:100])
# check II
'''
    vol = 100000
# NOTE: the image stacks/tif and segmentation results/csv are not matching! For debug only
    df1 = pd.read_csv("../data/raw/P4P7_T0-_RUN_s170480-nyzzsrssfx_tissuumaps_20230428-2219_GLOBAL.csv", usecols=['X_centroid','Y_centroid'], nrows=vol)
    df2 = pd.DataFrame(label[:vol], columns=['markers'])
    fig = go.Figure(go.Scatter(x=df1['X_centroid'], y=df1['Y_centroid'], mode='markers', marker=dict(color=df2['markers'], colorscale='Viridis', showscale=True)))
    fig.show()
    pio.write_image(fig, "../data/outputs/test.png", width=1080, height=1080, scale=2)
'''

if __name__ == '__main__':
    main()
