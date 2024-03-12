import os
import sys

import numpy as np

from mydataset import MyTIFFDataset_v3, MyTransform_v3
from myutils import save_tiff_from_tensor
from myhelpermethods import get_first_line_of_txt, get_tiff_info, get_samples_from_raw_data, get_samples_from_raw_data_test_mode, get_registration_results, get_segmentation_results, \
                            get_inputs_for_feature_extraction, \
                            get_features_extracted, \
                            get_clustering_labels, get_labels_visualized

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import pandas as pd
import matplotlib.pyplot as plt

# set the paths
## pipeline inputs ##
#raw_path = "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/Huang/TMA_0622/16bit_raw_TMA_0622_ROI_A01N.ome.tiff"
## work path for pipeline
work_path = "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/ky/mini_pipeline"

# set the files
## image information file
img_info_file = "/image_information.txt"
## segmentation inputs/sub-group of the pipeline, generated from sample extract methods
samples_file = "/samples.txt"
## feature extraction inputs
#segmentation_file = "/singleCellQuant_Cells_dataFrame_regTMA_0622_A01N_0517-24Jun2023.csv"
segmentation_file = "/_P2T1_roi_metadata.csv"
inputs_file = "/inputs.txt"
inputs_file_v2 = "/inputs.pkl"
## clustering inputs
features_file = "/features.pkl"
## output labels
labels_file = "/labels.pkl"
## output images
outputs_file = "/clustering.png"

# set subpaths
## sample extract outputs/segmentation inputs
samples_path = "/samples"
## segemntation outputs/feature extraction inputs
inputs_path = "/inputs"
## feature extraction outputs/clustering inputs
features_path = "/features"
## clustering outputs
outputs_path = "/outputs"

# set file names
samples_name = "/sample_"
inputs_name = "/input_"

## train/inference settings ##
device = "cuda"

# for the channels, select certain maps and combine certain maps(use dict)
#channels = (1, 2, 3)
#means = (0.485, 0.456, 0.406)
#stds = (0.229, 0.224, 0.225)

channels = (0, 1, 2, 3)
means = (0.485, 0.456, 0.406, 0.449)
stds = (0.229, 0.224, 0.225, 0.226)

embed_map = {}

resize = 256
central_crop = 224

#num_samples = 4
#samples_window_size = 25000

#num_inputs = 1000
num_inputs = 298464
inputs_window_size = 32

# clustering srttings
dim_reduction_method = 0
clustering_method = 0

# visualization settings
num_points = num_inputs

# Tranform
mytransform = transforms.Compose([
    MyTransform_v3(central_crop, 0.5, 0.5, 0.5, 0.5, 0.5),
])

# models
dino_model_01 = "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/ky/Demo_dinov2/outputs/eval/training_12499/teacher_checkpoint.pth"
dino_model_02 = "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/ky/Demo_dinov2/outputs/eval/training_24999/teacher_checkpoint.pth"

from dinov2.models.vision_transformer import vit_base_CODEX

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from dinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block

# data
model_pretrained = "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/ky/Demo_dinov2/models/dinov2_vitb14_pretrain.pth"

from tqdm import tqdm
from torch.utils.data.dataloader import default_collate

# files
seg = work_path+segmentation_file
res = work_path+outputs_path+outputs_file
feat = work_path+features_path+features_file

import cupy as cp
import pandas as pd
import cudf
import pickle
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt

from myclustering import reduce_pca, kmeans_clustering_gpu
from myutils import get_keys_from_txt

import argparse

import scanpy as sc
import rapids_singlecell as rsc

import cuml
from cuml.cluster import KMeans
from cuml import PCA

import anndata as ad

def main():
    print("START!")
# parse args
    parser = argparse.ArgumentParser(description='dsa:clustering')
    parser.add_argument('--src', type=str, help='data path of csv')
    parser.add_argument('--out', type=str, help='data path of outputs')
    parser.add_argument('--method', type=str, help='clustering method')
    parser.add_argument('--method_arg', type=float, help='arg for clustering method')
    args = parser.parse_args()

# clustering
    ## cols
    list_int = list(range(768))
    list_string = map(str, list_int)
    cols = list(list_string)
    ## data
    df = pd.read_csv(args.src, usecols=cols)
    ## check
    print(df)
    ## branch
    if args.method == "KMeans":
        n_comps = int(args.method_arg)
        print("KMeans with " + str(n_comps) + " components")
        # get df
        features = cudf.DataFrame.from_pandas(df).to_cupy()
        # get labels
        k_means = KMeans(n_clusters=n_comps)
        k_means.fit(features)
        labels = k_means.predict(features).get()
    elif args.method == "Leiden":
        resol = args.method_arg
        print("Leiden with " + str(resol) + " resolution")
        # get df
        features = df.to_numpy()
        ## create AnnData from DataFrame
        n_obs = len(features)
        obs = pd.DataFrame()
        obs['CellID'] = list(range(1, n_obs+1))
        var_names = cols
        n_vars = len(cols)
        var = pd.DataFrame(index=var_names)
        adata = ad.AnnData(X=features, obs=obs, var=var)
        # Leiden Clustering
        ## PCA
        adata_20dim = adata.copy()
        pca = PCA(n_components=20)
        adata_20dim.obsm['X_pca'] = pca.fit_transform(adata_20dim.X)
        ## neighbor
        sc.pp.neighbors(adata_20dim, use_rep='X_pca', method='rapids')
        rsc.tl.leiden(adata_20dim, resolution=resol)
        ## check
        print(adata_20dim.obs['leiden'])
        # get labels
        labels_str = adata_20dim.obs['leiden'].to_numpy()
        labels = np.array(labels_str, dtype=int)
        labels = labels/max(labels)
    elif args.method == "Louvain":
        resol = args.method_arg
        print("Louvain with " + str(resol) + " resolution")
        # get df
        features = df.to_numpy()
        ## create AnnData from DataFrame
        n_obs = len(features)
        obs = pd.DataFrame()
        obs['CellID'] = list(range(1, n_obs+1))
        var_names = cols
        n_vars = len(cols)
        var = pd.DataFrame(index=var_names)
        adata = ad.AnnData(X=features, obs=obs, var=var)
        # Leiden Clustering
        ## PCA
        adata_20dim = adata.copy()
        pca = PCA(n_components=20)
        adata_20dim.obsm['X_pca'] = pca.fit_transform(adata_20dim.X)
        ## neighbor
        sc.pp.neighbors(adata_20dim, use_rep='X_pca', method='rapids')
        rsc.tl.louvain(adata_20dim, resolution=resol)
        ## check
        print(adata_20dim.obs['louvain'])
        # get labels
        labels_str = adata_20dim.obs['louvain'].to_numpy()
        labels = np.array(labels_str, dtype=int)
        labels = labels/max(labels)
    else:
        print("Not Implemented!")

# attach features and labels to csv file
    df_ori = pd.read_csv(args.src)
    df_labels = pd.DataFrame(labels, columns=['n_class'])
    frames = [df_ori, df_labels]
    results = pd.concat(frames, axis=1)
    print(results)
    res1 = args.out + "_" + args.method + str(args.method_arg) + "_clustering.csv"
    results.to_csv(res1)
    print("features are attached to " + res1 + "...")
# visualization
    df1 = pd.read_csv(args.src, usecols=['X_centroid','Y_centroid'])
    df2 = pd.DataFrame(labels, columns=['markers'])
    fig = go.Figure(go.Scatter(x=df1['X_centroid'], y=df1['Y_centroid'], \
                               mode='markers', \
                               marker=dict(color=df2['markers'], colorscale='Viridis', showscale=True, size=1.5)))
## flip
    fig.update_layout(
        yaxis = dict(autorange="reversed")
    )
# outputs
    res2 = args.out + "_" + args.method + str(args.method_arg) + "_clustering.png"
    pio.write_image(fig, res2, width=1080, height=1080, scale=2)
    print("The clustering image is saved to " + res2 + "...")

    print("DONE!")

if __name__ == '__main__':
    main()
