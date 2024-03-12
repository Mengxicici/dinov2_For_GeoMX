import os
import sys

import numpy as np

from mydataset import MyTIFFDataset_v3, MyTransform_v3, MyTransform_v3_vis
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
#mytransform = transforms.Compose([
#    MyTransform_v3_vis(central_crop, 0.1, 0.1, 0.1, 0.1, 0.1),
#])

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
from cuml.manifold import TSNE

def main():
    print("START!")
# parse args
    parser = argparse.ArgumentParser(description='dsa:t-SNE')
    parser.add_argument('--src', type=str, help='data path of csv')
    parser.add_argument('--out', type=str, help='data path of outputs')
    args = parser.parse_args()

# visualization
    ## cols
    list_int = list(range(768))
    list_string = map(str, list_int)
    cols = list(list_string)
    ## data
    df1 = pd.read_csv(args.src, usecols=cols)
    df2 = pd.read_csv(args.src, usecols=['n_class'])
    ## t-SNE
    tsne = TSNE(n_components=2, perplexity=50, learning_rate=20)
    z = tsne.fit_transform(df1.to_numpy())
    df = pd.DataFrame()
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    fig = go.Figure(go.Scatter(x=df['comp-1'], y=df['comp-2'], \
                               mode='markers', \
                               marker=dict(color=df2['n_class'], colorscale='Viridis', showscale=True, size=1.5)))
    res = args.out + "_tSNE.png"
    pio.write_image(fig, res, width=1080, height=1080, scale=2)
    print("The clustering image is saved to " + res + "...")

    print("DONE!")

if __name__ == '__main__':
    main()
