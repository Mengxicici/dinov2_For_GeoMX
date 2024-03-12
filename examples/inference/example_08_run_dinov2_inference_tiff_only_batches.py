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
import json
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

def main():
    print("START!")
# parse args
    parser = argparse.ArgumentParser(description='dinov2:inference')
    parser.add_argument('--src', type=str, help='backbone or checkpoint')
    parser.add_argument('--model', type=str, help='data path of model')
    parser.add_argument('--data', type=str, help='data path of images')
    parser.add_argument('--seg', type=str, help='data path of segementation results')
    parser.add_argument('--ws', type=int, help='window_size of single-cell images')
    parser.add_argument('--chans', type=int, help='number of channels')
    parser.add_argument('--names', type=str, help='data path of channelNames')
    parser.add_argument('--picks', type=str, help='data path of channelPicks')
    #parser.add_argument('--add', type=str, help='data path of HE images')
    parser.add_argument('--outdir', type=str, help='data path of csv file to adding features and labels info')
    parser.add_argument('--labels', type=json.loads, help='labels dict of samples')
    #=======================================
    # parser.add_argument('--custom_embedding_map',default={
    #                                                     0: 0,
    #                                                     1: 1,
    #                                                     2: 2,
    #                                                     3: 3,
                                                        
    #                                                 }, type=dict, help='custom embedding map')
    # parser.add_argument('--selected_channels', default=list(range(0, 4)), type=list, help='selected channels')
    #======================================
    args = parser.parse_args()

# check state_dict of backbone
    model = vit_base_CODEX(
            img_size=224,
            patch_size=14,
            in_chans=args.chans,
            qkv_bias=True,
            ffn_bias=True,
            proj_bias=True,
            drop_path_rate=0.0,
            drop_path_uniform=False,
            init_values=None,
            embed_layer=PatchEmbed,
            act_layer=nn.GELU,
            ffn_layer="mlp",
            block_chunks=1,
        )
    
    print("Before if-else, args.src:", args.src)
    if args.src == "checkpoint":
        print("Inside checkpoint branch")
        ckpts_full = torch.load(args.model)['teacher']
        ckpts_backbone = {k : v for k, v in ckpts_full.items() if k.startswith('backbone')}
        ## match the keys, if needed
        model_state_dict = {}
        for k, v in ckpts_backbone.items():
            k1 = k.replace('backbone.', '')
            model_state_dict[k1] = v
        #===========================================   
        # def load_pretrained_weights(model, pretrained_weights, checkpoint_key='teacher'):

        #     state_dict = torch.load(pretrained_weights, map_location="cpu")
        #     if checkpoint_key is not None and checkpoint_key in state_dict:
        #         print(f"Take key {checkpoint_key} in provided checkpoint dict")
        #         state_dict = state_dict[checkpoint_key]
        #     # remove `module.` prefix
        #     state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        #     # remove `backbone.` prefix induced by multicrop wrapper
        #     state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        #     msg = model.load_state_dict(state_dict, strict=False)
        #     print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
            
        # def embedding_seq(custom_embedding_map, selected_channels):
        #     #map_dict = ast.literal_eval(custom_embedding_map)
        #     map_dict = custom_embedding_map
        #     map_dict = {int(k):int(v) for k,v in map_dict.items()}
        #     embedding_seq = [map_dict[int(input_channel)] for input_channel in selected_channels]
        #     return embedding_seq

        # def build_weight_emb(embedding_seq, model):
        #     weights = model.patch_embed.proj.weight
        #     print(f"weights shape {weights.shape}")
        #     weights = weights[:,embedding_seq,:,:]
        #     model.patch_embed.proj.weight = nn.Parameter(weights)
        #     return model 
        
        # load_pretrained_weights(model, args.model)
        # embedding_seq = embedding_seq(args.custom_embedding_map, args.selected_channels)
        # model = build_weight_emb(embedding_seq, model)   
            
            
            
            
            
        #===========================================      
    else:
        print("Inside else branch")
        bb_full = torch.load(args.model)
        bb_blocks = {k : v for k, v in bb_full.items() if k.startswith('blocks') or \
                                                          k.startswith('cls') or \
                                                          k.startswith('mask') or \
                                                          k.startswith('patch_embed.proj.bias') or \
                                                          k.startswith('norm.weight') or \
                                                          k.startswith('norm.bias')}

        model_state_dict = bb_blocks

# load state_dict of backbone then to device
    #model.load_state_dict(model_state_dict)
    model.load_state_dict(model_state_dict, strict=False)
    model.to(device)

    names=[os.path.join(args.names, file_name) for file_name in os.listdir(args.names) if os.path.isfile(os.path.join(args.names, file_name))]
    picks=[os.path.join(args.picks, file_name) for file_name in os.listdir(args.picks) if os.path.isfile(os.path.join(args.picks, file_name))]
    datas=[os.path.join(args.data, file_name) for file_name in os.listdir(args.data) if os.path.isfile(os.path.join(args.data, file_name))]
    segs=[os.path.join(args.seg, file_name) for file_name in os.listdir(args.seg) if os.path.isfile(os.path.join(args.seg, file_name))]


    labels=args.labels
    
    def sorting_key(path):
        # Extract the filename without extension
        base_name = os.path.splitext(os.path.basename(path))[0]
        # Split the filename by "_" and take the last part
        return base_name.split("_")[-1]

    names_s = sorted(names, key=sorting_key)
    picks_s = sorted(picks, key=sorting_key)
    datas_s = sorted(datas, key=sorting_key)
    segs_s = sorted(segs, key=sorting_key)
    
    print(len(names),len(picks),len(datas),len(segs),len(labels.values()))

    assert len(names)==len(picks)==len(datas)==len(segs)==len(labels), "The number of files is not equal!"

    for name,pick,data,seg in zip(names_s,picks_s,datas_s,segs_s):
        label=labels[name.split('/')[-1].split('.')[0]]
        #print(name,pick,label)

        # load keys
        key = get_keys_from_txt(name, pick)

    # load data to device
        data_set = MyTIFFDataset_v3(data, key, seg, args.ws//2, mytransform)
        data_loader = DataLoader(data_set, batch_size=32, shuffle=False, num_workers=0)
    # init features
        features = []
    # inference
        with torch.no_grad():
            for i, image in tqdm(enumerate(data_loader)):
                feature = model(image)
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.tolist()
                features.extend(feature)

        print("features are generated...")
        '''
    # save features
        features_df = pd.DataFrame(features)
        features_df.to_pickle(feat)
        #print(features_df)
        print("features are saved to " + feat)

    # clustering
        df = pd.read_pickle(feat)
    # check features
        print(df)
    ## KMeans
        features = cudf.DataFrame.from_pandas(df).to_cupy()
        tmp = features
        labels = kmeans_clustering_gpu(tmp, 12, 42)
    # check labels
        print(labels)
        '''
    # attach features and labels to csv file
        df_ori = pd.read_csv(seg)
        df = pd.DataFrame(features)
        #print(df_ori)
        #df_labels = pd.DataFrame(labels, columns=['n_class'])
        #frames = [df_ori, df_labels, df]
        frames = [df_ori, df]
        results = pd.concat(frames, axis=1)
        print(results)
        results.to_csv(f"{args.outdir}/{name.split('/')[-1].split('.')[0]}_{label}.csv")
        print(f"features of {name.split('/')[-1].split('.')[0]} {label} are attached to {args.outdir}")
        '''
    # visualization
        df1 = pd.read_csv(args.seg, usecols=['X_centroid','Y_centroid'], nrows=num_points)
        df2 = pd.DataFrame(labels[:num_points], columns=['markers'])
        fig = go.Figure(go.Scatter(x=df1['X_centroid'], y=df1['Y_centroid'], \
                                mode='markers', \
                                marker=dict(color=df2['markers'], colorscale='Viridis', showscale=True, size=1.5)))
    ## flip
        fig.update_layout(
            yaxis = dict(autorange="reversed")
        )
        pio.write_image(fig, res, width=1080, height=1080, scale=2)
        print("The clustering image is saved to " + res + "...")

    ## Clustering
        #TODO AttributeError: 'NoneType' object has no attribute 'obsp'
    ### create AnnData
    #### UMAP
        df_coords = pd.read_csv(seg, usecols=['X_centroid','Y_centroid'])
        #TODO UMAP?
    #### adata
        adata = sc.AnnData(df)
    ### Leiden Clustering
        adata_neighbors = sc.pp.neighbors(adata)
        labels = rsc.tl.leiden(adata_neighbors, resolution=0.3)
        print(labels)
        '''
        print(f"{label} DONE!")

if __name__ == '__main__':
    main()
