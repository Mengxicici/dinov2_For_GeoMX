import os
import sys

import numpy as np

from mydataset import MyTransform, MyTransform_v2, MyTransform_v3
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
raw_path = "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/Huang/TMA_0622/16bit_raw_TMA_0622_ROI_A01N.ome.tiff"
## work path for pipeline
work_path = "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/ky/mini_pipeline"

# set the files
## image information file
img_info_file = "/image_information.txt"
## segmentation inputs/sub-group of the pipeline, generated from sample extract methods
samples_file = "/samples.txt"
## feature extraction inputs
segmentation_file = "/singleCellQuant_Cells_dataFrame_regTMA_0622_A01N_0517-24Jun2023.csv"
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
means = (0.5, 0.5, 0.5, 0.5)
stds = (0.25, 0.25, 0.25, 0.25)

embed_map = {}

resize = 256
central_crop = 224

#num_samples = 4
#samples_window_size = 25000

num_inputs = 298464
inputs_window_size = 48

# clustering srttings
dim_reduction_method = 0
clustering_method = 0

# visualization settings
num_points = num_inputs

# Tranform
mytransform = transforms.Compose([
    MyTransform_v2(channels, resize, central_crop, means, stds, device),
])

# model used in dino
#dino_model = 'dinov2_vitl14'
dino_model = "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/ky/mini_pipeline/zero_shot/dinov2_vits14_pretrain.pth"

def main():
###########
# STAGE 1 #
###########
# The dimensions are examples of ROI A01N from Dr. Huang's Dataset #
## image information
    #get_tiff_info(raw_path, work_path+img_info_file) # 4 x 16880 x 26500
## samples
    #get_samples_from_raw_data(raw_path, work_path+samples_path, samples_name, work_path+samples_file, num_samples, channels, samples_window_size//2)
## registration
    #get_registration_results()
## segmentation information
    #get_segmentation_results(raw_path, "random", work_path+segmentation_file, channels, num_inputs) # out: 298464 x 3 x 48 x 48, 3: PanCK, Vimentin and CD45 are picked; 48: window size for single-cell image
## segmentation
    get_inputs_for_feature_extraction(raw_path, work_path+segmentation_file, work_path+inputs_path, inputs_name, work_path+inputs_file_v2, num_inputs, channels, inputs_window_size//2) # in: 298464 x 3 x 48 x 48, out: 298464 x 3 x 224 x 224, accoring to the transforms
###########
# STAGE 2 #
###########
## feature extraction
    #get_features_extracted(work_path+inputs_file_v2, work_path+features_path+features_file, mytransform, dino_model, device) # in: 298464 x 3 x 224 x 224 # out: 298464 x 384, since vits14 is used, dim = 384
## clustering
    #get_clustering_labels(work_path+features_path+features_file, dim_reduction_method, clustering_method, work_path+outputs_path+labels_file) # in: 298464 x 384, out: labels, 298464 x 1, labels are numbers from 0 to (n_class-1)
## final result visualization
    #get_labels_visualized(work_path+outputs_path+labels_file, work_path+segmentation_file, num_points, work_path+outputs_path+outputs_file)

if __name__ == '__main__':
    main()
