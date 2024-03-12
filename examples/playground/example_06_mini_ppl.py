import os
import sys

import numpy as np

from mydataset import MyTransform
from myutils import save_tiff_from_tensor
from myhelpermethods import get_first_line_of_txt, get_tiff_info, get_samples_from_raw_data, get_samples_from_raw_data_test_mode, get_registration_results, get_segmentation_results, \
                            get_inputs_for_feature_extraction, \
                            get_features_extracted, \
                            get_clustering_labels, get_labels_visualized

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# set the paths
## pipeline inputs
raw_path  = "/home2/s224636/Documents/Spatial_Biology_Project/data/raw/INNATE_P4_P7_tumor_37plex_opt1_Scan1.qptiff"
## work path for pipeline
work_path = "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/ky/mini_pipeline"

# set the files
## image information file
img_info_file = "/image_information.txt"
## segmentation inputs/sub-group of the pipeline, generated from sample extract methods
samples_file = "/samples.txt"
## feature extraction inputs
segmentation_file = "/segmentations.csv"
inputs_file = "/inputs.txt"
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

# train/inference settings
device = "cuda"

channels = (0, 1, 2) # embeeding map is needed
resize = 256
central_crop = 224
means = (0.485, 0.456, 0.406)
stds = (0.229, 0.224, 0.225)

num_samples = 4
num_inputs = 1000
samples_window_size = 25000
inputs_window_size = 48

# clustering srttings
dim_reduction_method = 0
clustering_method = 0

# visualization settings
num_points = 100000

# Tranform
mytransform = transforms.Compose([
    MyTransform(channels, resize, central_crop, means, stds, device),
])
# model used in dino
#dino_model = 'dinov2_vitl14'
dino_model = "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/ky/mini_pipeline/zero_shot/dinov2_vits14_pretrain.pth"

def main():
###########
# STAGE 1 #
###########
## image information
    #get_tiff_info(raw_path, work_path+img_info_file)
## samples
    #get_samples_from_raw_data(raw_path, work_path+samples_path, samples_name, work_path+samples_file, num_samples, channels, samples_window_size//2)
## registration
    #get_registration_results()
## segmentation information
    #get_segmentation_results(raw_path, "random", work_path+segmentation_file, channels, num_inputs)
## segmentation
    #get_inputs_for_feature_extraction(raw_path, work_path+segmentation_file, work_path+inputs_path, inputs_name, work_path+inputs_file, num_inputs, channels, inputs_window_size//2)
###########
# STAGE 2 #
###########
## feature extraction
    get_features_extracted(work_path+inputs_file, work_path+features_path+features_file, mytransform, dino_model, device)
## clustering
    #get_clustering_labels(work_path+features_path+features_file, dim_reduction_method, clustering_method, work_path+outputs_path+labels_file)
## final result visualization
    #get_labels_visualized(work_path+outputs_path+labels_file, work_path+segmentation_file, num_points, work_path+outputs_path+outputs_file)

if __name__ == '__main__':
    main()
