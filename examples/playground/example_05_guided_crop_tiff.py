import os
import sys

import numpy as np

from mydataset import MyDataset, MyTransform
from myutils import save_tiff_from_tensor, gen_guided_cropped_tiff

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import torch

def main():
# random crop
    # NOTE: the image stacks/tif and segmentation results/csv are not matching! The coordinates in csv file matches the centrosymetric-rotated image stacks! For debug only
    gen_guided_cropped_tiff("../data/raw/sroi_P4P7_T0_reg000.tif", "../data/raw/P4P7_T0-_RUN_s170480-nyzzsrssfx_tissuumaps_20230428-2219_GLOBAL.csv", (0, 1, 2), 40, 20, 20, "", "")
# Tranform
    mytransform = transforms.Compose([
        MyTransform((0, 1, 2), 256, 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), "cpu"),
    ])
# Load data
    test_set = MyDataset("../data/list_qptiff.txt", mytransform)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)
# Check data briefly
    for idx, data in enumerate(test_loader):
        image, idx = data
        save_tiff_from_tensor(image, idx[0].numpy(), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), "")

if __name__ == '__main__':
    main()
