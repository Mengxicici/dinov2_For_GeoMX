import os
import sys

import numpy as np

from mydataset import MyDataset, MyTransform
from myutils import save_tiff_from_tensor, gen_random_cropped_tiff

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import torch

def main():
# qptiff to tiff
    gen_random_cropped_tiff("../data/raw/INNATE_P4_P7_tumor_37plex_opt1_Scan1.qptiff", (0, 1, 2), 4, 1, 0, 10000, 10000, "auto-track", "", "")
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
