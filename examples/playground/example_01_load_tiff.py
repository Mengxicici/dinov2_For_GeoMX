import os
import sys

import numpy as np

from mydataset import MyDataset, MyTransform
from myutils import save_tiff_from_tensor

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import torch

def main():
# Tranform
    mytransform = transforms.Compose([
        MyTransform((0, 3, 12), 256, 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), "cpu"),
    ])
# Load data
    test_set = MyDataset("../data/list_tiff.txt", mytransform)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)
# Check data briefly
    for idx, data in enumerate(test_loader):
        image, idx = data
        save_tiff_from_tensor(image, idx[0].numpy(), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), "")

if __name__ == '__main__':
    main()
