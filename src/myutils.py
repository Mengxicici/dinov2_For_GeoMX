import os
import numpy as np
import pandas as pd
import random
import pickle
from tqdm import tqdm

from tifffile import imread, imwrite, TiffFile

import torch

def save_tiff_from_tensor(image, idx, mean, std, path):
    '''
    @args:
        image (tensor)      0~1, torch.float32, CHW
        idx   (int/ndarray) index of the tif image stacks
        mean  (list)        default (0.485, 0.456, 0.406)
        std   (list)        default (0.229, 0.224, 0.225)
    '''
# set path
    if path == "":
        path = "../data/inputs/test"
# put images back from Normalization
    a = torch.Tensor(mean).unsqueeze(-1).unsqueeze(-1)
    m = torch.Tensor(std).unsqueeze(-1).unsqueeze(-1)
# save tif image stacks, if from gpu, copy to cpu firstly
    if image.device.type == 'cuda':
        a_dev = a.to(device='cuda')
        m_dev = m.to(device='cuda')
        image = (image * m_dev + a_dev) * 255
        imwrite(path + str(idx) + ".tiff", np.array(image.to(dtype=torch.uint8, device='cpu')), imagej=True, metadata={'axes': 'ZCYX'})
    else:
        image = (image * m + a) * 255
        imwrite(path + str(idx) + ".tiff", np.array(image.to(dtype=torch.uint8)), imagej=True, metadata={'axes': 'ZCYX'})
# print information
    #print(image)
    print("shape:  ", end = "")
    print(image.shape)
    print("dtype:  ", end = "") 
    print(image.dtype)
    print("device: ", end = "")
    print(image.device)

def dart_throw(image_page, height, width, num_samples, yaxis, xaxis, idx, max_throw, step, min_intensity):
    '''
    @args:
        image_page    (ndarray) the selected page in .qptiff file, CHW, C=1
        height, width (int)     shape of the selected page
        num_samples   (int)     total number of samples
        xaxis, yaxis  (int)     default 1/0, specify how the huge page will be roughly divided
        idx           (int)     index of samples
        max_throw     (int)     maximum number of trails
        step          (int)     step of random.rangrange()
        min_intensity (float)   minimum intensity for locating a dart
    '''
# randomly get height and width
    if yaxis == 0:
        h = height
        y = random.randrange(0, h, step)
    else:
        h = height // num_samples
        y = random.randrange(h*idx, h*(idx+1), step)
    if xaxis == 0:
        w = width
        x = random.randrange(0, w, step)
    else:
        w = width // num_samples
        x = random.randrange(w*idx, w*(idx+1), step)
    c = 1
    while image_page[y, x] < min_intensity and c < max_throw:
        if yaxis == 0:
            y = random.randrange(0, h, step)
        else:
            y = random.randrange(h*idx, h*(idx+1), step)
        if xaxis == 0:
            x = random.randrange(0, w, step)
        else:
            x = random.randrange(w*idx, w*(idx+1), step)
        c = c + 1

    return x, y

def is_boundary(image_page, height, width, x, y, a, b, tol, counts):
    '''
    @args:
        image_page    (ndarray) the selected page in .qptiff file, CHW, C=1
        height, width (int)     shape of the selected page
        x, y          (int)     current location
        a, b          (int)     shape of the rectangular area used to check boundary, a for height, b for width
        tol           (float)   minimum intensity recognized as boundary
        counts        (int)     counts for minimum intensity recognized as boundary
    '''
# get coordinates
    upper = max(y - a//2, 0)
    lower = min(y + a//2, height)
    left  = max(x - b//2, 0)
    right = min(x + b//2, width)
# count numbers of non-boundary points
    c = 0
    for i in range(upper, lower):
        for j in range(left, right):
            if image_page[i, j] > tol:
                c = c + 1
                if c >= counts:
                    return False
    return True

def get_samples_boundaries(image_page, height, width, num_samples, yaxis, xaxis, ori_x, ori_y, w_x, w_y, method, d, tol, counts):
    '''
    @args:
        image_page    (ndarray) the selected page in .qptiff file, CHW, C=1
        height, width (int)     shape of the selected page
        num_samples   (int)     total number of samples
        xaxis, yaxis  (int)     default 1/0, specify how the huge page will be roughly divided
        ori_x, ori_y  (int)     starting point for searching non-zero intensity area
        w_x, w_y      (int)     maximum window size for crop
        method        (string)  default "fixed-window"/"auto-track"
        d             (int)     height for searching non-zero intensity area
        tol           (float)   minimum intensity recognized as boundary
        counts        (int)     counts for minimum intensity recognized as boundary
    '''
# fix-sized window
    if method == "fixed-window":
        upper = max(ori_y - w_y, 0)
        lower = min(ori_y + w_y, height)
        left  = max(ori_x - w_x, 0)
        right = min(ori_x + w_x, width)
# auto track
    elif method == "auto-track":
# init
        if yaxis == 0:
            h = height
        else:
            h = height // num_samples
        if xaxis == 0:
            w = width
        else:
            w = width // num_samples
        upper = ori_y - d//2
        lower = ori_y + d//2
        left  = ori_x - d//2
        right = ori_x + d//2
# search the boundaries
        while not is_boundary(image_page, height, width, ori_x, upper, d, w//4, tol, counts) and upper > 0:
            upper = upper - d//2
        while not is_boundary(image_page, height, width, ori_x, lower, d, w//4, tol, counts) and lower < height:
            lower = lower + d//2
        while not is_boundary(image_page, height, width, left,  ori_y, h//4, d, tol, counts) and left > 0:
            left  = left  - d//2
        while not is_boundary(image_page, height, width, right, ori_y, h//4, d, tol, counts) and right < width:
            right = right + d//2
# fine-tune the boundaries
        dlt_y = (lower-upper)//2 - w_y
        dlt_x = (right-left)//2  - w_x
        if dlt_y < 0:
            dlt_y = 0
        if dlt_x < 0:
            dlt_x = 0
        upper = max(upper + dlt_y, 0)
        lower = min(lower - dlt_y, height)
        left  = max(left  + dlt_x, 0)
        right = min(right - dlt_x, width)
# print info
        print("boundaries: " + str(upper) + ", "  + str(lower) + ", " + str(left) + ", " + str(right))
# default
    else:
        upper = max(ori_y - 1000, 0)
        lower = min(ori_y + 1000, height)
        left  = max(ori_x - 1000, 0)
        right = min(ori_x + 1000, width)

    return upper, lower, left, right

def gen_random_cropped_tiff(images_file, keys, num_samples, yaxis, xaxis, w_x, w_y, method, path, to_file):
    '''
    @args:
        images_file  (string) path/to/qptiff/file
        keys         (list)   page indexes of .qptiff file
        num_samples  (int)    total number of samples, also number of .tiff files
        w_x, w_y     (int)    maximum window size for crop
        method       (string) default "fixed-window"/"auto-track"
        xaxis, yaxis (int)    default 1/0, specify how the huge page will be roughly divided
    @tuning: the performance of the random crop is largely decided by the parameters including:
                 max_throw:    500,   large value for sparse image
                 min_intensity 50.00, large value for bright image
                 d             2000,  large value leads to large cropped chunks
                 tol           15.00, small value for sharpe boundary
                 counts        1000,  small value for sharpe boundary
    @reference: https://pypi.org/project/tifffile/
    '''
# set path and to_file
    if path == "":
        path = "../data/raw/temp"
    if to_file == "":
        to_file = "../data/list_qptiff.txt"
# get information
    tif = TiffFile(images_file)
    page = tif.pages[0]
    height, width = page.shape
    dtype         = page.dtype
    axes          = page.axes
    image         = page.asarray()
    tif.close()
# print information
    print("page height: ", end = "")
    print(height)
    print("page width: ", end = "")
    print(width)
    print(image)
# using dummy random crop for test
    images = imread(images_file, key=keys)
    for idx in range(0, num_samples):
        x, y                      =             dart_throw(image, height, width, num_samples, yaxis, xaxis, idx, 500, 100, 50.00)
        upper, lower, left, right = get_samples_boundaries(image, height, width, num_samples, yaxis, xaxis, x, y, w_x, w_y, method, 2000, 20.00, 500)
# write to image stacks, CHW
        imwrite(path + str(idx) + ".tiff", images[:, upper:lower, left:right], imagej=True, metadata={'axes': 'CYX'})
# generate list_qptiff.txt
    f = open(to_file, "w")
    list_qptiff_context = ""
    for line_num in range(0, num_samples):
        list_qptiff_context += path + str(line_num) + ".tiff " + str(line_num) + "\n"
    f.write(list_qptiff_context[:-1])
    f.close()

def gen_guided_cropped_tiff(images_file, csv_file, keys, num_samples, w_x, w_y, path, to_file):
    '''
    @args:
        images_file  (string) path/to/qptiff/file
        csv_file     (string) path/to/csv/file, containing the segmentation results
        keys         (list)   page indexes of .qptiff file
        num_samples  (int)    total number of samples, also number of .tiff files
        w_x, w_y     (int)    maximum window size for crop
    '''
# set path and to_file
    if path == "":
        path = "../data/raw/temp"
    if to_file == "":
        to_file = "../data/list_qptiff.txt"
# get information
    tif = TiffFile(images_file)
    page = tif.pages[0]
    height, width = page.shape
    dtype         = page.dtype
    axes          = page.axes
    image         = page.asarray()
    tif.close()
# print information
    print("page height: ", end = "")
    print(height)
    print("page width: ", end = "")
    print(width)
    print(image)
# load x, y from csv file
    coords_df = pd.read_csv(csv_file, usecols=['X_centroid','Y_centroid'], nrows=num_samples)
    coords_x = coords_df['X_centroid'].to_numpy().astype(int)
    coords_y = coords_df['Y_centroid'].to_numpy().astype(int)
# using dummy random crop for test
    images = imread(images_file, key=keys)
    for idx in range(0, num_samples):
        x                         = coords_x[idx]
        y                         = coords_y[idx]
        upper, lower, left, right = get_samples_boundaries(image, height, width, num_samples, 0, 0, x, y, w_x, w_y, "fixed-window", 0, 0.0, 0)
# write to image stacks, CHW
        imwrite(path + str(idx) + ".tiff", images[:, upper:lower, left:right], imagej=True, metadata={'axes': 'CYX'})
# generate list_qptiff.txt
    f = open(to_file, "w")
    list_qptiff_context = ""
    for line_num in range(0, num_samples):
        list_qptiff_context += path + str(line_num) + ".tiff " + str(line_num) + "\n"
    f.write(list_qptiff_context[:-1])
    f.close()

def gen_random_cropped_tiff_to_uint8(images_file, keys, num_samples, yaxis, xaxis, w_x, w_y, method, path, to_file):
    '''
    @args:
        images_file  (string) path/to/qptiff/file
        keys         (list)   page indexes of .qptiff file
        num_samples  (int)    total number of samples, also number of .tiff files
        w_x, w_y     (int)    maximum window size for crop
        method       (string) default "fixed-window"/"auto-track"
        xaxis, yaxis (int)    default 1/0, specify how the huge page will be roughly divided
    @tuning: the performance of the random crop is largely decided by the parameters including:
                 max_throw:    500,   large value for sparse image
                 min_intensity 50.00, large value for bright image
                 d             2000,  large value leads to large cropped chunks
                 tol           15.00, small value for sharpe boundary
                 counts        1000,  small value for sharpe boundary
    @reference: https://pypi.org/project/tifffile/
    '''
# set path and to_file
    if path == "":
        path = "../data/raw/temp"
    if to_file == "":
        to_file = "../data/list_qptiff.txt"
# get information
    tif = TiffFile(images_file)
    page = tif.pages[1] # DAPI preferred?
    height, width = page.shape
    dtype         = page.dtype
    axes          = page.axes
    image         = page.asarray()
    tif.close()
# print information
    print("page height: ", end = "")
    print(height)
    print("page width: ", end = "")
    print(width)
    print(image)
# using dummy random crop for test
    images = imread(images_file, key=keys)
    for idx in range(0, num_samples):
        x, y                      =             dart_throw(image, height, width, num_samples, yaxis, xaxis, idx, 500, 100, 100.00)
        upper, lower, left, right = get_samples_boundaries(image, height, width, num_samples, yaxis, xaxis, x, y, w_x, w_y, method, 2000, 15.00, 1000)
# write to image stacks, CHW
        #convert to uint8
        data = images[:, upper:lower, left:right]
        if data.dtype == np.uint8:
            print("dtype: uint8")
            depth = 1
        elif data.dtype == np.uint16:
            print("dtype: uint16")
            depth = 8
        else:
            print("dtype: not uint8/uint16, but consider as uint16!")
            depth = 8
        data = (data.astype(np.float64) / (2**depth + 1)).astype(np.uint8)
        imwrite(path + str(idx) + ".tiff", data, imagej=True, metadata={'axes': 'CYX'})
# generate list_qptiff.txt
    f = open(to_file, "w")
    list_qptiff_context = ""
    for line_num in range(0, num_samples):
        list_qptiff_context += path + str(line_num) + ".tiff " + str(line_num) + "\n"
    f.write(list_qptiff_context[:-1])
    f.close()

def gen_pkl_of_guided_cropped_tiff(images_file, csv_file, keys, num_samples, w_x, w_y, path, to_file):
    '''
    @args:
        images_file  (string) path/to/qptiff/file
        csv_file     (string) path/to/csv/file, containing the segmentation results
        keys         (list)   page indexes of .qptiff file
        num_samples  (int)    total number of samples, also number of .tiff files
        w_x, w_y     (int)    maximum window size for crop
    '''
# set path and to_file
    if path == "":
        path = "../data/raw/temp"
    if to_file == "":
        to_file = "../data/list_qptiff.txt"
# get information
    tif = TiffFile(images_file)
    page = tif.pages[0]
    height, width = page.shape
    dtype         = page.dtype
    axes          = page.axes
    image         = page.asarray()
    tif.close()
# print information
    print("page height: ", end = "")
    print(height)
    print("page width: ", end = "")
    print(width)
    print(image)
# load x, y from csv file
    if num_samples > 0:
        coords_df = pd.read_csv(csv_file, usecols=['X_centroid','Y_centroid'], nrows=num_samples)
    else:
        coords_df = pd.read_csv(csv_file, usecols=['X_centroid','Y_centroid'])
    coords_x = coords_df['X_centroid'].to_numpy().astype(int)
    coords_y = coords_df['Y_centroid'].to_numpy().astype(int)
# using dummy random crop for test
    Z = len(coords_x)
    C = len(keys)
    H = w_y*2
    W = w_x*2
    res = np.zeros((Z, C, H, W))
    images = imread(images_file, key=keys)
    for idx in tqdm(range(0, num_samples)):
        x                         = coords_x[idx]
        y                         = coords_y[idx]
        upper, lower, left, right = get_samples_boundaries(image, height, width, num_samples, 0, 0, x, y, w_x, w_y, "fixed-window", 0, 0.0, 0)
        res[idx] = images[:, upper:lower, left:right]
        if idx < 100:
            imwrite(path + str(idx) + ".tiff", images[:, upper:lower, left:right], imagej=True, metadata={'axes': 'CYX'})
    with open(to_file, "wb") as fp:
        pickle.dump(res, fp)

def save_tiff_from_tensor_one_batch_v2(images, num, mean, std, path):
    '''
    @args:
        images (tensor)      0~1, torch.float32, CHW
        num    (int)         batch size
        mean   (list)        default (0.485, 0.456, 0.406)
        std    (list)        default (0.229, 0.224, 0.225)
        path   (string)      default "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/ky/mini_pipeline/visualization/dino_inputs/input_"
    '''
# set path
    if path == "":
        path = "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/ky/mini_pipeline/visualization/dino_inputs/input_"
# put images back from Normalization
    a = torch.Tensor(mean).unsqueeze(-1).unsqueeze(-1)
    m = torch.Tensor(std).unsqueeze(-1).unsqueeze(-1)
# save tif image stacks, if from gpu, copy to cpu firstly
    for idx in range(num):
        image = images[idx]
        if image.device.type == 'cuda':
            a_dev = a.to(device='cuda')
            m_dev = m.to(device='cuda')
            image = (image * m_dev + a_dev) * 255
            imwrite(path + str(idx) + ".tiff", np.array(image.to(dtype=torch.uint8, device='cpu')), imagej=True, metadata={'axes': 'CYX'})
        else:
            image = (image * m + a) * 255
            imwrite(path + str(idx) + ".tiff", np.array(image.to(dtype=torch.uint8)), imagej=True, metadata={'axes': 'CYX'})

def get_keys_from_txt(names, picks):
    keys = []
    with open(names, "r") as f1:
        lines1 = f1.read()
        list1 = lines1.split('\n')
    with open(picks, "r") as f2:
        line2 = f2.readline()
        while line2:
            keys.append(list1.index(line2.replace('\n', '')))
            line2 = f2.readline()

    return keys
