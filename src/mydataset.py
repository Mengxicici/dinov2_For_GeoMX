import numpy as np
import cv2

import torchvision.transforms as transforms
import torch

from torch.utils.data import Dataset

from tifffile import imread, imwrite
import pickle

class MyDataset(Dataset):
    '''
    @args:
        data_list (String) a .txt file containing list of tif/tiff image stacks
                           for each line, the format is: "/path/to/image/stacks ID"
    '''
    def __init__(self, data_list, transform):
        f = open(data_list, 'r')
        data = []
        for line in f:
            line = line.rstrip()
            words = line.split()
            data.append((words[0], words[1]))
        self.data = data
        self.transform = transform
 
    def __getitem__(self, idx):
        path, index = self.data[idx]
        image = imread(path)
        image = self.transform(image)

        return image, int(index)

    def __len__(self):
        return len(self.data)

class MyPKLDataset(Dataset):
    def __init__(self, data_list, transform):
        with open(data_list, 'rb') as f:
            data = pickle.load(f)
        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        image = self.data[idx] 
        image = self.transform(image)

        return image

    def __len__(self):
        return len(self.data)

def label_encode(index, num_class):
    labels = np.zeros(num_class, dtype=np.float32)
    i = int(index)
    if i < num_class-1:
        labels[i] = 1.0
    else:
        labels[num_class-1] = 1.0

    return labels

class MyFineTuneDataset(Dataset):
    def __init__(self, data_list, transform, num_class, label_encode):
        f = open(data_list, 'r')
        data = []
        for line in f:
            line = line.rstrip()
            words = line.split()
            data.append((words[0], words[1]))
        self.data = data
        self.transform = transform
        self.num_class = num_class
        self.label_encode = label_encode

    def __getitem__(self, idx):
        path, i = self.data[idx]
        image = imread(path)
        image = self.transform(image)
        index = self.label_encode(i, self.num_class)

        return image, index

    def __len__(self):
        return len(self.data)

class MyTransform(object):
    '''
    @args:
        selected_channels (list)    default (0, 1 ,2)
        resize_length     (int)     default 256
        center_corp       (int)     default 224
        mean              (list)    default (0.485, 0.456, 0.406)
        std               (list)    default (0.229, 0.224, 0.225)
        device            (string)  default "cpu"/"cuda"
    '''
    def __init__(self, selected_channels, resize_length, center_crop, mean, std, device):
        self.selected_channels = selected_channels
        self.resize_length     = resize_length
        self.center_crop       = center_crop
        self.mean              = mean
        self.std               = std
        self.device            = device

    def __call__(self, image):
# select channels, CHW
        if not self.selected_channels:
            selected_image = image
        else:
            selected_image = image[self.selected_channels, :, :]
# resize
        if not self.resize_length:
            resize  = 256
        else:
            resize  = self.resize_length
        channels      = selected_image.shape[0]
        height        = selected_image.shape[1]
        width         = selected_image.shape[2]
        resized_image = cv2.resize(selected_image.reshape((height, width, channels)), [resize, resize], interpolation=cv2.INTER_AREA).reshape((channels, resize, resize))
# center crop
        if not self.center_crop:
            dlt = 224
        else:
            dlt = self.center_crop
        
        center_x     = resized_image.shape[1]
        center_y     = resized_image.shape[2] 
        x            = (center_x - dlt)//2
        y            = (center_y - dlt)//2
        croped_image = resized_image[:, x:x+dlt, y:y+dlt]
# tensor, cpu or gpu/cuda
        if self.device == "cuda":
            tensor_image = torch.from_numpy(croped_image.astype(np.int32)).to(dtype=torch.float, device='cuda')
# normalization
            a = torch.Tensor(self.mean).unsqueeze(-1).unsqueeze(-1).to(device='cuda')
            m = torch.Tensor(self.std).unsqueeze(-1).unsqueeze(-1).to(device='cuda')
        else:
            tensor_image = torch.from_numpy(croped_image).to(dtype=torch.float)
            a = torch.Tensor(self.mean).unsqueeze(-1).unsqueeze(-1)
            m = torch.Tensor(self.std).unsqueeze(-1).unsqueeze(-1)

        tensor_image = (tensor_image/255 - a) / m

        return tensor_image

class MyTransform_v2(object):
    '''
    @args:
        resize_length     (int)     default 256
        center_corp       (int)     default 224
        mean              (list)    default (0.485, 0.456, 0.406)
        std               (list)    default (0.229, 0.224, 0.225)
        device            (string)  default "cpu"/"cuda"
    '''
    def __init__(self, resize_length, center_crop, mean, std, device):
        self.resize_length     = resize_length
        self.center_crop       = center_crop
        self.mean              = mean
        self.std               = std
        self.device            = device

    def __call__(self, image):
# remove selected_channels
        selected_image = image
# resize
        if not self.resize_length:
            resize  = 256
        else:
            resize  = self.resize_length
        channels      = selected_image.shape[0]
        height        = selected_image.shape[1]
        width         = selected_image.shape[2]
        resized_image = cv2.resize(selected_image.reshape((height, width, channels)), [resize, resize], interpolation=cv2.INTER_AREA).reshape((channels, resize, resize))
# center crop
        if not self.center_crop:
            dlt = 224
        else:
            dlt = self.center_crop

        center_x     = resized_image.shape[1]
        center_y     = resized_image.shape[2]
        x            = (center_x - dlt)//2
        y            = (center_y - dlt)//2
        croped_image = resized_image[:, x:x+dlt, y:y+dlt]
# tensor, cpu or gpu/cuda
        if self.device == "cuda":
            tensor_image = torch.from_numpy(croped_image.astype(np.int32)).to(dtype=torch.float, device='cuda')
# normalization
            a = torch.Tensor(self.mean).unsqueeze(-1).unsqueeze(-1).to(device='cuda')
            m = torch.Tensor(self.std).unsqueeze(-1).unsqueeze(-1).to(device='cuda')
        else:
            tensor_image = torch.from_numpy(croped_image).to(dtype=torch.float)
            a = torch.Tensor(self.mean).unsqueeze(-1).unsqueeze(-1)
            m = torch.Tensor(self.std).unsqueeze(-1).unsqueeze(-1)

        tensor_image = (tensor_image/255 - a) / m

        return tensor_image

class MyPKLDataset_v2(Dataset):
    def __init__(self, data_list, transform):
        with open(data_list, 'rb') as f:
            data = pickle.load(f)
        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        image = self.data[idx]
        image = self.transform(image)

        return image, idx

    def __len__(self):
        return len(self.data)

class MyPKLDataset_v3(Dataset):
    def __init__(self, data_list, transform):
        with open(data_list, 'rb') as f:
            data = pickle.load(f)
        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        image = self.data[idx] 
        image = self.transform(image)

        return image.type(torch.float).cuda()

    def __len__(self):
        return len(self.data)

###########
# dataset #
###########
import pandas as pd

class MyTIFFDataset_v3(Dataset):
    def __init__(self, data_list, keys, csv_file, window_size, transform):
## keys ##
        #keys = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
## csv_file ##
        #csv_file = "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/ky/Demo_CODEX/_P2T1_roi_metadata.csv"
## window_size ##
        #window_size = 64
        w_x = window_size
        w_y = window_size
## imgs
        imgs = []
        df = pd.read_csv(csv_file, usecols=['X_centroid','Y_centroid'])
        x = df['X_centroid'].to_numpy().astype(int)
        y = df['Y_centroid'].to_numpy().astype(int)
        with open(data_list, 'rb') as f:
            tmp = imread(f, key=keys)
            page = tmp[0]
            height, width = page.shape
            for idx in range(len(x)):
                ori_x = x[idx]
                ori_y = y[idx]
                upper = max(ori_y - w_y, 0)
                lower = min(ori_y + w_y, height)
                left  = max(ori_x - w_x, 0)
                right = min(ori_x + w_x, width)
                img = tmp[:, upper:lower, left:right]
                imgs.append(img)

        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, idx):
        image = self.imgs[idx]
        image = self.transform(image)

        return image.type(torch.float).cuda()

    def __len__(self):
        return len(self.imgs)


class MyHEDataset_v3(Dataset):
    def __init__(self, data_list, keys, csv_file, window_size, transform, add_file):
## keys ##
        #keys = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
## csv_file ##
        #csv_file = "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/ky/Demo_CODEX/_P2T1_roi_metadata.csv"
## window_size ##
        #window_size = 64
        w_x = window_size
        w_y = window_size
## imgs
        imgs = []
        df = pd.read_csv(csv_file, usecols=['X_centroid','Y_centroid'])
        x = df['X_centroid'].to_numpy().astype(int)
        y = df['Y_centroid'].to_numpy().astype(int)
        with open(data_list, 'rb') as f:
            tmp = imread(f, key=keys)
            page = tmp[0]
            height, width = page.shape
            for idx in range(len(x)):
                ori_x = x[idx]
                ori_y = y[idx]
                upper = max(ori_y - w_y, 0)
                lower = min(ori_y + w_y, height)
                left  = max(ori_x - w_x, 0)
                right = min(ori_x + w_x, width)
                ## make sure that each of the inputs are of the same size
                if upper == 0:
                    lower = 2 * w_y
                if lower == height:
                    upper = height - 2*w_y
                if left == 0:
                    right = 2 * w_x
                if right == width:
                    left = width - 2*w_x
                img = tmp[:, upper:lower, left:right]
                imgs.append(img)

        if not(add_file):
            self.imgs = imgs
        else:
################
# load HE file #
################
## imgs_adds
            imgs_adds = []
            with open(add_file, 'rb') as f:
                tmp_hwc = imread(f)
                ## tmp: RGBA image, hwc -> chw
                channels = tmp_hwc.shape[2]
                height = tmp_hwc.shape[0]
                width = tmp_hwc.shape[1]
                tmp = np.transpose(tmp_hwc, (2, 0, 1))
                #page = tmp[0]
                #height, width = page.shape
                for idx in range(len(x)):
                    ori_x = x[idx]
                    ori_y = y[idx]
                    upper = max(ori_y - w_y, 0)
                    lower = min(ori_y + w_y, height)
                    left  = max(ori_x - w_x, 0)
                    right = min(ori_x + w_x, width)
                    ## make sure that each of the inputs are of the same size
                    if upper == 0:
                        lower = 2 * w_y
                    if lower == height:
                        upper = height - 2*w_y
                    if left == 0:
                        right = 2 * w_x
                    if right == width:
                        left = width - 2*w_x
                    img = tmp[:, upper:lower, left:right]
                    imgs_adds.append(img)
#########
# merge #
#########
            print(len(imgs_adds))
            print(len(imgs_adds[0]))

            self.imgs = np.concatenate((imgs_adds, imgs), axis=1)
            print(self.imgs.shape)

        self.transform = transform

    def __getitem__(self, idx):
        image = self.imgs[idx]
        image = self.transform(image)

        return image.type(torch.float).cuda()

    def __len__(self):
        return len(self.imgs)


#################
# augmentation  #
#################
import random
from typing import Sequence
from torch.nn.functional import normalize

# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def make_std_mean(image):
    # tensor 24x224x224
    flatten_image = torch.flatten(image, start_dim=1)
    # tensor 24x1x1
    std, mean = torch.std_mean(flatten_image, dim=1, keepdim=True)
    # tensor 24x1
    return torch.flatten(std)+1e-8, torch.flatten(mean)

def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)

def make_scale_to_zero_one(image):
    # tensor 24x224x224
    t = torch.flatten(image, start_dim=1)
    # tensor 24x1x1
    t_min, _ = torch.min(t, dim=1, keepdim=True)
    t_max, _ = torch.max(t, dim=1, keepdim=True)
    # tensor 24x1x1
    return (image-t_min.unsqueeze(-1)) / (t_max.unsqueeze(-1)-t_min.unsqueeze(-1)+1e-8), t_max, t_min

def randomly_apply_image_method(image, method, keep_p):
    tmp = random.randrange(10)
    if tmp <= 10*keep_p:
        return method(image)
    return image

def image_method_reshape(image):
    channels      = image.shape[0]
    height        = image.shape[1]
    width         = image.shape[2]
    return np.transpose(image, (1, 2, 0))

def image_method_resize(image):
    resize = 256
    return cv2.resize(image, [resize,resize], interpolation=cv2.INTER_AREA)

def gen_rand_x_y(size, random_crop_size):
    assert size - random_crop_size >= 0
    rand_min = random_crop_size//2
    rand_max = size - random_crop_size//2
    tmpx = random.randrange(rand_min, rand_max)
    tmpy = random.randrange(rand_min, rand_max)
    return tmpx, tmpy

def image_method_rand_crop(image, random_crop_size):
    if not random_crop_size:
        dlt = 224
    else:
        dlt = random_crop_size
## find random center_x and center_y
    x, y = gen_rand_x_y(image.shape[0], dlt)
    return image[x-dlt//2:x+dlt//2, y-dlt//2:y+dlt//2, :]

def image_method_blur(image):
    return cv2.GaussianBlur(image, (3,3), 0.1, 2.0)

def image_method_horizontal_flip(image):
    axis = 0
    return cv2.flip(image, axis)

def image_method_color(image):
    brightness = 0.5
    contract = 128
    return cv2.convertScaleAbs(image, contract, brightness)

def image_method_gray_scale(image):
    h, w, c = image.shape
    gray_image = np.empty((h, w, c))
    # DAPI included
    targets = [0, 1, 2]
    tmp = image[:, :, targets]
    ## 3-channel inputs are required
    gray_tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    for i in range(c):
        gray_image[:, :, i] = gray_tmp
    return gray_image

def image_method_solarize(image):
## use threshold to black/white out 10% channels
    h, w, c = image.shape
    n = int(0.1*c) + 1
    target = 0.5
    rand_chans = np.random.randint(0, c-1, n)
    for rand_chan in rand_chans:
        image[:, :, rand_chan] = np.ones((h, w)) * target
    return image

class MyTransform_v3(object):
    '''
    @args:
        keep_p            (float) default 0.5
        random_corp_size  (int)   default 224
    '''
    def __init__(self, random_crop_size, p_flip, p_color, p_gray, p_blurr, p_solar):
        self.random_crop_size = random_crop_size
        self.p_flip  = 1 - p_flip
        self.p_color = 1 - p_color
        self.p_gray  = 1 - p_gray
        self.p_blurr = 1 - p_blurr
        self.p_solar = 1 - p_solar

    def __call__(self, image):
# reshape: chw -> hwc
        hwc_image = image_method_reshape(image)
# resize
        resized_hwc_image = image_method_resize(hwc_image)
# random crop
        croped_hwc_image_1 = image_method_rand_crop(resized_hwc_image, self.random_crop_size)
        croped_hwc_image = randomly_apply_image_method(croped_hwc_image_1, image_method_horizontal_flip, self.p_flip)
# color jitting
        colored_hwc_image_1 = randomly_apply_image_method(croped_hwc_image, image_method_color, self.p_color)
        colored_hwc_image = randomly_apply_image_method(colored_hwc_image_1, image_method_gray_scale, self.p_gray)
# global transfo
        blurred_hwc_image_1 = randomly_apply_image_method(colored_hwc_image, image_method_blur, self.p_blurr)
        blurred_hwc_image = randomly_apply_image_method(blurred_hwc_image_1, image_method_solarize, self.p_solar)
# to tensor: hwc -> chw
        t = torch.from_numpy(blurred_hwc_image.astype(float))
        tensor_image, t_max, t_min = make_scale_to_zero_one(t.permute(2, 0 ,1))
# get std and mean, per page
        t_std, t_mean = make_std_mean(tensor_image)
# normalization
        normalized_tensor_image = make_normalize_transform(t_mean, t_std)(tensor_image)

        return normalized_tensor_image

###############
# ImageFolder #
###############
import os
from torchvision.datasets import ImageFolder

def find_classes(root):
    class_to_idx = {}
    classes = os.listdir(root)
# order the classes
    classes.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))

    for i, cls in enumerate(classes):
        class_to_idx[cls.split(".")[0]] = i

    return classes, class_to_idx

def make_dataset(root, csv_root, class_to_idx):
## csv_file ##
    #csv_file = "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/ky/Demo_CODEX/_P2T1_roi_metadata.csv"
    imgs = []
# locate class name/folder and class index
    for class_name, class_idx in class_to_idx.items():
    # traverse all files in the folder
        #root_class = os.path.join(root, class_name)
        #file_name = class_name + ".pkl"
        #full_file_name = os.path.join(root, file_name)
        csv_name = class_name + ".csv"
        full_csv_name = os.path.join(csv_root, csv_name)
    # attach the class index to full file name
        #with open(full_file_name, 'rb') as f:
            #tmps = pickle.load(f)
        tmps = pd.read_csv(full_csv_name, usecols=['X_centroid','Y_centroid'])
        for i in range(len(tmps)):
            dummy_file_name = class_name + "_" + str(i)
            dummy_full_file_name = os.path.join(root, dummy_file_name)
            imgs.append((dummy_full_file_name, class_idx))

    return imgs

class MyImageFolder(ImageFolder):
    '''
    @args:
        classes      (list) class names
        class_to_idx (dict) (class_name:class_idx) 
        imgs         (list of tuple) (image_path, class_idx)
    '''
    def __init__(self, root, csv_root):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, csv_root, class_to_idx)
        self.root = root
        self.samples = imgs

    def __getitem__(self, index):
        sample, target = self.samples[index]

        return sample, target

    def __len__(self):
        return len(self.samples)

#################
# visualization #
#################
def save_image_from_chw_ndarray(arr, to_file, index):
    imwrite(to_file+"/sample"+str(index)+".tiff", arr.astype('uint8'), imagej=True, metadata={'axes':'CYX'})

def save_image_from_ndarray(arr, to_file, index):
    arr_chw = np.transpose(arr, (2, 0, 1))
    imwrite(to_file+"/sample"+str(index)+".tiff", arr_chw.astype('uint8'), imagej=True, metadata={'axes':'CYX'})

def save_image_from_tensor(tensor, to_file, index):
    imwrite(to_file+"/sample"+str(index)+".tiff", np.array(tensor.to(dtype=torch.uint8)), imagej=True, metadata={'axes':'CYX'})

def save_image_from_norm_tensor(tensor, to_file, t_mean, t_std, t_max, t_min, index):
    # tensor 24x*
    a = t_std.unsqueeze(-1).unsqueeze(-1)
    m = t_mean.unsqueeze(-1).unsqueeze(-1)
    ma = t_max.unsqueeze(-1)
    mi = t_min.unsqueeze(-1)
    # tensor 24x1x1
    t = (tensor*a+m) * (ma-mi) + mi
    imwrite(to_file+"/sample"+str(index)+".tiff", np.array(t.to(dtype=torch.uint8)), imagej=True, metadata={'axes':'CYX'})

class MyTransform_v3_vis(object):
    '''
    @args:
        keep_p            (float) default 0.5
        random_corp_size  (int)   default 224
    '''
    def __init__(self, random_crop_size, p_flip, p_color, p_gray, p_blurr, p_solar, path):
        self.random_crop_size = random_crop_size
        self.p_flip  = 1 - p_flip
        self.p_color = 1 - p_color
        self.p_gray  = 1 - p_gray
        self.p_blurr = 1 - p_blurr
        self.p_solar = 1 - p_solar
        self.path = path

    def __call__(self, image):
        idx = random.randint(0, 9)
#### path ####
        vis_path = self.path + "/"

        save_image_from_chw_ndarray(image, vis_path+"raw", idx)

# reshape: chw -> hwc
        hwc_image = image_method_reshape(image)

        save_image_from_ndarray(hwc_image, vis_path+"reshape", idx)

# resize
        resized_hwc_image = image_method_resize(hwc_image)

        save_image_from_ndarray(resized_hwc_image, vis_path+"resize", idx)

# random crop
        croped_hwc_image_1 = image_method_rand_crop(resized_hwc_image, self.random_crop_size)
        croped_hwc_image = randomly_apply_image_method(croped_hwc_image_1, image_method_horizontal_flip, self.p_flip)

        save_image_from_ndarray(croped_hwc_image, vis_path+"crop", idx)

# color jitting
        colored_hwc_image_1 = randomly_apply_image_method(croped_hwc_image, image_method_color, self.p_color)
        colored_hwc_image = randomly_apply_image_method(colored_hwc_image_1, image_method_gray_scale, self.p_gray)

        save_image_from_ndarray(colored_hwc_image, vis_path+"color", idx)

# global transfo
        blurred_hwc_image_1 = randomly_apply_image_method(colored_hwc_image, image_method_blur, self.p_blurr)
        blurred_hwc_image = randomly_apply_image_method(blurred_hwc_image_1, image_method_solarize, self.p_solar)

        save_image_from_ndarray(blurred_hwc_image, vis_path+"blurr", idx)

# to tensor: hwc -> chw
        t = torch.from_numpy(blurred_hwc_image.astype(float))
        tensor_image, t_max, t_min = make_scale_to_zero_one(t.permute(2, 0 ,1))

        #save_image_from_tensor(t.permute(2, 0 ,1), vis_path+"tensor", idx)
        #print(t.permute(2, 0 ,1))
        #print(tensor_image)

# get std and mean, per page
        t_std, t_mean = make_std_mean(tensor_image)

        #print(t_std)
        #print(t_mean)
        #print(t_max)
        #print(t_min)

# normalization
        normalized_tensor_image = make_normalize_transform(t_mean, t_std)(tensor_image)

        save_image_from_norm_tensor(normalized_tensor_image, vis_path+"tensor", t_mean, t_std, t_max, t_min, idx)
        #print(normalized_tensor_image)


        return normalized_tensor_image
