import numpy as np
import cupy as cp
import pandas as pd
import cuml
from tqdm import tqdm

from mydataset import MyDataset, MyFineTuneDataset, MyPKLDataset, MyTransform
from myutils import save_tiff_from_tensor, save_tiff_from_tensor_one_batch_v2

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.multiprocessing import set_start_method

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
import math
from tifffile import imread, imwrite
import pickle

import time

from dinov2.models.vision_transformer import DinoVisionTransformer

#############
# zero-shot #
#############
def zero_shot_inference(model, dev, data_list_file, transform):
# set device
    device = torch.device(dev)
# set start method if use workers >= 2
    #set_start_method('spawn')
# get features
    features = []
# load model
    # from url
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
# load data
    data_set = MyDataset(data_list_file, transform)
    data_loader = DataLoader(data_set, batch_size=100, shuffle=False, num_workers=0)
# inference
    with torch.no_grad():
        for i, data in tqdm(enumerate(data_loader)):
            image, idx = data
            feature = dinov2_model(image)
            feature /= feature.norm(dim=-1, keepdim=True)
            feature = feature.tolist()
            features.extend(feature)

    return features

def zero_shot_visualization(model, dev, data_list_file, transform):
    '''
    @notes:
        The keys of feature_dict are: ['x_norm_clstoken', 'x_norm_patchtokens', 'x_prenorm', 'masks']
        The size of 'x_norm_patchtokens' is patch_num x (inputs_size//patch_size)^2  x dim, 3D
        The size of features for PCA is    (patch_num x (inputs_size//patch_size)^2) x dim, 2D
    '''
# set device
    device = torch.device(dev)
# set start method if use workers >= 2
    #set_start_method('spawn')
# get features 
    features = []
# load model
    # from url
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
# check model structure
    print(dinov2_model)
# load data
    data_set = MyDataset(data_list_file, transform)
    data_loader = DataLoader(data_set, batch_size=100, shuffle=False, num_workers=0)
# get features
    with torch.no_grad():
        for i, image in tqdm(enumerate(data_loader)):
            feature_dict = dinov2_model.forward_features(image)
            feature = feature_dict['x_norm_patchtokens']
            features.append(feature)
    features = torch.cat(features, dim=0).cpu()
# prepare to reshape the Tensor
    patch_num = features.shape[0]
    patch_h = patch_w = 224 // 14 # according to input size(224) and vits14 patch size(14)
    dim = features.shape[2]
# PCA
    features = features.reshape(patch_num*patch_h*patch_w, dim) # patch_num x 16 x 16 x 384, 3D -> 2D
    pca = PCA(n_components=3)
    pca.fit(features)
    pca_features = pca.transform(features) # (patch_num x 16 x 16) x n_components, 2D 
## histogram ##
    plt.subplot(2, 2, 1)
    plt.hist(pca_features[:, 0])
    plt.subplot(2, 2, 2)
    plt.hist(pca_features[:, 1])
    plt.subplot(2, 2, 3)
    plt.hist(pca_features[:, 2])
    plt.show()
## min max scale ##
    img_c = 2
    pca_features[:, img_c] = (pca_features[:, img_c] - pca_features[:, img_c].min()) / \
                         (pca_features[:, img_c].max() - pca_features[:, img_c].min())
# save to images
    img_dir = "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/ky/mini_pipeline/visualization/features"
    for i in range(patch_num):
        img_name = img_dir + "/img_comp" + str(img_c) + "_" + str(i) + ".png"
        matplotlib.image.imsave(img_name, pca_features[i*patch_h*patch_w:(i+1)*patch_h*patch_w, img_c].reshape(patch_h, patch_w))
## background ##
    pca_features_background = pca_features[:, img_c] > 0.5 # threshold, according to histogram(?)
    pca_features_foreground = ~pca_features_background
# save to images
    img_dir = "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/ky/mini_pipeline/visualization/masks"
    for i in range(patch_num):
        img_name = img_dir + "/img_comp" + str(img_c) + "_" + str(i) + ".png"
        matplotlib.image.imsave(img_name, pca_features_foreground[i*patch_h*patch_w:(i+1)*patch_h*patch_w].reshape(patch_h, patch_w))
## foreground RGB ##
    img_dir = "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/ky/mini_pipeline/visualization/RGB_features"
    pca.fit(features[pca_features_foreground])
    pca_features_left = pca.transform(features[pca_features_foreground])
    for i in range(3):
        # min_max scaling
        pca_features_left[:, i] = (pca_features_left[:, i] - pca_features_left[:, i].min()) / \
                                  (pca_features_left[:, i].max() - pca_features_left[:, i].min())
    pca_features_rgb = pca_features.copy()
    # for black background
    pca_features_rgb[pca_features_background] = 0
    # new scaled foreground features
    pca_features_rgb[pca_features_foreground] = pca_features_left
    # reshaping to numpy image format
    pca_features_rgb = pca_features_rgb.reshape(patch_num, patch_h, patch_w, 3)
    for i in range(patch_num):
        img_name = img_dir + "/img_comp" + str(img_c) + "_" + str(i) + ".png"
        matplotlib.image.imsave(img_name, pca_features_rgb[i])

###############
# fine-tuning #
###############
class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self, vit_model, features_in, categories_out):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = vit_model
        # simple linear layer for classification
        self.classifier = nn.Sequential(
            nn.Linear(features_in, 256),
            nn.ReLU(),
            nn.Linear(256, categories_out)
        )

    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x

def fine_tuned_inference(dinov2_model, dev, data_list_file, transform, fine_tune_labels, to_file):
# set device
    device = torch.device(dev)
# set parameters
    features_in = 384
    categories_out = 5
    num_epochs = 10
# set data
    data_set = MyFineTuneDataset(fine_tune_labels, transform, categories_out, label_encode)
    data_loader = DataLoader(data_set, batch_size=10, shuffle=False, num_workers=0)
# set model
    vit_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model = DinoVisionTransformerClassifier(vit_model, features_in, categories_out).to(device)
# set fine-tuning configs
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.000001)
# start to fine-tuning
    print("Start fine-tuning...")
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, data in tqdm(enumerate(data_loader)):
            optimizer.zero_grad()
# forward
            inputs, labels = data
            outputs = model(inputs)
# loss
            targets = labels.to(device)
            loss = criterion(outputs, targets)
# backward
            loss.backward()
# update
            optimizer.step()
# check convergency
            total_loss += loss.item()
        print("Average loss: " + str(total_loss / len(data_loader)))
    print("End fine-tuning...")
# save fine-tuned model
    torch.save(model, to_file)
    print("The fine_tuned model is saved to " + to_file + "...")
# validation
    print("Start validation...")
    ## load model
    fine_tuned_model = torch.load(to_file)
    ## load test data
    test_set = MyDataset(data_list_file, transform)
    test_loader = DataLoader(test_set, batch_size=10, shuffle=False, num_workers=0)
    ## get features
    features = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader)):
            image, idx = data
            feature = fine_tuned_model(image)
            feature /= feature.norm(dim=-1, keepdim=True)
            feature = feature.tolist()
            features.extend(feature)
    ## check results
    idxs = range(40, 80, 10)
    for idx in idxs:
        print(features[idx])

############
# training #
############
def forward_student_and_teacher(data, student, teacher, temperature):
# set Softmax
    m = nn.Softmax()
# student
    student_image = data
    ps = student(student_image)
    ps = m(ps)
# teacher
    teacher_image = data
    pt = teacher(teacher_image)
    pt = pt / temperature
    pt = m(pt)

    return ps, pt

def backward_student(loss, student, teacher):
    loss.backward()

def update_student(optimizer):
    optimizer.step()

def update_teacher(m, student, teacher):
    with torch.no_grad():
        for param_q, param_k in zip(student.parameters(), teacher.parameters()):
            param_k.data.mul_(m).add_((1-m)*param_q.detach().data)

def do_train_student(model, dev, data_list_file, transform, to_file):
# set device
    device = torch.device(dev)
# load student and teacher, use vits14
    student = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
    teacher = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
    embed_dim = 384
# load data
    data_set = MyDataset(data_list_file, transform)
    data_loader = DataLoader(data_set, batch_size=10, shuffle=False, num_workers=0)
# set training configs
    learning_rate = 0.000001
    temperature = 0.1
    momentum = 0.9
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(student.parameters(), lr=learning_rate, momentum=momentum)
# train student
    num_epochs = 100
    for i in range(num_epochs):
        total_loss = 0.0
        for i, data in tqdm(enumerate(data_loader)):
            optimizer.zero_grad()
            inputs, idxs = data
# forward, get ps and pt
            ps, pt = forward_student_and_teacher(inputs, student, teacher, temperature)
# loss, crossEntropyLoss of ps and pt
            loss = criterion(ps, pt)
# backward, student only
            backward_student(loss, student, teacher)
# update, student and teacher
            update_student(optimizer)
            update_teacher(momentum, student, teacher)
# logging
            total_loss += loss.item()
        print("Average loss: " + str(total_loss / len(data_loader)))
# save student
    torch.save(student, to_file)
    print("The model is saved to " + to_file + "...")

def trained_inference(model, dev, data_list_file, transform, to_file):
# train
    print("Start training...")
    do_train_student(model, dev, data_list_file, transform, to_file)
# validation
    print("Start validation...")
    ## load model
    trained_model = torch.load(to_file)
    ## load test data
    test_set = MyDataset(data_list_file, transform)
    test_loader = DataLoader(test_set, batch_size=10, shuffle=False, num_workers=0)
    ## get features
    features = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader)):
            image, idx = data
            feature = trained_model(image)
            feature /= feature.norm(dim=-1, keepdim=True)
            feature = feature.tolist()
            features.extend(feature)

    #print(features)

############
# pkl ver. #
############
def zero_shot_inference_v2(model, dev, data_list_file, transform):
# set device
    device = torch.device(dev)
# set start method if use workers >= 2
    #set_start_method('spawn')
# get features
    features = []
# load model
    # from url
    #dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
    # from path
    dinov2_model = torch.load(model)
# load data
    data_set = MyPKLDataset(data_list_file, transform)
    data_loader = DataLoader(data_set, batch_size=100, shuffle=False, num_workers=0)
# inference
    with torch.no_grad():
        for i, image in tqdm(enumerate(data_loader)):
            feature = dinov2_model(image)
            feature /= feature.norm(dim=-1, keepdim=True)
            feature = feature.tolist()
            features.extend(feature)

    return features

def zero_shot_visualization_v2(model, dev, data_list_file, transform):
    '''
    @notes:
        The keys of feature_dict are: ['x_norm_clstoken', 'x_norm_patchtokens', 'x_prenorm', 'masks']
        The size of 'x_norm_patchtokens' is patch_num x (inputs_size//patch_size)^2  x dim, 3D
        The size of features for PCA is    (patch_num x (inputs_size//patch_size)^2) x dim, 2D
    '''
# set device
    device = torch.device(dev)
# set start method if use workers >= 2
    #set_start_method('spawn')
# get features 
    features = []
# load model
    # from url
    #dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
    # from path
    dinov2_model = torch.load(model)
# check model structure
    print(dinov2_model)
# load data
    data_set = MyPKLDataset(data_list_file, transform)
    data_loader = DataLoader(data_set, batch_size=100, shuffle=False, num_workers=0)
# get features
    with torch.no_grad():
        for i, image in tqdm(enumerate(data_loader)):
            if i == 0:
                save_tiff_from_tensor_one_batch_v2(image, 100, transform.mean, transform.std, \
                                                   "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/ky/mini_pipeline/visualization/dino_inputs/input_")
            feature_dict = dinov2_model.forward_features(image)
            feature = feature_dict['x_norm_patchtokens']
            features.append(feature)
    features = torch.cat(features, dim=0).cpu()
# number of images to be saved
    num = 100
# prepare to reshape the Tensor
    patch_num = features.shape[0]
    inputs_size = 224 #central croped image size
    patch_size = 14 # patch size of vits14 
    patch_h = patch_w = inputs_size // patch_size # patch_h == patch_w == 224//14 = 16
    dim = features.shape[2]
# PCA
    features = features.reshape(patch_num*patch_h*patch_w, dim) # patch_num x 16 x 16 x 384, 3D -> 2D
    pca = PCA(n_components=3)
    pca.fit(features)
    pca_features = pca.transform(features) # (patch_num x 16 x 16) x n_components, 2D
## histogram ##
    #plt.subplot(2, 2, 1)
    #plt.hist(pca_features[:, 0])
    #plt.subplot(2, 2, 2)
    #plt.hist(pca_features[:, 1])
    #plt.subplot(2, 2, 3)
    #plt.hist(pca_features[:, 2])
    #plt.show()
## min max scale ##
    img_c = 0
    pca_features[:, img_c] = (pca_features[:, img_c] - pca_features[:, img_c].min()) / \
                         (pca_features[:, img_c].max() - pca_features[:, img_c].min())
# save to images
    img_dir = "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/ky/mini_pipeline/visualization/features"
    for i in range(num):
        img_name = img_dir + "/img_comp" + str(img_c) + "_" + str(i) + ".png"
        matplotlib.image.imsave(img_name, pca_features[i*patch_h*patch_w:(i+1)*patch_h*patch_w, img_c].reshape(patch_h, patch_w))
## background ##
    pca_features_background = pca_features[:, img_c] > 0.3 # threshold, according to histogram(?)
    pca_features_foreground = ~pca_features_background
# save to images
    img_dir = "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/ky/mini_pipeline/visualization/masks"
    for i in range(num):
        img_name = img_dir + "/img_comp" + str(img_c) + "_" + str(i) + ".png"
        matplotlib.image.imsave(img_name, pca_features_foreground[i*patch_h*patch_w:(i+1)*patch_h*patch_w].reshape(patch_h, patch_w))
## foreground RGB ##
    img_dir = "/archive/bioinformatics/Jamieson_lab/shared/spatial-core/ky/mini_pipeline/visualization/RGB_features"
    pca.fit(features[pca_features_foreground])
    pca_features_left = pca.transform(features[pca_features_foreground])
    for i in range(3):
        # min_max scaling
        pca_features_left[:, i] = (pca_features_left[:, i] - pca_features_left[:, i].min()) / \
                                  (pca_features_left[:, i].max() - pca_features_left[:, i].min())
    pca_features_rgb = pca_features.copy()
    # for black background
    pca_features_rgb[pca_features_background] = 0
    # new scaled foreground features
    pca_features_rgb[pca_features_foreground] = pca_features_left
    # reshaping to numpy image format
    pca_features_rgb = pca_features_rgb.reshape(patch_num, patch_h, patch_w, 3)
    for i in range(num):
        img_name = img_dir + "/img_comp" + str(img_c) + "_" + str(i) + ".png"
        matplotlib.image.imsave(img_name, pca_features_rgb[i])

######################
# multi-channel ver. #
######################
def zero_shot_inference_v3(model, dev, data_list_file, transform):
# set device
    device = torch.device(dev)
# set start method if use workers >= 2
    #set_start_method('spawn')
# get features
    features = []
# load model
    # from url
    #dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
    # from path
    #dinov2_model = torch.load(model)
    ## from state_dict
    #### change size of patch embed
    dinov2_model = DinoVisionTransformer(img_size=224, patch_size=14, in_chans=4, embed_dim=384)
    #### exclude the weights of patch embed
    tmp_state_dict = torch.load(model)
    trimed_state_dict = {k : v for k, v in tmp_state_dict.items() if k.startswith('blocks')}
    #### match the keys of blocks
    trimed_state_dict_v2 = {}
    for k, v in trimed_state_dict.items():
        k2 = k.replace('blocks.', 'blocks.0.')
        trimed_state_dict_v2[k2] = v
    #### ignore the unmatched keys
    #dinov2_model.load_state_dict(trimed_state_dict_v2)
    dinov2_model.load_state_dict(trimed_state_dict_v2, strict=False)
    dinov2_model.to(device)
# check
    print("ViT-DINO:")
    print(dinov2_model.state_dict().keys())
    print("vits14:")
    print(trimed_state_dict_v2.keys())
    print(dinov2_model)
# load data
    data_set = MyPKLDataset(data_list_file, transform)
    data_loader = DataLoader(data_set, batch_size=100, shuffle=False, num_workers=0)
# inference
    with torch.no_grad():
        for i, image in tqdm(enumerate(data_loader)):
            feature = dinov2_model(image)
            feature /= feature.norm(dim=-1, keepdim=True)
            feature = feature.tolist()
            features.extend(feature)

    return features

def trained_inference_v3_step_1(model, dev, to_file):
# set device
    device = torch.device(dev)
# set start method if use workers >= 2
    #set_start_method('spawn')
# load model
    # from url
    #dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
    # from path
    #dinov2_model = torch.load(model)
    ## from state_dict
    #### change size of patch embed
    dinov2_model = DinoVisionTransformer(img_size=224, patch_size=14, in_chans=4, embed_dim=384)
    #### exclude the weights of patch embed
    tmp_state_dict = torch.load(model)
    trimed_state_dict = {k : v for k, v in tmp_state_dict.items() if k.startswith('blocks')}
    #### match the keys of blocks
    trimed_state_dict_v2 = {}
    for k, v in trimed_state_dict.items():
        k2 = k.replace('blocks.', 'blocks.0.')
        trimed_state_dict_v2[k2] = v
    #### ignore the unmatched keys
    #dinov2_model.load_state_dict(trimed_state_dict_v2)
    dinov2_model.load_state_dict(trimed_state_dict_v2, strict=False)
    dinov2_model.to(device)
# check
    print("ViT-DINO:")
    print(dinov2_model.state_dict().keys())
    print("vits14:")
    print(trimed_state_dict_v2.keys())
    print(dinov2_model)
# save model to file - GPU ver.
    ## change to_file for saving the checkpoint with modified inputs
    tmp_to_file = to_file[:-4] + "_ckpt.pth"
    torch.save(dinov2_model, tmp_to_file)
    print("The model is saved to " + tmp_to_file + "...")

def trained_inference_v3_step_2(data_list_file, transform, to_file):
## change to_file for testing
    if to_file:
        #tmp_to_file = to_file
        tmp_to_file = to_file[:-4] + "_ckpt.pth"
# get features
    features = []
# load model
    print("The model is loaded from " + tmp_to_file + "...")
    trained_model = torch.load(tmp_to_file)
# load data
    data_set = MyPKLDataset(data_list_file, transform)
    data_loader = DataLoader(data_set, batch_size=100, shuffle=False, num_workers=0)
# inference
    with torch.no_grad():
        for i, image in tqdm(enumerate(data_loader)):
            feature = trained_model(image)
            feature /= feature.norm(dim=-1, keepdim=True)
            feature = feature.tolist()
            features.extend(feature)

    return features

def trained_inference_v3(model, dev, data_list_file, transform, to_file):
    '''
    @call after training
    @inference only
    @load model from /archive/bioinformatics/Jamieson_lab/shared/spatial-core/ky/Demo_dinov2/outputs/eval
    '''
# set device
    device = torch.device(dev)
# use vitb14
    trained_model = DinoVisionTransformer(img_size=224, patch_size=14, in_chans=4, embed_dim=768)
# get features
    features = []
# load model
    print("The model is loaded from " + model + "...")
    trained_model.load_state_dict(torch.load(model), strict=False)
    trained_model.to(device)
# load data
    data_set = MyPKLDataset(data_list_file, transform)
    data_loader = DataLoader(data_set, batch_size=100, shuffle=False, num_workers=0)
# inference
    with torch.no_grad():
        for i, image in tqdm(enumerate(data_loader)):
            feature = trained_model(image)
            feature /= feature.norm(dim=-1, keepdim=True)
            feature = feature.tolist()
            features.extend(feature)

    return features
