#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import json
import sys
import os
import random

# matplotlib and PIL imports
import numpy as np
import matplotlib.pyplot as plt        # (load and display images)
# import matplotlib.gridspec as gridspec
# from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
from collections import OrderedDict    #storing state_dict - all the weights and parameter


# General pytorch imports:
import torch                           # root package

from torch.utils.data import Dataset   # dataset representation and loading

# Neural Network API imports
#import torch.autograd as autograd     # computation graph
from torch.autograd import Variable    # variable node in computation graph               
from torch import nn                   # neural networks
import torch.nn.functional as F        # layers, activations and more, to use nn functions to # define your layers
from torch import optim                # optimizers e.g. gradient descent, ADAM, etc.
# from torch.optim import lr_scheduler

# pytorch vision imports
import torchvision                     #enables use of CNN neural nets, ResNet VGG and other pretrained models
from torchvision import datasets, models, transforms     # vision datasets, architectures & transforms 
from torchvision.utils import make_grid
#cli project imports
#import utility


# create a mapping from the label number (some number between 1 and 102) and the actual flower name. 
def map_catalog_to_name():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name 
        
def load_and_transform(data_dir, batch_size=32):
    """
        Creates pytorch training, validation and testing pytorch dataloaders and applies transformations
        respectively, and then through our network for training, testing and prediction. Uses ImageFolder and Dataloader
        Parameters:
            data_dir - Path to data to be used
        Returns:
            training - Normalized training data loader with random crops, flipping and resizing applied
            testing - Normalized testing data loader with fixed cropping and resizing
            validation - Normalized validation data loader with fixed cropping and resizing
    """
    # set the data filepaths for the pictures and data
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #The images have large scale, pose and light variations.
    #In addition, there are categories that have large variations within the category and several very similar categories.
    #The dataset is visualized using isomap with shape and colour features.

    # Define your transforms for the training, validation, and testing sets
    # Data augmentation and normalization for training
    # Just normalization for validation
    norm_mean = [0.485, 0.456, 0.406]
    norm_std  = [0.229, 0.224, 0.225]
    data_transforms = {
    # For the training, you'll want to apply transformations such as random scaling, cropping, and flipping
    # The validation and testing sets are used to measure the model's performance on data it hasn't seen yet.
    # For this you don't want any scaling or rotation transformations,
    # but you'll need to resize then crop the images to the appropriate size.
    # training - Normalized training data loader with random crops, flipping and resizing applied
    # validation - Normalized validation data loader with fixed cropping and resizing
    # testing - Normalized testing data loader with fixed cropping and resizing
    'training_transforms' :    transforms.Compose([
            transforms.RandomRotation(25),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),
    'validation_transforms' :  transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),
    'testing_transforms'     :  transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])
    }
    print("transformations completed")
    # Load the datasets with ImageFolder
    print("Initializing Datasets and Dataloaders...")
    # Use ImageFolder and Dataloader 
    # Load the datasets with ImageFolder - make sure data is in flowers folder.
    image_datasets = {
        'training_dataset'    : datasets.ImageFolder(train_dir, transform=data_transforms['training_transforms']),
        'validation_dataset'  : datasets.ImageFolder(valid_dir, transform=data_transforms['validation_transforms']),
        'testing_dataset'     : datasets.ImageFolder(test_dir,  transform=data_transforms['testing_transforms'])
    }
    # Using the image datasets and the trainforms, define the dataloaders
    # Batch size for training
    
    # The training, validation, testing returns images and labels(labels are the class that the image belongs to)
    dataloaders = {
        'training'    : torch.utils.data.DataLoader(image_datasets['training_dataset'], batch_size, shuffle=True ),
        'validation'  : torch.utils.data.DataLoader(image_datasets['validation_dataset'], batch_size, shuffle=True ),
        'testing'     : torch.utils.data.DataLoader(image_datasets['testing_dataset'], batch_size, shuffle=True )
    }
    trainloader      = dataloaders['training']
    vloader          = dataloaders['validation']
    testloader       = dataloaders['testing']

    dataset_sizes = {x: len(image_datasets[x]) for x in ['training_dataset', 'validation_dataset', 'testing_dataset']}
    training_set_percent = len(testloader)/len(trainloader) * 100
    print ("dataloading complete:/ sizes:", dataset_sizes, 'batch_size:', batch_size)
    # Get the class ids and Store class_to_idx into a model property
    class_to_idx = image_datasets['training_dataset'].class_to_idx
    # print("class ids as a list: ", class_to_idx)

    return dataloaders, trainloader, vloader, testloader, class_to_idx, dataset_sizes
