import time
import json
import sys
import os
 
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
import torchvision
from torchvision import datasets, models, transforms     # vision datasets, architectures & transforms 
from torchvision.utils import make_grid


# Process a PIL image for use in a PyTorch model
def process_image(image_path):
    ''' 
    This function opens the image using the PIL package, 
    applies the  necessery transformations and returns the image as a tensor ready to be fed to the network
    Arguments: The image's path
    Returns: The image as a tensor
    '''
    from PIL import Image
    img = Image.open(image_path)

    make_img_good = transforms.Compose([ # Do the same as the training data we will define a set of
        # transfomations that we will apply to the PIL image
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Preprocess the image, transform and make a tensor float
    tensor_image = make_img_good(img)
    
    return tensor_image


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    #image = image.transpose((1, 2, 0))
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

# Process a PIL image for use in a PyTorch model
def process_tensor_image(image_path):
    ''' 
    This function opens the image using the PIL package, 
    Scales, crops, and normalizes a PIL image for a PyTorch model
    Arguments: The image's path
    Returns: The image as a tensor
    '''
    from PIL import Image
    test_image = Image.open(image_path)

    transform_to = transforms.Compose([ # Do the same as the training data we will define a set of
        # transformations that we will apply to the PIL image
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Preprocess the image, transform and make a tensor float
    tensor_image = transform_to(test_image).float()
    # Add an extra batch dimension since pytorch treats all images as batches
    #image.unsqueeze_(0)
    
    return tensor_image  

   # Process a PIL image for use in a PyTorch model / returns a Numpy array
def process_npimage(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    transform = transforms.Compose([
        transforms.Resize(256),      
        transforms.CenterCrop(224), 
        transforms.ToTensor()])
    
    pil_image = Image.open(image)
    pil_image = transform(pil_image).float()
    
    np_image = np.array(pil_image)    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
            
    return np_image   # return as a np image

# process_npimage(image_path) 

