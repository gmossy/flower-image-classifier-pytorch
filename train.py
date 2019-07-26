#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date Last Modified:  05/06/2019
@author:  glenn
"""
#cli project imports
import argparse
import utility
import setupmodel
import datasetprep
#
from time import time, strftime
import json
import sys
import os
import logging

# matplotlib imports
import numpy as np
import matplotlib.pyplot as plt        # (load and display images)
from PIL import Image
from collections import OrderedDict    #storing state_dict - all the weights and parameter

# General pytorch imports:
import torch                           # root package
print("pytorch", torch.__version__)    # print the current version of pytorch
from torch.utils.data import Dataset   # dataset representation and loading

# Neural Network API imports
import torch.autograd as autograd     # computation graph
from torch.autograd import Variable    # variable node in computation graph
from torch import nn                   # neural networks
import torch.nn.functional as F        # layers, activations and more, to use nn functions to # define your layers
from torch import optim                # optimizers e.g. gradient descent, ADAM, etc.
from torch.optim import lr_scheduler

# pytorch vision imports
import torchvision
from torchvision import datasets, models, transforms     # vision datasets, architectures & transforms
from torchvision.utils import make_grid

from workspace_utils import active_session
#from __future__ import print_function, division

# get command line arguments if required, if you use no argments, then all defaults will be used.
def get_input_args():
    parser =  argparse.ArgumentParser(description='train.py')
    #Define arguments Example use:  python train.py data_dir ./flowers --arch vgg16 --epochs 20 --dropout .5  --lr .001 --hidden_layers 4096,4096 --gpu True
    parser.add_argument('data_dir', default="./flowers", type=str,  nargs='*', help='Directory filepath to locate dataset')  #    data_dir = '/home/workspace/ImageClassifier/flowers', action="store"
    valid_archs = {'vgg16', 'vgg19','alexnet', 'densenet121', 'densenet161'}
    parser.add_argument('--arch', dest="arch", action="store", default="vgg19", type = str, choices=valid_archs, help='model architecture to use: vgg16 or 19, alexnet, densenet121 or 161')
    parser.add_argument('--save-dir', type=str, help='Save the trained model checkpoint to file')
    parser.add_argument('--hidden_layers', dest="hidden_layers",  nargs=2, action="store", type=int,  default = [4096,4096], help='enter 2 int values w comma: for ex. --hidden_layers 4096, 4096')
    parser.add_argument('--lr', default=0.001, type=float, help='the learning rate hyperparameter' )
    parser.add_argument('--scheduler', action='store_false', default=False, dest="schdlr", help='turn the scheduler on/off - off by default')
    parser.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5, type=float, help='dropout rate hyperparameter' )
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', dest="batch_size", action="store", type=int, default=32, help='number of images of training data that gets loaded at one time')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--log', default='tlog', type=str, help=' log processing information')
    parser.print_help()
    return parser.parse_args()
 
def main():
    debug = 'true'
    l = 1
    while l == 1 :
        in_args = get_input_args()
        print("***************Training Starting***************")
        print(" data_dir:       ", in_args.data_dir)
        print(" arch:           = {!r}".format(in_args.arch))
        print(" learning_rate:  = {!r}".format(in_args.lr))
        print(" scheduler:      = {!r}".format(in_args.schdlr))
        print(" dropout:        = {!r}".format(in_args.dropout))
        print(" hidden_layers:  = {!r}".format(in_args.hidden_layers))
        print(" epochs:         = {!r}".format(in_args.epochs))
        print(" batch_size      = {!r}".format(in_args.batch_size))
        print(" gpu:            = {!r}".format(in_args.gpu))
        print(" checkpoint:     = {!r}".format(in_args.save_dir))
        print(" log:            = {!r}".format(in_args.log))
        #print(in_args)
        if in_args.batch_size > 64 or in_args.batch_size < 1:
            print("--batch_size: must range from 1 to 64.")
            sys.exit(1)
    
        yn = str(input("Would you like to continue training with these choices Y/N?  "))
        if (yn == 'y' or 'Y' or 'YES' or 'yes'):l = 0
        else: 
            sys.exit(1)                 
    print("\nSet the directory, Path and Name for the Checkpoint-------------")
    if in_args.save_dir:
        # Create save directory if required
        if not os.path.exists(in_args.save_dir):
            os.makedirs(in_args.save_dir)
         # Save checkpoint in save directory
        chkpoint_filepath = in_args.save_dir + '/' + in_args.arch + '_checkpoint.pth'
    else:
        # Save checkpoint in current directory
        chkpoint_filepath = in_args.arch + '_checkpoint.pth'
   
    # create logger
    logfile = in_args.log
    logging.basicConfig(filename=logfile, filemode='a', level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info('train logging started')
    # log selected arguments
    logging.info('%s %s %s %s',"architecture:", in_args.arch,"checkpoint:", chkpoint_filepath)

    #check gpu and set a flag for it.
    device = setupmodel.gpu_check()
    print(device)
    if torch.cuda.is_available() and in_args.gpu:
           gpu = True
           print("gpu is ENABLED and set, Training on GPU")
    else: gpu = False
    print("device:= ", device, "and gpu is:", gpu, "in_args.gpu:", in_args.gpu)

    # Load the data and do the transforms for the training, validation, and testing sets
    print("\ncall load_and_transform-----------------------")
    # Set the default top level folder for the data
    data_dir = in_args.data_dir
    dataloaders, trainloader, vloader, testloader, class_to_idx, dataset_sizes = datasetprep.load_and_transform(in_args.data_dir, in_args.batch_size)

    # create a mapping from the label number and the actual flower name.
    cat_to_name = datasetprep.map_catalog_to_name()

    if debug:
      # Explore the current batch, ids and labels
      print("Now show only data batch tensor, ids, and labels")
      inputs, labels = next(iter(dataloaders['training']))
      print(inputs.size())   # gets the batch tensor info
      print(labels)

    print("\nData load_and_transforms completed-----------------------------------")
    logging.info('%s',"Data load_and_transforms completed-------------------------")

    print("\nget the model and features sizes-----------------------",in_args.arch)
    model, input_size =  setupmodel.get_model(in_args.arch)
    # print out the model information
    if debug:
        print("architecture:", in_args.arch)
        output_size = len(class_to_idx)
        print("output_size:= ", output_size)
        print("input_size:= ", input_size)
        print("output_size= ", len(class_to_idx))
        print("hidden_layers:  = {!r}".format(in_args.hidden_layers))
        learning_rate = in_args.lr
        print("hyperparameters:")
        print("batch_size: =",in_args.batch_size,"epochs: =",in_args.epochs,"dropout: =", in_args.dropout, "learning_rate= ", learning_rate)
        # How to load and view all the class indexes, and then get the output_size of the dataset:
        #model.class_to_idx = image_datasets['training_dataset'].class_to_idx
        #model.class_to_idx = trainloader.class_to_idx
        # print(model.class_to_idx)
    logging.info('%s %s %s %s %s %s %s %s',"architecture:", in_args.arch, "input_size:= ", input_size, "output_size:= ", output_size, "hidden_layers: ", in_args.hidden_layers )
    logging.info('%s %s %s %s %s %s %s %s',"batch_size: =",in_args.batch_size,"epochs: =",in_args.epochs,"dropout: =", in_args.dropout, "learning_rate= ", learning_rate)
    print("\n apply hyperparameters and run the classifier-----------------------")
    # Create the classifier
    print("\nSetting Neural Network / create Classifer------")
    print('class_to_idx: ', class_to_idx)
    model , criterion, optimizer = setupmodel.create_classifier(model, input_size, in_args.hidden_layers, output_size, learning_rate, in_args.dropout, class_to_idx)
    print('criterion=', criterion)
    logging.info('%s %s %s %s %s %s %s %s',"learning_rate: =",in_args.lr,"hidden_layers: =",in_args.hidden_layers," batch_size: =", in_args.batch_size,"checkpoint:", chkpoint_filepath)
    print("model", model)

    print("\nTraining Neural Network------------------------")
    # Model Training, train the final layers, also we will get an idea of how well the training is working
    # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#load-data
    # Initialize with the hyperparameters
    start_time = time()
    epochs = in_args.epochs
    print("Network architecture:", in_args.arch)
    print("Number of epochs:    ", in_args.epochs)  # Number of epochs  to train for
    print('Learning rate:       ', in_args.lr)
    print("dropout:=            ", in_args.dropout)
    print('device=              ', device)
    # print("schedular not active")
    logging.info('%s %s %s %s',"architecture:", in_args.arch, "Number of epochs: ", in_args.epochs )
    # Train the network
    print("\nTrain the network---")
    logging.info('%s'," Training Starting-----")
    
    from workspace_utils import keep_awake
    for i in keep_awake(range(1)): 
        print("active session started")
    # The training loss, validation loss, and validation accuracy are printed out as a network trains
        model, criterion, optimizer = setupmodel.train_model(model, criterion, optimizer, epochs, trainloader, vloader)
        logging.info('%s'," Training Completed-----")
    
        # Save trained model
        print("\n Save the checkpoint -------------------")
        # Create `class_to_idx` attribute in model before saving to checkpoint
        # model.class_to_idx = image_datasets['training_dataset'].class_to_idx
        setupmodel.save_checkpoint(model, chkpoint_filepath, in_args.arch, in_args.epochs, criterion, optimizer)
        print('Model saved at {}'.format(chkpoint_filepath))
        logging.info('%s %s %s %s  %s'," checkpoint saved-----",in_args.arch, model, chkpoint_filepath,  in_args.epochs)
        print("\n checkpoint saved-------------------")

    # Calculate and print overall runtime
    time_elapsed = time() - start_time
    print('Training time_elapsed: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('%s','Training time_elapsed: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

# main program
if __name__ == '__main__':
    main()
