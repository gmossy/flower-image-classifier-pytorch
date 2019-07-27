#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date Last Modified:  05/12/2019
@author:  glenn
"""
#import utilities
import argparse
from time import time, strftime
import json
import random
import sys
import os
import logging

# impport ImageClassifier  programs
import utility
import setupmodel
import datasetprep
import processimage

# matplotlib and PIL imports
import numpy as np

# General pytorch imports:
import torch                           # root package
from torch.utils.data import Dataset   # dataset representation and loading

# pytorch vision imports
import torchvision
from torchvision import datasets, models, transforms     # vision datasets, architectures & transforms
from torchvision.utils import make_grid

# project imports
#from workspace_utils import active_session
#from __future__ import print_function, division
def get_input_args():
    parser = argparse.ArgumentParser(description='predict.py')
    #Define arguments
    parser.add_argument('checkpoint', type=str, help='Enter the Model checkpoint filename to use for prediction')
    parser.add_argument('--image_path', default='./flowers/test/1/image_06752.jpg', type = str, action="store", help="/path/to/image to process and predict")
    parser.add_argument('--test_dir', nargs='*', action="store", default="./flowers/test", type=str, help='/path/to/testdir data folder name   ')
    parser.add_argument("--r", action="store_true", required=False, default=False, help="randomly pick a flower image to process and then predict")   
    parser.add_argument('--top_k', default=5,  action="store", type=int, help='Returns the top k most likely classes')
    parser.add_argument('--category_names',  action="store", default='cat_to_name.json', type=str, help='json Mapping file used to map categories to real names' )
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--log', default='plog', type=str, help=' log processing information')
    parser.add_argument('--verbose', dest='verbose',action='store_true', default=True, help='Display additional processing information')
    parser.add_argument('--accuracy', dest='accuracy',action='store_true', help='Process test accuracy information')
    parser.print_help()
    return parser.parse_args()

# get a filename from the data directory
# reference:  https://stackoverflow.com/questions/31346593/how-to-list-all-directories-that-do-not-contain-a-file-type
def get_filepaths(directory):
    """
    This function will generate the file names in a directory  tree by walking the tree either top-down or bottom-up. For each
    directory in the tree rooted at directory top (including top itself),  it yields a 3-tuple (dirpath, dirnames, filenames).
    example:
    full_file_paths = get_filepaths(test_dir)
    print(full_file_paths[i])  # where i = n range of files in the directory
    image_path = full_file_paths[file_indx]
    /flowers/test/1/image_06752.jpg
    """
    file_paths = []  # List which will store all of the full filepaths.
    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.
    return file_paths  # list filepaths

# Get a random flower fullfilename from the a data directory
def pick_random_flower():
    # generate random integer
    ridx = random.randint(0,102)  #102 is the output_size
    return ridx

def get_randomimage(ridx, data_directory):
    get_filepaths(data_directory)
    full_file_paths = get_filepaths(data_directory)
    image_path = full_file_paths[ridx]
    return image_path

def get_mapping():
    with open('cat_to_name.json', 'r') as json_file:
       cat_to_name = json.load(json_file)
    return cat_to_name

def get_label(label, cat_to_name):
    try:
        return cat_to_name[label]
    except KeyError:
        return "unknown label"

# Implement the code to predict the class from an image file
# Create a function for prediction to Predict the class (or classes) of an image using a trained deep learning model.
def predict(image_path, model,  use_gpu, topk):
   # This method should take a path to an image and a model checkpoint, then return the probabilities and classes.
    # usage example:  probs, classes = predict(image_path, model,  use_gpu, in_args.top_k):
    #Predict the class (or classes) of an image using a trained deep learning model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # make a device switch, added for cli application
    model.eval()
    # use_gpu = False                               # move model and tensor to device, cpu or cuda, based on command line argument --gpu
    print(use_gpu)
    if torch.cuda.is_available():
        use_gpu = True
        model = model.cuda()
    else:
        model = model.cpu()
        
    np_array = processimage.process_npimage(image_path)
    tensor = torch.from_numpy(np_array)
    
    # Run image through model (from either device gpu or cpu) 
    if use_gpu:
        inputs = (tensor.float().cuda())
        # inputs = (tensor.cuda())
        # inputs = tensor.to(device)
    else:       
        inputs = tensor.float()
        # inputs = tensor.to(device)
        
    inputs = inputs.unsqueeze(0)           # unsqueeze the tensor image

    output = model.forward(inputs)        # run tensor through model
    
    # compute the top probabilities, and classes (labels)     
    ps = torch.exp(output).data.topk(topk)
    probabilities = ps[0].cpu() if use_gpu else ps[0]
    classes = ps[1].cpu() if use_gpu else ps[1]
    
    # invert the dictionary so you get a mapping from index to class 
    idx_to_class = {model.class_to_idx[k]: k for k in model.class_to_idx}
    top_classes = list()
    for label in classes.numpy()[0]:
        top_classes.append(idx_to_class[label])
        
    top_prob_array = probabilities.numpy()[0]  # get top_probs as numpy array
    return top_prob_array,  top_classes

def main():
    in_args = get_input_args()
    print("************************Start predict************************")
    if not in_args.r: print(" image_path:     = ", in_args.image_path)
    print(" random image filepick= ", in_args.r)
    print(" checkpoint:     = ", in_args.checkpoint)
    print(" top_k:          = ", in_args.top_k)
    print(" category_names  = {!r}".format(in_args.category_names))
    print(" gpu:            = {!r}".format(in_args.gpu))
    print("\n\n List the in.args: ")
    # print (in_args)

    data_dir = '/home/workspace/ImageClassifier/flowers'
    test_dir = in_args.test_dir

    # Check in_args.gpu and set use_gpu for GPU
    use_gpu = torch.cuda.is_available() and in_args.gpu
    # Check for GPU processing
    if in_args.verbose:
        print("Prediction on {} using {}".format( "GPU" if use_gpu else "CPU", in_args.checkpoint)) 
        
    if in_args.top_k >  len(in_args.category_names):
        print("top_k is out of range, it must be less than the number of labels") 
        sys.exit()    
    
    # create logger
    ti = str(time())[1:10]
    logfile = in_args.log
    logging.basicConfig(filename=logfile, filemode='a',  level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info('%s', 'predict logging started')
    # log selected arguments
    logging.info('%s %s %s %s %s %s %s',"checkpoint:", in_args.checkpoint," top_k: ",in_args.top_k, in_args.category_names,'use_gpu: ',  use_gpu)
     # Run the get_filepaths(directory) function and store its results in a variable.
    '''
    file_indx = 2
    full_file_paths = get_filepaths(test_dir)
    image_path = full_file_paths[file_indx]
    print (image_path)
    '''
    if in_args.r:
        ridx = pick_random_flower()
        image_path = get_randomimage(ridx, test_dir)
        print("Random image_path selection is: \t", image_path)
    else:
         image_path = in_args.image_path
         print("using the default image_path")
         print("The default or input path selection is: \t", in_args.image_path)

    print("The flower to be predicted is: ", image_path)
    #show the tensor for the chosen image , #process_image(image_path)  # put in a flower filename here and show the tensor image
    logging.info('%s %s  %s',"The flower to be predicted is: ", "image_path:",image_path)

    print("\nLoading the pretrained model from a Checkpoint-----------------------------")
    # Loads a pretrained model
    #model.class_to_idx =  image_datasets['training_dataset'].classes
    # dataloaders, trainloader, vloader, testloader, class_to_idx, dataset_sizes = datasetprep.load_and_transform(data_dir, 64)
    # model.class_to_idx = image_datasets['training_dataset'].class_to_idx
    # model.class_to_idx = testloader.class_to_idx
    model, arch, criterion, epochs, optimizer  =   setupmodel.load_checkpoint(in_args.checkpoint)
    print(arch)
    # get index to class mapping
    
    print("checkpoint loaded", arch, model.classifier.fc1.in_features,  model.classifier.fc2.out_features, model.classifier.fc2.in_features, model.classifier.fc1.out_features,"epochs: ", epochs)
    logging.info('%s %s %s %s %s %s %s', 'with the following trained model: ', arch, model.classifier.fc1.in_features,  model.classifier.fc2.out_features, model.classifier.fc1.out_features, "epochs: ", epochs)
    print('Pre-Trained with {} epochs.'.format(epochs))
    print(criterion)
    print(optimizer)
    print("\nCheckpoint Loaded------------------------------------")
    # Move tensors to GPU if available
    if use_gpu:
        model.cuda()
    # pause
    if not in_args.verbose: utility.pause()
    
    # Load category mapping dictionary
    use_mapping_file = False
    if len(in_args.category_names):
        with open(in_args.category_names, 'r') as f:
            cat_to_name = json.load(f)
            use_mapping_file = True

    cat_to_name = get_mapping()
    # make sure model class to idx mapping is saved to allow retrieval of indexs
    # model.class_to_idx = image_datasets['training_dataset'].class_to_idx
    
    print("Start Predict running")
    start_time = time()
    # Get prediction
    # This method should take a path to an image and a model checkpoint, then return the probabilities and classes.
    probs, classes = predict(image_path, model,  use_gpu, in_args.top_k)
    print("\n\n")
    print("\nThe Following Top {} Classes are predicted for '{}':".format(len(classes), image_path))
    print("probs: ",probs)
    print("classes ", classes)
    max_index = np.argmax(probs)
    max_label = classes[max_index]
    flower_name = cat_to_name[max_label]
    print("\n", "The top flower name prediction is: ", flower_name)
    # print("\nMost likely image class is '{}' with probability of {:.2f}".format(get_label(classes[0], cat_to_name)))
    # Now do some output and log formatting. number_of_results, use_mapping_file
    logging.info('%s %s %s', probs, classes, flower_name)
    logging.info('%s %s %s %s %s %s %s %s',"image_path:",image_path, "probs:",probs, " classes:", classes, "flower_name:",flower_name)
    print("\nPrediction completed-----------------------------")

    # Testing checkpoint
    if in_args.accuracy:
            dataloaders, trainloader, vloader, testloader, class_to_idx, dataset_sizes = datasetprep.load_and_transform(data_dir)
            # Test out your network, from video, 12:15/13:02 Inference
            #evaluate loaded_model from checkpoint
            print("evaluate loaded_model from checkpoint")
            setupmodel.check_accuracy(model, testloader)      # uses validation function 
            setupmodel.check_accuracy_on_testset(model, testloader) #uses (predicted == labels).sum()
    
    # Calculate and print overall runtime
    time_elapsed = time() - start_time
    print('Accuracy Testing Time_elapsed: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('%s','Accuracy Testing time_elapsed: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
# main program    
if __name__ == '__main__':
    main()

