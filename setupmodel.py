"""
Date Last Modified:  05/12/2019
@author:  glenn
"""
# General pytorch imports:
#cli project imports
from time import time
import logging

# matplotlib imports
import numpy as np
import matplotlib.pyplot as plt        # (load and display images)
from PIL import Image
from collections import OrderedDict    #storing state_dict - all the weights and parameter

# General pytorch imports:
import torch                           # root package
from torch.utils.data import Dataset   # dataset representation and loading

# Neural Network API imports
import torch.autograd as autograd     # computation graph
from torch.autograd import Variable    # variable node in computation graph
from torch import nn                   # neural networks
import torch.nn.functional as F        # layers, activations and more, to use nn functions to # define your layers
from torch import optim                # optimizers e.g. gradient descent, ADAM, etc.
# from torch.optim import lr_scheduler

# pytorch vision imports
import torchvision                     #enables use of CNN neural nets, ResNet VGG and other pretrained models
from torchvision import datasets, models, transforms     # vision datasets, architectures & transforms

def gpu_check():
    # Check for gpu or cuda, then set parameters, and provide info
    # reference:  https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:",device)
    print("GPU {}".format("Enabled" if torch.cuda.is_available() else "Disabled"))
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    #Additional Info when using cuda
    if device.type == 'cuda':
        print("{}  {}:  {}{}{}{}{}{}".format(torch.cuda.get_device_name(0),
        '  Memory Usage-->','Allocated: ',1024,' GB',' Cached: ',round(torch.cuda.memory_cached(0)/1024**3,1),' GB'))
    return device

# get_model
# Reference:  https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
# Downloads and returns model provided. Returns model architecture for the CNN and the associated input_size.
def get_model(arch):
    # Initialize the variables, for which are model specific, they are from the torchvision library
    # the models are pre-trained models from the ImageNet Challenge
    # https://en.wikipedia.org/wiki/ImageNet#ImageNet_Challenge
    #Parameters:
    #   arch - Used to select which architecture to use for prepare
    # Returns:
    #    model[arch] - selects the variable out of a dictionary and returns the
    #         model associated with arch
    #    input_size[arch] - selects the associated input size for the model selected
    # vgg16 = models.vgg16(pretrained=True)            # in_features=25088
    vgg16=''
    vgg19=''
    alexnet=''
    densenet161=''
    densenet121=''
    if arch == 'vgg16':
        vgg16 = models.vgg16(pretrained=True)
    elif arch == 'vgg19':
        vgg19 = models.vgg19(pretrained=True)
    elif arch == 'alexnet':
        alexnet = models.alexnet(pretrained=True)
    elif arch == 'densenet121':
        densenet121 = models.densenet121(pretrained=True)
    elif arch == 'densenet161':
        densenet161 = models.densenet161(pretrained=True)
    else:
        print('{} architecture not recognized..use vgg16 / 19, alexnet, densenet121 / 161', format(arch))
        exit()
    model_select = {
          "vgg16":vgg16,
          "vgg19":vgg19,
          "alexnet":alexnet,
          "densenet161":densenet161,
          "densenet121":densenet121
     }
    input_size = {
          "vgg16":25088,
          "vgg19":25088,
          "alexnet":9216,
          "densenet161":2208,
          "densenet121":1024
     }
    return model_select[arch], input_size[arch]

# Build the network, Create classifier
def create_classifier(model, input_size, hidden_layers, output_size, learning_rate, dropout, class_to_idx):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # make a device switch, added for cli application
    """
    Takes a pretrained CNN, freezes the features and creates a untrained classifier. Returns
    model with an untrained classifier, loss function critierion (NLLLoss) and Adam optimizer.
    Parameters:
        model - Pretrained CNN, i'm going to use vgg19 as a default
        input_size -    integer, size of the input layer
        hidden_layers - list of hidden_layer sizes, get via indexing
        output_size -   integer, size of the output layer
        learning_rate (lr) - determines the learning rate for the optimizer
        dropout - determines the dropout probability for the classifier(default- 0.5)
        class_to_idx - maps class to index of the labels
    Returns:
        model - Pretrained feature CNN with untrained classifier  # should change this to classifier instead of model
        criterion - loss function to train on (torch.nn.NLLLoss())
        optimizer - optimizer for new, untrained classifier (torch.optim.Adam)
        schedular - decay adj the lr at (n)step size epochs (optional)
    """
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():  # To use an existing model is equivalent to freeze some
        param.requires_grad = False   # of its layers and parameters and not train those.

    #classifier_input_size = model.classifier.in_features
    print("Input size: ",  input_size)
    print("hidden layers", hidden_layers)
    print("output_size",   output_size)    # = number of classes = 102 for flower data
    print("dropout", dropout)

    classifier = nn.Sequential(OrderedDict([
             ('fc1', nn.Linear(input_size,  hidden_layers[0])), # fully connected layer,( "size" - in_features, out_features, bias)
             ('relu', nn.ReLU()),                                 # the activation layer
             ('dropout',nn.Dropout(dropout)),
             ('fc2', nn.Linear( hidden_layers[1], output_size)),
             ('output', nn.LogSoftmax(dim=1))
             ]))

    #Apply new classifier and generate criterion and optimizer
    model.classifier = classifier   # Now replace the classifier portion of the model
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Save class to index mapping
    model.class_to_idx = class_to_idx
    model = model.to(device)              # send model to device

    return model , criterion, optimizer

# ref https://stackoverflow.com/questions/52176178/pytorch-model-accuracy-test/52178638
# Function to do Test Valication and get test loss and accuracy
def validation(model, criterion, data_loader):
    print(" Now validating mode, getting validation_loss, accuracy")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # make a device switch, added for cli application
    model.to(device)
    model.eval()    # Set model to evaluate mode,  turns dropout off, we don't want it on while doing validation

    validation_loss = 0
    validation_accuracy = 0
    for inputs, labels in data_loader:  # Iterate over validation data(via data_loader)
        if torch.cuda.is_available():
            inputs = inputs.float().cuda()
            labels = labels.long().cuda()
        else:
            inputs, labels = inputs.to(device), labels.to(device)

        output = model.forward(inputs)
        validation_loss += criterion(output, labels).item()   # this is validation loss, per
        # print('validation_loss',validation_loss, 'validation_accuracy', validation_accuracy)
        # get the probablities, by taking the exponential
        ps = torch.exp(output)
        ps.max(dim=1)   #  #tells us the predicted classes by probablity (0-1) second tensor tells use which classes
                        # have the highest probablites within the softmax output
        equality = (labels.data == ps.max(dim=1)[1]) # Class with the highest probability is our predicted class
        equality
        #print("equality:{:.3f}".format(equality))
        # Accuracy is number of correct predictions divided by all predictions it made
        validation_accuracy += equality.type(torch.FloatTensor).mean()

    return validation_loss, validation_accuracy

# Model Training, Train a deep learning model using a training dataset, with validation using validation data,
#                 also we will get an idea of how well the training is working
# referenced: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#load-data\
# https://www.kaggle.com/pvlima/use-pretrained-pytorch-models
#  There are six inputs to the model:
#  The first argument is model that we will use.
#  The second argument criteria is the method used to evaluate the model fit.
#  The optimizer is the optimization technique used to update the weights.
#  The epoch, as described in the lessons is full run of feedforward and backpropagation through the network.
#  The training data = trainloader and 
#  the validation data = vloader
# ref:  https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
def train_model(model, criterion, optimizer, epochs, trainloader, vloader):
    since = time()
    print('Training start: {:.0f}m {:.0f}s'.format(since // 60, since % 60))
    steps = 0
    running_loss = 0
    print_every = 40
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # make a device switch, added for cli application
    model.to(device)

    for epoch in range(epochs):     # Each epoch has a training and validation phase
        # print('Epoch {}/{}'.format(epoch, epochs - 1))
        # print('-' * 10)
        model.train()  # Set model to training mode
        print('-' * 10, 'Training mode', '-' * 10)

        #scheduler.step()           # scheduler needs testing, add on/off, the schedular will manipulate the learning rate
        #for param_group in optimizer.param_groups:
        #     print(param_group['lr'])     # can print out current lr when using scheduler
        #Training forward pass and backpropagation
        for inputs, labels in trainloader:   # ("interating over the training data w loader")
                steps += 1         # take a loop step count

                # pytorch versions 0.4 & higher - Variable depreciated so that it returns a tensor.
                if torch.cuda.is_available():        # Move tensors to GPU if available
                    inputs = inputs.float().cuda()
                    labels = labels.long().cuda()
                else:
                    inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()                # zero out the gradients during the training,weights in a neural network
                # print('model.forward')             # are adjusted based on gradients accumulated for each batch
                outputs = model.forward(inputs)      # pass the data into the model,call forward method.store the output from log softmax of the nn for the batch.
                loss = criterion(outputs, labels)    # Calculate loss (how far is prediction from label)

                # backward + optimize during training data phase
                loss.backward()                     # Backward pass to calculate and propagate gradients

                optimizer.step()                    # Update weights using optimizer in accordance with the propagated gradients.
                running_loss += loss.item()         # Track the loss as we are training the network running_loss
                                                    # To get the # out of a scalar tensor, use .item()
                                                    # https://github.com/pytorch/pytorch/releases/tag/v0.4.0
                if steps % print_every == 0:
                    model.eval()                    # make sure dropout is off while we are doing validation
                    # print("****** Eval mode ******", "#steps:",steps)
                    with torch.no_grad():  # no grad in validation
                        validation_loss, validation_accuracy = validation(model, criterion, vloader)   # do validation func, gets test_loss and accuracy
                        # print("finished validation function",validation_loss, validation_accuracy)

                        # Val Accuracy is number of correct predictions divided by all predictions it made in the batch
                    print("\nEpoch: {}/{} | ".format(epoch+1, epochs), "Training Loss: {:.4f} ".format(running_loss),
                          "Validation Loss: {:.4f} ".format(validation_loss/len(vloader)),
                          "Validation Accuracy: {:.3f}".format(100*validation_accuracy/len(vloader)," %"),
                          "Steps: ", steps)
                    logging.info('%s %s %s %s  %s %s', \
                          "Epoch: {}/{} | ".format(epoch+1, epochs), \
                          "Training Loss: {:.3f} ".format(running_loss),  \
     			          "Validation Loss: {:.3f} ".format(validation_loss/len(vloader)), \
    			          "Validation Accuracy: {:.3f}".format(100*validation_accuracy/len(vloader),"%"),"  Steps:", steps)
                running_loss = 0
                model.train()                       # put model back in training mode,i.e. turns dropout back on

    time_elapsed = time() - since
    print('Training time_elapsed: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('training completed with {} epochs.'.format(epochs))
    return model, criterion, optimizer

# Compute the average test_accuracy over all test images
# Do validation on the test set, outputs Test Accuracy %
# referenced, https://www.kaggle.com/pvlima/use-pretrained-pytorch-models and
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
# Do validation on the test set - using validation function
def check_accuracy(model, testloader):
    model.eval()    # Set model to evaluate model
    print("Checking accuracy of the test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # make a device switch, added for cli application
    model.to(device)
    criterion = nn.NLLLoss()
    with torch.no_grad():  # no grad in validation
        validation_loss, validation_accuracy = validation(model, criterion, testloader)   # do validation func, gets test_loss and accuracy
        # Accuracy = number of correct predictions divided by all predictions
        print("Test Accuracy: {:.3f}".format(100*validation_accuracy/len(testloader)," %"))
        print("Val. Loss: {:.3f}".format(validation_loss/len(testloader)))
    return
# Check average test_accuracy over all test images, Used as a check on my validation code,
def check_accuracy_on_testset(model, testloader):
    # will sum up the predictions and provide number of correct predictions out of of the tatal
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # make a device switch, added for cli application
    model.to(device)
    print('check_accuracy_on_testset started')
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            # now accumulate the total images interated over
            total += labels.size(0)
        print("Number of Images with correct predictions:", correct)
        print("Total images to labels compared:", total)
        print('Test Accuracy of the network on the test dataset of images:{:.2f}%'.format(100 * correct / total))
        logging.info('%s %s %s  %s %s','Test_accuracy: {:.2f}%'.format(100 * correct / total), '#Correct:',correct, 'Total:',total)
    return

# check_accuracy_on_testset(testloader, model)

# Save the checkpoint
# ref: https://pytorch.org/tutorials/beginner/saving_loading_models.html
# Refer https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch
def save_checkpoint(model, chkpoint_filepath, arch, epochs, criterion, optimizer):
    # This function saves the model specified by the user path
    # Arguments: The saving path and the hyperparameters of the network
    # save the classifier parameters on the pretrained network in the checkpoint
    checkpoint = { 'arch': arch,
                   'classifier':model.classifier,     
                    'epochs': epochs,
                    'class_to_idx':model.class_to_idx,
                    'optimizer': optimizer.state_dict(),
                    'criterion': criterion,
                    'state_dict':model.state_dict()}
    
    torch.save(checkpoint, chkpoint_filepath)
    print("Checkpoint Saved: '{}'".format(chkpoint_filepath))
    return 

    # How to use save_checkpoint
    # filepath = 'checkpoint.pth'
    # chkpoint_filepath = arch + 'checkpoint.pth'
    # print("Saving checkpoint with: ", chkpoint_filepath, arch)
    # save_checkpoint(model, chkpoint_filepath, arch, epochs)

#  Write a function that loads a checkpoint and rebuilds the model
# ref: https://pytorch.org/tutorials/beginner/saving_loading_models.html
#
#    Arguments: The path of the checkpoint file
#    Returns: The Neural Network with all hyperparameters, weights and biases
#
def load_checkpoint(chkpoint_filepath):
    # Loads deep learning model checkpoint.
    # Load the saved file, on gpu or else cpu
    if torch.cuda.is_available():        # Move tensors to GPU if available
        checkpoint = torch.load(chkpoint_filepath)
        print('model on gpu')
    else: 
        # Needed to add this map_location parameters when on CPU
        print('model on cpu')
        checkpoint = torch.load(chkpoint_filepath, map_location=lambda storage, loc: storage)
        # checkpoint = torch.load(chkpoint_filepath)
    # model, input_size = get_model(arch)
    arch = checkpoint['arch'] 
    model = models.__getattribute__(arch)(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = checkpoint['classifier']
    
    # get the hyperparameters, etc, layer feature sizes from the classifier-
    input_size = model.classifier.fc1.in_features
    output_size = model.classifier.fc2.out_features
    try:                                    # added try for chkpts saved w/o optimzizer key
       optimizer = checkpoint['optimizer'] 
    except KeyError:
      print("optizmer key check")
      learning_rate = .001
      optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    #end try
    criterion = checkpoint['criterion']
    # load the class names from the idx
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict']) 
    epochs = checkpoint['epochs']
    
    return model, arch, criterion, epochs, optimizer

# Here is how to load the checkpoint:
def show_loaded_checkpoint(chkpoint_filepath):
    print(chkpoint_filepath)
    # Here is how to load the checkpoint:
    model= load_checkpoint(chkpoint_filepath)
    print("model loaded")
    print("Loaded '{}' (arch={}, input_size={}, output_size={}, hidden_layers={}, epochs={})".format(
        chkpoint_filepath,
        checkpoint['arch'],
        checkpoint['input_size'],
        checkpoint['output_size'],
        checkpoint['hidden_layers'],
        checkpoint['epochs'],
        ))
    class_to_idx = model.class_to_idx
    print(criterion)
    print(optimizer)
    return

# How to use show_loaded_checkpoint:
# print(show_loaded_checkpoint('checkpoint.pth')
