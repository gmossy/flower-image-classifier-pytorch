# Udacity's AI Programming with Python Nanodegree program, Image classification using Pytorch models.

AI Programming with Python Project. Image classification (102 flower categories) using Pytorch models.
Project code for Udacity's AI Programming with Python Nanodegree program. In this project, code developed for an image classifier built with PyTorch, then converted into a command line applications: train.py, predict.py.
AI Programming with Python Project. Image classification (102 flower categories) using Pytorch models.
Project code for Udacity's AI Programming with Python Nanodegree program. In this project, code developed for an image classifier built with PyTorch, then converted into a command line applications: train.py, predict.py.

The image classifier to recognize different species of flowers. Dataset contains 102 flower categories.

In Image Classifier Project.ipynb Alexnet from torchvision.models pretrained models was used. It was loaded as a pre-trained network, based on which defined a new, untrained feed-forward network as a classifier, using ReLU activations and dropout. Trained the classifier layers using backpropagation using the pre-trained network to get the features. The loss and accuracy on the validation set were tracked to determine the best hyperparameters


Command line applications train.py and predict.py
For command line applications there is an option to select either Alexnet, VGG, and Densenet models.

Following arguments mandatory or optional for train.py

'data_dir'. 'Provide data directory. Mandatory argument', type = str
'--save_dir'. 'Provide saving directory. Optional argument', type = str
'--arch'. 'Vgg13 can be used if this argument specified, otherwise Alexnet will be used', type = str
'--lr'. 'Learning rate, default value 0.001', type = float
'--hidden_units'. 'Hidden units in Classifier. Default value is 2048', type = int
'--epochs'. 'Number of epochs', type = int
'--GPU'. "Option to use GPU", type = str
Following arguments mandatory or optional for predict.py

'image_dir'. 'Provide path to image. Mandatory argument', type = str
'load_dir'. 'Provide path to checkpoint. Mandatory argument', type = str
'--top_k'. 'Top K most likely classes. Optional', type = int
'--category_names'. 'Mapping of categories to real names. JSON file name to be provided. Optional', type = str
'--GPU'. "Option to use GPU. Optional", type = str


Viewing the Jyputer Notebook
In order to better view and work on the jupyter Notebook I encourage you to use nbviewer . You can simply copy and paste the link to this website and you will be able to edit it without any problem. Alternatively you can clone the repository using

git clone https://github.com/gmossy/flower-image-classifier-pytorch/
then in the command Line type, after you have downloaded jupyter notebook type

jupyter notebook
locate the notebook and run it.


Json file
In order for the network to print out the name of the flower a .json file is required. If you aren't familiar with json you can find information here. By using a .json file the data can be sorted into folders with numbers and those numbers will correspond to specific names specified in the .json file.

Data and the json file
The data used specifically for this assignemnt are a flower database are not provided in the repository as it's larger than what github allows. Nevertheless, feel free to create your own databases and train the model on them to use with your own projects. The structure of your data should be the following:
The data need to comprised of 3 folders, test, train and validate. Generally the proportions should be 70% training 10% validate and 20% test.
Inside the train, test and validate folders there should be folders bearing a specific number which corresponds to a specific category, clarified in the json file. For example if we have the image a.jpj and it is a rose it could be in a path like this /test/5/a.jpg and json file would be like this {...5:"rose",...}. Make sure to include a lot of photos of your catagories (more than 10) with different angles and different lighting conditions in order for the network to generalize better.

GPU
As the network makes use of a sophisticated deep convolutional neural network the training process is impossible to be done by a common laptop. In order to train your models to your local machine you have three options

Cuda -- If you have an NVIDIA GPU then you can install CUDA from here. With Cuda you will be able to train your model however the process will still be time consuming
Cloud Services -- There are many paid cloud services that let you train your models like AWS or Google Cloud
Coogle Colab -- Google Colab gives you free access to a tesla K80 GPU for 12 hours at a time. Once 12 hours have ellapsed you can just reload and continue! The only limitation is that you have to upload the data to Google Drive and if the dataset is massive you may run out of space.
However, once a model is trained then a normal CPU can be used for the predict.py file and you will have an answer within some seconds.

Hyperparameters
As you can see you have a wide selection of hyperparameters available and you can get even more by making small modifications to the code. Thus it may seem overly complicated to choose the right ones especially if the training needs at least 15 minutes to be completed. So here are some hints:

By increasing the number of epochs the accuracy of the network on the training set gets better and better however be careful because if you pick a large number of epochs the network won't generalize well, that is to say it will have high accuracy on the training image and low accuracy on the test images. Eg: training for 12 epochs training accuracy: 85% Test accuracy: 82%. Training for 30 epochs training accuracy 95% test accuracy 50%.
A big learning rate guarantees that the network will converge fast to a small error but it will constantly overshot
A small learning rate guarantees that the network will reach greater accuracies but the learning process will take longer
Densenet121 works best for images but the training process takes significantly longer than alexnet or vgg16
*My settings were lr=0.001, dropoup=0.5, epochs= 15 and my test accuracy was 86% with densenet121 as my feature extraction model.

Pre-Trained Network
The checkpoint.pth file contains the information of a network trained to recognise 102 different species of flowers. I has been trained with specific hyperparameters thus if you don't set them right the network will fail. In order to have a prediction for an image located in the path /path/to/image using my pretrained model you can simply type python predict.py /path/to/image checkpoint.pth

Contributing
Authors
Glenn Mossy- Initial work
Udacity - Final Project of the AI with Python Nanodegree
This project is licensed under the MIT License - see the LICENSE.md file for details


