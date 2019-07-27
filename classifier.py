
from collections import OrderedDict    #storing state_dict - all the weights and parameter

# Build the network, Create classifier
def create_classifier(model, input_size, hidden_layers, output_size, learning_rate, dropout=0.5):
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
