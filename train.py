# Imports here
import matplotlib.pyplot as plt
from torchvision import  models
from collections import OrderedDict
import torch.nn as nn
import torch
import numpy as np
import os
import argparse
from data_handling import data_handler
import torch.optim as optim
from workspace_utils import active_session



parser = argparse.ArgumentParser()
parser.add_argument("Data_directory", help="Directory for training dataset")
parser.add_argument("--arch", help="model architecture to use")
parser.add_argument("--learning_rate", help="models learning rate", type=float)
parser.add_argument("--hidden_units", help="model hidden units", nargs=2, type=int)
parser.add_argument("--epochs", help="number of trainig epochs", type=int)
parser.add_argument("--gpu", help="use gpu for training")



args = parser.parse_args()


train_data_loader, val_data_loader, test_data_loader, train_image_datasets = data_handler(args.Data_directory)
first_h, second_h = args.hidden_units

# Building the model
def create_model(model):
    # freeze the features for the vgg model
    # ie ensures that we do not update the weights of the vgg model when training
    # our own model or rather when performing the back propagation step
    for param in model.parameters():
        param.requires_grad = False

    # define our model classsifier
    classifier = nn.Sequential(OrderedDict([
        ("fc1", nn.Linear(25088, first_h)),
        ("ReLU", nn.ReLU()),
        ("fd1", nn.Dropout(p=0.5)),
        ("fc2", nn.Linear(first_h, second_h)),
        ("ReLU", nn.ReLU()),
        ("fd2", nn.Dropout(p=0.5)),
        ("fc3", nn.Linear(second_h, 102)),
        ("Output", nn.LogSoftmax(dim=1))
    ]))

    # modify vgg model classifeir
    model.classifier = classifier
    return model

# selects GPU if available for model training
device = torch.device("cuda" if (args.gpu == "yes") else "cpu")

# First import an already trained model
if args.arch == "vgg19":
    model = models.vgg19(pretrained=True)
elif args.arch == "vgg13":
    model = models.vgg13(pretrained=True)

model_1 = create_model(model)


def train_model(model):
    # Trainig process: ie we are training the model with our own classifier
    criterion = nn.NLLLoss()
    # notice that we access only the classifier parameters
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    # moves model to GPU to ensure faster training
    model.to(device)

    epochs = args.epochs
    train_losses, test_losses = [], []
    for e in range(epochs):
        current_loss = 0
        for images, labels in train_data_loader:


            images = images.to(device)
            labels = labels.to(device)


            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            current_loss += loss.item()
        else:
            total_test_loss = 0  # Number of incorrect predictions on the test set
            total_test_correct = 0  # Number of correct predictions on the test set


            # turn off gradients
            with torch.no_grad():
                model.eval() # turns of dropouts
                # validation pass here
                for images, labels in val_data_loader:


                    images = images.to(device)
                    labels = labels.to(device)


                    output = model(images)
                    loss = criterion(output, labels)
                    total_test_loss += loss.item()

                    predicted_prob = torch.exp(output)
                    top_p, top_class = predicted_prob.topk(1, dim=1)
                    eq = top_class == labels.view(*top_class.shape)
                    total_test_correct += eq.sum().item()

            model.train() # turns dropouts back on
            # Get mean loss to enable comparison between train and test sets
            train_loss = current_loss / len(train_data_loader.dataset)
            test_loss = total_test_loss / len(val_data_loader.dataset)

            # At completion of epoch
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(train_loss),
                  "validation Loss: {:.3f}.. ".format(test_loss),
                  "Model Accuracy: {:.3f}".format(total_test_correct / len(val_data_loader.dataset)))
    return model



def save_checkpoint(model):
    model.class_to_idx = train_image_datasets.class_to_idx
    Checkpoint = {
        "inputs": 25088,
        "outputs": 102,
        "fc1_output": first_h,
        "fc2_output": second_h,
        "class_to_idx": model.class_to_idx,
        "state_dict": model.state_dict()
    }
    torch.save(Checkpoint, "checkpoint.pth")
    print("Model Checkpoint Saved")

    
# A function that loads a checkpoint and rebuilds the model
def load_checkpoint(path):
    checkpoint = torch.load(path)
    
    # Rebuild model
    new_model = models.vgg19(pretrained=True)
    for param in new_model.parameters():
        param.requires_grad = False
    
    new_model.class_to_idx = checkpoint["class_to_idx"]
    
    classifier = nn.Sequential(OrderedDict([
        ("fc1", nn.Linear(checkpoint["inputs"], checkpoint["fc1_output"])),
        ("ReLU", nn.ReLU()),
        ("fd1", nn.Dropout(p=0.5)),
        ("fc2", nn.Linear(checkpoint["fc1_output"], checkpoint["fc2_output"])),
        ("ReLU", nn.ReLU()),
        ("fd2", nn.Dropout(p=0.5)),
        ("fc3", nn.Linear(checkpoint["fc2_output"], checkpoint["outputs"])),
        ("Output", nn.LogSoftmax(dim=1))
    ]))
    
    # update classifier for pretrained network
    new_model.classifier = classifier
    
    new_model.load_state_dict(checkpoint["state_dict"])
    
    return new_model

with active_session():
    # trains our model
    tr_model = train_model(model_1)
    
save_checkpoint(tr_model)
model_1 = load_checkpoint("checkpoint.pth")



