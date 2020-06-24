"""
Code adapted from:
https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627

"""

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import os
from util import cudaify
from loss import NLLA, AWNLL, CAWNLL
from torch.nn import NLLLoss
import json
from os.path import join


# TODO: refactor this into something more modular

file_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = join(file_dir, "data")
train_dir = join(data_dir, "train")
test_dir = join(data_dir, "test")
model_dir = join(file_dir, "saved")
validation_dir = join(file_dir, "validations")
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)
if not os.path.isdir(train_dir):
    os.mkdir(train_dir)
if not os.path.isdir(test_dir):
    os.mkdir(test_dir)
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
if not os.path.isdir(validation_dir):
    os.mkdir(validation_dir)

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

trainset = datasets.MNIST(train_dir, download=True, train=True, transform=transform)
valset = datasets.MNIST(test_dir, download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

ABSTAIN = 10

input_size = 784
hidden_sizes = [128, 64]
output_size = 11

def FFN():
    print("constructing model...")
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[1], output_size),
                          nn.LogSoftmax(dim=1))
    return cudaify(model)

def confuse(labels):
    labels = labels.clone()
    one_and_sevens = (labels == 1) + (labels == 7)
    one_seven_shape = labels[one_and_sevens].shape
    new_labels = torch.randint(0, 2, one_seven_shape)
    new_labels[new_labels == 0] = 7
    #print(new_labels)
    #print(labels)
    labels[one_and_sevens] = new_labels
    return labels

def train(criterion):
    model = FFN()
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    time0 = time()
    epochs = 15
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
            # randomly assign labels between 1 and 7
            labels = confuse(labels)
            # Training pass
            optimizer.zero_grad()
                                                                
            output = model(cudaify(images))
            loss = criterion(output, cudaify(labels))
                                                                                        
            #This is where the model learns by backpropagating
            loss.backward()
                                                                                                                
            #And optimizes its weights here
            optimizer.step()
                                                                                                                                        
            running_loss += loss.item()
        _, acc = validate_and_analyze(model, criterion)
        print("Epoch {} - Training loss: {}; Dev accuracy: {}".format(e, 
                                                                      running_loss/len(trainloader),
                                                                      acc))
        print("\nTraining Time (in minutes) =",(time()-time0)/60)
    torch.save(model.state_dict(), join(model_dir, "params.pt"))

def validate_and_analyze(model, criterion):
    correct_count, all_count = 0, 0
    data_dict = {}
    for i in range(output_size):
        data_dict[i] = [0, 0, 0] # [n_correct, n_wrong, n_abstain]
    for images,labels in valloader:
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            # Turn off gradients to speed up this part
            with torch.no_grad():
                logps = model(cudaify(img))

            # Output of the network are log-probabilities, need to take 
            # exponential for probabilities
            ps = torch.exp(logps)
            probab = list(ps.cpu().numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if true_label == 1 or true_label == 7:
                print(ps)
            if (pred_label == ABSTAIN):
                data_dict[true_label][2] += 1
            elif (true_label == pred_label):
                correct_count += 1
                data_dict[true_label][0] += 1
            else:
                data_dict[true_label][1] += 1 
            all_count += 1
    print(data_dict)
    return data_dict, correct_count / all_count



if __name__ == "__main__":
    pass

