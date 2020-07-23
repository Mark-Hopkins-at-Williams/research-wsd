"""
Code adapted from:
https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627

"""

import torch
import copy
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import os
from reed_wsd.util import cudaify
from reed_wsd.mnist.loss import NLLA, AWNLL, CAWNLL, ConfidenceLoss1
from reed_wsd.mnist.loss import ConfidenceLoss2, ConfidenceLoss4, PairwiseConfidenceLoss
from os.path import join
from reed_wsd.mnist.loader import PairLoader
from collections import defaultdict
from reed_wsd.plot import PYCurve, plot_curves


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
                          nn.Softmax(dim=1))
    return cudaify(model)

def confuse(labels):
    labels = labels.clone()
    one_and_sevens = (labels == 1) + (labels == 7)
    one_seven_shape = labels[one_and_sevens].shape
    new_labels = torch.randint(0, 2, one_seven_shape) #change the second argument for different weights
    new_labels[new_labels == 0] = 7    
    labels[one_and_sevens] = new_labels
    """
    one_and_sevens = (labels == 2) + (labels == 3)
    one_seven_shape = labels[one_and_sevens].shape
    new_labels = torch.randint(0, 2, one_seven_shape)
    new_labels[new_labels > 0] = 3
    new_labels[new_labels == 0] = 2
    labels[one_and_sevens] = new_labels
    one_and_sevens = (labels == 4) + (labels == 5)
    one_seven_shape = labels[one_and_sevens].shape
    new_labels = torch.randint(0, 4, one_seven_shape)
    new_labels[new_labels > 0] = 4
    new_labels[new_labels == 0] = 5
    labels[one_and_sevens] = new_labels
    """
    return labels

def decode_gen(confidence):
    assert(confidence == "baseline" or
           confidence == "neg_abs")
    def decode(net, data):
        net.eval()
        for images, labels in data:
            for i in range(len(labels)):
                img = images[i].view(1, 784)
                # Turn off gradients to speed up this part
                with torch.no_grad():
                    ps = net(cudaify(img))
                ps = ps.squeeze(dim=0)
                if confidence == "baseline":
                    c, _ = ps[:-1].max(dim=0)
                    c = c.item()
                if confidence == "neg_abs":
                    c = (1 - ps[-1]).item()
                pred = ps[:-1].argmax(dim=0).item()
                gold = labels[i].item()
                yield {'pred': pred, 'gold': gold, 'confidence': c}
    return decode

def train_pair(criterion):
    train_loader = PairLoader(trainset, bsz=64, shuffle=True)
    model = FFN()
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    time0 = time()
    epochs = 20
    best_model = None
    best_model_score = float('-inf')
    for e in range(epochs):
        running_loss = 0
        batch_iter = train_loader.batch_iter()
        for img_x, img_y, lbl_x, lbl_y in batch_iter:
            # randomly assign labels between 1 and 7
            lbl_x = confuse(lbl_x)
            lbl_y = confuse(lbl_y)
            # Training pass
            optimizer.zero_grad()
                                                                
            output_x = model(cudaify(img_x))
            output_y = model(cudaify(img_y))
            loss = criterion(output_x, output_y, cudaify(lbl_x), cudaify(lbl_y))
                                                                                        
            #This is where the model learns by backpropagating
            loss.backward()
                                                                                                                
            #And optimizes its weights here
            optimizer.step()
                                                                                                                                        
            running_loss += loss.item()
        _, precision = validate_and_analyze(model)
        if precision > best_model_score:
            print("Updating best model.")
            best_model = copy.deepcopy(model)
            best_model_score = precision
        print("Epoch {} - Training loss: {}; Dev precision: {}".format(e, 
                                                                      running_loss/len(trainloader),
                                                                      precision))
        print("\nTraining Time (in minutes) =",(time()-time0)/60)
    return best_model
    
def train(criterion):
    model = FFN()
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    time0 = time()
    epochs = 10
    best_model = None
    best_model_score = float('-inf')
    for e in range(epochs):
        if e == 2:
            criterion.p0 = 0.5
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
        _, precision = validate_and_analyze(model)
        if precision > best_model_score:
            print("Updating best model.")
            best_model = copy.deepcopy(model)
            best_model_score = precision
        print("Epoch {} - Training loss: {}; Dev precision: {}".format(e, 
                                                                      running_loss/len(trainloader),
                                                                      precision))
        print("\nTraining Time (in minutes) =",(time()-time0)/60)
    outfile = "params_" + str(criterion) + ".pt"
    torch.save(best_model.state_dict(), join(model_dir, "params_" + str(criterion) + ".pt"))
    return best_model

def validate_and_analyze(model):
    correct_count, n_confident = 0, 0
    avg_err_conf = 0
    avg_crr_conf = 0
    n_error = 0
    data_dict = {}
    error_dict = defaultdict(int)
    for i in range(output_size):
        data_dict[i] = [0, 0, 0] # [n_correct, n_wrong, n_abstain]
    for images,labels in valloader:
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            # Turn off gradients to speed up this part
            with torch.no_grad():
                ps = model(cudaify(img))

            # Output of the network are log-probabilities, need to take 
            # exponential for probabilities
            probab = list(ps.cpu().numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if (pred_label == ABSTAIN):
                data_dict[true_label][2] += 1
            elif (true_label == pred_label):
                data_dict[true_label][0] += 1
            else:
                data_dict[true_label][1] += 1 
            best_nonabstain_label = probab.index(max(probab[:-1]))
            if (true_label == best_nonabstain_label):
                avg_crr_conf += probab[best_nonabstain_label]
                correct_count += 1
                n_confident += 1
            else:
                avg_err_conf += probab[best_nonabstain_label]
                error_dict[true_label] += 1
                n_confident += 1            
                n_error += 1
    for lbl in error_dict:
        error_dict[lbl] /= n_error
    print(error_dict)
    print('average error confidence:', avg_err_conf / n_error)
    print('average correct confidence:', avg_crr_conf / correct_count)
    return data_dict, correct_count / n_confident


if __name__ == "__main__":
    """
    criterion = PairwiseConfidenceLoss('baseline')
    net = train_pair(criterion)
    with open('saved/pair_baseline.pt', 'w') as f:
        torch.save(net.state_dict(), 'saved/pair_baseline.pt')
    criterion = PairwiseConfidenceLoss('neg_abs')
    net = train_pair(criterion)
    with open('saved/pair_negabs.pt', 'w') as f:
        torch.save(net.state_dict(), 'saved/pair_neg_abs.pt')
    model = FFN()
    print('evaluating baseline...')
    model.load_state_dict(torch.load('saved/pair_baseline.pt'))
    validate_and_analyze(model)
    """
    net_base = FFN()
    net_base.load_state_dict(torch.load('saved/pair_baseline.pt', map_location=torch.device('cpu')))
    net_neg = FFN()
    net_neg.load_state_dict(torch.load('saved/pair_neg_abs.pt', map_location=torch.device('cpu')))
    pyc_base = PYCurve.from_data(net_base, valloader, decode_gen('baseline'))
    pyc_neg = PYCurve.from_data(net_neg, valloader, decode_gen('neg_abs'))
    plot_curves([pyc_base, 'baseline'], [pyc_neg, 'neg_abs'])
