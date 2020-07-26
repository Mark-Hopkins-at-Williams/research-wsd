from reed_wsd.mnist.mnist import train, train_pair, validate_and_analyze, ConfidentFFN
from reed_wsd.mnist.loss import PairwiseConfidenceLoss, ConfidenceLoss1
from reed_wsd.mnist.loader import PairLoader, ConfusedMnistLoader, MnistLoader
import torch
from os.path import join
import json
from torchvision import datasets, transforms
import os


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
trainloader = ConfusedMnistLoader(trainset, batch_size=64, shuffle=True)
valloader = ConfusedMnistLoader(valset, batch_size=64, shuffle=True)


def pairwise1():
    train_loader = PairLoader(trainset, bsz=64, shuffle=True)
    criterion = PairwiseConfidenceLoss()
    net = run(criterion, train_pair, train_loader, valloader)
    torch.save(net.state_dict(), 'saved/pair_baseline.pt')

def pairwise2():
    train_loader = PairLoader(trainset, bsz=64, shuffle=True)
    criterion = PairwiseConfidenceLoss('neg_abs')
    net = run(criterion, train_pair, train_loader, valloader)
    torch.save(net.state_dict(), 'saved/pair_neg_abs.pt')

def closs1(p0):
    criterion = ConfidenceLoss1(p0)
    net = run(criterion, train, trainloader, valloader)
    torch.save(net.state_dict(), 'saved/closs1.pt')

def run(criterion, trainer, train_loader, val_loader, name = None):
    if name is None:
        name = type(criterion).__name__
    print("================{} EXPERIMENT======================".format(name))
    outfile = "params_" + str(criterion) + ".pt"
    net = trainer(criterion, train_loader, val_loader, join(model_dir, outfile))
    data_dict, _ = validate_and_analyze(net, val_loader)
    results_file = "{}.json".format(name.lower())
    with open(join(validation_dir, results_file), "w") as f:
        json.dump(data_dict, f)
    return net
    
