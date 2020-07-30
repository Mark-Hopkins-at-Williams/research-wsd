from reed_wsd.mnist.loss import PairwiseConfidenceLoss, ConfidenceLoss1, CrossEntropyLoss
from reed_wsd.mnist.loss import NLLLoss
from reed_wsd.mnist.loader import PairLoader, ConfusedMnistLoader, MnistLoader
from reed_wsd.mnist.train import SingleTrainer, PairwiseTrainer
from reed_wsd.mnist.train import validate_and_analyze
from reed_wsd.mnist.networks import BasicFFN, AbstainingFFN
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
valloader = MnistLoader(valset, batch_size=64, shuffle=True)

def basic1():
    criterion = NLLLoss()
    trainer = SingleTrainer(criterion, trainloader, valloader, n_epochs = 20)
    net = run(trainer, starting_model = BasicFFN())
    torch.save(net.state_dict(), 'saved/simple.pt')
    return net

def pairwise1():
    #pretrain
    criterion = NLLLoss()
    trainer = SingleTrainer(criterion, trainloader, valloader, n_epochs = 10)
    net = run(trainer, starting_model = AbstainingFFN())
    #pairwise train 
    train_loader = PairLoader(trainset, bsz=64, shuffle=True)
    criterion = PairwiseConfidenceLoss()
    trainer = PairwiseTrainer(criterion, train_loader, valloader, n_epochs = 30)
    net = run(trainer, starting_model=AbstainingFFN(confidence_extractor='max_non_abs'))
    torch.save(net.state_dict(), 'saved/pair_baseline.pt')

def pairwise2():
    # pretrain
    criterion = NLLLoss()
    trainer = SingleTrainer(criterion, trainloader, valloader, n_epochs = 10)
    net = run(trainer)
    # train
    train_loader = PairLoader(trainset, bsz=64, shuffle=True)
    criterion = PairwiseConfidenceLoss()
    trainer = PairwiseTrainer(criterion, train_loader, valloader, n_epochs = 30)
    net = trainer(net)
    torch.save(net.state_dict(), 'saved/pair_baseline.pt')


def closs1(p0):
    criterion = ConfidenceLoss1(p0)
    trainer = SingleTrainer(criterion, trainloader, valloader, n_epochs = 10)
    net = run(trainer)
    torch.save(net.state_dict(), 'saved/closs1.pt')

def run(trainer, starting_model = None, name = None):
    if name is None:
        name = type(trainer.criterion).__name__
    print("================{} EXPERIMENT======================".format(name))
    net = trainer(starting_model)
    data_dict, _ = validate_and_analyze(net, trainer.val_loader)
    results_file = "{}.json".format(name.lower())
    with open(join(validation_dir, results_file), "w") as f:
        json.dump(data_dict, f)
    return net

if __name__ == "__main__":
    pairwise1()
    
