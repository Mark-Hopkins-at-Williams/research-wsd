from reed_wsd.mnist.model import BasicFFN, AbstainingFFN, ConfidentFFN
from reed_wsd.loss import AbstainingLoss, NLLLoss, PairwiseConfidenceLoss
from reed_wsd.mnist.train import MnistDecoder, MnistAbstainingDecoder
from reed_wsd.mnist.train import SingleTrainer, PairwiseTrainer
from reed_wsd.mnist.loader import ConfusedMnistLoader, ConfusedMnistPairLoader
import torch
from os.path import join
from torchvision import datasets, transforms
import os


file_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = join(file_dir, "data")
train_dir = join(data_dir, "train")
test_dir = join(data_dir, "test")
model_dir = join(file_dir, "saved")
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

trainset = datasets.MNIST(train_dir, download=True, train=True, 
                          transform=transform)
valset = datasets.MNIST(test_dir, download=True, train=False, 
                        transform=transform)
trainloader = ConfusedMnistLoader(trainset, bsz=64, shuffle=True)
valloader = ConfusedMnistLoader(valset, bsz=64, shuffle=True)

def random_guesser():
    criterion = NLLLoss()
    #input_size: 784, hidden_size: [128, 64], output_size: 10
    model = BasicFFN(confidence_extractor = 'random')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    decoder = MnistDecoder()
    n_epochs = 20
    trainer = SingleTrainer(criterion, optimizer, trainloader, 
                            valloader, decoder, n_epochs)
    best_model = trainer(model)
    return best_model

def baseline():
    criterion = NLLLoss()
    #input_size: 784, hidden_size: [128, 64], output_size: 10
    model = BasicFFN()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    decoder = MnistDecoder()
    n_epochs = 20
    trainer = SingleTrainer(criterion, optimizer, trainloader, 
                            valloader, decoder, n_epochs)
    best_model = trainer(model)
    return best_model

def abstaining():
    criterion = AbstainingLoss(0.5)
    model = AbstainingFFN(confidence_extractor='max_non_abs')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    decoder = MnistAbstainingDecoder()
    n_epochs = 30
    trainer = SingleTrainer(criterion, optimizer, trainloader, 
                            valloader, decoder, n_epochs)
    best_model = trainer(model)
    return best_model

def confidence_twin():
    criterion = PairwiseConfidenceLoss()
    trainloader = ConfusedMnistPairLoader(trainset, bsz = 64, shuffle=True)
    valloader = ConfusedMnistLoader(valset, bsz = 64, shuffle=True)
    model = ConfidentFFN()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    decoder = MnistAbstainingDecoder()
    n_epochs = 20
    trainer = PairwiseTrainer(criterion, optimizer, trainloader, 
                              valloader, decoder, n_epochs)
    best_model = trainer(model)
    return best_model

if __name__ == "__main__":
    abstaining()
    
