from reed_wsd.mnist.model import BasicFFN, AbstainingFFN, ConfidentFFN
from reed_wsd.loss import AbstainingLoss, NLLLoss, PairwiseConfidenceLoss
from reed_wsd.mnist.train import MnistAbstainingDecoder, MnistSimpleDecoder
from reed_wsd.mnist.train import SingleTrainer as MnistSingleTrainer
from reed_wsd.mnist.train import PairwiseTrainer as MnistPairwiseTrainer
from reed_wsd.mnist.loader import ConfusedMnistLoader, ConfusedMnistPairLoader
import os
import torch
from os.path import join
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

file_dir = os.path.dirname(os.path.realpath(__file__))
mnist_dir = join(file_dir, 'mnist')
mnist_data_dir = join(mnist_dir, 'data')
mnist_train_dir = join(mnist_data_dir, 'train')
mnist_test_dir = join(mnist_data_dir, 'test')

mnist_trainset = datasets.MNIST(mnist_train_dir, download=True, train=True,
                                transform=transform)
mnist_valset = datasets.MNIST(mnist_test_dir, download=True, train=False,
                              transform=transform)
mnist_trainloader = ConfusedMnistLoader(mnist_trainset, bsz=64, shuffle=True)
mnist_valloader = ConfusedMnistLoader(mnist_valset, bsz=64, shuffle=True)

def abstaining():
    criterion = AbstainingLoss(alpha=0.5, warmup_epochs=3)
    model = AbstainingFFN(confidence_extractor='max_non_abs')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    decoder = MnistAbstainingDecoder()
    n_epochs = 30
    trainer = MnistSingleTrainer(criterion, optimizer, mnist_trainloader,
                                 mnist_valloader, decoder, n_epochs)
    best_model = trainer(model)
    return best_model

class MnistTaskFactory:

    def __init__(self, config):
        self.config = config
        self._ds_lookup = {'train': datasets.MNIST(mnist_train_dir, download=True, train=True,
                                                   transform=transform),
                           'test': datasets.MNIST(mnist_test_dir, download=True, train=False,
                                                  transform=transform)
                           }
        self._decoder_lookup = {'simple': MnistSimpleDecoder,
                                'abstaining': MnistAbstainingDecoder
                                }
        self._model_lookup = {'simple': BasicFFN,
                              'abstaining': AbstainingFFN
                              }
        self._trainer_lookup = {'single': MnistSingleTrainer,
                                'pairwise': MnistPairwiseTrainer
                                }
    
    def init_loader(self, stage, style, confuse, bsz):
        ds = self._ds_lookup[stage]
        if style == "single":
            loader = ConfusedMnistLoader(ds, bsz, confuser=confuse, shuffle=True)
        if style == "pairwise":
            loader = ConfusedMnistPairLoader(ds, bsz, confuser=confuse, shuffle=True)
        return loader

    def train_loader_factory(self):
        return self.init_loader('train', 
                                self.config['style'], 
                                self.config['confuse'], 
                                self.config['bsz'])
    
    def val_loader_factory(self):
        return self.init_loader('test', 
                                self.config['style'], 
                                self.config['confuse'], 
                                self.config['bsz']) 

    def criterion_factory(self):
        config = self.config['criterion']
        return AbstainingLoss(config['alpha'], config['warmup_epochs'])
    
    def decoder_factory(self):
        return self._decoder_lookup[self.config['architecture']]()

    def model_factory(self):
        return self._model_lookup[self.config['architecture']](confidence_extractor=self.config['confidence'])

    def optimizer_factory(self, model):
        return torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    def select_trainer(self):
        return self._trainer_lookup[self.config['style']]
        

if __name__ == "__main__":
    config = {"task": "mnist",
          "architecture": "abstaining",
          "confidence": "max_non_abs",
          "criterion": {"name": "conf1",
                        "alpha": 0.5,
                        "warmup_epochs": 3},
          "confuse": "all",
          "style": "single",
          "dev_corpus": None,
          "bsz": 64,
          "n_epochs": 30}
    task = MnistTaskFactory(config)
    train_loader = task.train_loader_factory()
    val_loader = task.val_loader_factory()
    decoder = task.decoder_factory()
    model = task.model_factory()
    optimizer = task.optimizer_factory(model)
    trainer = task.select_trainer()
    criterion = task.criterion_factory()
    n_epochs = config['n_epochs']
    trainer = task.select_trainer()(criterion, optimizer, train_loader,
                                    val_loader, decoder, n_epochs)
    trainer(model)
    """
    abstaining()
    """
