import copy
from reed_wsd.allwords.model import SimpleFFN, SimpleAbstainingFFN, BEMforWSD
from reed_wsd.mnist.model import BasicFFN, AbstainingFFN
from reed_wsd.allwords.wordsense import SenseInstanceDataset, SenseTaggedSentences, SenseInstanceLoader, TwinSenseInstanceLoader
from reed_wsd.allwords.vectorize import DiskBasedVectorManager
from reed_wsd.allwords.blevins import BEMDataset, BEMLoader
from reed_wsd.mnist.train import MnistSimpleDecoder, MnistAbstainingDecoder
from reed_wsd.allwords.evaluate import AllwordsSimpleEmbeddingDecoder, AllwordsAbstainingEmbeddingDecoder, AllwordsBEMDecoder
from reed_wsd.loss import CrossEntropyLoss, NLLLoss, ConfidenceLoss1, ConfidenceLoss4, PairwiseConfidenceLoss
from reed_wsd.allwords.train import SingleEmbeddingTrainer, PairwiseEmbeddingTrainer, BEMTrainer
from reed_wsd.mnist.train import MnistSingleTrainer, MnistPairwiseTrainer
from reed_wsd.mnist.loader import ConfusedMnistLoader, MnistLoader, MnistPairLoader, ConfusedMnistPairLoader
import os
from os.path import join
import torch.optim as optim
from torchvision import datasets, transforms

file_dir = os.path.dirname(os.path.realpath(__file__))
mnist_dir = join(file_dir, 'mnist')
allwords_dir = join(file_dir, 'allwords')
allwords_data_dir = join(allwords_dir, 'data')
mnist_data_dir = join(mnist_dir, 'data')
mnist_train_dir = join(mnist_data_dir, 'train')
mnist_test_dir = join(mnist_data_dir, 'test')

transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                               ])

def init_allwords_loader(stage, architecture, style, corpus_id):
    data_dir = allwords_data_dir
    filename = join(data_dir, 'raganato.json')
    sents = SenseTaggedSentences.from_json(filename, corpus_id)
    if architecture == "bem":
        ds = BEMDataset(sents)
        loader = BEMLoader(ds, config['bsz'])
    if architecture == 'simple' or architecture == 'abstaining': 
        vecmgr = DiskBasedVectorManager(join(join(data_dir, 'vecs'), corpus_id))
        ds = SenseInstanceDataset(sents, vecmgr)
        if stage == 'train':
            if style == 'single':
                loader = SenseInstanceLoader(ds, batch_size=config['bsz'])
            if style == 'pairwise':
                loader = TwinSenseInstanceLoader(ds, batch_size=config['bsz'])
        if stage == 'test':
            loader = SenseInstanceLoader(ds, batch_size=config['bsz'])
    return loader

def init_mnist_loader(stage, style, confused):
    if stage == 'train':
        ds = datasets.MNIST(mnist_train_dir, download=True, train=True, transform=transform)
        if style == 'single':
            if confused:
                loader = ConfusedMnistLoader(ds, config['bsz'], shuffle=True)
            else:
                loader = MnistLoader(ds, config['bsz'], shuffle=True)
        if style == 'pairwise':
            if confused:
                loader = ConfusedMnistPairLoader(ds, config['bsz'], shuffle=True)
            else:
                loader = MnistPairLoader(ds, config['bsz'], shuffle=True)
    if stage == 'test':
        ds = datasets.MNIST(mnist_test_dir, download=True, train=False, transform=transform)
        if confused:
            loader = ConfusedMnistLoader(ds, config['bsz'], shuffle=True)
        else:
            loader = MnistLoader(ds, config['bsz'], shuffle=True)
    return loader

corpus_id_lookup = {'semcor': 'data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml',
                        'semev07': 'data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.data.xml',
                        'semev13': 'data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2013/semeval2013.data.xml',
                        'semev15': 'data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2015/semeval2015.data.xml',
                        'sensev2': 'data/WSD_Evaluation_Framework/Evaluation_Datasets/senseval2/senseval2.data.xml',
                        'sensev3': 'data/WSD_Evaluation_Framework/Evaluation_Datasets/senseval3/senseval3.data.xml'}

allwords_model_lookup = {'simple': SimpleFFN,
                         'abstaining': SimpleAbstainingFFN,
                         'bem': BEMforWSD}
mnist_model_lookup = {'simple': BasicFFN,
                      'abstaining': AbstainingFFN}
allwords_decoder_lookup = {'simple': AllwordsSimpleEmbeddingDecoder,
                         'abstaining': AllwordsAbstainingEmbeddingDecoder,
                         'bem': AllwordsBEMDecoder}
mnist_decoder_lookup = {'simple': MnistSimpleDecoder,
                        'abstaining': MnistAbstainingDecoder}
criterion_lookup = {'crossentropy': CrossEntropyLoss,
                    'nll': NLLLoss,
                    'conf1': ConfidenceLoss1,
                    'conf4': ConfidenceLoss4,
                    'pairwise': PairwiseConfidenceLoss}

def train_loader_factory(config):
    if config['task'] == "allwords":
        loader = init_allwords_loader('train', config['architecture'], config['style'], corpus_id_lookup['semcor'])
    if config['task'] == "mnist":
        loader = init_mnist_loader('train', config['style'], config['confused'])
    return loader

def val_loader_factory(config):
    if config['task'] == "allwords":
        loader = init_allwords_loader('test', config['architecture'], config['style'], corpus_id_lookup[config['dev_corpus']])
    if config['task'] == "mnist":
        loader = init_mnist_loader('test', config['style'], config['confused'])
    return loader

def decoder_factory(config):
    if config['task'] == 'mnist':
        return mnist_decoder_lookup[config['architecture']]()
    if config['task'] == 'allwords':
        return allwords_decoder_lookup[config['architecture']]()

def model_factory(config, data):
    if config['task'] == 'allwords':
        if config['architecture'] == 'bem':
            model = allwords_model_lookup[config['architecture']](gpu=True)
        else:
            model = allwords_model_lookup[config['architecture']](input_size=768,
                                                                  output_size=data.num_senses(),
                                                                  zone_applicant=config['confidence'])
    if config['task'] == 'mnist':
        model = mnist_model_lookup[config['architecture']](confidence_extractor=config['confidence'])
    return model

def optimizer_factory(config, model):
    if config['task'] == 'allwords':
        return optim.Adam(model.parameters(), lr=0.001)
    if config['task'] == 'mnist':
        return optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

def criterion_factory(config):
    if config['name'] in ['conf1', 'conf4']:
        criterion = criterion_lookup[config['name']](p0=config['p0'])
    if config['name'] in ['crossentropy', 'nll', 'pairwise']:
        criterion = criterion_lookup[config['name']]()
    return criterion

def select_trainer(config):
    if config['task'] == 'allwords':
        if config['architecture'] == 'bem':
            trainer = BEMTrainer
        else:
            if config['style'] == 'single':
                trainer = SingleEmbeddingTrainer
            if config['style'] == 'pairwise':
                trainer = PairwiseEmbeddingTrainer
    if config['task'] == 'mnist':
        if config['style'] == 'single':
            trainer = MnistSingleTrainer
        if config['style'] == 'pairwise':
            trainer = MnistPairwiseTrainer
    return trainer



def trainer_factory(config):
    assert(config['task'] in ['mnist', 'allwords'])
    if config['task'] == 'mnist':
        assert(config['architecture'] != 'bem')
    if config['architecture'] == 'simple':
        assert(config['confidence'] == 'max_prob')
        assert(config['criterion']['name'] == 'nll')
        assert(config['train_style'] == 'single')
    if config['architecture'] == 'abstaining':
        assert(config['confidence'] in ['max_non_abs', 'inv_abs', 'abs'])
        assert(config['criterion']['name'] in ['conf1', 'conf4', 'pairwise'])
    if config['architecture'] == 'bem':
        assert(config['criterion']['name'] == 'crossentropy')
        assert(config['train_style'] == 'single')
    if config['train_style'] == 'single':
        assert(config['criterion']['name'] in ['crossentropy', 'nll', 'conf1', 'conf4'])
    if config['train_style'] == 'pairwise':
        assert(config['criterion']['name'] == 'pairwise')
        assert(config['architecture'] == 'abstaining')

    #construct train_loader
    train_loader_config = {}
    train_loader_config['task'] = config['task']
    train_loader_config['style'] = config['train_style']
    train_loader_config['architecture'] = config['architecture']
    train_loader_config['bsz'] = config['bsz']
    train_loader_config['confused'] = config['confused']
    train_loader = train_loader_factory(train_loader_config)

    #construct validation loader
    val_loader_config = {}
    val_loader_config['task'] = config['task']
    val_loader_config['style'] = config['train_style']
    val_loader_config['architecture'] = config['architecture']
    val_loader_config['bsz'] = config['bsz']
    val_loader_config['confused'] = config['confused']
    val_loader_config['dev_corpus'] = config['dev_corpus']
    val_loader = val_loader_factory(val_loader_config)

    #construct decoder
    decoder_config = {}
    decoder_config['architecture'] = config['architecture']
    decoder_config['task'] = config['task']
    decoder = decoder_factory(decoder_config)

    #construct model
    model_config = {}
    model_config['task'] = config['task']
    model_config['architecture'] = config['architecture']
    model_config['confidence'] = config['confidence']
    model = model_factory(model_config, train_loader)

    #construct optimizer
    optimizer_config = {}
    optimizer_config['task'] = config['task']
    optimizer = optimizer_factory(optimizer_config, model)

    #construct criterion
    criterion_config = config['criterion']
    criterion = criterion_factory(criterion_config)

    #n_epochs
    n_epochs = config['n_epochs']
    
    #construct trainer
    trainer_config = {}
    trainer_config['task'] = config['task']
    trainer_config['style'] = config['train_style']
    trainer_config['architecture'] = config['architecture']
    trainer_class = select_trainer(trainer_config)
    trainer = trainer_class(criterion, optimizer, train_loader, val_loader, decoder, n_epochs)

    print('model:', type(model).__name__)
    print('criterion:', type(criterion).__name__)
    print('optimizer:', type(optimizer).__name__)
    print('train loader:', type(train_loader).__name__)
    print('val loader:', type(val_loader).__name__)
    print('decoder:', type(decoder).__name__)
    print('n_epochs', n_epochs)

    return trainer, model

def run_experiment(config):
    trainer, model = trainer_factory(config)
    best_model = trainer(model)
    return best_model

if __name__ == "__main__":
    config =  {'task': 'allwords',
	       'architecture': 'bem',
	       'confidence': 'inv_abs',
	       'criterion': {'name': 'crossentropy',
                             'p0': None},
	       'confused': None,
	       'train_style': 'single',
	       'dev_corpus': 'semev07',
	       'bsz': 16,
	       'n_epochs': 20
	     }
    run_experiment(config)
