import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from os.path import join
import torch
from torch import optim
from reed_wsd.allwords.model import SimpleFFN, SimpleAbstainingFFN
from reed_wsd.util import cudaify, Logger
from reed_wsd.allwords.evaluate import AllwordsEmbeddingDecoder
from reed_wsd.allwords.wordsense import SenseTaggedSentences, SenseInstanceDataset, SenseInstanceLoader, TwinSenseInstanceLoader
from reed_wsd.allwords.vectorize import DiskBasedVectorManager
from reed_wsd.loss import NLLLoss, ConfidenceLoss1, ConfidenceLoss4, PairwiseConfidenceLoss
from reed_wsd.allwords.train import SingleEmbeddingTrainer, PairwiseEmbeddingTrainer
from reed_wsd.plot import PYCurve, plot_curves
from tqdm import tqdm
import copy

file_dir = os.path.dirname(os.path.realpath(__file__))


def init_dataset(data_dir, stage, style, sense_sz=-1, gloss=None):
    assert(stage == "train" or stage == "dev")
    assert(style == "bem" or style == "fnn")
    if gloss is not None:
        assert(style == "bem")
    if stage == "dev":
        corpus_id = 'data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.data.xml'
    elif stage == "train":
        corpus_id = 'data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml'

    filename = join(data_dir, 'raganato.json')
    sents = SenseTaggedSentences.from_json(filename, corpus_id) 
    if style == "bem":
        ds = BEMDataset(sents, sense_sz=sense_sz, gloss=gloss)
    if style == 'fnn':
        vecmgr = DiskBasedVectorManager(join(join(data_dir, 'vecs'), corpus_id))
        ds = SenseInstanceDataset(sents, vecmgr)
    return ds

def baseline():
    train_ds = init_dataset("./data", stage="train", style="fnn")
    val_ds = init_dataset("./data", stage="dev", style="fnn")
    train_loader = SenseInstanceLoader(train_ds, batch_size=16)
    val_loader = SenseInstanceLoader(val_ds, batch_size=16)
    criterion = NLLLoss()
    model = SimpleFFN(768, train_loader.num_senses())
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    decoder = AllwordsEmbeddingDecoder()
    n_epochs = 20
    trainer = SingleEmbeddingTrainer(criterion, optimizer, train_loader, val_loader, decoder, n_epochs)
    best_net = trainer(model)

def confidence_max1():
    train_ds = init_dataset("./data", stage="train", style="fnn")
    val_ds = init_dataset("./data", stage="dev", style="fnn")
    train_loader = SenseInstanceLoader(train_ds, batch_size=16)
    val_loader = SenseInstanceLoader(val_ds, batch_size=16)
    criterion = ConfidenceLoss1(0.5)
    model = SimpleAbstainingFFN(768, train_loader.num_senses(), zone_applicant='max_non_abs')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    decoder = AllwordsEmbeddingDecoder()
    n_epochs = 20
    trainer = SingleEmbeddingTrainer(criterion, optimizer, train_loader, val_loader, decoder, n_epochs)
    best_net = trainer(model)

def confidence_inv1():
    train_ds = init_dataset("./data", stage="train", style="fnn")
    val_ds = init_dataset("./data", stage="dev", style="fnn")
    train_loader = SenseInstanceLoader(train_ds, batch_size=16)
    val_loader = SenseInstanceLoader(val_ds, batch_size=16)
    criterion = ConfidenceLoss1(0.5)
    model = SimpleAbstainingFFN(768, train_loader.num_senses(), zone_applicant='inv_abs')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    decoder = AllwordsEmbeddingDecoder()
    n_epochs = 20
    trainer = SingleEmbeddingTrainer(criterion, optimizer, train_loader, val_loader, decoder, n_epochs)
    best_net = trainer(model)

def confidence_max4():
    train_ds = init_dataset("./data", stage="train", style="fnn")
    val_ds = init_dataset("./data", stage="dev", style="fnn")
    train_loader = SenseInstanceLoader(train_ds, batch_size=16)
    val_loader = SenseInstanceLoader(val_ds, batch_size=16)
    criterion = ConfidenceLoss4(0.5)
    model = SimpleAbstainingFFN(768, train_loader.num_senses(), zone_applicant='max_non_abs')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    decoder = AllwordsEmbeddingDecoder()
    n_epochs = 20
    trainer = SingleEmbeddingTrainer(criterion, optimizer, train_loader, val_loader, decoder, n_epochs)
    best_net = trainer(model)

def confidence_inv4():
    train_ds = init_dataset("./data", stage="train", style="fnn")
    val_ds = init_dataset("./data", stage="dev", style="fnn")
    train_loader = SenseInstanceLoader(train_ds, batch_size=16)
    val_loader = SenseInstanceLoader(val_ds, batch_size=16)
    criterion = ConfidenceLoss4(0.5)
    model = SimpleAbstainingFFN(768, train_loader.num_senses(), zone_applicant='inv_abs')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    decoder = AllwordsEmbeddingDecoder()
    n_epochs = 20
    trainer = SingleEmbeddingTrainer(criterion, optimizer, train_loader, val_loader, decoder, n_epochs)
    best_net = trainer(model)

def confidence_pair():
    train_ds = init_dataset("./data", stage="train", style="fnn")
    val_ds = init_dataset("./data", stage="dev", style="fnn")
    train_loader = TwinSenseInstanceLoader(train_ds, batch_size=16)
    val_loader = SenseInstanceLoader(val_ds, batch_size=16)
    criterion = PairwiseConfidenceLoss()
    model = SimpleAbstainingFFN(768, train_loader.num_senses(), zone_applicant='abs')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    decoder = AllwordsEmbeddingDecoder()
    n_epochs = 20
    trainer = PairwiseEmbeddingTrainer(criterion, optimizer, train_loader, val_loader, decoder, n_epochs)
    best_net = trainer(model)

if __name__ == "__main__":
    confidence_pair()
