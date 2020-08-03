import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from os.path import join
import torch
from torch import optim
from reed_wsd.allwords.evaluate import evaluate, decode_gen, decode_BEM
from reed_wsd.allwords.networks import AffineClassifier, DropoutClassifier, BEMforWSD
from reed_wsd.util import cudaify, Logger
from reed_wsd.allwords.wordsense import SenseTaggedSentences, SenseInstanceDataset, SenseInstanceLoader
from reed_wsd.allwords.blevins import BEMDataset, BEMLoader
from reed_wsd.allwords.vectorize import DiskBasedVectorManager
from reed_wsd.allwords.loss import NLLLossWithZones, PairwiseConfidenceLossWithZones
from reed_wsd.plot import PYCurve, plot_curves
from tqdm import tqdm
import copy

file_dir = os.path.dirname(os.path.realpath(__file__))


def init_loader(data_dir, stage, style, batch_size = 16, sense_sz=-1, gloss=None):
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
        loader = BEMLoader(ds, batch_size)
    if style == 'fnn':
        vecmgr = DiskBasedVectorManager(join(join(data_dir, 'vecs'), corpus_id))
        ds = SenseInstanceDataset(sents, vecmgr)
        loader = SenseInstanceLoader(ds, batch_size)
    return loader

if __name__ == "__main__":
    pass
