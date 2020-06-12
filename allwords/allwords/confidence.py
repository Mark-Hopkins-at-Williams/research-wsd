import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from os.path import join
import torch
from torch import optim
from evaluate import evaluate, decode, precision_yield_curve 
from allwords.networks import AffineClassifier, DropoutClassifier
from allwords.util import cudaify, Logger
from allwords.wordsense import SenseTaggedSentences, SenseInstanceDataset, SenseInstanceLoader
from allwords.vectorize import DiskBasedVectorManager

file_dir = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    batch_size = 16
    data_dir = sys.argv[1]
    dev_corpus_id = 'data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.data.xml'
    filename = join(data_dir, 'raganato.json')
    dev_sents = SenseTaggedSentences.from_json(filename, dev_corpus_id) 
    dev_vecmgr = DiskBasedVectorManager(join(join(data_dir, 'vecs'), dev_corpus_id))
    dev_ds = SenseInstanceDataset(dev_sents, dev_vecmgr)
    dev_loader = SenseInstanceLoader(dev_ds, batch_size)
    input_size = 768 # TODO: what is it in general?
    output_size = dev_loader.num_senses()
    net = AffineClassifier(input_size, output_size)
    net.load_state_dict(torch.load(join(file_dir, "../saved/bert_simple.pt")))
    net = cudaify(net)
    precision_yield_curve(net, dev_loader)
