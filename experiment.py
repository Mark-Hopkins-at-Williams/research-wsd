# -*- coding: utf-8 -*-
import torch
import pandas as pd

from train import train_net
from networks import SimpleClassifier, DropoutClassifier
from util import cudaify
from lemmas import all_sense_histograms, sample_sense_pairs
from compare import getExampleSentencesBySense
def tensor_batcher(t, batch_size):
    def shuffle_rows(a):
        return a[torch.randperm(a.size()[0])]        
    neg = t[(t[:, 0] == 0).nonzero().squeeze(1)] # only negative rows
    pos = t[(t[:, 0] == 1).nonzero().squeeze(1)] # only positive rows
    neg = shuffle_rows(neg)
    pos = shuffle_rows(pos)
    min_row_count = min(neg.shape[0], pos.shape[0])
    epoch_data = torch.cat([neg[:min_row_count], pos[:min_row_count]])    
    epoch_data = shuffle_rows(epoch_data)
    for i in range(0, len(epoch_data), batch_size):
        yield epoch_data[i:i+batch_size]
        
def test_training(d, k):
    def nth_dim_positive_data(n, d, k):
        data = torch.randn(d, k)
        u = torch.cat([torch.clamp(torch.sign(data[2:3]), min=0), data])
        return u.t()

    train = nth_dim_positive_data(2, d, k)
    dev = nth_dim_positive_data(2, d, 500)
    #test = nth_dim_positive_data(2, d, 500)   
    classifier = SimpleClassifier(d,100,2)
    train_net(classifier, train, dev, tensor_batcher,
              batch_size=96, n_epochs=30, learning_rate=0.001,
              verbose=True)


def create_and_train_net(training_data, test_data):
    training_data = cudaify(training_data)
    test_data = cudaify(test_data)
    print("training size:", training_data.shape)
    print("testing size:", test_data.shape)
    classifier = cudaify(DropoutClassifier(1536, 700, 2))
    train_net(classifier, training_data, test_data, tensor_batcher,
                batch_size=96, n_epochs=30, learning_rate=0.001,
                verbose=True)
    
def train_from_csv(train_csv, dev_csv):
    print('loading train')
    train = torch.tensor(pd.read_csv(train_csv).values).float()
    print('train size: {}'.format(train.shape[0]))
    print('loading dev')
    dev = torch.tensor(pd.read_csv(dev_csv).values).float()
    print('dev size: {}'.format(dev.shape[0]))
    return create_and_train_net(train, dev)

def train_lemma_classifiers(min_sense2_freq, max_sense2_freq):
    for (lemma, sense_hist) in all_sense_histograms():
        if len(sense_hist) > 1 and sense_hist[1][0] >= min_sense2_freq and sense_hist[1][0] <= max_sense2_freq:
            sense1 = sense_hist[0][1]
            sense2 = sense_hist[1][1]   
            print(lemma)
            print(sense1)
            for sent in getExampleSentencesBySense(sense1):
                print(sent)
            print(sense2)               
            for sent in getExampleSentencesBySense(sense2):
                print(sent)      
            training_data, test_data = sample_sense_pairs(300, lemma, sense1, sense2)
            create_and_train_net(training_data, test_data)

   