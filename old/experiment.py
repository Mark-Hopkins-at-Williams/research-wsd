# -*- coding: utf-8 -*-
import torch
from train import train_net
from networks import DropoutClassifier
from util import cudaify
from wordsense import SenseInstanceDatabase
from collections import defaultdict
from elmo import ElmoVectorizer
from bert import BertVectorizer

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
    

def create_and_train_net(net, training_data, test_data, verbose):
    training_data = cudaify(training_data)
    test_data = cudaify(test_data)
    if verbose:
        print("training size:", training_data.shape)
        print("testing size:", test_data.shape)
    classifier = cudaify(net)

    best_net, best_acc = train_net(classifier, training_data, test_data, tensor_batcher,
                batch_size=2, n_epochs=10, learning_rate=0.001,
                verbose=verbose)
    return best_acc



def train_lemma_classifiers(vectorize, lemmas, n_fold, max_sample_size,
                            instance_db, cached, verbose=True):
    lemma_info_dict = defaultdict(tuple)
    for lemma in lemmas:
        print('Training classifier for: {}.'.format(lemma))                    
        sense_hist = instance_db.sense_histogram(lemma)
        sense1 = sense_hist[0][1]
        sense2 = sense_hist[1][1]   
        print('  ...sampling sense pairs.')
        data = instance_db.sample_sense_pairs(vectorize, max_sample_size//2, lemma, 
                                              sense1, sense2, n_fold)

        sum_acc = 0
        fold_count = 0
        for training_data, test_data in data:
            print('  ...training fold {}.'.format(fold_count+1))
            sum_acc += create_and_train_net(DropoutClassifier(2 * vectorize.dim(), 100, 2), 
                                            training_data, test_data, verbose)
            fold_count += 1
        avg_acc = sum_acc / fold_count
        print("  ...best epoch accuracy average = {:.2f}".format(avg_acc))
        lemma_info_dict[lemma] = (avg_acc, sense1, sense2)
    return dict(lemma_info_dict)


def second_sense_support_filter(instance_db, min_sense2_freq, max_sense2_freq):
    def f(lemma):
        sense_hist = instance_db.sense_histogram(lemma)
        return (len(sense_hist) > 1 and 
                sense_hist[1][0] >= min_sense2_freq and 
                sense_hist[1][0] <= max_sense2_freq)
    
    return f



if __name__ == '__main__':
    #vectorizer = BertVectorizer()
    vectorizer = ElmoVectorizer()
    instance_db = SenseInstanceDatabase('senses_elmo')
    filt = second_sense_support_filter(instance_db, 50, 51)   
    lemmas = ['race']
    #lemmas = instance_db.lemmas(filt)
    results = train_lemma_classifiers(vectorizer, lemmas, 
                                      1, 2000, instance_db, 
                                      cached=False, verbose=True)   
    print(results)
