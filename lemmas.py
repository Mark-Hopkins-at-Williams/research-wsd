# -*- coding: utf-8 -*-
import json
import os
from collections import defaultdict

import torch

from pytorch_transformers import BertModel, BertConfig, BertTokenizer
import random

from os.path import join
import pandas as pd 
from bert import vectorize_instance
from wordsense import SenseInstance

config = BertConfig.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel(config)    

def lemmadata_iter():
    """
    Iterates through the lemmas. Every call to `next` provides a tuple
    (lemma, instances) where:
        - lemma is a string representation of a lemma (e.g. 'record')
        - instances is a list of SenseInstances corresponding to that lemma
    
    """
    with open("lemmadata/id_to_sent.json") as sent_id_dict_file:
        sent_id_dict = json.load(sent_id_dict_file)
    for dir_item in os.listdir("lemmadata"):
        filename = join("lemmadata", dir_item)
        if os.path.isfile(filename):            
            if (dir_item.endswith(".json") and 
                not dir_item.endswith("id_to_sent.json")):
                with open(filename, "r") as f:
                    lemma_data = json.load(f)    
                    result = []
                    for instance in lemma_data:
                        inst_sent_id = instance["sent_id"]
                        inst_sense = instance["sense"]
                        inst_sent = sent_id_dict[str(inst_sent_id)]
                        result.append(SenseInstance(inst_sent, 
                                                    instance['pos'], 
                                                    inst_sense))
                    yield((dir_item[:-5], result))

def lemmadata(lemma):
    """
    Returns a list of SenseInstances for a specified lemma. 
    
    """
    for (other_lemma, data) in lemmadata_iter():
        if other_lemma == lemma:
            return data
    return []

def sense_histogram(instances):
    """
    Given a list of SenseInstances, this counts how many instances
    correspond to each sense and returns a dictionary mapping each
    word sense to its instance count.
    
    """    
    result = defaultdict(int)
    for instance in instances:
        result[instance.sense] += 1
    return result


def all_sense_histograms():
    """
    Iterates through each lemma. Each call to `next` returns a tuple
    (lemma, histogram), where histogram is a list of pairs of the
    form (k, sense) such that there are k SenseInstances of sense
    corresponding to the lemma. This list is sorted by k, from greatest
    to least.
    
    """
    
    lemma_iter = lemmadata_iter()
    for (lemma, instances) in lemma_iter:
        histogram = sense_histogram(instances)
        sense_freqs = reversed(sorted([(histogram[sense], sense) for 
                              sense in histogram]))
        yield (lemma, list(sense_freqs))
            
   

    

def contextualized_vectors_by_sense(lemma, use_cached = False):
    """
    Returns a dictionary mapping each word sense (of a particular lemma)
    to a list of vector encodings of its SenseInstances.
    
    If use_cached == True, then this assumes that a file called
    `lemmadata/vectors/LEMMA.csv` exists and contains the serialized
    vectors in CSV format. If use_cached == False, the vectors are
    computed from scratch.
    
    """
    def cached_contextualized_vectors(lemma):
        filename = join("lemmadata/vectors", lemma + ".csv")
        data = pd.read_csv(filename, delimiter=",")
        for k in range(len(data.index)):
            curr = data.iloc[k]
            yield (curr.iloc[2], torch.tensor(curr.iloc[4:]))
    
    def contextualized_vectors(lemma):    
        for instance in lemmadata(lemma):
            # Bert can only handle sentences with a maximum of 512 tokens
            if len(instance.tokens) > 511:
                continue
            yield (instance.sense, vectorize_instance(instance))

    if use_cached:
        context_vecs = cached_contextualized_vectors
    else:
        context_vecs = contextualized_vectors
    sense_vectors = defaultdict(list)
    for (sense, vec) in context_vecs(lemma):
        sense_vectors[sense].append(vec)
    return dict(sense_vectors)    

         
def sample_sense_pairs(n_pairs, lemma, sense1, sense2, n_fold, train_percent = 0.8):
    """
    Creates training and test data for classifying whether two SenseInstances
    of a particular lemma correspond to the same sense (positive) or a
    different sense (negative).
    
    First, the SenseInstances for the specified senses are split into train
    and test, according to the train_percent. Then we randomly generate n 
    positive training examples and n negative training examples.
    
    """
    
    def sample_negative_sense_pair(sense_vecs):
        assert len(sense_vecs) >= 2, "sample_negative_sense_pair cannot be called with only one sense!"
        sense1, sense2 = random.sample(list(range(len(sense_vecs))), 2)
        vec1 = random.choice(sense_vecs[sense1])
        vec2 = random.choice(sense_vecs[sense2])
        return torch.cat([vec1, vec2])
       
    def sample_positive_sense_pair(sense_vecs):
        # Currently may break if every sense doesn't have at least two vectors.
        assert len(sense_vecs) >= 1, "sample_positive_sense_pair cannot be called with zero senses!"
        sense = random.choice(list(range(len(sense_vecs))))
        vec1, vec2 = random.sample(sense_vecs[sense], 2)
        return torch.cat([vec1, vec2])

    def train_test_split(vecs, percent_train):
        random.shuffle(vecs)
        cutoff = int(percent_train * len(vecs))
        return vecs[:cutoff], vecs[cutoff:]
    
    def train_test_split_nfold(vecs, percent_train, nthfold, n_fold):
        random.shuffle(vecs)
        cutoff_start = int((nthfold / n_fold) * len(vecs))
        cutoff_end = int(((nthfold + 1) / n_fold) * len(vecs))
        return (vecs[:cutoff_start] + vecs[cutoff_end:]), vecs[cutoff_start:cutoff_end]


    def sample(sense1_vecs, sense2_vecs):
        vecs = [sense1_vecs, sense2_vecs]
        positives = torch.stack([sample_positive_sense_pair(vecs) for 
                                 _ in range(n_pairs)])
        negatives = torch.stack([sample_negative_sense_pair(vecs) for
                                 _ in range(n_pairs)])
        positives = torch.cat([torch.ones(len(positives), 1), positives], 
                               dim=1)
        negatives = torch.cat([torch.zeros(len(negatives), 1), negatives], 
                               dim=1)
        return torch.cat([negatives, positives])

    vecs_by_sense = contextualized_vectors_by_sense(lemma)
    sense1_vecs = vecs_by_sense[sense1]
    sense2_vecs = vecs_by_sense[sense2]
    print('sampling {} with senses of magnitude {} and {}'.format(lemma, 
          len(sense1_vecs), len(sense2_vecs)))

    if n_fold == 1:
        sense1_vecs_train, sense1_vecs_test = train_test_split(sense1_vecs,
                                                               train_percent) 
        sense2_vecs_train, sense2_vecs_test = train_test_split(sense2_vecs,
                                                               train_percent)
        train = sample(sense1_vecs_train, sense2_vecs_train)
        test = sample(sense1_vecs_test, sense2_vecs_test)

        return [(train, test)]
    elif n_fold > 1:
        data = []
        for i in range(n_fold):
            sense1_vecs_train, sense1_vecs_test = train_test_split_nfold(sense1_vecs,
                                                               train_percent,
                                                               i, n_fold) 
            sense2_vecs_train, sense2_vecs_test = train_test_split_nfold(sense2_vecs,
                                                               train_percent,
                                                               i, n_fold)
            train = sample(sense1_vecs_train, sense2_vecs_train)
            test = sample(sense1_vecs_test, sense2_vecs_test)

            data.append((train, test))

        return data

    
