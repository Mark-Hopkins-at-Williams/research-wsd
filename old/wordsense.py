import os
from os.path import join
import json
from collections import defaultdict
import torch
from torch import tensor
import random
import pandas as pd
from torch.utils.data import Dataset
from util import cudaify
from align import align


class SenseTaggedSentences(Dataset):

    def __init__(self, st_sents, inventory):
        self.st_sents = []
        self.inventory = inventory
        self.senses = set()
        for sent in st_sents:
            #words = []
            for word in sent['words']:
                #if 'sense' in word:
                #    fields = word['sense'].split('%')
                #    sense = fields[-1]
                #    self.senses.add(sense)
                #    word['sense'] = sense
                #words.append(word)
                if 'sense' in word:
                    sense = word['sense']
                    self.senses.add(sense)
            #sent['words'] = words
            self.st_sents.append(sent)
        self.senses = sorted(self.senses)  
        print(self.senses)

    def sense_inventory(self):
        return self.senses

    def num_senses(self):
        return len(self.senses)

    def onehot(self, sense):
        result = torch.zeros(self.num_senses())
        result[self.senses.index(sense)] = 1.0
        return result

    def sense_id(self, sense):
        return self.senses.index(sense)

    def __getitem__(self, index):
        return self.st_sents[index]

    def __len__(self):
        return len(self.st_sents)   

    @staticmethod
    def from_json_str(json_str, corpus_id):        
        st_sents = json.loads(json_str)
        return SenseTaggedSentences(st_sents['corpora'][corpus_id], 
                                    st_sents['inventory'])

    @staticmethod
    def from_json(json_file, corpus_id):
        with open(json_file) as reader:
            st_sents = json.load(reader)        
        result = SenseTaggedSentences(st_sents['corpora'][corpus_id], 
                                      st_sents['inventory'])
        return result

class SenseInstanceDataset(Dataset):

    def __init__(self, st_sents, vec_manager):
        self.st_sents = st_sents
        self.vec_manager = vec_manager
        self.instance_index = 0
        self.instance_iter = self.item_iter()
        self.current_instance = next(self.instance_iter)

    def onehot(self, sense):
        return self.st_sents.onehot(sense)
        
    def sense_id(self, sense):
        return self.st_sents.sense_id(sense)
            
    def num_senses(self):
        return self.st_sents.num_senses()
    
    def item_iter(self):
        for i, index in enumerate(range(len(self.st_sents))):            
            st_sent = self.st_sents[index]                
            vecs = self.vec_manager.get_vector(st_sent['sentid'])
            old_toks = [wd['word'] for wd in st_sent['words']]    
            new_toks = vecs['tokens']
            alignment = align(old_toks, new_toks)
            if alignment is not None:
                for i, word in enumerate(st_sent['words']):
                    if 'sense' in word:
                        sense_inst = SenseInstance(old_toks, i, word['sense'], word['tag'])
                        (projection_start, projection_stop) = alignment[i]
                        embeddings = []
                        for j in range(projection_start, projection_stop):
                            embeddings.append(tensor(vecs['vecs'][j]))
                        embedding = torch.stack(embeddings).sum(dim=0).tolist()
                        sense_inst.add_embedding('embed', embedding)
                        yield sense_inst
        
    def __getitem__(self, index):
        if index < self.instance_index:
            self.instance_index = 0
            self.instance_iter = self.item_iter()
            self.current_instance = next(self.instance_iter)
        while self.instance_index < index:
            self.instance_index += 1
            self.current_instance = next(self.instance_iter)            
        return self.current_instance

    def __len__(self):
        return 200000   

class SenseInstanceLoader:
    
    def __init__(self, inst_ds, batch_size, desired_ids = None):
        self.inst_ds = inst_ds
        self.batch_size = batch_size
        if desired_ids is None:
            self.desired_ids = list(range(len(self.inst_ds)))
        else:
            self.desired_ids = desired_ids            

    def sense_id(self, sense):
        return self.inst_ds.sense_id(sense)
            
    def num_senses(self):
        return self.inst_ds.num_senses()
        
    def batch_iter(self):
        """Currently doesn't yield the incomplete last batch."""
        evidence_batch = []
        response_batch = []
        for i in self.desired_ids:
            evidence_batch.append(self.inst_ds[i].get_embedding('embed'))
            response_batch.append(self.inst_ds.sense_id(self.inst_ds[i].sense))
            if len(evidence_batch) == self.batch_size:
                yield cudaify(torch.tensor(evidence_batch)), cudaify(torch.tensor(response_batch))
                evidence_batch = []
                response_batch = []
    
                
    

 

class SenseInstance:
    """
    A SenseInstance corresponds to a single annotation of a word sense
    in a sentence/document. It has the following fields:
        - 'tokens' (a list of strings): the BERT tokens in the sentence
        - 'pos' (integer): the position of the sense-annotated token
        - 'sense' (string): the sense of the annotated token
    
    """   
    def __init__(self, tokens, pos, sense, sensetag):
        self.tokens = tokens
        self.pos = pos
        self.sense = sense
        self.sensetag = sensetag
        self.embeddings = dict()
        
    def __str__(self):
        return json.dumps(self.to_json())
        
    def has_embedding(self, label):
        return (label in self.embeddings)
    
    def get_embedding(self, label):
        if label in self.embeddings:
            return self.embeddings[label]
        else:
            return None
        
    def add_embedding(self, label, embedding):
        self.embeddings[label] = embedding
    
    def to_json(self):
        result = {'tokens': self.tokens,
                  'position': self.pos,
                  'sense': self.sense,
                  'sensetag': self.sensetag}
        result.update(self.embeddings)
        return result
    
    @staticmethod
    def from_json(jsonobj):
        instance = SenseInstance(jsonobj['tokens'],
                                 jsonobj['position'],
                                 jsonobj['sense'],
                                 jsonobj['sensetag'])
        for key in jsonobj:
            if key not in ["tokens", "position", "sense", "sensetag"]:
                instance.add_embedding(key, jsonobj[key])
        return instance

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



class SenseInstanceDatabase:
    
    def __init__(self, directory):
        self.directory = directory
        self.lemma_instances = dict()
        for dir_item in sorted(os.listdir(self.directory)):
            filename = join(self.directory, dir_item)
            if os.path.isfile(filename):            
                if (dir_item.endswith(".json") and 
                    not dir_item.endswith("id_to_sent.json")):
                    with open(filename, "r") as f:
                        lemma_data = json.load(f)    
                        result = []
                        for instance in lemma_data:
                            inst_sent = instance['tokens']
                            if len(inst_sent) > 511: continue
                            sense_inst = SenseInstance.from_json(instance)
                            result.append(sense_inst)
                        self.lemma_instances[dir_item[:-5]] = result
         
    def lemmas(self, lemma_filter = lambda x: True):
        for lemma in self.lemma_instances:
            if lemma_filter(lemma):
                yield lemma


    def all_instances(self):
        """
        Iterates through the lemmas. Every call to `next` provides a tuple
        (lemma, instances) where:
            - lemma is a string representation of a lemma (e.g. 'record')
            - instances is a list of SenseInstances corresponding to that lemma
        
        """
        for lemma in self.lemma_instances:
            yield (lemma, self.lemma_instances[lemma])

    def instances(self, lemma):
        """
        Returns a list of SenseInstances for a specified lemma. 
        
        """
        if lemma in self.lemma_instances:
            return self.lemma_instances[lemma]
        else:
            return []

    def sense_histogram(self, lemma):
        """
        Iterates through each lemma. Each call to `next` returns a tuple
        (lemma, histogram), where histogram is a list of pairs of the
        form (k, sense) such that there are k SenseInstances of sense
        corresponding to the lemma. This list is sorted by k, from greatest
        to least.
        
        """ 
        insts = self.instances(lemma)
        histogram = sense_histogram(insts)
        sense_freqs = reversed(sorted([(histogram[sense], sense) for 
                              sense in histogram]))
        return list(sense_freqs)


    def embeddings_by_sense(self, lemma, vectorize):
        """
        Returns a dictionary mapping each word sense (of a particular lemma)
        to a list of vector encodings of its SenseInstances.
        
        If use_cached == True, then this assumes that a file called
        `lemmadata/vectors/LEMMA.csv` exists and contains the serialized
        vectors in CSV format. If use_cached == False, the vectors are
        computed from scratch.
        
        """        
        def contextualized_vectors(lemma):    
            for instance in self.instances(lemma):
                # Bert can only handle sentences with a maximum of 512 tokens
                if len(instance.tokens) > 511:
                    continue
                yield (instance.sense, vectorize(instance))
           
        context_vecs = contextualized_vectors
        sense_vectors = defaultdict(list)
        for (sense, vec) in context_vecs(lemma):
            sense_vectors[sense].append(vec)
        return dict(sense_vectors)    



    def sample_sense_pairs(self, vectorize, n_pairs, lemma,
                           sense1, sense2, n_fold, cached=False, 
                           train_percent = 0.8):
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
            assert len(sense_vecs) >= 2, "sample_positive_sense_pair cannot be called with zero senses!"
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
            return ((vecs[:cutoff_start] + vecs[cutoff_end:]), vecs[cutoff_start:cutoff_end])
    
    
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
            return torch.cat([negatives, positives]).detach()
    
        vecs_by_sense = self.embeddings_by_sense(lemma, vectorize)
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





def sample_cross_lemma(vec, threshold, n_fold, n_pairs_each_lemma):

    def get_lemmas(threshold):
        data = pd.read_csv("data/classifier_data8_20-max.csv")
        lemmas = []
        for i in data.index:
            if data.iloc[i]["best_avg_acc"] >= threshold:
                lemmas.append(data.iloc[i]["lemma"])
        return lemmas

    vecdir = 'lemmadata'
    lemmas = get_lemmas(threshold)
    print("# of lemmas: " + str(len(lemmas)))
    cutoff = int(0.8 * len(lemmas))
    train_lemmas = lemmas[:cutoff]
    test_lemmas = lemmas[cutoff:]
    print(len(lemmas))
    n_fold_train = [None] * n_fold
    n_fold_test = [None] * n_fold
    n_fold_data = [None] * n_fold
    # sample train data
    for (lemma, sense_hist) in all_sense_histograms('lemmadata'):
        if len(sense_hist) > 1 and sense_hist[1][0] >= 21 and lemma in train_lemmas:
            sense1 = sense_hist[0][1]
            sense2 = sense_hist[1][1]
            print("Sampling for {}".format(lemma))
            train_data = sample_sense_pairs(vec, n_pairs_each_lemma//2, lemma, vecdir, sense1, sense2, n_fold)
            for i, fold in enumerate(train_data):
                if n_fold_train[i] is None:
                    n_fold_train[i] = torch.cat([train_data[i][0], train_data[i][1]]) 
                else:
                    n_fold_train[i] = torch.cat([n_fold_train[i], train_data[i][0], train_data[i][1]]) 
    # sample test data
    for (lemma, sense_hist) in all_sense_histograms('lemmadata'):
        if len(sense_hist) > 1 and sense_hist[1][0] >= 21 and lemma in test_lemmas:
            sense1 = sense_hist[0][1]
            sense2 = sense_hist[1][1]
            test_data = sample_sense_pairs(vec, n_pairs_each_lemma//2, lemma, vecdir, sense1, sense2, n_fold, cached=True)
            for i, fold in enumerate(test_data):
                if n_fold_test[i] is None:
                    n_fold_test[i] = torch.cat([test_data[i][0], test_data[i][1]]) 
                else:
                    n_fold_test[i] = torch.cat([n_fold_test[i], test_data[i][0], test_data[i][1]]) 
    for i in range(len(n_fold_train)):
        n_fold_train[i] = n_fold_train[i][torch.randperm(n_fold_train[i].shape[0])]
        n_fold_test[i] = n_fold_test[i][torch.randperm(n_fold_test[i].shape[0])]
        n_fold_data[i] = (n_fold_train[i][:10000], n_fold_test[i][:10000])
    return n_fold_data


def second_sense_support_filter(instance_db, min_sense2_freq, max_sense2_freq):
    def f(lemma):
        sense_hist = instance_db.sense_histogram(lemma)
        return (len(sense_hist) > 1 and 
                sense_hist[1][0] >= min_sense2_freq and 
                sense_hist[1][0] <= max_sense2_freq)
    
    return f

def precompute_embeddings(inputdir, outputdir, vectorizer, lemma_filter):
    filenames = sorted(os.listdir(inputdir))
    for dir_item in filenames:
        filename = join(inputdir, dir_item)
        if os.path.isfile(filename) and dir_item.endswith(".json"):
            if lemma_filter(dir_item[:-5]):
                print('Processing {}.'.format(dir_item))
                results = []            
                with open(filename, "r") as f:
                    lemma_data = json.load(f) 
                    for datum in lemma_data:
                        result = SenseInstance.from_json(datum)
                        vectorizer(result)
                        results.append(result.to_json())
                with open(join(outputdir, dir_item), 'w') as writer:
                    writer.write(json.dumps(results, indent=4))


                
if __name__ == "__main__":
    #make_explicit('lemmadata', 'senses')
    from elmo import ElmoVectorizer
    instance_db = SenseInstanceDatabase('senses')
    filt = second_sense_support_filter(instance_db, 100, 101)       
    precompute_embeddings('senses','senses_elmo', ElmoVectorizer(), filt)
            
            