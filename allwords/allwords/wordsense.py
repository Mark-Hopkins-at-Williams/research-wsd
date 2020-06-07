import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import torch
from torch import tensor
from torch.utils.data import Dataset
from allwords import util
import allwords.align as align
from collections import defaultdict

class SenseInventory:
    def __init__(self, senses_by_lemma):
        self.senses_by_lemma = {x: tuple(sorted(senses_by_lemma[x])) 
                                for x in senses_by_lemma}
        self.all_senses = []
        self.lemma_ranges = dict()
        current_id = 0
        for lemma in sorted(self.senses_by_lemma):
            for sense in self.senses_by_lemma[lemma]:
                self.all_senses.append(sense)
            prev_id = current_id
            current_id += len(self.senses_by_lemma[lemma])
            self.lemma_ranges[lemma] = (prev_id, current_id)
             
    def contains_lemma(self, lemma):
        return lemma in self.senses_by_lemma
            
    def sense_lemma(self, sense):
        return SenseInventory.deconstruct_sense(sense)[0]
    
    def sense_id(self, sense):
        lemma = self.sense_lemma(sense)
        lemma_range = self.lemma_ranges[lemma]
        lemma_subindex = self.senses_by_lemma[lemma].index(sense)
        return lemma_range[0] + lemma_subindex
        
    def sense(self, sense_id):
        return self.all_senses[sense_id]
    
    def sense_range(self, lemma):
        return self.lemma_ranges[lemma]

    def get_senses(self, lemma):
        return self.senses_by_lemma[lemma]

    def max_senses_per_target(self):
        num_senses = [len(self.senses_by_lemma[key]) 
                      for key in self.senses_by_lemma]
        return max(num_senses)

    def all_senses(self):
        result = []
        for key in self.senses_by_lemma:
            result += self.senses_by_lemma[key]
        return result
    
    def num_senses(self):
        return len(self.all_senses)

    @staticmethod
    def deconstruct_sense(sense):
        return sense.split('%')

    @staticmethod
    def from_sense_iter(senses):
        senses_by_lemma = defaultdict(set)
        for sense in senses:
            (lemma, _) = SenseInventory.deconstruct_sense(sense)
            senses_by_lemma[lemma].add(sense)
        return SenseInventory(senses_by_lemma)
        
        
        
class SenseTaggedSentences(Dataset):

    def __init__(self, st_sents, inventory):
        self.st_sents = st_sents
        self.inventory = SenseInventory(inventory)

    def onehot(self, sense):
        result = torch.zeros(self.num_senses())
        result[self.senses.index(sense)] = 1.0
        return result

    def get_inventory(self):
        return self.inventory

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


class SenseInstance:
    """
    A SenseInstance corresponds to a single annotation of a word sense
    in a sentence/document. It has the following fields:
        - 'tokens' (a list of strings): the BERT tokens in the sentence
        - 'pos' (integer): the position of the sense-annotated token
        - 'sense' (string): the sense of the annotated token
    
    """   
    def __init__(self, inst_id, tokens, pos, sense, sensetag):
        self.inst_id = inst_id
        self.tokens = tokens
        self.pos = pos
        self.sense = sense        
        self.sensetag = sensetag
        self.embeddings = dict()
        
    def __str__(self):
        return json.dumps(self.to_json())
    
    def get_target(self):
        fields = self.sense.split('%')
        return fields[0]
        
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
        result = {'id': self.inst_id,
                  'tokens': self.tokens,
                  'position': self.pos,
                  'sense': self.sense,
                  'sensetag': self.sensetag}
        result.update(self.embeddings)
        return result
    
    @staticmethod
    def from_json(jsonobj):
        instance = SenseInstance(jsonobj('id'),
                                 jsonobj['tokens'],
                                 jsonobj['position'],
                                 jsonobj['sense'],
                                 jsonobj['sensetag'])
        for key in jsonobj:
            if key not in ["id", "tokens", "position", "sense", "sensetag"]:
                instance.add_embedding(key, jsonobj[key])
        return instance




class SenseInstanceDataset(Dataset):

    def __init__(self, st_sents, vec_manager):
        self.st_sents = st_sents
        self.vec_manager = vec_manager
        self.instance_index = 0
        self.instance_iter = self.item_iter()
        self.current_instance = next(self.instance_iter)
        self.num_insts = 250000

    def onehot(self, sense):
        return self.st_sents.onehot(sense)

    def get_inventory(self):
        return self.st_sents.get_inventory()
         
    def item_iter(self):
        for i, index in enumerate(range(len(self.st_sents))):            
            st_sent = self.st_sents[index] 
            vecs = self.vec_manager.get_vector(st_sent['sentid'])
            old_toks = [wd['word'] for wd in st_sent['words']]    
            new_toks = vecs['tokens']
            alignment = align.align(old_toks, new_toks)
            if alignment is not None:
                for i, word in enumerate(st_sent['words']):
                    if 'sense' in word:
                        sense_inst = SenseInstance(word['id'], old_toks, i, word['sense'], word['tag'])
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
        return self.num_insts

class SenseInstanceLoader:
    
    def __init__(self, inst_ds, batch_size, desired_ids = None):
        self.inst_ds = inst_ds
        self.batch_size = batch_size
        if desired_ids is None:
            self.desired_ids = list(range(len(self.inst_ds)))
        else:
            self.desired_ids = desired_ids            
        self.inventory = self.inst_ds.get_inventory()
            
    def sense_id(self, sense):
        return self.inventory.sense_id(sense)
            
    def num_senses(self):
        return self.inventory.num_senses()
    
    def sense(self, sense_id):
        return self.inventory.sense(sense_id)
            
    def batch_iter(self):
        inst_ids = []
        target_batch = []
        evidence_batch = []
        response_batch = []
        zones = []
        for i in self.desired_ids:
            inst_ids.append(self.inst_ds[i].inst_id)
            target_batch.append(self.inst_ds[i].get_target())
            evidence_batch.append(self.inst_ds[i].get_embedding('embed'))
            response_batch.append(self.sense_id(self.inst_ds[i].sense))
            lemma = self.inst_ds[i].sense.split("%")[0]
            zones.append(self.inventory.sense_range(lemma))
            if len(evidence_batch) == self.batch_size:
                yield (inst_ids,
                       target_batch, 
                       util.cudaify(torch.tensor(evidence_batch)), 
                       util.cudaify(torch.tensor(response_batch)),
                       zones)                
                inst_ids = []
                target_batch = []
                evidence_batch = []
                response_batch = []
                zones = []
        if len(inst_ids) > 0:
            yield (inst_ids,
                   target_batch, 
                   util.cudaify(torch.tensor(evidence_batch)), 
                   util.cudaify(torch.tensor(response_batch)),
                   zones)
            
                
    



