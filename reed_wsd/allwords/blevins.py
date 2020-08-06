import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math
import random
from torch.utils.data import Dataset
from reed_wsd.allwords.bert import tokenize_with_target
from reed_wsd.allwords.wordnet import wn_example, wn_definition_with_target
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer
import nltk
from reed_wsd.allwords.wordsense import SenseInventory

class BEMDataset(Dataset):

    def __init__(self, st_sents, randomize_sents = True, sense_sz=-1, 
                 gloss='defn_cls', random_wneg=False):
        assert sense_sz == -1 or sense_sz > 0, "sense_sz must be either positive integer or -1"
        assert(gloss in ['defn_cls', 'defn_tgt', 'wneg'])
        if random_wneg:
            assert gloss == 'wneg', 'need wneg gloss to turn on random examples'
        self.st_sents = st_sents
        self.instance_index = 0
        self.gloss = gloss
        self.random_wneg = random_wneg
        self.randomize_sents = randomize_sents
        self.tknz = BertTokenizer.from_pretrained('bert-base-uncased')
        self.num_insts = self.st_sents.get_n_insts()
        self.inv = self.st_sents.get_inventory()
        self.instance_iter = self.item_iter()
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.current_instance = next(self.instance_iter)
        if sense_sz > 0:
            all_senses_by_lemma = self.inv.get_senses_by_lemma()
            key_samples = random.sample(all_senses_by_lemma.keys(), sense_sz)
            new_senses_by_lemma = {}
            for k in key_samples:
                new_senses_by_lemma[k] = all_senses_by_lemma[k]
            self.inv = SenseInventory(new_senses_by_lemma)
            # recount n_insts
            self.count_insts()

    def count_insts(self):
        n_insts = 0
        sent_ids = list(range(len(self.st_sents)))
        for i, index in enumerate(sent_ids):            
            st_sent = self.st_sents[index] 
            for i, word in enumerate(st_sent['words']):
                if 'sense' in word:
                    lemma = self.inv.sense_lemma(word['sense'])
                    if lemma in self.inv.get_senses_by_lemma():
                        n_insts += 1
        self.num_insts = n_insts
        
    def onehot(self, sense):
        return self.st_sents.onehot(sense)

    def get_inventory(self):
        return self.inv

    def set_inventory(self, inv):
        self.inv = inv
        self.count_insts()
    
    def set_randomize_sents(self, value):
        self.randomize_sents = value

    def item_iter(self):
        sent_ids = list(range(len(self.st_sents)))
        if self.randomize_sents:
            random.shuffle(sent_ids)
        for i, index in enumerate(sent_ids):            
            st_sent = self.st_sents[index] 
            old_toks = [wd['word'] for wd in st_sent['words']]    
            for i, word in enumerate(st_sent['words']):
                if 'sense' in word:
                    s = word['sense']
                    lemma = self.inv.sense_lemma(s)
                    if lemma in self.inv.get_senses_by_lemma():
                        input_ids, target_range = tokenize_with_target(self.tknz, old_toks, i)
                        senses = self.inv.get_senses(lemma)
                        correct_sense_i = senses.index(s) 
                        if self.gloss == 'defn_cls':
                            glosses = [wn.lemma_from_key(sense).synset().definition() for sense in senses]
                            glosses_ids = self.tknz(glosses, padding=True, return_tensors='pt')
                            gloss_span = [[0, 1]] * len(glosses)
                            glosses_ids['span'] = gloss_span
                        if self.gloss == 'defn_tgt':
                            glosses = []
                            rep_spans = []
                            for sense in senses:
                                wn_lemma = wn.lemma_from_key(sense)
                                gloss, rep_span = wn_definition_with_target(self.tknz, wn_lemma)
                                glosses.append(gloss)
                                rep_spans.append(rep_span)
                            glosses_ids = self.tknz(glosses, padding=True, return_tensors='pt')
                            glosses_ids['span'] = rep_spans
                        if self.gloss == 'wneg':
                            glosses = []
                            rep_spans = []
                            for sense in senses:
                                wn_lemma = wn.lemma_from_key(sense)
                                gloss, rep_span = wn_example(wn_lemma, word['word'], self.tknz, rand=self.random_wneg)
                                glosses.append(gloss)
                                rep_spans.append(rep_span)
                            glosses_ids = self.tknz(glosses, padding=True, return_tensors='pt')
                            glosses_ids['span'] = rep_spans
                        yield {'input_ids': input_ids, 'pos': target_range,
                               'glosses_ids': glosses_ids, 'sense_id': correct_sense_i}

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
                    

class BEMLoader:

    def __init__(self, bem_ds, batch_size, desired_ids = None):
        self.ds = bem_ds
        self.n_insts = len(self.ds)
        self.batch_size = batch_size
        if desired_ids is None:
            self.desired_ids = list(range(len(self.ds)))
        else:
            self.desired_ids = desired_ids            
        self.inventory = self.ds.get_inventory()
        self.tknz = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_dataset(self):
        return self.ds

    def get_inventory(self):
        return self.inventory

    def set_inventory(self, inv):
        self.ds.set_inventory(inv)

    def sense_id(self, sense):
        return self.inventory.sense_id(sense)
            
    def num_senses(self):
        return self.inventory.num_senses()
    
    def sense(self, sense_id):
        return self.inventory.sense(sense_id)

    def __len__(self):
        return math.ceil(self.n_insts / self.batch_size)

    def __iter__(self):
        input_sents_batch = []
        glosses_ids_batch = []
        pos_batch = []
        gold_batch = []
        for i in self.desired_ids:
            input_ids = self.ds[i]['input_ids']
            string_sent = self.tknz.decode(input_ids, skip_special_tokens=True)
            input_sents_batch.append(string_sent)
            glosses_ids_batch.append(self.ds[i]['glosses_ids'])
            pos_batch.append(self.ds[i]['pos'])
            gold_batch.append(self.ds[i]['sense_id'])
            if len(input_sents_batch) == self.batch_size:
                contexts = self.tknz(input_sents_batch, padding=True, return_tensors='pt')
                yield {'contexts': contexts,
                       'glosses': glosses_ids_batch,
                       'span': pos_batch,
                       'gold': gold_batch}
                input_sents_batch = []
                glosses_ids_batch = []
                pos_batch = []
                gold_batch = []
        if len(input_sents_batch) > 0:
            contexts = self.tknz(input_sents_batch, padding=True, return_tensors='pt')
            yield {'contexts': contexts,
                   'glosses': glosses_ids_batch,
                   'span': pos_batch,
                   'gold': gold_batch}

