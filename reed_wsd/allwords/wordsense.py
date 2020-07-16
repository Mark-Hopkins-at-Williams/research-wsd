import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math
import json
import torch
import random
from torch import tensor
from torch.utils.data import Dataset
import reed_wsd.util as util
import reed_wsd.allwords.align as align
from reed_wsd.allwords.bert import tokenize_with_target
from collections import defaultdict
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer
from nltk.tokenize import word_tokenize
import nltk

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

    def get_senses_by_lemma(self):
        return self.senses_by_lemma

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

    def __init__(self, st_sents, inventory, n_insts):
        self.st_sents = st_sents
        self.inventory = SenseInventory(inventory)
        self.n_insts = n_insts

    def onehot(self, sense):
        result = torch.zeros(self.num_senses())
        result[self.senses.index(sense)] = 1.0
        return result

    def get_inventory(self):
        return self.inventory

    def set_inventory(self, inv):
        self.inv = inv

    def get_n_insts(self):
        return self.n_insts

    def __getitem__(self, index):
        return self.st_sents[index]

    def __len__(self):
        return len(self.st_sents)

    @staticmethod
    def from_json_dict(st_sents, corpus_id):
        sents = st_sents['corpora'][corpus_id]['sents']
        inv = st_sents['inventory']
        n_insts = st_sents['corpora'][corpus_id]['n_insts']
        result = SenseTaggedSentences(sents, inv, n_insts)
        return result
       
    @staticmethod
    def from_json_str(json_str, corpus_id):        
        st_sents = json.loads(json_str)
        return SenseTaggedSentences.from_json_dict(st_sents, corpus_id)

    @staticmethod
    def from_json(json_file, corpus_id):
        with open(json_file) as reader:
            st_sents = json.load(reader)        
        return SenseTaggedSentences.from_json_dict(st_sents, corpus_id)



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
    
    def get_sense(self):
        return self.sense

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


class BEMDataset(Dataset):

    def __init__(self, st_sents, randomize_sents = True, sense_sz=-1, gloss='defn_cls'):
        assert(sense_sz == -1 or sense_sz > 0,
               "sense_sz must be either positive integer or -1")
        assert(gloss in ['defn_cls', 'defn_tgt', 'wneg']
        self.st_sents = st_sents
        self.instance_index = 0
        self.gloss = gloss
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
                lemma = self.inv.sense_lemma(word['sense'])
                if 'sense' in word and lemma in self.inv.get_senses_by_lemma():
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

    @staticmethod
    def random_wn_example(lemma, word):
        def nltk_tag_to_wordnet_tag(nltk_tag):
	    if nltk_tag.startswith('J'):
		return wordnet.ADJ
	    elif nltk_tag.startswith('V'):
		return wordnet.VERB
	    elif nltk_tag.startswith('N'):
		return wordnet.NOUN
	    elif nltk_tag.startswith('R'):
		return wordnet.ADV
	    else:          
		return None

        def lemmatize_sentence(sentence):
	    """
            takes a str sentence and return a list of lemmatized tokens
            """
	    #tokenize the sentence and find the POS tag for each token
	    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
	    #tuple of (token, wordnet_tag)
	    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
	    lemmatized_sentence = []
	    for word, tag in wordnet_tagged:
		if tag is None:
		    #if there is no available tag, append the token as is
		    lemmatized_sentence.append(word)
		else:        
		    #else use the tag to lemmatize the token
		    lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
	    return lemmatized_sentence
        syns_egs = lemma.synset().examples()
        lemmatized_syns_egs = [lemmatize_sentence(sent) for sent in syns_egs]
	tgt_lemma = lemmatize_sentence(word)[0]
        example_pairs = []
        for i, eg in enumerate(lammatized_syns_egs):
            if tgt_lemma in eg:
                index = eg.index(tgt_lemma)
                example_paris.append([syns_egs[i], index])
        if len(example_pairs) == 0:
            gloss, span = wn_definition_with_target(lemma)
            return gloss, span
        random_pair = random.sample(example_pairs, 1)[0]
        _, span = tokenize_with_target(self.tknz, random_pair[0], random_pair[1])
        return random_pair[0], span

    @staticmethod
    def wn_definition_with_target(wn_lemma):
        pos = wn_lemma.synset().pos()
        defn = wn_lemma.synset().definition()
        if pos = 'n':
            gloss = wn_lemma.name() + ' is ' + defn
            rep_span = [1, 1 + len(self.tknz.tokenize(wn_lemma.name()))]
        if pos in ['a', 's', 'r']:
            gloss = wn_lemma.name() + ' means ' + defn
            rep_span = [1, 1 + len(self.tknz.tokenize(wn_lemma.name()))]
        if pos == 'v':
            gloss = 'to ' + wn_lemma.name() ' is to ' + defn
            rep_span = [2, 2 + len(self.tknz.tokenize(wn_lemma.name()))]
        return gloss, rep_span


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
                    if lemma in self.inv:
                        input_ids, target_range = tokenize_with_target(self.tknz, old_toks, i)
                        senses = self.inv.get_senses(lemma)
                        correct_sense_i = senses.index(s) 
                        if self.gloss = 'defn_cls':
                            glosses = [wn.lemma_from_key(sense).synset().definition() for sense in senses]
                            glosses_ids = self.tknz(glosses, padding=True, return_tensors='pt')
                            gloss_span = [[0, 1]] * len(glosses)
                            glosses_ids['span'] = gloss_span
                        if self.gloss = 'defn_tgt':
                            glosses = []
                            rep_spans = []
                            for sense in senses:
                                wn_lemma = wn.lemma_from_key(sense)
                                gloss, rep_span = self.wn_definition_with_target(wn_lemma)
                                glosses.append(gloss)
                                rep_spans.append(rep_span)
                            glosses_ids = self.tknz(glosses, padding=True, return_tensors='pt')
                            glosses_ids['span'] = rep_spans
                        if self.gloss = 'wneg':
                            glosses = []
                            rep_spans = []
                            for sense in senses:
                                wn_lemma = wn.lemma_from_key(sense)
                                gloss, rep_span = self.random_wn_example(wn_lemma, word['word'])
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
                    

        

class SenseInstanceDataset(Dataset):

    def __init__(self, st_sents, vec_manager, randomize_sents=True, sense_sz=-1):
        self.st_sents = st_sents
        self.inv = self.st_sents.get_inventory()
        self.vec_manager = vec_manager
        self.instance_index = 0
        self.instance_iter = self.item_iter()
        self.randomize_sents = randomize_sents
        self.num_insts = self.st_sents.get_n_insts()
        self.current_instance = next(self.instance_iter)
        if sense_sz > 0:
            all_senses_by_lemma = self.inv.get_senses_by_lemma()
            key_samples = random.sample(all_senses_by_lemma.keys(), sense_sz)
            new_senses_by_lemma = {}
            for k in key_samples:
                new_senses_by_lemma[k] = all_senses_by_lemma[k]
            self.inv = SenseInventory(new_senses_by_lemma)

            # recount n_insts
            n_insts = 0
            sent_ids = list(range(len(self.st_sents)))
            for i, index in enumerate(sent_ids):            
                st_sent = self.st_sents[index] 
                for i, word in enumerate(st_sent['words']):
                    w = word['word']
                    if 'sense' in word and w in self.inv.get_senses_by_lemma():
                        n_insts += 1
            self.num_insts = n_insts

    def onehot(self, sense):
        return self.st_sents.onehot(sense)

    def get_inventory(self):
        return self.inv
    
    def set_randomize_sents(self, value):
        self.randomize_sents = value
         
    def item_iter(self):
        sent_ids = list(range(len(self.st_sents)))
        if self.randomize_sents:
            random.shuffle(sent_ids)
        for i, index in enumerate(sent_ids):            
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

    def sense_id(self, sense):
        return self.inventory.sense_id(sense)
            
    def num_senses(self):
        return self.inventory.num_senses()
    
    def sense(self, sense_id):
        return self.inventory.sense(sense_id)

    def __len__(self):
        return math.ceil(self.n_insts / self.batch_size)

    def batch_iter(self):
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



class SenseInstanceLoader:
    
    def __init__(self, inst_ds, batch_size, desired_ids = None):
        self.inst_ds = inst_ds
        self.n_insts = len(self.inst_ds)
        self.batch_size = batch_size
        if desired_ids is None:
            self.desired_ids = list(range(len(self.inst_ds)))
        else:
            self.desired_ids = desired_ids            
        self.inventory = self.inst_ds.get_inventory()

    def __len__(self):
        return math.ceil(self.n_insts / self.batch_size)
            
    def get_instance_dataset(self):
        return self.inst_ds

    def get_inventory(self):
        return self.inventory

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
            
                
    



