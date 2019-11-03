import os
from os.path import join
import json
from elmo import elmo_vectorize_instance

class SenseInstance:
    """
    A SenseInstance corresponds to a single annotation of a word sense
    in a sentence/document. It has the following fields:
        - 'tokens' (a list of strings): the BERT tokens in the sentence
        - 'pos' (integer): the position of the sense-annotated token
        - 'sense' (string): the sense of the annotated token
    
    """   
    def __init__(self, tokens, pos, sense):
        self.tokens = tokens
        self.pos = pos
        self.sense = sense
        self.embeddings = dict()
        
    def add_embedding(self, label, embedding):
        self.embeddings[label] = embedding
    
    def to_json(self):
        result = {'tokens': self.tokens,
                  'position': self.pos,
                  'sense': self.sense}
        result.update(self.embeddings)
        return result
    
    @staticmethod
    def from_json(jsonobj):
        instance = SenseInstance(jsonobj['tokens'],
                                 jsonobj['position'],
                                 jsonobj['sense'])
        for key in jsonobj:
            if key not in ["tokens", "position", "sense"]:
                instance.add_embedding(key, jsonobj[key])
        return instance
    
def precompute_embeddings(inputdir, outputdir, vectorizer):
    """
    Iterates through the lemmas. Every call to `next` provides a tuple
    (lemma, instances) where:
        - lemma is a string representation of a lemma (e.g. 'record')
        - instances is a list of SenseInstances corresponding to that lemma
    
    """
    for dir_item in os.listdir(inputdir):
        filename = join(inputdir, dir_item)
        if os.path.isfile(filename) and dir_item.endswith(".json"):
            print('Processing {}.'.format(dir_item))
            results = []            
            with open(filename, "r") as f:
                lemma_data = json.load(f) 
                for datum in lemma_data:                    
                    result = SenseInstance.from_json(datum)
                    result = vectorizer(result)
                    results.append(result.to_json())
            with open(join(outputdir, dir_item), 'w') as writer:
                writer.write(json.dumps(results, indent=4))
                
if __name__ == "__main__":
    precompute_embeddings('senses','senses2',elmo_vectorize_instance)
            
            