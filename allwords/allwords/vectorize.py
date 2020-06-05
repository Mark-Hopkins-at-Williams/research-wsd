import os
from os.path import join
import json
import sys
from .bert import BertSentenceVectorizer
    
class VectorManager:
    def get_vector(self, sent_id):
        raise NotImplementedError('Cannot call .get_vector on abstract class.')

class RamBasedVectorManager(VectorManager):
    def __init__(self, vec_map):
        super().__init__()
        self.vec_map = vec_map
        
    def get_vector(self, sent_id):
        return self.vec_map[sent_id]
         

class DiskBasedVectorManager(VectorManager):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
    
    def get_filename(self, sent_id):
        code = sent_id // 100
        subdir = join(self.root_dir, 'batch{}'.format(code))
        if not os.path.exists(subdir):
            print('Making subdirectory: {}'.format(subdir))
            os.makedirs(subdir)
        filename = join(subdir, 'sent{}.json'.format(sent_id))
        return filename

    def get_vector(self, sent_id):
        filename = self.get_filename(sent_id)
        if not os.path.exists(filename):
            return None
        else:
            with open(filename) as reader:
                data = json.load(reader)
            return data
    
    def write(self, sent_id, toks, vectors):
        with open(self.get_filename(sent_id), 'w') as writer:
            output = {'sentid': sent_id,
                      'tokens': toks,
                      'vecs': vectors}
            writer.write(json.dumps(output))

def normalize(word):
    words = word.split("_")
    return ' '.join(words)

def vectorize_sent(sent, vectorizer):
    sent_id = sent['sentid']
    words = [normalize(wd['word']) for wd in sent['words']]
    sent_str = ' '.join(words)
    toks, vectors = vectorizer(sent_str)
    return sent_id, toks, vectors
        
def vectorize_sents(sents, vectorizer, writer):
    for sent in sents:
        sent_id, toks, vectors = vectorize_sent(sent, vectorizer)
        writer.write(sent_id, toks, vectors.tolist())
        
def vectorize_json(json_file, vectorizer, vector_dir):
    if not os.path.exists(vector_dir):
        os.makedirs(vector_dir)
    writer = DiskBasedVectorManager(vector_dir)
    with open(json_file) as f:
        sents = json.load(f)
        vectorize_sents(sents, vectorizer, writer)
        
if __name__ == '__main__':
    vectorize_json(sys.argv[1], BertSentenceVectorizer(), sys.argv[2])
    