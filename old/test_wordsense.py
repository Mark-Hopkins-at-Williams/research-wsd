import unittest
from wordsense import SenseInstanceDatabase
from bert import BertVectorizer
import torch

class TestWordsense(unittest.TestCase):
    
    def setUp(self):
        self.db = SenseInstanceDatabase('testdata')
        self.vectorize_instance = BertVectorizer()
    
    def test_lemmadata_iter(self):
        itr = self.db.lemmas()
        (focus, instances) = next(itr)
        assert focus == 'embolden'
        assert len(instances) == 2
        expected_tokens = """i suppose i am naive ##ly driven to consider 
                            that the bio ##sphere , with its urgent diversity 
                            in which , em ##bold ##ened by all our know - how , 
                            we do get on with a very rich conversation , may 
                            very early already have harbor ##ed all the levels 
                            of which den ##nett speaks ."""
        expected_position = 21
        expected_sense = '/dictionary/sense/en_us_NOAD3e_2012/m_en_us1243778.001'
        assert ' '.join(instances[0].tokens) == ' '.join(expected_tokens.split())
        assert instances[0].pos == expected_position
        assert instances[0].sense == expected_sense


    def test_contextualized_vectors_by_sense(self):
        vecs = self.db.embeddings_by_sense('stock', self.vectorize_instance)
        senses = set(vecs.keys())
        expected_senses = ['/dictionary/sense/en_us_NOAD3e_2012/m_en_us1294508.007', 
                           '/dictionary/sense/en_us_NOAD3e_2012/m_en_us1294508.001', 
                           '/dictionary/sense/en_us_NOAD3e_2012/m_en_us1294508.032', 
                           '/dictionary/sense/en_us_NOAD3e_2012/m_en_us1294508.020', 
                           '/dictionary/sense/en_us_NOAD3e_2012/m_en_us1294508.029']
        assert set(senses) == set(expected_senses)
        sense_vecs = vecs['/dictionary/sense/en_us_NOAD3e_2012/m_en_us1294508.032']
        assert len(sense_vecs) == 4
        assert sense_vecs[0].shape == torch.Size([768])
        
    def test_sample_sense_pairs(self):
        sense_pairs = self.db.sample_sense_pairs(self.vectorize_instance, 3, 
                                         'stock',
                                         '/dictionary/sense/en_us_NOAD3e_2012/m_en_us1294508.007', 
                                         '/dictionary/sense/en_us_NOAD3e_2012/m_en_us1294508.001', 
                                         n_fold=2, cached=False, train_percent=.75)
        assert len(sense_pairs) == 2
        (fold1, fold2) = sense_pairs
        (train, test) = fold1
        assert len(train) == 6
        assert len(test) == 6
        assert(train[0].shape == torch.Size([1537]))
        assert(train[0][0] == 0.0)   # negative examples
        assert(train[1][0] == 0.0)
        assert(train[2][0] == 0.0)
        assert(train[3][0] == 1.0)   # positive examples
        assert(train[4][0] == 1.0)
        assert(train[5][0] == 1.0)
 

if __name__ == "__main__":
	unittest.main()
