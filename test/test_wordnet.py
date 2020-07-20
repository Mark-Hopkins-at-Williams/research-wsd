import unittest
from reed_wsd.allwords.wordnet import wn_example
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer

class TestBEMDataset(unittest.TestCase):

    def setUp(self):
        self.tknz = BertTokenizer.from_pretrained('bert-base-uncased')
        

    def test_wn_example1(self):
        word = 'is'
        be = wn.lemma_from_key('be%2:42:03::')
        example_sent, span = wn_example(be, word, self.tknz)
        assert(example_sent == 'John is rich')        
        assert(span == [2, 3]) # note that [CLS] is token 0

    def test_wn_example2(self):
        word = 'cat'
        cat = wn.lemma_from_key('cat%1:05:00::')
        example_sent, span = wn_example(cat, word, self.tknz)
        expected = (word + ' is ' + 'feline mammal usually having ' + 
                    'thick soft fur and no ability to roar: domestic cats; ' + 
                    'wildcats')
        assert(example_sent == expected)
        print(span)
        assert(span == [1, 2])



        
if __name__ == "__main__":
	unittest.main()
