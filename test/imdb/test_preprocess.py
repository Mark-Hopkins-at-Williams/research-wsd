import unittest
from reed_wsd.imdb.preprocess import sequence_to_vec
from transformers import BertModel, BertTokenizer

class TestPreprocess(unittest.TestCase):
    def setUp(self):
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def test_sequence_to_vec(self):
        seq = 'This is an example movie comment. Honestly, this movie sucks.'
        cls_embedding = sequence_to_vec(seq, self.tokenizer, self.bert)
        print(cls_embedding.shape)

if __name__ == '__main__':
    unittest.main()
