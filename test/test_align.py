import unittest
from reed_wsd.allwords.align import align

class TestAlign(unittest.TestCase):
    
    def test_align(self):
        orig_sent = ("The jury said it did find that many_of Georgia 's " +
                     "registration and election laws `` are outmoded or " +
                     "inadequate and often ambiguous '' .")
        bert_sent = ("[CLS] the jury said it did find that many of " +
                     "georgia ' s registration and election laws ` ` are " +
                     "out ##mo ##ded or inadequate and often " +
                     "ambiguous ' ' . [SEP]")
        orig_toks = orig_sent.lower().split()
        bert_toks = bert_sent.lower().split()
        special_toks =  ['[CLS]', '[SEP]']
        result = align(orig_toks, bert_toks, special_toks)
        expected = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), 
                    (8, 10), (10, 11), (11, 13), (13, 14), (14, 15), (15, 16), 
                    (16, 17), (17, 19), (19, 20), (20, 23), (23, 24), 
                    (24, 25), (25, 26), (26, 27), (27, 28), (28, 30), (30, 31)]
        assert(result == expected)
  
    def test_align2(self):
        orig_sent = ("The jury said it did")
        bert_sent = ("the jury said it did [SEP]")
        orig_toks = orig_sent.lower().split()
        bert_toks = bert_sent.lower().split()
        special_toks =  ['[CLS]', '[SEP]']
        result = align(orig_toks, bert_toks, special_toks)
        expected = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        assert(result == expected)    
    
if __name__ == "__main__":
	unittest.main()
