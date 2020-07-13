import unittest
from reed_wsd.allwords.networks import BEMforWSD
import torch
from transformers import BertModel

class TestBEMforWSD(unittest.TestCase):

    def test_target_rep(self):
        # shape is 3x3x4 (batch size x num tokens x encoding length)
        # note [0, 0, 0, 0] is a "padding" token encoding
        context_rep = torch.tensor([[[1, 2, 3, 4],
                                     [4, 3, 2, 1],
                                     [0, 0, 0, 0]],
                                    [[1, 1, 1, 1],
                                     [2, 2, 2, 2],
                                     [3, 3, 3, 3]],
                                    [[1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 1, 0]]]).float()
        # spans of the target token(s)
        spans = [[0, 1],
                 [1, 3],
                 [2, 3]]

        expected_result = torch.tensor([[1, 2, 3, 4],
                                        [2.5, 2.5, 2.5, 2.5],
                                        [0, 0, 1, 0]]).float()
        result = BEMforWSD.target_representation(context_rep, spans)
        assert( torch.equal(result, expected_result))

    def test_forward(self):
        model = BEMforWSD()
        model.eval()
        # there are 3 contexts, each a list of token ids
        # "attention mask" tells us which tokens are not padding
        contexts = {'input_ids': torch.tensor([[ 101, 2023, 2003, 6251, 2028, 1012, 102, 0, 0, 0],
                                               [ 101, 2023, 2003, 1996, 2936, 6251, 2048, 1012, 102, 0],
                                               [ 101, 2023, 2003, 1996, 2130, 2936, 6251, 2093, 1012,  102]]),
                    'token_type_ids': torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                    'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])} 
        # each of the 3 contexts has 2 glosses
        glosses = [{'input_ids': torch.tensor([[101, 1037, 5164, 1997, 2616, 17087, 1996, 24402, 3513, 1997, 1037, 2653, 102],
                                               [101, 1996, 2558, 1997, 2051, 1037, 7267, 2003, 8580, 102, 0, 0, 0]]),
                    'token_type_ids': torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                    'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]])},
                   {'input_ids': torch.tensor([[101, 1037, 5164, 1997, 2616, 17087, 1996, 24402, 3513, 1997, 1037, 2653, 102],
                                               [101, 1996, 2558, 1997, 2051, 1037, 7267, 2003, 8580, 102, 0, 0, 0]]),
                    'token_type_ids': torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                    'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]])},
                   {'input_ids': torch.tensor([[101, 1037, 5164, 1997, 2616, 17087, 1996, 24402, 3513, 1997, 1037, 2653, 102],
                                               [101, 1996, 2558, 1997, 2051, 1037, 7267, 2003, 8580, 102, 0, 0, 0]]),
                    'token_type_ids': torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                    'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]])}]
        spans = [[3, 4], [5, 6], [6, 7]]
        # each context is transformed by the NN into scores over its 2 glosses (senses)
        # note that these are not normalized, but the loss function will apply
        # a softmax prior to scoring loss
        # e.g. 38.4328 is the dot product of the first context 0 with the second gloss
        expected_output = torch.tensor([[36.7255, 38.4328],
        			                        [34.4906, 29.1099],
        			                        [32.1247, 29.1898]])	
        scores = model(contexts, glosses, spans).detach()
        assert(scores.allclose(expected_output))

if __name__ == '__main__':
    unittest.main()

