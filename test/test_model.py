import unittest
from reed_wsd.allwords.model import BEMforWSD
from reed_wsd.allwords.model import abstention
import torch

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
        #defn_cls
        glosses = [{'input_ids': torch.tensor([[101, 1037, 5164, 1997, 2616, 17087, 1996, 24402, 3513, 1997, 1037, 2653, 102],
                                               [101, 1996, 2558, 1997, 2051, 1037, 7267, 2003, 8580, 102, 0, 0, 0]]),
                    'token_type_ids': torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                    'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]),
                    'span': [[0, 1], [0, 1]]},
                   {'input_ids': torch.tensor([[101, 1037, 5164, 1997, 2616, 17087, 1996, 24402, 3513, 1997, 1037, 2653, 102],
                                               [101, 1996, 2558, 1997, 2051, 1037, 7267, 2003, 8580, 102, 0, 0, 0]]),
                    'token_type_ids': torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                    'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]),
                    'span': [[0, 1], [0, 1]]},
                   {'input_ids': torch.tensor([[101, 1037, 5164, 1997, 2616, 17087, 1996, 24402, 3513, 1997, 1037, 2653, 102],
                                               [101, 1996, 2558, 1997, 2051, 1037, 7267, 2003, 8580, 102, 0, 0, 0]]),
                    'token_type_ids': torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                    'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]),
                    'span': [[0, 1], [0, 1]]}]
        pos = [[3, 4], [5, 6], [6, 7]]
        contexts = {'input_ids': torch.tensor([[ 101, 2023, 2003, 6251, 2028, 1012, 102, 0, 0, 0],
                                               [ 101, 2023, 2003, 1996, 2936, 6251, 2048, 1012, 102, 0],
                                               [ 101, 2023, 2003, 1996, 2130, 2936, 6251, 2093, 1012,  102]]),
                    'token_type_ids': torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                    'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])} 

        expected_output = torch.tensor([[36.7255, 38.4328],
        			  [34.4906, 29.1099],
        			  [32.1247, 29.1898]])	
        with torch.no_grad():
            scores = model(contexts, glosses, pos)
        assert(scores.allclose(expected_output))

class TestFFN(unittest.TestCase):
    def test_abstention(self):
        input_vec = torch.tensor([[1., 1, 1, 3],
                                 [1., 0, 1, 2]])
        zones = [[1, 3], [0, 2]]
        normalized, confidence = abstention(input_vec, zones)
        assert(torch.allclose(normalized, torch.tensor([[0, 0.5, 0.5, 3],
                                                 [0.7310, 0.2690, 0, 2]]),
                                   atol=0.0001))
        assert(torch.allclose(confidence, torch.tensor([3., 2]), atol=0.0001))

        

if __name__ == '__main__':
    unittest.main()

