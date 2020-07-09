import unittest
import torch
from torch import tensor
from reed_wsd.allwords.loss import LossWithZones, NLLLossWithZones, zone_based_loss, ConfidenceLossWithZones
from reed_wsd.allwords.loss import CrossEntropyLossBEM

def approx(x, y, num_digits = 4):
    return abs(x-y) < 1.0 * (10 ** -num_digits)

class TestLoss(unittest.TestCase):
    
    def test_nll_loss(self):
        loss = torch.nn.NLLLoss()
        predicted = tensor([[1, 0, 0, 0.], [0, 1, 0, 0.]])
        gold = tensor([0, 1])
        assert(loss(predicted, gold).item() == -1.0)
        predicted = tensor([[0.8, 0, 0, 0.2], [0.2, 0.8, 0, 0.]])
        gold = tensor([0, 1])
        assert(approx(loss(predicted, gold).item(), -0.8))
        predicted = tensor([[0.2, 0, 0, 0.8], [0.2, 0.8, 0, 0.]])
        gold = tensor([0, 1])
        assert(approx(loss(predicted, gold).item(), -0.5))
   
    def test_zone_based_loss(self):
        f = lambda x: -x
        predicted = tensor([[0.2, 0, 0.2, 0.6], [0.6, 0.2, 0, 0.2]])
        gold = tensor([0, 3])
        zones = [(0, 2), (2, 4)]
        assert(approx(zone_based_loss(predicted, gold, zones, f).item(), -1.0))
        predicted = tensor([[0.2, 0.2, 0, 0.6], [0.6, 0, 0.2, 0.2]])
        gold = tensor([0, 3])
        zones = [(0, 2), (2, 4)]
        assert(approx(zone_based_loss(predicted, gold, zones, f).item(), -0.5))
        predicted = tensor([[0.6, 0.2, 0, 0.2], [0.2, 0, 0.2, 0.6]])
        gold = tensor([0, 3])
        zones = [(0, 2), (2, 4)]
        assert(approx(zone_based_loss(predicted, gold, zones, f).item(), -0.75))
        predicted = tensor([[0.6, 0.2, 0, 0.2], [0.2, 0, 0.6, 0.2]])
        gold = tensor([0, 3])
        zones = [(0, 2), (2, 4)]
        assert(approx(zone_based_loss(predicted, gold, zones, f).item(), -0.5))
    
     
    def test_loss_with_zones(self):
        loss = LossWithZones()
        predicted = tensor([[0, 0, 0, 0.], [0, 0, 0, 0.]])
        gold = tensor([0, 3])
        zones = [(0, 2), (2, 4)]
        assert(approx(loss(predicted, gold, zones).item(), -0.5))
 

    def test_nll_loss_with_zones(self):
        loss = NLLLossWithZones()
        predicted = tensor([[0, 0, 0, 0.], [0, 0, 0, 0.]])
        gold = tensor([0, 3])
        zones = [(0, 2), (2, 4)]
        assert(approx(loss(predicted, gold, zones).item(), 0.6931))

    def test_confidence_loss_with_zones(self):
        loss = ConfidenceLossWithZones(0.5)
        predicted = tensor([[0, 0, 0, 0., 0], [0, 0, 0, 0., 0]])
        gold = tensor([0, 3])
        zones = [(0, 2), (2, 4)]
        e = loss(predicted, gold, zones).item()
        assert(approx(e, 0.6931))

    def test_cross_entropy_loss_BEM(self):
        loss = CrossEntropyLossBEM()
        scores = torch.tensor([[1., 0, 0, float('-inf'), float('-inf')],
                                 [0.5, 0.5, 0.1, 0.1, 0.1],
                                 [0.8, 0.6, 0.2, 0.1, float('-inf')]])
        gold = [0, 1, 3]
        expected_output = (0.5514 + 1.3890 + 1.7523) / 3
        output = loss(scores, gold).item()
        assert( approx(expected_output, output) )

     
if __name__ == "__main__":
    unittest.main()

