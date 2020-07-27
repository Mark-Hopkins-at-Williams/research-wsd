import unittest
import torch
from torch import tensor
from reed_wsd.allwords.loss import LossWithZones, NLLLossWithZones, zone_based_loss, ConfidenceLossWithZones
from reed_wsd.allwords.loss import PairwiseConfidenceLossWithZones

def approx(x, y, num_digits = 4):
    return abs(x-y) < 1.0 * (10 ** -num_digits)

class TestLoss(unittest.TestCase):
    def test_pairwise_confidence_loss_with_zones(self):
        criterion = PairwiseConfidenceLossWithZones('abs')

        output_x = torch.tensor([[1,1,1,1,1.],
                                 [0,1,0,0,0.]])
        output_y = torch.tensor([[1,1,0,0,0.],
                                 [0,0,1,1,1.]])
        zones_x = [[0, 2], [1, 3]]
        zones_y = [[2, 4], [3, 4]]
        gold_x = [1, 1]
        gold_y = [2, 3]
        expected_losses = torch.tensor((0.6931 + 0.0842) / 2)

        losses = criterion(output_x, output_y, gold_x, gold_y, zones_x, zones_y)
        assert(torch.allclose(losses, expected_losses, atol=10**(-4)))

    
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
     
if __name__ == "__main__":
    unittest.main()

