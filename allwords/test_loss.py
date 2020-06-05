import unittest
from torch import tensor
import torch
from loss import LossWithZones, NLLLossWithZones

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
   
    
     
    def test_loss_with_zones(self):
        loss = LossWithZones()
        predicted = tensor([[0.2, 0, 0.2, 0.6], [0.6, 0.2, 0, 0.2]])
        gold = tensor([0, 3])
        zones = [(0, 2), (2, 4)]
        assert(approx(loss(predicted, gold, zones).item(), -1.0))
        predicted = tensor([[0.2, 0.2, 0, 0.6], [0.6, 0, 0.2, 0.2]])
        gold = tensor([0, 3])
        zones = [(0, 2), (2, 4)]
        assert(approx(loss(predicted, gold, zones).item(), -0.5))
        predicted = tensor([[0.6, 0.2, 0, 0.2], [0.2, 0, 0.2, 0.6]])
        gold = tensor([0, 3])
        zones = [(0, 2), (2, 4)]
        assert(approx(loss(predicted, gold, zones).item(), -0.75))
        predicted = tensor([[0.6, 0.2, 0, 0.2], [0.2, 0, 0.6, 0.2]])
        gold = tensor([0, 3])
        zones = [(0, 2), (2, 4)]
        assert(approx(loss(predicted, gold, zones).item(), -0.5))

    def test_nll_loss_with_zones(self):
        loss = NLLLossWithZones()
        predicted = tensor([[0.2, 0, 0.2, 0.6], [0.6, 0.2, 0, 0.2]])
        gold = tensor([0, 3])
        zones = [(0, 2), (2, 4)]
        assert(approx(loss(predicted, gold, zones).item(), 0.0))
        predicted = tensor([[0.2, 0.2, 0, 0.6], [0.6, 0, 0.2, 0.2]])
        gold = tensor([0, 3])
        zones = [(0, 2), (2, 4)]
        assert(approx(loss(predicted, gold, zones).item(), 0.6931471805599453))
        predicted = tensor([[0.6, 0.2, 0, 0.2], [0.2, 0, 0.2, 0.6]])
        gold = tensor([0, 3])
        zones = [(0, 2), (2, 4)]
        assert(approx(loss(predicted, gold, zones).item(), 0.2877))
        predicted = tensor([[0.6, 0.2, 0, 0.2], [0.2, 0, 0.6, 0.2]])
        gold = tensor([0, 3])
        zones = [(0, 2), (2, 4)]
        assert(approx(loss(predicted, gold, zones).item(), 0.8370))


        
     
if __name__ == "__main__":
	unittest.main()
