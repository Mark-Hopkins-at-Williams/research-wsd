import unittest
import math
import torch
from torch import tensor
from reed_wsd.allwords.loss import LossWithZones, NLLLossWithZones, zone_based_loss, ConfidenceLossWithZones
from reed_wsd.mnist.loss import PairwiseConfidenceLoss
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
    
    def test_pairwise_confidence_loss(self):
        criterion = PairwiseConfidenceLoss()
        output_x = torch.tensor([[0.2, 0.5, 0.3],   # distribution 1  over classA, classB, abstain
                                 [0.5, 0.5, 0]])    # distribution 2  over classA, classB, abstain
        gold_x = torch.tensor([0, 1])            # gold labels for instances 1, 2
        output_y = torch.tensor([[0.5, 0.2, 0.3],
                                 [0.5, 0.1, 0.4]])
        gold_y = torch.tensor([1, 0])        
        confidence_x = torch.tensor([0., 0.]) # confidences from first network
        confidence_y = torch.tensor([0., 0.]) # confidences from second network

        # loss should be -log [ ((0.2 * 0.5 + 0.2 * 0.5) + (0.5 * 0.5 + 0.5 * 0.5)) / 2 ]
        # which equals 1.0498
        expected_loss = torch.tensor(1.0498)
        loss = criterion(output_x, output_y, gold_x, gold_y, confidence_x, confidence_y)
        assert(torch.allclose(loss, expected_loss, atol=10**(-4)))

    def test_pairwise_confidence_loss2(self):
        criterion = PairwiseConfidenceLoss()
        output_x = torch.tensor([[0.2, 0.5, 0.3],   # distribution 1  over classA, classB, abstain
                                 [0.5, 0.5, 0]])    # distribution 2  over classA, classB, abstain
        gold_x = torch.tensor([1, 0])            # gold labels for instances 1, 2
        output_y = torch.tensor([[0.5, 0.2, 0.3],
                                 [0.5, 0.1, 0.4]])
        gold_y = torch.tensor([1, 0])        
        confidence_x = torch.tensor([0., 0.]) # confidences from first network
        confidence_y = torch.tensor([0., 0.]) # confidences from second network

        # loss should be -log [ ((0.5 * 0.5 + 0.2 * 0.5) + (0.5 * 0.5 + 0.5 * 0.5)) / 2 ]
        # which equals 0.8557
        expected_loss = torch.tensor(0.8557)
        loss = criterion(output_x, output_y, gold_x, gold_y, confidence_x, confidence_y)
        assert(torch.allclose(loss, expected_loss, atol=10**(-4)))
    
    
    def test_pairwise_confidence_loss3(self):
        criterion = PairwiseConfidenceLoss()
        output_x = torch.tensor([[0.2, 0.3, 0.5],   # distribution 1  over classA, classB, abstain
                                 [0.1, 0.5, 0.4]])    # distribution 2  over classA, classB, abstain
        gold_x = torch.tensor([1, 0]) # assigned gold probs are 0.3 and 0.1
        output_y = torch.tensor([[0.4, 0.5, 0.1],
                                 [0.6, 0.2, 0.2]])
        gold_y = torch.tensor([0, 1]) # assigned gold probs are 0.4 and 0.2
        confidence_x = torch.tensor([-math.log(2), math.log(3)]) # confidences from first network
        confidence_y = torch.tensor([math.log(2), -math.log(3)]) # confidences from second network

        # the weighted prob of instance 1 is 0.2 * 0.3 + 0.8 * 0.4
        # the weighted prob of instance 2 is 0.9 * 0.4 + 0.1 * 0.2
        # loss should be -log [ ((0.2 * 0.3 + 0.8 * 0.4) + (0.9 * 0.4 + 0.1 * 0.2)) / 2 ]
        # which equals 0.8557
        expected_loss = torch.tensor(1.406497)
        loss = criterion(output_x, output_y, gold_x, gold_y, confidence_x, confidence_y)
        assert(torch.allclose(loss, expected_loss, atol=10**(-4)))

     
if __name__ == "__main__":
    unittest.main()

