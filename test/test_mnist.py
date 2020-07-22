import unittest
import os
import math
from os.path import join
import sys
file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, join(file_dir, ".."))
import torch
from reed_wsd.mnist.loss import NLL, NLLA, AWNLL, CAWNLL, CRANLL, LRANLL
from reed_wsd.mnist.loss import CABNLL, ConfidenceLoss1, ConfidenceLoss2
from reed_wsd.mnist.loss import PairwiseConfidenceLoss
from reed_wsd.mnist.mnist import confuse

class Test(unittest.TestCase):
    
    def close_enough(self, x, y):
        return (round(x * 1000) / 1000 == round(y * 1000) / 1000)

    def test_nll(self):
        pred1 = [0.1, 0.2, 0.3, 0.3, 0.1] # loss = 1.4485
        pred2 = [0.25, 0.1, 0.3, 0.05, 0.2] # loss = 0.96318
        pred3 = [0.05, 0.02, 0.02, 0.01, 0.9] # loss = 0.46051
        # mean_loss = 0.95740
        gold = torch.tensor([1, 2, 3])
        preds = torch.tensor([pred1, pred2, pred3])
        criterion = NLL()
        expected_loss = - (math.log(0.2) + math.log(0.3) + math.log(0.01)) / 3
        loss = criterion(preds, gold).item()
        assert self.close_enough(loss, expected_loss)

    def test_nlla(self):
        pred1 = [0.1, 0.2, 0.3, 0.3, 0.1] # loss = 1.4485
        pred2 = [0.25, 0.1, 0.3, 0.05, 0.2] # loss = 0.96318
        pred3 = [0.05, 0.02, 0.02, 0.01, 0.9] # loss = 0.46051
        # mean_loss = 0.95740
        gold = torch.tensor([1, 2, 3])
        preds = torch.tensor([pred1, pred2, pred3])
        criterion = NLLA()
        expected_loss = (1.4485 + 0.96318 + 0.46051) / 3
        loss = criterion(preds, gold, abstain_i=4).item()
        assert self.close_enough(loss, expected_loss)

    def test_confuse(self):
        torch.manual_seed(1234567)
        labels = torch.tensor([4, 2, 9, 0, 1, 8, 0, 1, 7, 7])
        new_labels1 = confuse(labels)
        new_labels2 = confuse(labels)
        new_labels3 = confuse(labels)
        assert(new_labels1.equal(torch.tensor([4, 2, 9, 0, 1, 8, 0, 1, 7, 7])))
        assert(new_labels2.equal(torch.tensor([4, 2, 9, 0, 1, 8, 0, 7, 7, 1])))
        assert(new_labels3.equal(torch.tensor([4, 2, 9, 0, 7, 8, 0, 1, 7, 7])))

    def test_closs1(self):
        pred1 = [0.1, 0.2, 0.3, 0.3, 0.1] # 0.2, 0.1
        pred2 = [0.25, 0.1, 0.3, 0.05, 0.2] # 0.3 0.2
        pred3 = [0.05, 0.02, 0.02, 0.01, 0.9] # 0.01, 0.9
        gold = torch.tensor([1, 2, 3])
        preds = torch.tensor([pred1, pred2, pred3])
        criterion = ConfidenceLoss1(p0 = 0.5)
        expected_loss = 1.0264
        assert self.close_enough(criterion(preds, gold, abstain_i=4).item(),
                                 expected_loss)

    def test_closs2(self):
        pred1 = [0.1, 0.2, 0.3, 0.3, 0.1] # 0.2, 0.1
        pred2 = [0.25, 0.1, 0.3, 0.05, 0.2] # 0.3 0.2
        pred3 = [0.05, 0.02, 0.02, 0.01, 0.9] # 0.01, 0.9
        gold = torch.tensor([1, 2, 3])
        preds = torch.tensor([pred1, pred2, pred3])
        criterion = ConfidenceLoss2(p0 = 0.5)
        expected_loss = (-math.log(.2) + -math.log(.3) + -math.log(.45)) / 3
        assert self.close_enough(criterion(preds, gold, abstain_i=4).item(),
                                 expected_loss)


    def test_awnll(self):
        pred1 = [0.1, 0.2, 0.3, 0.3, 0.1] # 0.2, 0.1
        pred2 = [0.25, 0.1, 0.3, 0.05, 0.2] # 0.3 0.2
        pred3 = [0.05, 0.02, 0.02, 0.01, 0.9] # 0.01, 0.9
        gold = torch.tensor([1, 2, 3])
        preds = torch.tensor([pred1, pred2, pred3])

        a1, b1 = 1, 1
        criterion = AWNLL(a1, b1)
        expected_loss1 = (1.89712 + 1.38629 + 0.78746) / 3
        loss = criterion(preds, gold, abstain_i=4).item()
        assert self.close_enough(loss, expected_loss1)

        a2, b2 = 2, 1
        criterion = AWNLL(a2, b2)
        expected_loss2 = (1.79176 + 1.32176 + 1.18199) / 3
        loss = criterion(preds, gold, abstain_i=4).item()
        assert self.close_enough(loss, expected_loss2)
 
    def test_cawnll(self):
        pred1 = [0.1, 0.2, 0.3, 0.3, 0.1] # 0.2, c = 0.9343
        pred2 = [0.25, 0.1, 0.3, 0.05, 0.2] # 0.3 c = 0.8394
        pred3 = [0.05, 0.02, 0.02, 0.01, 0.9] # 0.01, c = 0.8338
        gold = torch.tensor([1, 2, 3])
        preds = torch.tensor([pred1, pred2, pred3])
        
        a1, b1 = 1, 1
        criterion = CAWNLL(a1, b1)
        expected_loss1 = (1.69909 + 1.10775 + 0.75099) / 3
        loss = criterion(preds, gold, abstain_i=4).item()
        assert self.close_enough(loss, expected_loss1)

        a2, b2 = 2, 1
        criterion = CAWNLL(a2, b2)
        expected_loss2 = (1.66830 + 1.13881 + 1.14591) / 3
        loss = criterion(preds, gold, abstain_i=4).item()
        assert(self.close_enough(loss, expected_loss2))

    def test_cranll(self):
        pred1 = [0.1, 0.2, 0.3, 0.3, 0.1] # -log(0.2)
        pred2 = [0.25, 0.1, 0.3, 0.05, 0.2] # -log(0.3)
        pred3 = [0.05, 0.02, 0.02, 0.01, 0.9] # -log(0.5)
        gold = torch.tensor([1, 2, 3])
        preds = torch.tensor([pred1, pred2, pred3])
        p0 = 0.5

        criterion = CRANLL(p0)
        expected_loss = (-math.log(0.2) + -math.log(0.3) + -math.log(p0)) / 3
        loss = criterion(preds, gold, 4).item()
        assert self.close_enough(loss, expected_loss)

        
    def test_lranll(self):
        pred1 = [0.1, 0.2, 0.3, 0.3, 0.1] # -log(0.2)
        pred2 = [0.25, 0.1, 0.3, 0.05, 0.2] # -log(0.3)
        pred3 = [0.05, 0.02, 0.02, 0.01, 0.9] # -log(0.9)
        gold = torch.tensor([1, 2, 3])
        preds = torch.tensor([pred1, pred2, pred3])
        expected_loss = - (math.log(0.2) + math.log(0.3) + math.log(0.9)) / 3
        
        criterion = LRANLL()
        loss = criterion(preds, gold, abstain_i=4).item()
        assert self.close_enough(loss, expected_loss)

    def test_cabnll(self):
        pred1 = [0.1, 0.2, 0.3, 0.3, 0.1] # -log(0.2)
        pred2 = [0.25, 0.1, 0.3, 0.05, 0.2] # -log(0.3)
        pred3 = [0.05, 0.02, 0.02, 0.01, 0.9] # c = 0.8338
        gold = torch.tensor([1, 2, 3])
        preds = torch.tensor([pred1, pred2, pred3])
        expected_loss = (-math.log(0.2) - math.log(0.3) - math.log (0.9 - 0.8338)) / 3

        criterion = CABNLL()
        loss = criterion(preds, gold, abstain_i=4).item()
        assert self.close_enough(loss, expected_loss)

    def test_pairwise_confidence_loss(self):
        criterion = PairwiseConfidenceLoss('neg_abs')
        #test compute_loss
        gold_probs_x = torch.tensor([0.2, 0.5, 1])
        gold_probs_y = torch.tensor([0.5, 0.5, 0.2])
        confidence_x = torch.tensor([0.5, 0.5, 0.8])
        confidence_y = torch.tensor([0.6, 0.6, 0.6])
        
        expected_losses = torch.tensor([1.1284, 0.6931, 0.7246])
        losses = criterion.compute_loss(confidence_x, confidence_y, gold_probs_x, gold_probs_y)
        assert(torch.allclose(expected_losses, losses, atol=10**(-4)))
        
        #test loss function
        #neg_abs
        
        output_x = torch.tensor([[0.2, 0.2, 0.5],
                                 [0.5, 0.4, 0.5],
                                 [1, 0.2, 0.2]])
        output_y = torch.tensor([[0.5, 0.1, 0.4],
                                 [0.5, 0.1, 0.4],
                                 [0.2, 0.2, 0.4]])
        gold_x = [1, 0, 0]
        gold_y = [0, 0, 0]

        expected_loss = torch.tensor(0.8487)
        loss = criterion(output_x, output_y, gold_x, gold_y)
        assert(torch.allclose(loss, expected_loss, atol=10**(-4)))

        #baseline
        criterion = PairwiseConfidenceLoss('baseline')
        output_x = torch.tensor([[0.2, 0.5, 0.3],
                                 [0.5, 0.5, 0],
                                 [1, 0., 0.]])
        gold_x = torch.tensor([0, 1, 0])
        output_y = torch.tensor([[0.5, 0.6, 0],
                                 [0.5, 0.6, 0],
                                 [0.6, 0.2, 0.4]])
        gold_y = torch.tensor([0, 0, 1])
        expected_loss = torch.tensor(0.8225)
        loss = criterion(output_x, output_y, gold_x, gold_y)
        assert(torch.allclose(loss, expected_loss, atol=10**(-4)))


if __name__ == '__main__':
        unittest.main()