import unittest
import torch
from reed_wsd.mnist.loss import NLLLoss, PairwiseConfidenceLoss
from reed_wsd.mnist.loss import AbstainingLoss


class TestMnistLoss(unittest.TestCase):
    def test_nll_loss(self):
        criterion = NLLLoss()
        in_vec = torch.tensor([[0.3, 0.2, 0.5],
                               [0.1, 0.1, 0.8]])
        gold = torch.tensor([2, 1])
        expected_loss1 = 0.6931
        expected_loss2 = 2.3026
        expected_loss = torch.tensor((expected_loss1 + expected_loss2) / 2)
        loss = criterion(in_vec, None, gold)
        assert(torch.allclose(loss, expected_loss, atol=0.0001))

    def test_abstaining_loss1(self):
        criterion = AbstainingLoss(alpha = 0.5)
        criterion.notify(5)
        in_vec = torch.tensor([[0.25, 0.25, 0.1, 0.4]])
        gold = torch.tensor([2])
        abstains = torch.tensor([0.4])
        loss = criterion(in_vec,  abstains, gold)
        expected_loss = torch.tensor(1.2039728043259361) # i.e., -log(0.3)
        assert(torch.allclose(loss, expected_loss, atol=0.0001))

        
    def test_abstaining_loss2(self):
        criterion = AbstainingLoss(alpha = 0.5)
        in_vec = torch.tensor([[0.3, 0.2, 0.1, 0.4],
                               [0.1, 0.1, 0.6, 0.2]])
        gold = torch.tensor([2, 1])
        abstains = torch.tensor([0.4, 0.2])
        loss = criterion(in_vec,  abstains, gold)


class TestPairwiseConfidenceLoss(unittest.TestCase):
    def test_call(self):
        criterion = PairwiseConfidenceLoss()
        in_vec_x = torch.tensor([[0.3, 0.2, 0.5],
                               [0.1, 0.1, 0.8]])
        gold_x = torch.tensor([1, 1])
        in_vec_y = torch.tensor([[0.6, 0.3, 0.1],
                                 [0.3, 0.3, 0.4]])
        gold_y = torch.tensor([1, 1])
        conf_x = torch.tensor([0.5, 0.2])
        conf_y = torch.tensor([0.9, 0.6])

        expected_loss_x = 1.3667
        expected_loss_y = 1.6448

        expected_loss = torch.tensor((expected_loss_x + expected_loss_y) / 2)
        loss = criterion(in_vec_x, in_vec_y, gold_x, gold_y, conf_x, conf_y)
        assert(torch.allclose(expected_loss, loss, atol=0.0001))


if __name__ == "__main__":
    unittest.main()
