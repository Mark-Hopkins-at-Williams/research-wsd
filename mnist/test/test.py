import unittest
import os
from os.path import join
import sys
file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, join(file_dir, ".."))
import torch
from loss import NLLA, AWNLL, CAWNLL, ConfidenceLoss1
from mnist import confuse

class Test(unittest.TestCase):
    def test_nlla(self):
        pred1 = [0.1, 0.2, 0.3, 0.3, 0.1] # loss = 1.4485
        pred2 = [0.25, 0.1, 0.3, 0.05, 0.2] # loss = 0.96318
        pred3 = [0.05, 0.02, 0.02, 0.01, 0.9] # loss = 0.46051
        # mean_loss = 0.95740
        gold = torch.tensor([1, 2, 3])
        preds = torch.tensor([pred1, pred2, pred3])
        logpreds = torch.log(preds)
        criterion = NLLA()
        expected_loss = (1.4485 + 0.96318 + 0.46051) / 3
        loss = criterion(logpreds, gold, abstain_i=4).item()
        assert (round(loss * 1000) / 1000 == round(expected_loss * 1000) / 1000)

    def test_confuse(self):
        labels = torch.tensor([4, 2, 9, 0, 1, 8, 0, 1, 7, 7])
        # print(labels)
        new_labels = confuse(labels)
        #print(labels)

    def test_closs1(self):
        pred1 = [0.1, 0.2, 0.3, 0.3, 0.1] # 0.2, 0.1
        pred2 = [0.25, 0.1, 0.3, 0.05, 0.2] # 0.3 0.2
        pred3 = [0.05, 0.02, 0.02, 0.01, 0.9] # 0.01, 0.9
        gold = torch.tensor([1, 2, 3])
        preds = torch.tensor([pred1, pred2, pred3])
        criterion = ConfidenceLoss1(p0 = 0.5)
        print('hi')
        print(criterion(preds, gold, abstain_i=4))

    def test_awnll(self):
        pred1 = [0.1, 0.2, 0.3, 0.3, 0.1] # 0.2, 0.1
        pred2 = [0.25, 0.1, 0.3, 0.05, 0.2] # 0.3 0.2
        pred3 = [0.05, 0.02, 0.02, 0.01, 0.9] # 0.01, 0.9
        gold = torch.tensor([1, 2, 3])
        preds = torch.tensor([pred1, pred2, pred3])
        logpreds = torch.log(preds)
        criterion = AWNLL()

        a1, b1 = 1, 1
        expected_loss1 = (1.89712 + 1.38629 + 0.78746) / 3
        loss = criterion(logpreds, gold, a1, b1,  abstain_i=4).item()
        assert (round(loss * 1000) / 1000 == round(expected_loss1 * 1000) / 1000)

        a2, b2 = 2, 1
        expected_loss2 = (1.79176 + 1.32176 + 1.18199) / 3
        loss = criterion(logpreds, gold, a2, b2,  abstain_i=4).item()
        assert (round(loss * 1000) / 1000 == round(expected_loss2 * 1000) / 1000)

    def test_cawnll(self):
        pred1 = [0.1, 0.2, 0.3, 0.3, 0.1] # 0.2, c = 0.9343
        pred2 = [0.25, 0.1, 0.3, 0.05, 0.2] # 0.3 c = 0.8394
        pred3 = [0.05, 0.02, 0.02, 0.01, 0.9] # 0.01, c = 0.8338
        gold = torch.tensor([1, 2, 3])
        preds = torch.tensor([pred1, pred2, pred3])
        logpreds = torch.log(preds)
        criterion = CAWNLL()
        
        a1, b1 = 1, 1
        expected_loss1 = (1.69909 + 1.10775 + 0.75099) / 3
        loss = criterion(logpreds, gold, a1, b1,  abstain_i=4).item()
        assert (round(loss * 1000) / 1000 == round(expected_loss1 * 1000) / 1000)

        a2, b2 = 2, 1
        expected_loss2 = (1.66830 + 1.13881 + 1.14591) / 3
        loss = criterion(logpreds, gold, a2, b2,  abstain_i=4).item()
        assert (round(loss * 1000) / 1000 == round(expected_loss2 * 1000) / 1000)
        



if __name__ == '__main__':
        unittest.main()
