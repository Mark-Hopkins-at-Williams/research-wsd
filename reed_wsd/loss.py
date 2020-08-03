import torch
import math
import torch.nn.functional as F

class ConfidenceLoss:
    def notify(self, epoch):
        pass

def confidence_weighted_loss(confidence_x, confidence_y, gold_probs_x, gold_probs_y):
    nll_x = - torch.log(gold_probs_x)
    nll_y = - torch.log(gold_probs_y)
    confidence_pair = torch.stack([confidence_x, confidence_y], dim=-1)                                                                                                                                                                                                                      softmaxed_pair = F.softmax(confidence_pair, dim=-1)                                                                                                                                                                                                                                      nll_pair = torch.stack([nll_x, nll_y], dim=-1)                                                                                                                                                                                                                                           losses = torch.sum(nll_pair * softmaxed_pair, dim=-1)
    return losses
