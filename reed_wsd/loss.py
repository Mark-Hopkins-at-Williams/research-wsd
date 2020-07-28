import torch
import math
import torch.nn.functional as F

class PairwiseConfidenceLoss:
    def __init__(self, confidence='neg_abs'):
        assert(confidence in ['max_non_abs', 'neg_abs', 'abs'])
        self.confidence = confidence

    @staticmethod
    def compute_loss(confidence_x, confidence_y, gold_probs_x, gold_probs_y):
        nll_x = - torch.log(gold_probs_x)
        nll_y = - torch.log(gold_probs_y)
        confidence_pair = torch.stack([confidence_x, confidence_y], dim=-1)
        softmaxed_pair = F.softmax(confidence_pair, dim=-1)
        nll_pair = torch.stack([nll_x, nll_y], dim=-1)
        losses = torch.sum(nll_pair * softmaxed_pair, dim=-1)
        return losses

    def __call__(self, output_x, output_y, gold_x, gold_y):
        if self.confidence != 'abs':
            output_x = F.softmax(output_x, dim=-1)
            output_y = F.softmax(output_y, dim=-1)
            probs_x, probs_y = output_x[:, :-1], output_y[:, :-1] # 2d
            gold_probs_x = output_x[list(range(output_x.shape[0])), gold_x]
            gold_probs_y = output_y[list(range(output_y.shape[0])), gold_y]
            if self.confidence == 'neg_abs':
                confidence_x, confidence_y = 1 - output_x[:, -1], 1 - output_y[:, -1] #1d
                losses = self.compute_loss(confidence_x, confidence_y, gold_probs_x, gold_probs_y)
            elif self.confidence == 'max_non_abs':
                confidence_x, max_ids_x = probs_x.max(dim=-1)
                confidence_y, max_ids_y = probs_y.max(dim=-1)
                confidence_x = torch.clamp(confidence_x, min=0.000000001)
                confidence_y = torch.clamp(confidence_y, min=0.000000001)
                losses = self.compute_loss(confidence_x, confidence_y, gold_probs_x, gold_probs_y)
        else:
            abstention_x = output_x[:, -1]
            abstention_y = output_y[:, -1]
            losses_x = F.cross_entropy(output_x[:, :-1], gold_x)
            losses_y = F.cross_entropy(output_y[:, :-1], gold_y)
            abstention_pair = torch.stack([abstention_x, abstention_y], dim=-1)
            softmaxed_pair = F.softmax(abstention_pair, dim=-1)
            losses_pair = torch.stack([losses_x, losses_y], dim=-1)
            losses = torch.sum(losses_pair * softmaxed_pair, dim=-1)
        return losses.mean()
            
