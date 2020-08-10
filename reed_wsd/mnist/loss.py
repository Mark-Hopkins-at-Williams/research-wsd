import torch
import torch.nn.functional as F
from reed_wsd.loss import ConfidenceLoss

def confidence_weighted_loss(confidence_x, confidence_y, gold_probs_x, gold_probs_y):
    nll_x = - torch.log(gold_probs_x)
    nll_y = - torch.log(gold_probs_y)
    confidence_pair = torch.stack([confidence_x, confidence_y], dim=-1)
    softmaxed_pair = F.softmax(confidence_pair, dim=-1)
    nll_pair = torch.stack([nll_x, nll_y], dim=-1)
    losses = torch.sum(nll_pair * softmaxed_pair, dim=-1)
    return losses

class SingleConfidenceLoss(ConfidenceLoss):
    def __call__(self, output, confidence, gold):
        raise NotImplementedError("This feature has to be implemented in the child class.")

class PairwiseConfidenceLoss(ConfidenceLoss):

    def __call__(self, output_x, output_y, gold_x, gold_y, conf_x, conf_y):
        gold_probs_x = output_x[list(range(output_x.shape[0])), gold_x]
        gold_probs_y = output_y[list(range(output_y.shape[0])), gold_y]
        losses = confidence_weighted_loss(conf_x, conf_y, gold_probs_x, gold_probs_y)
        return losses.mean()

class CrossEntropyLoss(SingleConfidenceLoss):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def __call__(self, output, confidence, gold):
        return self.loss(output, gold)

    def __str__(self):
        return "CrossEntropyLoss"

class NLLLoss(SingleConfidenceLoss):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.NLLLoss()

    def __call__(self, output, confidence, gold):
        log_output = torch.log(output)
        return self.loss(log_output, gold)

    def __str__(self):
        return "NLLLoss"

class AbstainingLoss(SingleConfidenceLoss):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = 0.0
        self.target_alpha = alpha
        self.notify(0)

    def notify(self, epoch):
        if epoch >= 3:
            self.alpha = self.target_alpha

    def __call__(self, output, confidence, gold):
        #print(output[0])
        #print(confidence[0])
        #print(gold[0])
        label_ps = output[list(range(len(output))), gold]
        abstains = output[:,-1]
        losses = label_ps + (self.alpha * abstains)
        losses = torch.clamp(losses, min = 0.000000001)
        return -torch.mean(torch.log(losses))

    def __str__(self):
        return "Abstaining_p0_" + str(self.p0)

class ConfidenceLoss4(SingleConfidenceLoss):
    def __init__(self, p0):
        super().__init__()
        self.p0 = 0.0
        self.target_p0 = p0
        self.notify(0)

    def notify(self, epoch):
        if epoch >= 2:
            self.p0 = self.target_p0

    def __call__(self, output, confidence, gold):
        label_ps = output[list(range(len(output))), gold]
        label_ps_woa = output[:, :-1]
        label_ps_woa = F.normalize(label_ps_woa, p=1, dim=1)
        label_ps_woa = label_ps_woa[list(range(len(label_ps_woa))), gold]
        losses = label_ps_woa * (label_ps + (self.p0 * confidence))
        losses = torch.clamp(losses, min = 0.000000001)
        return -torch.mean(torch.log(losses))

    def __str__(self):
        return "ConfidenceLoss4_p0_" + str(self.p0)
