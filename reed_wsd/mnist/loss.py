import torch
import math
import torch.nn.functional as F

ABSTAIN = 10

class PairwiseConfidenceLoss:
    def __init__(self, confidence='neg_abs'):
        assert(confidence in ['baseline', 'neg_abs'])
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
        probs_x, probs_y = output_x[:, :-1], output_y[:, :-1] # 2d
        gold_probs_x = output_x[list(range(output_x.shape[0])), gold_x]
        gold_probs_y = output_y[list(range(output_y.shape[0])), gold_y]
        if self.confidence == 'neg_abs':
            confidence_x, confidence_y = 1 - output_x[:, -1], 1 - output_y[:, -1] #1d
            losses = self.compute_loss(confidence_x, confidence_y, gold_probs_x, gold_probs_y)
        elif self.confidence == 'baseline':
            confidence_x, max_ids_x = probs_x.max(dim=-1)
            confidence_y, max_ids_y = probs_y.max(dim=-1)
            confidence_x = torch.clamp(confidence_x, min=0.000000001)
            confidence_y = torch.clamp(confidence_y, min=0.000000001)
            losses = self.compute_loss(confidence_x, confidence_y, gold_probs_x, gold_probs_y)
        return losses.mean()
            

        


class ConfidenceLoss1:
    def __init__(self, p0):
        self.p0 = p0
    
    def __call__(self, output, gold, abstain_i=ABSTAIN):
        label_ps = output[list(range(len(output))), gold]
        losses = label_ps + (self.p0 * output[:,-1])
        losses = torch.clamp(losses, min = 0.000000001)
        return -torch.mean(torch.log(losses))
    
    def __str__(self):
        return "ConfidenceLoss_p0_" + str(self.p0)

class ConfidenceLoss2:
    def __init__(self, p0):
        self.p0 = p0
    
    def __call__(self, output, gold, abstain_i=ABSTAIN):
        label_ps = output[list(range(len(output))), gold]
        losses = torch.max(label_ps, (self.p0 * output[:,-1]))
        losses = torch.clamp(losses, min = 0.000000001)
        return -torch.mean(torch.log(losses))
    
    def __str__(self):
        return "ConfidenceLoss2_p0_" + str(self.p0)

class ConfidenceLoss3:
    def __init__(self, p0):
        self.p0 = p0
    
    def __call__(self, output, gold, abstain_i=ABSTAIN):
        label_ps = output[list(range(len(output))), gold]
        losses = label_ps - output[:,-1]
        return -torch.mean(losses)
    
    def __str__(self):
        return "ConfidenceLoss3_p0_" + str(self.p0)


class ConfidenceLoss4:
    def __init__(self, p0):
        self.p0 = p0
    
    def __call__(self, output, gold, abstain_i=ABSTAIN):
        label_ps = output[list(range(len(output))), gold]
        label_ps_woa = output[:, :-1]
        label_ps_woa = F.normalize(label_ps_woa, p=1, dim=1)
        label_ps_woa = label_ps_woa[list(range(len(label_ps_woa))), gold]
        losses = label_ps_woa * (label_ps + (self.p0 * output[:,-1]))
        losses = torch.clamp(losses, min = 0.000000001)
        return -torch.mean(torch.log(losses))
    
    def __str__(self):
        return "ConfidenceLoss4_p0_" + str(self.p0)


class NLL:
    def __call__(self, output, gold):
        label_ps = output[list(range(len(output))), gold]
        losses = - torch.log(label_ps)
        return torch.mean(losses)
    
class NLLA:
    def __call__(self, output, gold, abstain_i=ABSTAIN):
        maxes, preds = torch.max(output, dim=-1)
        aps = output[:, abstain_i]
        discounters = 1 - aps
        label_ps = output[list(range(len(output))), gold]
        losses = - torch.log(label_ps) * discounters
        return torch.mean(losses)
        
class AWNLL:
    # abstention-weighted Negative log likelihood loss
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, output, gold, abstain_i = ABSTAIN):
        a = self.a
        b = self.b
        maxes, preds = torch.max(output, dim=-1)
        aps = output[:, abstain_i]
        label_ps = output[list(range(len(output))), gold]
        losses = - torch.log((a * label_ps + b * aps) / (a + b))
        return torch.mean(losses)

class CAWNLL:
    # confusion-abstention weighted negative log likelihood loss
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, output, gold, abstain_i = ABSTAIN):
        a = self.a
        b = self.b
        maxes, preds = torch.max(output, dim=-1)
        aps = output[:, abstain_i]
        label_ps = output[list(range(len(output))), gold]
        label_ps_woa = output[:, :-1]
        label_ps_woa = F.normalize(label_ps_woa, p=1, dim=1)
        lens = label_ps_woa.norm(dim=-1)
        confusions = (1 - lens) / (1 - 1 / math.sqrt(output.shape[1] - 1))
        losses = - torch.log((a * label_ps + b * (1 - torch.abs(confusions - aps))) / (a + b))
        return torch.mean(losses)
        
class CRANLL:
    # constant reward for abstention negative log likelihood
    def __init__(self, p0):
        self.p0 = p0

    def __call__(self, output, gold, abstain_i = ABSTAIN):
        maxes, preds = torch.max(output, dim=-1)

        abstained = (preds == abstain_i)
        confident = (preds != abstain_i)

        results_confident = -torch.log(output[confident, gold[confident]])
        results_sum = results_confident.sum() + abstained.float().sum() * (-torch.log(torch.tensor(self.p0)))
        result = results_sum / output.shape[0]
        return result

class LRANLL:
    # linear reward for abstention negative log likelihood
    
    def __call__(self, output, gold, abstain_i = ABSTAIN):
        maxes, preds = torch.max(output, dim=-1)
        aps = output[:, abstain_i]

        abstained = (preds == abstain_i)
        confident = (preds != abstain_i)

        results_abstained = - torch.log(aps[abstained])
        results_confident = - torch.log(output[confident, gold[confident]])
        results = torch.cat([results_abstained, results_confident])
        return torch.mean(results)

class CABNLL:
    # confusion-abstention balanced negative log likelihood
    def __call__(self, output, gold, abstain_i = ABSTAIN):
        lens = F.normalize(output[:, :-1], p=1, dim=1).norm(dim=-1)
        confusions = (1 - lens) / (1 - 1 / math.sqrt(output.shape[1] - 1))
        maxes, preds = torch.max(output, dim=-1)
        aps = output[:, abstain_i]

        abstained = (preds == abstain_i)
        confident = (preds != abstain_i)

        results_abstained = - torch.log(torch.abs(confusions[abstained] - aps[abstained]))
        results_confident = - torch.log(output[confident, gold[confident]])
        results = torch.cat([results_abstained, results_confident])
        return torch.mean(results)
