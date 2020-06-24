import torch
import math
import torch.nn.functional as F

ABSTAIN = 10

class NLLA:
    def __call__(self, log_output, gold, abstain_i=ABSTAIN):
        maxes, preds = torch.max(log_output, dim=-1)
        aps = torch.exp(log_output[:, abstain_i])
        discounters = 1 - aps
        label_ps = log_output[list(range(len(log_output))), gold]
        losses = - label_ps * discounters
        return torch.mean(losses).item()
        
class AWNLL:
    # abstention-weighted Negative log likelihood loss
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, log_output, gold, abstain_i = ABSTAIN):
        a = self.a
        b = self.b
        output = torch.exp(log_output)
        maxes, preds = torch.max(output, dim=-1)
        aps = output[:, abstain_i]
        label_ps = output[list(range(len(log_output))), gold]
        losses = - torch.log((a * label_ps + b * aps) / (a + b))
        return torch.mean(losses)

class CAWNLL:
    # confusion-abstention weighted negative log likelihood loss
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, log_output, gold, abstain_i = ABSTAIN):
        a = self.a
        b = self.b
        output = torch.exp(log_output)
        maxes, preds = torch.max(output, dim=-1)
        aps = output[:, abstain_i]
        label_ps = output[list(range(len(log_output))), gold]
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

    def __call__(self, log_output, gold, abstain_i = ABSTAIN):
        maxes, preds = torch.max(log_output, dim=-1)

        abstained = (preds == abstain_i)
        confident = (preds != abstain_i)

        results_confident = -log_output[confident, gold[confident]]
        results_sum = results_confident.sum() + abstained.float().sum() * (-torch.log(torch.tensor(self.p0)))
        result = results_sum / log_output.shape[0]
        return result

class LRANLL:
    # linear reward for abstention negative log likelihood
    
    def __call__(self, log_output, gold, abstain_i = ABSTAIN):
        maxes, preds = torch.max(log_output, dim=-1)
        aps = log_output[:, abstain_i]

        abstained = (preds == abstain_i)
        confident = (preds != abstain_i)

        results_abstained = - aps[abstained]
        results_confident = -log_output[confident, gold[confident]]
        results = torch.cat([results_abstained, results_confident])
        return torch.mean(results)

class CABNLL:
    # confusion-abstention balanced negative log likelihood
    def __call__(self, log_output, gold, abstain_i = ABSTAIN):
        output = torch.exp(log_output)
        lens = F.normalize(output[:, :-1], p=1, dim=1).norm(dim=-1)
        confusions = (1 - lens) / (1 - 1 / math.sqrt(output.shape[1] - 1))
        maxes, preds = torch.max(log_output, dim=-1)
        aps = output[:, abstain_i]

        abstained = (preds == abstain_i)
        confident = (preds != abstain_i)

        results_abstained = -torch.log(torch.abs(confusions[abstained] - aps[abstained]))
        results_confident = -log_output[confident, gold[confident]]
        results = torch.cat([results_abstained, results_confident])
        return torch.mean(results)
