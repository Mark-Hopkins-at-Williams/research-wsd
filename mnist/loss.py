import torch
import math
import torch.nn.functional as F
ABSTAIN = 10


class ConfidenceLoss1:
    def __init__(self, p0):
        self.p0 = p0
    
    def __call__(self, output, gold, abstain_i=ABSTAIN):
        label_ps = output[list(range(len(output))), gold]
        losses = label_ps + (self.p0 * output[:,-1])
        losses = torch.clamp(losses, min = 0.000000001)
        return -torch.mean(torch.log(losses))
    
class NLLA:
    def __call__(self, log_output, gold, abstain_i=ABSTAIN):
        maxes, preds = torch.max(log_output, dim=-1)
        aps = torch.exp(log_output[:, abstain_i])
        discounters = 1 - aps
        label_ps = log_output[list(range(len(log_output))), gold]
        losses = - label_ps * discounters
        return torch.mean(losses)
        
class AWNLL:
    # abstention-weighted Negative log likelihood loss
    def __call__(self, log_output, gold, a=1, b=1, abstain_i = ABSTAIN):
        output = torch.exp(log_output)
        maxes, preds = torch.max(output, dim=-1)
        aps = output[:, abstain_i]
        label_ps = output[list(range(len(log_output))), gold]
        losses = - torch.log((a * label_ps + b * aps) / (a + b))
        return torch.mean(losses)

class CAWNLL:
    # confusion-abstention weighted negative log likelihood loss
    def __call__(self, log_output, gold, a=1, b=1, abstain_i = ABSTAIN):
        output = torch.exp(log_output)
        maxes, preds = torch.max(output, dim=-1)
        aps = output[:, abstain_i]
        label_ps = output[list(range(len(log_output))), gold]
        label_ps_woa = output[:, :-1]
        label_ps_woa = F.normalize(label_ps_woa, p=1, dim=1)
        lens = label_ps_woa.norm(dim=-1)
        confusions = (1 - lens) / (1 - 1 / math.sqrt(output.shape[1] - 1))
        losses = - torch.log((a * label_ps + b * (1 - torch.abs(confusions - aps))) / (a + b))
        print(losses)
        return torch.mean(losses)
        
        
        
