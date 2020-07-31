import torch
import torch.nn.functional as F
import reed_wsd.loss as loss

class ConfidenceLoss:
    
    def notify(self, epoch):
        pass

class PairwiseConfidenceLoss(ConfidenceLoss):
    def __init__(self):
        super().__init__()
    
    def __call__(self, output_x, output_y, gold_x, gold_y, conf_x, conf_y):
        gold_probs_x = output_x[list(range(output_x.shape[0])), gold_x]
        gold_probs_y = output_y[list(range(output_y.shape[0])), gold_y]
        losses = loss.PairwiseConfidenceLoss.compute_loss(conf_x, conf_y, gold_probs_x, gold_probs_y)
        return losses.mean()
 
class CrossEntropyLoss(ConfidenceLoss):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def __call__(self, output, gold):
        return self.loss(output, gold)
    
    def __str__(self):
        return "CrossEntropyLoss"

class NLLLoss(ConfidenceLoss):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.NLLLoss()

    def __call__(self, output, gold):
        log_output = torch.log(output)
        return self.loss(log_output, gold)
    
    def __str__(self):
        return "NLLLoss"

class ConfidenceLoss1(ConfidenceLoss):
    def __init__(self, p0):
        super().__init__()
        self.p0 = 0.0
        self.target_p0 = p0
        self.notify(0)

    def notify(self, epoch):
        if epoch >= 10:
            self.p0 = self.target_p0
    
    def __call__(self, output, gold):
        label_ps = output[list(range(len(output))), gold]
        losses = label_ps + (self.p0 * output[:,-1])
        losses = torch.clamp(losses, min = 0.000000001)
        return -torch.mean(torch.log(losses))
    
    def __str__(self):
        return "ConfidenceLoss1_p0_" + str(self.p0)

class ConfidenceLoss4(ConfidenceLoss):
    def __init__(self, p0):
        super().__init__()
        self.p0 = 0.0
        self.target_p0 = p0
        self.notify(0)
        
    def notify(self, epoch):
        if epoch >= 2:
            self.p0 = self.target_p0

    def __call__(self, output, gold):
        label_ps = output[list(range(len(output))), gold]
        label_ps_woa = output[:, :-1]
        label_ps_woa = F.normalize(label_ps_woa, p=1, dim=1)
        label_ps_woa = label_ps_woa[list(range(len(label_ps_woa))), gold]
        losses = label_ps_woa * (label_ps + (self.p0 * output[:,-1]))
        losses = torch.clamp(losses, min = 0.000000001)
        return -torch.mean(torch.log(losses))
    
    def __str__(self):
        return "ConfidenceLoss4_p0_" + str(self.p0)
