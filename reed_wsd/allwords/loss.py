import torch
import torch.nn.functional as F
from reed_wsd.loss import PairwiseConfidenceLoss

def zone_based_loss(predicted, gold, zones, f):
    """
    Computes a loss based on:
    - predicted: a matrix of predictions, where each row is a probability distribution over all outcomes - gold: a vector of correct outcomes - zones: a list of "zones", which define the set of valid outcomes for each prediction - f: post-processing function to apply to each probability
    
    For instance, suppose we have two predictions, each over the same 4 outcomes:
        predicted = tensor([[0.2, 0.3, 0.1, 0.4],
                            [0.1, 0.1, 0.2, 0.6]])        

    And the correct outcomes are 1 and 3: 
        gold = tensor([1, 3])
        
    And the valid zones are:
        zones = [(0, 2), (2, 4)]
        
    Then, the loss function first normalizes the probabilities over the
    valid outcomes (and zeroes the rest):
        tensor([[0.4, 0.6, 0, 0],
                [0, 0, 0.25, 0.75]])
        
    It then extracts the (normalized) predicted probability of the correct
    outcomes:
        [0.6, 0.75]
        
    It applies the post-processing function f (e.g. f(x) = -x):
        [-0.6, -0.75]
    
    Finally, the computed loss is the average of these scores:
        -0.675
    
    """
    revised_pred = apply_zones(predicted, zones)
    neglog_pred = f(revised_pred)
    result = neglog_pred[list(range(len(neglog_pred))), gold]
    return torch.mean(result)

def apply_zones(predicted, zones, abstain = False):
    """
    normalize the probabilities in the specified zone of a softmax output

    """
    revised_pred = torch.zeros(predicted.shape, device=predicted.device)
    for i, (zone_start, zone_stop) in enumerate(zones):
        normalizer = sum(predicted[i, zone_start:zone_stop])
        if abstain:
            normalizer += predicted[i, -1]
        revised_pred[i,zone_start:zone_stop] = (predicted[i, zone_start:zone_stop] 
                                                / normalizer) 
        if abstain:
            revised_pred[i, -1] = predicted[i, -1] / normalizer
    return revised_pred

class LossWithZones:
    
    def __call__(self, predicted, gold, zones):
        predicted = F.softmax(predicted.clamp(min=-10).clamp(max=10), dim=1)
        return zone_based_loss(predicted, gold, zones, lambda x: -x)

    
class NLLLossWithZones:
        
    def __call__(self, predicted, gold, zones):
        predicted = F.softmax(predicted.clamp(min=-10).clamp(max=10), dim=1)
        return zone_based_loss(predicted, gold, zones, lambda x: -torch.log(x))


class ConfidenceLossWithZones:
    def __init__(self, p0):
        self.p0 = p0
    
    def __call__(self, predicted, gold, zones):
        predicted = F.softmax(predicted.clamp(min=-10).clamp(max=10), dim=1)
        revised_pred = apply_zones(predicted, zones, abstain=True)
        label_ps = revised_pred[list(range(len(revised_pred))), gold]
        losses = label_ps + self.p0 * revised_pred[:, -1]
        return torch.mean(- torch.log(losses), dim=-1)

class PairwiseConfidenceLossWithZones(PairwiseConfidenceLoss):

    def __call__(self, output_x, output_y, gold_x, gold_y, zones_x, zones_y):
        if self.confidence != 'abs':
            output_x = F.softmax(output_x.clamp(min=-10, max=10), dim=-1)
            output_x = apply_zones(output_x, zones_x, abstain=True)
            output_y = F.softmax(output_y.clamp(min=-10, max=10), dim=-1)
            output_y = apply_zones(output_y, zones_y, abstain=True)
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
            confidence_x = output_x[:, -1]
            confidence_y = output_y[:, -1]
            probs_x = F.softmax(output_x[:, :-1].clamp(min=-10, max=10), dim=-1)
            probs_y = F.softmax(output_y[:, :-1].clamp(min=-10, max=10), dim=-1)
            probs_x = apply_zones(probs_x, zones_x, abstain=False)
            probs_y = apply_zones(probs_y, zones_y, abstain=False)
            gold_probs_x = probs_x[list(range(output_x.shape[0])), gold_x]
            gold_probs_y = probs_y[list(range(output_y.shape[0])), gold_y]
            losses = self.compute_loss(confidence_x, confidence_y, gold_probs_x, gold_probs_y)
        return losses.mean()

