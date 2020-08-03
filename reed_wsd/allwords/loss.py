import torch
import torch.nn.functional as F
from reed_wsd.loss import PairwiseConfidenceLoss
from reed_wsd.loss import ConfidenceLoss

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

class LossWithZones(ConfidenceLoss):
    
    def __call__(self, predicted, gold, conf, zones):
        return zone_based_loss(predicted, gold, zones, lambda x: -x)

    
class NLLLossWithZones(ConfidenceLoss):
        
    def __call__(self, predicted, gold, conf, zones):
        return zone_based_loss(predicted, gold, zones, lambda x: -torch.log(x))

class ConfidenceLossWithZones(ConfidenceLoss):
    def __init__(self, p0):
        self.target_p0 = p0
        self.p0 = 0

    def notify(self, e):
        if e >= 10:
            self.p0 = self.target_p0
    
    def __call__(self, predicted, gold, conf, zones):
        revised_pred = apply_zones(predicted, zones, abstain=True)
        label_ps = revised_pred[list(range(len(revised_pred))), gold]
        losses = - torch.log(label_ps + self.target_p0 * conf)
        return torch.mean(losses, dim=-1)

class PairwiseConfidenceLossWithZones(PairwiseConfidenceLoss):

    def __call__(self, output_x, output_y, gold_x, gold_y, conf_x, conf_y, zones_x, zones_y):
        output_x = apply_zones(output_x, zones_x, abstain=True)
        output_y = apply_zones(output_y, zones_y, abstain=True)
        gold_probs_x = output_x[list(range(output_x.shape[0])), gold_x]
        gold_probs_y = output_y[list(range(output_y.shape[0])), gold_y]
        losses = self.compute_loss(conf_x, conf_y, gold_probs_x, gold_probs_y)
        return losses.mean()

