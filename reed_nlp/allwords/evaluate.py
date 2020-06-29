import torch
import json
import os
from os.path import join
import torch.nn.functional as F
import matplotlib.pyplot as plt


LARGE_NEGATIVE = 0
ABSTAIN = -1.0
file_dir = os.path.dirname(os.path.realpath(__file__))

def predict(distribution):
    return distribution.argmax()

def accuracy(predicted_labels, gold_labels):
    assert(len(predicted_labels) == len(gold_labels))
    preds = torch.tensor(predicted_labels).double()
    gold = torch.tensor(gold_labels).double()
    n_confident = (preds != ABSTAIN).double().sum().item()
    n_correct = (preds == gold).double().sum().item()
    return n_correct, n_confident

def yielde(predicted_labels, gold_labels):
    assert(len(predicted_labels) == len(gold_labels))
    preds = torch.tensor(predicted_labels).double()
    gold = torch.tensor(gold_labels).double()
    n_correct = (preds == gold).double().sum().item()
    return n_correct, len(predicted_labels)
    
def apply_zone_masks(outputs, zones):
    revised = torch.empty(outputs.shape)
    revised = revised.fill_(LARGE_NEGATIVE)
    for row in range(len(zones)):
        (start, stop) = zones[row]
        revised[row][start:stop] = outputs[row][start:stop]
    revised = F.normalize(revised, dim=-1, p=1)
    return revised

def decode(net, data):
    """
    Runs a trained neural network classifier on validation data, and iterates
    through the top prediction for each datum.
    
    TODO: write some unit tests for this function
    
    """
    net.eval()
    val_loader = data.batch_iter()
    for inst_ids, targets, evidence, response, zones in val_loader:
        val_outputs = net(evidence)
        val_outputs = F.softmax(val_outputs, dim=1)
        revised = apply_zone_masks(val_outputs, zones)
        maxes, preds = revised.max(dim=-1)
        for element in zip(inst_ids, 
                           zip(targets,
                               zip(preds, zip(response.tolist(), maxes)))):                    
            (inst_id, (target, (prediction, (gold, confidence)))) = element
            yield {'pred': prediction.item(), 'gold': gold, 'confidence': confidence.item()}
    net.train()

def evaluate(net, data):
    """
    The accuracy (i.e. percentage of correct classifications) is returned.
    
    """
    decoded = list(decode(net, data))
    predictions = [pred for (_, _, pred, _, _) in decoded]
    gold = [g for (_, _, _, g, _) in decoded]
    correct, total = accuracy(predictions, gold)
    return correct/total

