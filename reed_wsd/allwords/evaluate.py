import torch
import json
import os
from os.path import join
import torch.nn.functional as F
import matplotlib.pyplot as plt


LARGE_NEGATIVE = 0
file_dir = os.path.dirname(os.path.realpath(__file__))

def predict(distribution):
    return distribution.argmax()

def accuracy(predicted_labels, gold_labels, abstain_i=-1):
    assert(len(predicted_labels) == len(gold_labels))
    preds = torch.tensor(predicted_labels).double()
    gold = torch.tensor(gold_labels).double()
    n_confident = (preds != abstain_i).double().sum().item()
    n_correct = (preds == gold).double().sum().item()
    return n_correct, n_confident

def yielde(predicted_labels, gold_labels):
    assert(len(predicted_labels) == len(gold_labels))
    preds = torch.tensor(predicted_labels).double()
    gold = torch.tensor(gold_labels).double()
    n_correct = (preds == gold).double().sum().item()
    return n_correct, len(predicted_labels)
    
def apply_zone_masks(outputs, zones, abstain=False):
    revised = torch.empty(outputs.shape)
    revised = revised.fill_(LARGE_NEGATIVE)
    for row in range(len(zones)):
        (start, stop) = zones[row]
        revised[row][start:stop] = outputs[row][start:stop]
        if abstain:
            revised[row][-1] = outputs[row][-1]
    revised = F.normalize(revised, dim=-1, p=1)
    return revised

def decode_BEM(net, data):
    net.eval()
    val_loader = data.batch_iter()
    with torch.no_grad():
        for batch in val_loader:
            contexts = batch['contexts']
            glosses = batch['glosses']
            span = batch['span']
            gold = batch['gold']
            scores = net(contexts, glosses, span)
            max_scores, preds = scores.max(dim=-1)
            for element in zip(max_scores,
                               zip(preds,
                                   gold)):
                (max_score, (pred, g)) = element
                yield({'pred': pred.item(), 'gold': g, 'confidence': max_scores})
            



def decode_gen(abstain, confidence):
    """
    abstain: boolean
    whether the model output has a abstention class

    confidence: string
    the confidence metric type
    baseline: the maximum class prob. among the non-abstain classes
    neg_abs: 1 - (class prob. of the abstention class)
    """
    assert(confidence == "baseline" or confidence == "neg_abs")
    assert(not (confidence == "neg_abs" and not abstain),
            "neg_abs must work with an abstention class!")

    def decode(net, data):
        """
        Runs a trained neural network classifier on validation data, and iterates
        through the top prediction for each datum.
        
        TODO: write some unit tests for this function
        
        """
        net.eval()
        val_loader = data.batch_iter()
        with torch.no_grad():
            for inst_ids, targets, evidence, response, zones in val_loader:
                val_outputs = net(evidence)
                val_outputs = F.softmax(val_outputs, dim=1)
                revised = apply_zone_masks(val_outputs, zones, abstain)
                if abstain:
                    maxes, preds = revised[:, :-1].max(dim=-1)
                    if confidence == "baseline":
                        cs = maxes
                    elif confidence == "neg_abs":
                        cs = 1 - revised[:, -1]
                else:
                    maxes, preds = revised.max(dim=-1)
                    if confidence == "baseline":
                        cs = maxes
                for element in zip(inst_ids, 
                                   zip(targets,
                                       zip(preds, zip(response.tolist(), cs)))):                    
                    (inst_id, (target, (prediction, (gold, c)))) = element
                    yield {'pred': prediction.item(), 'gold': gold, 'confidence': c.item()}
        net.train()
    return decode

def evaluate(net, data, decoder, abstain=False):
    """
    The accuracy (i.e. percentage of correct classifications) is returned.
    net: trained network used to decode the data
    abstain: Boolean, whether the outout has an abstention class
    
    """
    decoded = list(decoder(net, data))
    predictions = [inst['pred'] for inst in decoded]
    gold = [inst['gold'] for inst in decoded]
    if abstain:
        abstain_i = data.num_senses()
    else:
        abstain_i = -1
    correct, total = accuracy(predictions, gold, abstain_i)
    acc = correct / total if total > 0 else 0
    print('correct, total: ', correct, total)
    return acc

