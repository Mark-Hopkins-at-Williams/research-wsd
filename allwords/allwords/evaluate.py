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

def decode(net, data, threshold=LARGE_NEGATIVE):
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
        below_threshold_idx = (maxes < threshold)
        preds[below_threshold_idx] = ABSTAIN
        for element in zip(inst_ids, 
                           zip(targets,
                               zip(preds, response.tolist()))):                    
            (inst_id, (target, (prediction, gold))) = element
            yield inst_id, target, prediction, gold
    net.train()

def evaluate(net, data):
    """
    The accuracy (i.e. percentage of correct classifications) is returned.
    
    """
    decoded = list(decode(net, data))
    predictions = [pred for (_, _, pred, _) in decoded]
    gold = [g for (_, _, _, g) in decoded]
    correct, total = accuracy(predictions, gold)
    return correct/total

def py_curve(predictor, gold, thresholds):
    pys = {}
    for thres in thresholds:
        preds = predictor(thres)
        n_correct, n_confident = accuracy(preds, gold)
        precision = 0 if n_confident == 0 else n_correct / n_confident
        n_correct, n_all = yielde(preds, gold)
        y = n_correct / n_all
        pys[thres] = (precision, y)
    return pys    

def precision_yield_curve(net, data):
    def predictor_fn(thres):
        result = list(decode(net, data, thres))
        return [pred for (_, _, pred, _) in result]        
    thresholds = [x / 100 for x in range(0, 101, 5)]
    decoded = list(decode(net, data, 0))
    gold = [g for (_, _, _, g) in decoded]
    return py_curve(predictor_fn, gold, thresholds)

def plot_py_curve(py_curve):
    thresholds = sorted(py_curve.keys())
    x = [py_curve[thres][1] for thres in thresholds]
    y = [py_curve[thres][0] for thres in thresholds]    
    fig = plt.figure()
    fig.subplots_adjust(top=0.8)
    ax1 = fig.add_subplot(211)
    ax1.set_title('Precision-Recall Curve')
    ax1.set_ylabel('precision')
    ax1.set_xlabel('recall')
    ax1.plot(x, y)

    
def save_py_curve(curve):
    confidence_path = join(file_dir, "../confidence")
    if not os.path.isdir(confidence_path):
        os.mkdir(confidence_path)
    jsonfile = join(file_dir, "../confidence/precision_yield_curve.json")    
    with open(jsonfile, "w") as f:
        json.dump(curve, f)
