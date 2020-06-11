import torch
from collections import defaultdict
import json
import os
from os.path import join
from random import sample


LARGE_NEGATIVE = -10000000
file_dir = os.path.dirname(os.path.realpath(__file__))

def predict(distribution):
    return distribution.argmax()

def accuracy(predicted_labels, gold_labels):
    assert(len(predicted_labels) == len(gold_labels))
    confident = [l for l in predicted_labels if l != -1.0]
    correct = [l1 for (l1, l2) in zip(predicted_labels, gold_labels)
                  if l1 == l2]
    return len(correct), len(confident)

def apply_zone_masks(outputs, zones):
    revised = torch.empty(outputs.shape)
    revised = revised.fill_(LARGE_NEGATIVE)
    for row in range(len(zones)):
        (start, stop) = zones[row]
        revised[row][start:stop] = outputs[row][start:stop]
    return revised

def decode(net, data, threshold=LARGE_NEGATIVE):
    """
    Runs a trained neural network classifier on validation data, and iterates
    through the top prediction for each datum.
    
    """        
    net.eval()
    val_loader = data.batch_iter()
    for inst_ids, targets, evidence, response, zones in val_loader:
        val_outputs = net(evidence)
        revised = apply_zone_masks(val_outputs, zones)
        maxes, preds = revised.max(dim=-1)
        print(maxes)
        below_threshold_idx = (maxes < threshold)
        preds[below_threshold_idx] = -1.0
        #print(preds)
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
    print(predictions)
    gold = [g for (_, _, _, g) in decoded]
    correct, total = accuracy(predictions, gold)
    return correct/total

def confidence_analysis(net, data):
    id_range = list(range(len(data.instance_dataset())))
    print(len(id_range))
    sample_ids = sample(id_range, 5)
    positive_inv = []
    for i in sample_ids:
        print(i)
        positive_inv.append(data.instance_dataset()[i].get_sense())
    tprs = defaultdict(list)
    fprs = defaultdict(list)
    for curr_thres in range(0, 101, 10):
        decoded = list(decode(net, data, curr_thres))
        preds = torch.tensor([pred for (_, _, pred, _) in decoded])
        gold = torch.tensor([g for (_, _, _, g) in decoded])
        for positive_sense in positive_inv:
            print(positive_sense)
            positive_id = data.sense_id(positive_sense)
            pred_idx = (preds == positive_id)
            gold_idx = (gold == positive_id)
            """
            print(pred_idx.double().sum())
            print(gold_idx.double().sum())
            """
            tp = ((pred_idx * gold_idx) == 1.0).double().sum()
            n_s_p = ((preds != -1.0) * gold_idx).double().sum()
            fp = (pred_idx * (gold_idx == 0)).double().sum()
            n_abs = (preds == -1.0).double().sum()
            print("n_abs:", n_abs)
            n_s_n = gold.shape[0] - n_abs - n_s_p
            print("tp:", tp)
            print("fp:", fp)
            print("n_s_p:", n_s_p)
            print("n_s_n:", n_s_n)
            tpr = 0 if n_s_p == 0 else (tp / n_s_p).item()
            fpr = 0 if n_s_n == 0 else (fp / n_s_n).item()
            print("tpr:", tpr)
            print("fpr:", fpr)
            tprs[positive_sense].append(tpr)
            fprs[positive_sense].append(fpr)
    with open(join(file_dir, "../confidence/prc.json"), "w") as f:
        json.dump([tprs, fprs], f)

            
        
