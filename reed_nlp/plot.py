import torch
import json
import os
from os.path import join
import torch.nn.functional as F
import matplotlib.pyplot as plt


LARGE_NEGATIVE = 0
ABSTAIN = -1.0
file_dir = os.path.dirname(os.path.realpath(__file__))

def precision_yield_curve(net, data, decoder):
    """
    decode must be a iterable in which each element is of the form (prediction, gold, confidence)
    
    """
    decoded = list(decoder(net, data))
    decoded.sort(key = lambda inst: inst['confidence']) # sort decoded by confidence
    preds = [inst['pred'] for inst in decoded]
    gold = [inst['gold'] for inst in decoded]
    confidences = [inst['confidence'] for inst in decoded]
    return py_curve(preds, gold, confidences)

def py_curve(preds, gold, confidences):
    pr_curve = {}
    for c in confidences:
        pr_curve[c] = [0,0] # first is n_correct, second n_confident
    for thres in pr_curve:
        n_correct = 0
        n_confident = 0
        for i in range(len(preds)):
            p = preds[i]
            g = gold[i]
            c = confidences[i]
            is_correct = (p == g)
            if c >= thres:
                n_confident += 1
                if is_correct:
                    n_correct += 1
        pr_curve[thres][0] = n_correct / n_confident
        pr_curve[thres][1] = n_correct / len(preds)
    return pr_curve

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

