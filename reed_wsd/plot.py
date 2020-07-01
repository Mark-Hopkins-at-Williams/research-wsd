import json
import os
from os.path import join
import matplotlib.pyplot as plt


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
    triples = sorted([(-c,p,g) for (c,(p,g)) in zip(confidences, zip(preds, gold))])
    for c, p, g in triples:
        if p != g:
            print('with conf {}, classified a gold {} as {}'.format(-c, g, p))
    correct = [int(p == g) for (c,p,g) in triples]
    cumul_correct = []
    sum_so_far = 0
    for element in correct:
        sum_so_far += element
        cumul_correct.append(sum_so_far)
    precisions = [corr/(i+1) for i, corr in enumerate(cumul_correct)]
    recalls = [corr/len(cumul_correct) for corr in cumul_correct]
    return list(zip(precisions, recalls))

def plot_py_curves(py_curves):
    fig = plt.figure()
    fig.subplots_adjust(top=0.8)
    ax1 = fig.add_subplot(211)
    ax1.set_title('Precision-Recall Curve')
    ax1.set_ylabel('precision')
    ax1.set_xlabel('recall')
    for py_curve in py_curves:
        x = [pr[1] for pr in py_curve]
        y = [pr[0] for pr in py_curve]
        ax1.plot(x, y)

def save_py_curve(curve):
    confidence_path = join(file_dir, "../confidence")
    if not os.path.isdir(confidence_path):
        os.mkdir(confidence_path)
    jsonfile = join(file_dir, "../confidence/precision_yield_curve.json")
    with open(jsonfile, "w") as f:
        json.dump(curve, f)

