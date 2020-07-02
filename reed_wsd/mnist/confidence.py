import reed_wsd.plot as plt
import reed_wsd.mnist.mnist as mnist
from reed_wsd.mnist.loss import ConfidenceLoss1, NLL
import sys
from os.path import join, dirname, realpath
import os
import json
import torch

output_file = "../results.json"
if os.stat(output_file).st_size < 3:
    with open(output_file, "w") as f:
        json.dump([], f)

file_dir = dirname(realpath(__file__))
confidence_dir = join(file_dir, "confidence")
if not os.path.isdir(confidence_dir):
    os.mkdir(confidence_dir)

def closs_py(confidence):
    config = {'task': 'mnist',
              'loss': 'ConfidenceLoss_0.5',
              'confidence': confidence}
    decoder = mnist.decode_gen(confidence)
    criterion = ConfidenceLoss1(0)
    net = mnist.train(criterion)
    pyc = plt.precision_yield_curve(net, mnist.valloader, decoder)
    with open(output_file, "r") as f:
        results = json.load(f)
    with open(output_file, "w") as f:
        r = {'config': config,
             'result': pyc}
        results.append(r)
        json.dump(results, f)

def nll_py():
    config = {'task': 'mnist',
              'loss': 'NLLLoss',
              'confidence': 'baseline'}
    decoder = mnist.decode_gen('baseline')
    criterion = NLL()
    net = mnist.train(criterion)
    pyc = plt.precision_yield_curve(net, mnist.valloader, decoder)
    with open(output_file, "r") as f:
        results = json.load(f)
    with open(output_file, "w") as f:
        r = {'config': config,
             'result': pyc}
        results.append(r)
        json.dump(results, f)


if __name__ == "__main__":
    nll_py()
    closs_py('baseline')
    closs_py('neg_abs')
