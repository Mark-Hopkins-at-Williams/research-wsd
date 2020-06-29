import reed_wsd.plot as plt
import reed_wsd.mnist.mnist as mnist
from reed_wsd.mnist.loss import ConfidenceLoss1
import sys
from os.path import join, dirname, realpath
import os
import json
import torch

file_dir = dirname(realpath(__file__))
confidence_dir = join(file_dir, "confidence")
if not os.path.isdir(confidence_dir):
    os.mkdir(confidence_dir)

def closs_py(confidence):
    decoder = mnist.decode_gen(confidence)
    criterion = ConfidenceLoss1(0.5)
    net = mnist.FFN()
    net.load_state_dict(torch.load("saved/params_" + str(criterion) + ".pt"))
    pyc = plt.precision_yield_curve(net, mnist.valloader, decoder)
    f_name = "py_" + str(criterion) + "_" + confidence + ".json"
    with open(join(confidence_dir, f_name), "w") as f:
        json.dump(pyc, f)
    return pyc


if __name__ == "__main__":
    closs_py("baseline")
    closs_py("neg_abs")
