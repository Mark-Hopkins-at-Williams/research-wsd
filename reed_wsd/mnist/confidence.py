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

def closs_py(model_file, confidence = 'baseline'):
    decoder = mnist.decode_gen(confidence)
    #criterion = ConfidenceLoss1(0.4)
    net = mnist.FFN()
    #saved_model = "saved/params_" + str(criterion) + ".pt"
    saved_model = model_file
    print("loading {}".format(saved_model))
    net.load_state_dict(torch.load(saved_model))
    pyc = plt.precision_yield_curve(net, mnist.valloader, decoder)
    #f_name = "py_" + str(criterion) + "_" + confidence + ".json"
    #with open(join(confidence_dir, f_name), "w") as f:
    #    json.dump(pyc, f)
    return pyc


def main():
    return closs_py("baseline")
    #return closs_py("neg_abs")

if __name__ == '__main__':
    main()