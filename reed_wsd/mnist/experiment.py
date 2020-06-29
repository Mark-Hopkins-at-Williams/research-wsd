from reed_wsd.mnist.mnist import train, validate_and_analyze, FFN, model_dir, validation_dir
from reed_wsd.mnist.loss import AWNLL, CAWNLL, CRANLL, LRANLL, CABNLL, NLL
import torch
from os.path import join
import json

def nll():
    run(NLL())

def awnll():
    ab = [[6,1], [8, 1]]
    for (a, b) in ab:
        print("training with weight: ({}, {})".format(a, b))
        c = AWNLL(a, b)
        run(c, "awnll" + "_" + str(a) + "_" + str(b))
    

def cawnll():
    ab = [[1 + 0.1 * i, 1] for i in range(11)]
    for (a, b) in ab:
        print("training with weight: ({}, {})".format(a, b))
        c = CAWNLL(a, b)
        run(c, "cawnll" + "_" + str(a) + "_" + str(b))

def cranll():
    p0 = 0.5
    run(CRANLL(p0)) 

def lranll():
    run(LRANLL())

def cabnll():
    run(CABNLL())        

def run(criterion, name = None):
    if name is None:
        name = type(criterion).__name__
    print("================{} EXPERIMENT======================".format(name))
    net = train(criterion)
    data_dict = validate_and_analyze(net, criterion)
    results_file = "{}.json".format(name.lower())
    with open(join(validation_dir, results_file), "w") as f:
        json.dump(data_dict, f)
    
if __name__ == "__main__":
    nll()
