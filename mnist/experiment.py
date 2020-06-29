from reed_wsd.mnist.mnist import train, validate_and_analyze, FFN, model_dir, validation_dir
from reed_wsd.mnist.loss import AWNLL, CAWNLL, CRANLL, LRANLL, CABNLL, NLL
import torch
from os.path import join
import json

def nll():
    print("================NLL EXPERIMENT=======================")
    c = NLL()
    train(c)
    net = FFN()
    net.load_state_dict(torch.load(join(model_dir, "params.pt")))
    data_dict = validate_and_analyze(net, c)
    loss_name = "nll" + ".json"
    with open(join(validation_dir, loss_name), "w") as f:
        json.dump(data_dict, f)

def awnll():
    print("================AWNLL EXPERIMENT=======================")
    ab = [[6,1], [8, 1]]
    for (a, b) in ab:
        print("training with weight: ({}, {})".format(a, b))
        c = AWNLL(a, b)
        train(c)
        net = FFN()
        net.load_state_dict(torch.load(join(model_dir, "params.pt")))
        data_dict = validate_and_analyze(net, c)
        loss_name = "awnll" + "_" + str(a) + "_" + str(b) + ".json"
        with open(join(validation_dir, loss_name), "w") as f:
            json.dump(data_dict, f)

def cawnll():
    print("================CAWNLL EXPERIMENT=======================")
    ab = []
    for i in range(11):
        ab.append([1 + 0.1 * i, 1])
    for (a, b) in ab:
        print("training with weight: ({}, {})".format(a, b))
        c = CAWNLL(a, b)
        train(c)
        net = FFN()
        net.load_state_dict(torch.load(join(model_dir, "params.pt")))
        data_dict = validate_and_analyze(net, c)
        loss_name = "cawnll" + "_" + str(a) + "_" + str(b) + ".json"
        with open(join(validation_dir, loss_name), "w") as f:
            json.dump(data_dict, f)

def cranll():
    print("================CRANLL EXPERIMENT=======================")
    p0 = 0.5
    c = CRANLL(p0)
    train(c)
    net = FFN()
    net.load_state_dict(torch.load(join(model_dir, "params.pt")))
    data_dict = validate_and_analyze(net, c)
    loss_name = "cranll.json"
    with open(join(validation_dir, loss_name), "w") as f:
        json.dump(data_dict, f)

def lranll():
    print("================LRANLL EXPERIMENT=======================")
    c = LRANLL()
    train(c)
    net = FFN()
    net.load_state_dict(torch.load(join(model_dir, "params.pt")))
    data_dict = validate_and_analyze(net, c)
    loss_name = "lranll.json"
    with open(join(validation_dir, loss_name), "w") as f:
        json.dump(data_dict, f)

def cabnll():
    print("================CABNLL EXPERIMENT=======================")
    c = CABNLL()
    train(c)
    net = FFN()
    net.load_state_dict(torch.load(join(model_dir, "params.pt")))
    data_dict = validate_and_analyze(net, c)
    loss_name = "cabnll.json"
    with open(join(validation_dir, loss_name), "w") as f:
        json.dump(data_dict, f)

if __name__ == "__main__":
    nll()
