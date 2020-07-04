import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from os.path import join
import torch
import json
from reed_wsd.plot import PYCurve
from reed_wsd.allwords.networks import AffineClassifier
from reed_wsd.util import cudaify
from reed_wsd.allwords.run import init_loader, decode_gen, train_all_words_classifier
from reed_wsd.allwords.loss import ConfidenceLossWithZones, NLLLossWithZones
from reed_wsd.util import Logger

file_dir = os.path.dirname(os.path.realpath(__file__))
result_f = "../results.json"

def main(data_dir):
    batch_size = 16    
    print("Initializing data loader.")
    dev_loader = init_loader(data_dir, "dev", batch_size)
    input_size = 768 # TODO: what is it in general?
    output_size = dev_loader.num_senses()
    print("Loading saved neural network.")
    net = AffineClassifier(input_size, output_size)
    net.load_state_dict(torch.load(join(file_dir, "../saved/bert_simple.pt")))
    net = cudaify(net)
    print("Computing PR curve.")
    py_curve = precision_yield_curve(net, dev_loader)
    print(py_curve)
    return py_curve
    
def closs_py(confidence):
    """
    cofidence can be either 'baseline' or 'neg_abs'
    the 'baseline' confidence metric uses the maximum class prob. among 
    the non-abstention classes
    the 'neg_abs' metric uses 1 - abstention class prob. as the confidence
    metric
    """
    config = {"task": "allwords",
              "confidence": confidence,
              "loss": "ConfidenceLossWithZonesi_0.5",
              "n_epochs": 20}

    decoder = decode_gen(True, confidence)
    data_dir = "./data"
    batch_size = 16    

    print("Initializing data loader.")
    train_loader = init_loader(data_dir, "train", batch_size)
    dev_loader = init_loader(data_dir, "dev", batch_size)
    input_size = 768 # TODO: what is it in general?
    output_size = dev_loader.num_senses() + 1
    
    print("teaching to classification")
    net = AffineClassifier(input_size, output_size)
    loss1 = NLLLossWithZones()
    net = train_all_words_classifier(net, train_loader, dev_loader, loss1, n_epochs=20, logger=Logger(verbose=True), abstain=True)
    
    print("teaching abstention")
    loss2 = ConfidenceLossWithZones(0.5)
    net = train_all_words_classifier(net, train_loader, dev_loader, loss2, n_epochs=20, logger=Logger(verbose=True), abstain=True)

    print("Computing PR curve.")
    py_curve = PYCurve.from_data(net, dev_loader, decoder)
    with open(result_f, "r") as reader:
        results = json.load(reader)
    results.append({'config': config,
                    'result': py_curve.get_list()})
    with open(result_f, "w") as writer:
        json.dump(results, writer) 
    
    print(py_curve)
    
    
if __name__ == "__main__":
    decoder = decode_gen(True, "baseline")
    data_dir = "./data"
    batch_size = 16    

    print("Initializing data loader.")
    train_loader = init_loader(data_dir, "train", batch_size)
    dev_loader = init_loader(data_dir, "dev", batch_size)
    input_size = 768 # TODO: what is it in general?
    output_size = dev_loader.num_senses() + 1
    
    print("teaching to classification")
    net = AffineClassifier(input_size, output_size)
    loss1 = NLLLossWithZones()
    net = train_all_words_classifier(net, train_loader, dev_loader, loss1, n_epochs=20, logger=Logger(verbose=True), abstain=True)
    
    print("teaching abstention")
    loss2 = ConfidenceLossWithZones(0.5)
    net = train_all_words_classifier(net, train_loader, dev_loader, loss2, n_epochs=20, logger=Logger(verbose=True), abstain=True)

    with open("trained_models/abstain.pt", "w") as f:
        torch.save(net.state_dict(), "trained_models/abstain.pt")
