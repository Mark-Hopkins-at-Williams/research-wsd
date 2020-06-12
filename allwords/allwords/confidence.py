import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from os.path import join
import torch
from evaluate import precision_yield_curve, plot_py_curve
from allwords.networks import AffineClassifier
from allwords.util import cudaify
from allwords.run import init_dev_loader

file_dir = os.path.dirname(os.path.realpath(__file__))

def main(data_dir):
    batch_size = 16    
    print("Initializing data loader.")
    dev_loader = init_dev_loader(data_dir, batch_size)
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
    #plot_py_curve(py_curve)
    
    
if __name__ == "__main__":
    data_dir = sys.argv[1]    
    main(data_dir)