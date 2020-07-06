import reed_wsd.plot as plt
import reed_wsd.mnist.mnist as mnist
from reed_wsd.mnist.loss import ConfidenceLoss1, NLL
import json
from reed_wsd.plot import PYCurve, plot_curves


"""
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
"""

def create_pr_curve(config, output_file = '../results.json'):
    decoder = mnist.decode_gen(config['confidence'])
    criterion = config['loss']
    config['loss'] = str(criterion)
    net = mnist.train(criterion)
    pyc = plt.PYCurve.from_data(net, mnist.valloader, decoder)
    try:    
        with open(output_file, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        results = []
    with open(output_file, "w") as f:
        r = {'config': config,
             'result': pyc.get_list()}
        results.append(r)
        json.dump(results, f)
        
def closs_py(confidence):
    config = {'task': 'mnist',
              'loss': ConfidenceLoss1(0),
              'confidence': confidence}
    create_pr_curve(config)

def nll_py():
    config = {'task': 'mnist',
              'loss': NLL(),
              'confidence': 'baseline'}    
    create_pr_curve(config)
    

if __name__ == "__main__":
    closs_py('baseline')


