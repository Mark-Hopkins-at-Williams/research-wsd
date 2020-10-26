from reed_wsd.plot import pr_curve, roc_curve, plot_roc, plot_pr
from reed_wsd.util import cudaify
from collections import defaultdict
import copy

def validate_and_analyze(model, val_loader, decoder, output_size=None):
    model.eval()
    results = list(decoder(model, val_loader))
    _, _, auroc = roc_curve(results)
    _, _, aupr = pr_curve(results)
    plot_roc(results)
    avg_err_conf = 0
    avg_crr_conf = 0
    n_error = 0
    n_correct = 0
    data_dict = {}
    error_dict = defaultdict(int)
    n_total = len(results)
    if output_size is not None:
        for i in range(output_size):
            data_dict[i] = [0, 0] # [n_correct, n_wrong, n_abstain]
    for result in results:
        prediction = result['pred']
        gold = result['gold']
        confidence = result['confidence']
        if prediction == gold:
            if output_size is not None:
                data_dict[gold][0] += 1
            avg_crr_conf += confidence
            n_correct += 1
        else:
            #print("mistook {} for {}".format(gold, prediction))
            if output_size is not None:
                data_dict[gold][1] += 1
            avg_err_conf += confidence
            error_dict[gold] += 1
            n_error += 1            
    return {'prediction_by_class': data_dict,
            'avg_err_conf': avg_err_conf / n_error,
            'avg_crr_conf': avg_crr_conf / n_correct,
            'auroc': auroc,
            'aupr': aupr,
            'precision': n_correct / n_total}

class Decoder:

    def __call__(self, net, data):
        raise NotImplementedError("This feature needs to be implemented.")

class Trainer:
    
    def __init__(self, criterion, optimizer, train_loader, val_loader, decoder, n_epochs):
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_epochs = n_epochs
        self.decoder = decoder

    def _epoch_step(self, optimizer, model):
        raise NotImplementedError("Must be overridden by inheriting classes.")
    
    def __call__(self, model):
        model = cudaify(model)
        for e in range(self.n_epochs):
            self.criterion.notify(e)
            batch_loss = self._epoch_step(model)
            analytics = validate_and_analyze(model, self.val_loader, self.decoder, output_size=model.output_size)
            precision = analytics['precision']
            print(analytics)
            print("training loss: " + batch_loss)
        return best_model, analytics

