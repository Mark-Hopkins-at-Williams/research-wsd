from reed_wsd.plot import pr_curve, roc_curve, plot_roc, plot_pr
from collections import defaultdict
import copy

def validate_and_analyze(model, val_loader, decoder, output_size):
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
    for i in range(output_size):
        data_dict[i] = [0, 0] # [n_correct, n_wrong, n_abstain]
    for result in results:
        prediction = result['pred']
        gold = result['gold']
        confidence = result['confidence']
        if prediction == gold:
            data_dict[gold][0] += 1
            avg_crr_conf += confidence
            n_correct += 1
        else:
            #print("mistook {} for {}".format(gold, prediction))
            data_dict[gold][1] += 1
            avg_err_conf += confidence
            error_dict[gold] += 1
            n_error += 1            
    print('')
    print('average error confidence:', avg_err_conf / n_error)
    print('average correct confidence:', avg_crr_conf / n_correct)
    print('auroc: {}'.format(auroc))
    print('aupr: {}'.format(aupr))
    print(data_dict)
    return data_dict, n_correct / n_total

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
        best_model = None
        best_model_score = float('-inf')
        for e in range(self.n_epochs):
            self.criterion.notify(e)
            batch_loss = self._epoch_step(model)
            data_dict, precision = validate_and_analyze(model, self.val_loader, self.decoder, output_size=model.output_size)
            #print(data_dict)
            if precision > best_model_score:
                print("Updating best model.")
                best_model = copy.deepcopy(model)
                best_model_score = precision
            print("Epoch {} - Training loss: {}; Dev precision: {}".format(e,
                                                                          batch_loss,
                                                                          precision))
        data_dict, precision = validate_and_analyze(best_model, self.val_loader, self.decoder, output_size=10)
        print("Best Model Dev precision: {}".format(precision))
        return best_model

