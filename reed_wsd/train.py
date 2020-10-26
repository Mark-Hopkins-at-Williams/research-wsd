from reed_wsd.plot import pr_curve, roc_curve, plot_roc, plot_pr
from reed_wsd.util import cudaify
from collections import defaultdict
import copy
from reed_wsd.trustscore import TrustScore
import torch

class Decoder:

    def __call__(self, net, data, trust_model=None):
        raise NotImplementedError("This feature needs to be implemented.")

class Trainer:
    
    def __init__(self, criterion, optimizer, train_loader, val_loader, decoder, n_epochs, trust_model):
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_epochs = n_epochs
        self.decoder = decoder
        self.trust_model = trust_model

    def _epoch_step(self, optimizer, model):
        raise NotImplementedError("Must be overridden by inheriting classes.")
    
    def validate_and_analyze(self, model):
        model.eval()
        results = list(self.decoder(model, self.val_loader, self.trust_model))
        # results is a dictionary with keys {evidence, pred, gold, confidence, trustscore}
        #plot_roc(results)
        avg_err_conf = 0
        avg_crr_conf = 0
        n_error = 0
        n_correct = 0
        data_dict = {}
        error_dict = defaultdict(int)
        n_total = len(results)
        """
        for i in range(model.output_size):
            data_dict[i] = [0, 0] # [n_correct, n_wrong, n_abstain]
        """
        for result in results:
            prediction = result['pred']
            gold = result['gold']
            confidence = result['confidence']
            if prediction == gold:
                #data_dict[gold][0] += 1
                avg_crr_conf += confidence
                n_correct += 1
            else:
                #print("mistook {} for {}".format(gold, prediction))
                # data_dict[gold][1] += 1
                avg_err_conf += confidence
                error_dict[gold] += 1
                n_error += 1            
        _, _, auroc = roc_curve(results)
        _, _, aupr = pr_curve(results)
        return {#'prediction_by_class': data_dict,
                'avg_err_conf': avg_err_conf / n_error,
                'avg_crr_conf': avg_crr_conf / n_correct,
                'auroc': auroc,
                'aupr': aupr,
                'precision': n_correct / n_total}

    def __call__(self, model):
        model = cudaify(model)
        for e in range(self.n_epochs):
            self.criterion.notify(e)
            batch_loss = self._epoch_step(model)
            analytics = self.validate_and_analyze(model)
            #print(data_dict)
            print("Epoch {} - Training loss: {}".format(e,
                                                        batch_loss))
            print(analytics)
        print("Best Model analytics:")
        return model, analytics

