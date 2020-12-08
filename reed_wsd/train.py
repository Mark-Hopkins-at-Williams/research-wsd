from reed_wsd.plot import pr_curve, roc_curve, plot_roc, plot_pr, risk_coverage_curve
from reed_wsd.util import cudaify
from collections import defaultdict
import copy
from reed_wsd.util import ABS

class Decoder:

    def __call__(self, net, data):
        raise NotImplementedError("This feature needs to be implemented.")

class Trainer:
    
    def __init__(self, criterion, optimizer, train_loader, val_loader, decoder, n_epochs, trustmodel, scheduler):
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_epochs = n_epochs
        self.decoder = decoder
        self.trust_model = trustmodel
        self.scheduler = scheduler

    def _epoch_step(self, optimizer, model):
        raise NotImplementedError("Must be overridden by inheriting classes.")
    
    def __call__(self, model):
        model = cudaify(model)
        abs_rate_graph = []
        for e in range(self.n_epochs):
            print(self.optimizer.param_groups[0]['lr'])
            self.criterion.notify(e)
            batch_loss = self._epoch_step(model)
            analytics = self.validate_and_analyze(model)
            precision = analytics['precision']
            print(analytics)
            print("epoch {} training loss: ".format(e) + str(batch_loss))
            if self.scheduler is not None:
                self.scheduler.step()
            abs_rate_graph.append([e, 1-analytics['coverage']])
        print(abs_rate_graph)
        return model, analytics

    def validate_and_analyze(self, model):
        model.eval()
        results = list(self.decoder(model, self.val_loader, self.trust_model))
        _, _, auroc = roc_curve(results)
        _, _, aupr = pr_curve(results)
        _, _, capacity = risk_coverage_curve(results)
        avg_err_conf = 0
        avg_crr_conf = 0
        n_error = 0
        n_correct = 0
        n_published = 0
        n_total = len(results)
        for result in results:
            if result['abstained']:
                n_published += 1
            prediction = result['pred']
            gold = result['gold']
            confidence = result['confidence']
            if prediction == gold:
                avg_crr_conf += confidence
                n_correct += 1
            else:
                #print("mistook {} for {}".format(gold, prediction))
                avg_err_conf += confidence
                n_error += 1            
        return {'avg_err_conf': avg_err_conf / n_error if n_error > 0 else 0,
                'avg_crr_conf': avg_crr_conf / n_correct if n_correct > 0 else 0,
                'auroc': auroc,
                'aupr': aupr,
                'capacity': capacity,
                'precision': n_correct / n_total if n_published > 0 else 0,
                'coverage': n_published / n_total if n_total > 0 else 0}
