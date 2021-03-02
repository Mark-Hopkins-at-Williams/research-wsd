from reed_wsd.plot import pr_curve, roc_curve, plot_roc, plot_pr, risk_coverage_curve
from reed_wsd.util import cudaify
from collections import defaultdict
import copy
from reed_wsd.util import log
from datetime import datetime
from reed_wsd.plot import plot_and_save_confidence_distr, plot_coverage
import json

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
        best_model = None
        best_summary = 0
        best_validation = None
        coverages = []
        for e in range(self.n_epochs):
            #log(self.optimizer.param_groups[0]['lr'])
            self.criterion.notify(e)
            batch_loss = self._epoch_step(model)
            analytics, validation = self.validate_and_analyze(model)
            coverages.append(analytics['coverage'])
            if best_validation == None:
                best_validation = validation
            """
            summary = (analytics['auroc'] / 50  + 0.5 * (analytics['aupr/succ'] / analytics['precision'] 
                       + analytics['aupr/err'] / (1 - analytics['precision'])) + analytics['capacity'] / analytics['precision'])
            """
            summary = e
            log(analytics)
            log("epoch {} training loss: ".format(e) + str(batch_loss))
            if summary > best_summary:
                best_model = copy.deepcopy(model)
                best_analytics = analytics
                best_summary = summary
                best_validation = validation
            if self.scheduler is not None:
                self.scheduler.step()
            abs_rate_graph.append([e, 1-analytics['coverage']])
            log("\n")
        #print(abs_rate_graph)
        log("Best model performance\n" + str(best_analytics))
        curr_time = datetime.now().strftime("%Y_%b,%d_%H:%M:%S")
        graph_path = curr_time + ".conf_distr.png"
        # plot_and_save_confidence_distr(results, graph_path)
        '''
        with open("mnist_dac_coverages.json", "w") as f:
            json.dump(coverages, f)
        '''
        return best_model, best_analytics, best_validation

    def validate_and_analyze(self, model):
        model.eval()
        results = list(self.decoder(model, self.val_loader, self.trust_model))
        _, _, auroc = roc_curve(results)
        _, _, aupr_succ = pr_curve(results, succ_as_positive=True)
        _, _, aupr_err = pr_curve(results, succ_as_positive=False)
        _, _, capacity = risk_coverage_curve(results)
        avg_err_conf = 0
        avg_crr_conf = 0
        n_error = 0
        n_correct = 0
        n_published = 0
        n_total = len(results)
        for result in results:
            if not result['abstained']:
                n_published += 1
            prediction = result['pred']
            gold = result['gold']
            confidence = result['confidence']
            if prediction == gold:
                n_correct += 1
                avg_crr_conf = ((n_correct - 1) / n_correct) * avg_crr_conf + (confidence / n_correct)
            else:
                #print("mistook {} for {}".format(gold, prediction))
                n_error += 1
                avg_err_conf = ((n_error - 1) / n_error) * avg_err_conf + (confidence / n_error)

        return {'avg_err_conf': avg_err_conf,
                'avg_crr_conf': avg_crr_conf,
                'auroc': auroc,
                'aupr/succ': aupr_succ,
                'aupr/err': aupr_err,
                'capacity': capacity,
                'precision': n_correct / n_total if n_published > 0 else 0,
                'coverage': n_published / n_total if n_total > 0 else 0}, results
