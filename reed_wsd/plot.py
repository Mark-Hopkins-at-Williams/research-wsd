import os
import matplotlib.pyplot as plt
from sklearn import metrics 
from reed_wsd.util import ABS
import numpy as np

LARGE_NEGATIVE = 0
file_dir = os.path.dirname(os.path.realpath(__file__))

def plot_roc(predictions):
    fpr, tpr, auc = roc_curve(predictions)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUROC = %0.2f' % auc)
    plt.legend(loc = 'lower right')
    axes = plt.gca()
    axes.set_ylim([-0.05, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def plot_pr(predictions):
    precision, recall, auc = pr_curve(predictions)
    plt.title('Precision-Recall')
    plt.plot(recall, precision, 'b', label = 'AUPR = %0.2f' % auc)
    plt.legend(loc = 'lower right')
    axes = plt.gca()
    axes.set_ylim([-0.05, 1.05])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()
    
def pr_curve(predictions):
    y_true = [int(pred['pred'] == pred['gold']) for pred in predictions]
    y_scores = [pred['confidence'] for pred in predictions]
    if len(y_true) == 0 or len(y_scores) == 0:
        return None, None, None
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_scores)
    auc = metrics.auc(recall, precision)
    return precision, recall, auc

def roc_curve(predictions):
    y_true = [int(pred['pred'] == pred['gold']) for pred in predictions]
    y_scores = [pred['confidence'] for pred in predictions]
    if len(y_true) == 0 or len(y_scores) == 0:
        return None, None, None
    fpr, tpr, _ = metrics.roc_curve(y_true, y_scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return fpr, tpr, auc

def risk_coverage_curve(predictions):
    # this functions plots unconditional error rate against coverage
    y_true = [int(pred['pred'] == pred['gold']) for pred in predictions]
    y_scores = [pred['confidence'] for pred in predictions]
    if len(y_true) == 0 or len(y_scores) == 0:
        return None, None, None
    precision, _, thresholds = metrics.precision_recall_curve(y_true, y_scores)
    y_scores = sorted(y_scores)
    coverage = []
    N = len(y_scores)
    j = 0
    for i, t in enumerate(thresholds):
        while j < len(y_scores) and y_scores[j] < t:
            j += 1
        coverage.append((N - j) / N)
    coverage += [0.]
    conditional_err = 1 - precision
    unconditional_err = conditional_err * coverage
    coverage = np.array(coverage)
    capacity = 1 - metrics.auc(coverage, unconditional_err) 
    return coverage, unconditional_err, capacity



class PYCurve:
    def __init__(self, scatters):
        self.scatters = scatters

    def get_list(self):
        return self.scatters
    
    @classmethod
    def precision_yield_curve(cls, decoded):
        """
        decode must be a iterable in which each element is of the form (prediction, gold, confidence)
        
        """     
        decoded.sort(key = lambda inst: inst['confidence']) # sort decoded by confidence
        preds = [inst['pred'] for inst in decoded]
        gold = [inst['gold'] for inst in decoded]
        confidences = [inst['confidence'] for inst in decoded]
        return cls.py_curve(preds, gold, confidences)

    @staticmethod    
    def py_curve(preds, gold, confidences):
        triples = sorted([(-c,p,g) for (c,(p,g)) in zip(confidences, zip(preds, gold))])
        #for c, p, g in triples:
        #    if p != g:
        #        print('with conf {}, classified a gold {} as {}'.format(-c, g, p))
        correct = [int(p == g) for (_,p,g) in triples]
        cumul_correct = []
        sum_so_far = 0
        for element in correct:
            sum_so_far += element
            cumul_correct.append(sum_so_far)
        precisions = [corr/(i+1) for i, corr in enumerate(cumul_correct)]
        recalls = [corr/len(cumul_correct) for corr in cumul_correct]
        return list(zip(precisions, recalls))    
    
    @classmethod
    def from_data(cls, decoded):
        print(decoded)
        return cls(cls.precision_yield_curve(decoded))

    def aupy(self):
        area = 0
        prev_x = self.scatters[0][1]
        for yx in self.scatters:
            x = yx[1]
            y = yx[0]
            area += y * (x - prev_x)
            prev_x = x
        return area

    def plot(self, label=None):
        os.environ['KMP_DUPLICATE_LIB_OK']='True'
        #sns.set()
        x = [yx[1] for yx in self.scatters]
        y = [yx[0] for yx in self.scatters]
        if label != None:
            plt.plot(x, y, label=label)
        else:
            plt.plot(x, y)

def plot_curves(*pycs):
    for i in range(len(pycs)):
        curve = pycs[i][0]
        label = pycs[i][1]
        label = label + "; aupy = {:.3f}".format(curve.aupy())
        curve.plot(label)
    plt.legend()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()
