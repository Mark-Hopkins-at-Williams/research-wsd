import json
import os
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns

LARGE_NEGATIVE = 0
file_dir = os.path.dirname(os.path.realpath(__file__))

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
