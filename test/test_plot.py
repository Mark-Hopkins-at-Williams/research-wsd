import unittest
from reed_wsd.plot import pr_curve, roc_curve
import numpy as np

def compare(a1, a2, num_decimal_places = 4):
    comparison = a1.round(num_decimal_places) == a2.round(num_decimal_places)
    return comparison.all()

def approx(x, y, num_decimal_places = 4):
    return round(x, num_decimal_places) == round(y, num_decimal_places)


class TestPlot(unittest.TestCase):
    
    def setUp(self):
        self.preds = [ {'pred': 1, 'gold': 7, 'confidence': 0.1}, 
                       {'pred': 8, 'gold': 8, 'confidence': 0.3}, 
                       {'pred': 4, 'gold': 9, 'confidence': 0.5},
                       {'pred': 6, 'gold': 6, 'confidence': 0.7}, 
                       {'pred': 3, 'gold': 3, 'confidence': 0.9}]
        
        
    
    def test_pr_curve(self):
        precision, recall, auc = pr_curve(self.preds)
        assert compare(precision, np.array([0.75, 0.66666667, 1., 1., 1.]))
        assert compare(recall, np.array([1., 0.66666667, 0.66666667, 
                                         0.33333333, 0.]))
        assert approx(auc, 0.9027777777777777)
        
    def test_roc_curve(self):
        fpr, tpr, auc = roc_curve(self.preds)
        assert compare(fpr, np.array([0., 0., 0., 0.5, 0.5, 1. ]))
        assert compare(tpr, np.array([0., 0.33333333, 0.66666667, 
                                      0.66666667, 1., 1. ]))
        assert approx(auc, 0.8333333333333333)
        
    
if __name__ == "__main__":
	unittest.main()
