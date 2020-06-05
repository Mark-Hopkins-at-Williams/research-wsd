import unittest
from evaluate import predict, accuracy
from torch import tensor


class TestEvaluate(unittest.TestCase):
    
    def test_predict(self):
        t = tensor([0.35, 0.2, 0.1, 0.05, 0.3])
        assert(predict(t) == 0)
        t = tensor([0.1, 0.35, 0.2, 0.05, 0.3])
        assert(predict(t) == 1)
        t = tensor([0.1, 0.2, 0.35, 0.05, 0.3])
        assert(predict(t) == 2)
        t = tensor([0.1, 0.2, 0.05, 0.35, 0.3])
        assert(predict(t) == 3)
        t = tensor([0.1, 0.2, 0.3, 0.05, 0.35])
        assert(predict(t) == 4)
        
    def test_accuracy(self):
        predicted = [3,6,5,1,2,2,2,4,1,3]
        gold = [3,6,5,2,2,2,1,4,7,3]
        assert(accuracy(predicted, gold) == (7, 10))
     
if __name__ == "__main__":
	unittest.main()
