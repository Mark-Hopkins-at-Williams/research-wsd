import unittest
from torch import tensor
from allwords.evaluate import predict, accuracy, yielde, ABSTAIN, py_curve

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
        
    def test_accuracy1(self):
        predicted = [3,6,5,1,2,2,2,4,1,3]
        gold = [3,6,5,2,2,2,1,4,7,3]
        assert(accuracy(predicted, gold) == (7, 10))

    def test_accuracy2(self):
        predicted = [3,6,5,ABSTAIN,2,2,2,4,ABSTAIN,3]
        gold = [3,6,5,2,2,2,1,4,7,3]
        assert(accuracy(predicted, gold) == (7, 8))

    def test_accuracy3(self):
        predicted = [3,ABSTAIN,5,ABSTAIN,2,2,2,4,ABSTAIN,3]
        gold = [3,6,5,2,2,2,1,4,7,3]
        assert(accuracy(predicted, gold) == (6, 7))

    def test_yield(self):
        predicted = [3,6,5,1,2,2,2,4,1,3]
        gold = [3,6,5,2,2,2,1,4,7,3]
        assert(yielde(predicted, gold) == (7, 10))
 
    def test_yield2(self):
        predicted = [3,6,5,ABSTAIN,2,2,2,4,ABSTAIN,3]
        gold = [3,6,5,2,2,2,1,4,7,3]
        assert(yielde(predicted, gold) == (7, 10))

    def test_yield3(self):
        predicted = [3,ABSTAIN,5,ABSTAIN,2,2,2,4,ABSTAIN,3]
        gold = [3,6,5,2,2,2,1,4,7,3]
        assert(yielde(predicted, gold) == (6, 10))

    def test_py_curve(self):
        def predictor(thres):
            result = [3,6,5,1,2,2,2,4,1,3]
            if thres < 0.2:
                pass
            elif thres < 0.4:
                result[3] = ABSTAIN
                result[8] = ABSTAIN
            else:
                result[3] = ABSTAIN
                result[8] = ABSTAIN
                result[1] = ABSTAIN
                result[0] = ABSTAIN
                result[4] = ABSTAIN
            return result
                    
        gold = [3,6,5,2,2,2,1,4,7,3]
        expected ={0.1: (0.7, 0.7), 
                   0.3: (0.875, 0.7), 
                   0.5: (0.8, 0.4)}
        assert py_curve(predictor, gold, [0.1, 0.3, 0.5]) == expected


if __name__ == "__main__":
	unittest.main()
