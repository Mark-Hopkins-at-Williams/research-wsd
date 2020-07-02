import unittest
from reed_wsd.plot import PYCurve
from reed_wsd.util import approx

class TestPYCurve(unittest.TestCase):

    def setUp(self):
        self.dict = {0.3: [0.6, 0.5],
                     0.6: [0.75, 0.3],
                     1.0: [0.9, 0.1]}
        self.aupy = 0.1 * 0.9 + 0.2 * 0.75 + 0.2 * 0.6
        self.pycurve = PYCurve(self.dict)

    def test_aupy(self):
        aupy = self.pycurve.aupy()
        assert approx(aupy, self.aupy)
       
if __name__ == "__main__":
    unittest.main()
