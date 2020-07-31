import unittest
from reed_wsd.plot import PYCurve
from reed_wsd.util import approx

class TestPYCurve(unittest.TestCase):

    def test_aupy(self):
        scatters = [[0.9, 0.1],
                         [0.75, 0.3],
                         [0.6, 0.5]]
        expected_aupy = 0.2 * 0.75 + 0.2 * 0.6
        pycurve = PYCurve(scatters)
        aupy = pycurve.aupy()
        assert approx(aupy, expected_aupy)

    def test_from_data(self):
        decoded = [{'pred': 0, 'gold': 0, 'confidence': 0.9},
                   {'pred': 0, 'gold': 1, 'confidence': 0.6},
                   {'pred': 1, 'gold': 1, 'confidence': 1.0},
                   {'pred': 1, 'gold': 0, 'confidence': 0.5}]
        pyc = PYCurve.from_data(decoded)
        expected_aupy = 0.25
        aupy = pyc.aupy()
        assert(aupy == expected_aupy)
       
if __name__ == "__main__":
    unittest.main()
