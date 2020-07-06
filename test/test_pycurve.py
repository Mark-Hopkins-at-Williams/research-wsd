import unittest
from reed_wsd.plot import PYCurve
from reed_wsd.util import approx

class TestPYCurve(unittest.TestCase):

    def test_aupy(self):
        scatters = [[0.9, 0.1],
                         [0.75, 0.3],
                         [0.6, 0.5]]
        expected_aupy = 0.1 * 0.9 + 0.2 * 0.75 + 0.2 * 0.6
        pycurve = PYCurve(scatters)
        aupy = pycurve.aupy()
        assert approx(aupy, expected_aupy)
       
if __name__ == "__main__":
    unittest.main()
