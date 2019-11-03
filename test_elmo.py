import unittest
from wordsense import SenseInstance
from elmo import elmo_vectorize_instance
import torch
from torch import tensor

class TestElmo(unittest.TestCase):
    def test_elmo(self):
        instance = SenseInstance(["I", "swung", "the", "bat"], 3, 
                                 "bat_like_a_fruit_bat")
        result = elmo_vectorize_instance(instance)
        first_seven = tensor([ 0.0500, -0.2725,  0.3081, -0.7795, 
                              0.0522, -0.2052, -0.2959])
        last_seven = tensor([ 0.1043,  0.0660,  0.2928,  0.1568, 
                             -0.1257,  0.2225,  0.4422])
        n_digits = 4
        rounded = torch.round(result * 10**n_digits) / (10**n_digits)
        assert torch.all(torch.eq(rounded[:7], first_seven))
        assert torch.all(torch.eq(rounded[-7:], last_seven))
    

if __name__ == "__main__":
	unittest.main()
