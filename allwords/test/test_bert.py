import unittest
from allwords.wordsense import SenseInstance
from allwords.bert import BertVectorizer
from torch import tensor

class TestBert(unittest.TestCase):

    def compare_tensors(self, expected, result, num_digits):
        compare = zip(expected.tolist(), result.tolist())
        for i, (exp, actual) in enumerate(compare):
            if abs(exp - actual) > 10**(-num_digits):
                print("Element {} is incorrect: {} vs {}.".format(i, 
                                                                  expected[i], 
                                                                  result[i]))
                return False
        return True
        
    
    def test_bert(self):
        instance = SenseInstance(1, ["I", "swung", "the", "bat"], 3, "bat_like_a_fruit_bat", None)
        vectorize_instance = BertVectorizer()
        result = vectorize_instance(instance)
        expected = tensor(
            [-1.4628e-01,  2.4449e-01, -6.3506e-01, -3.9560e-01, -2.1649e-01,
            -5.0135e-01,  3.0898e-01,  5.0446e-01, -4.6515e-01, -3.0516e-01,
            -2.7876e-03, -2.9204e-02, -5.2181e-01,  3.6784e-01, -7.6221e-01,
             3.5340e-01,  6.3304e-01, -2.0991e-01,  1.1405e-01, -9.3629e-02,
             1.7420e-02, -2.8997e-01, -6.2381e-01,  6.8522e-01,  1.9712e-01,
             4.9819e-01,  3.3094e-01,  5.8903e-01, -1.8841e-02, -1.4805e+00,
             1.3536e-01,  3.9375e-01, -8.5668e-01, -7.6163e-01, -1.8073e-01,
             2.8041e-02, -2.3941e-01, -4.5448e-01, -6.8300e-01,  3.3108e-01,
             2.5510e-01,  1.7707e-01,  4.4185e-01, -3.5877e-01, -6.5479e-01,
            -2.0526e-01,  1.3045e-01, -4.5716e-01, -3.9572e-01, -1.4111e-01,
             2.2468e-01, -7.8193e-02, -2.9920e-01, -1.2476e-01, -4.7897e-01,
             1.1255e+00,  4.9035e-01, -6.1252e-01, -5.5520e-01,  4.0355e-01,
            -5.8457e-01, -1.7967e-01,  8.7223e-01, -5.5095e-01, -2.4339e-01,
            -2.8955e-01,  9.9863e-02,  1.1554e+00, -5.9919e-01,  3.3909e-01,
            -7.5828e-01, -5.1539e-01,  1.7033e-01, -1.2511e-01, -7.7684e-01,
            -4.1243e-01, -8.2184e-01,  6.0856e-01, -1.0223e-01,  2.1126e-01,
             3.5382e-01,  3.2824e-02, -9.1197e-01,  6.4068e-01,  5.6103e-01,
             3.3907e-01, -4.2050e-01, -7.2875e-01,  4.4904e-01,  7.5991e-01,
            -5.2431e-02, -3.8508e-01, -8.2581e-01,  7.2414e-01,  4.7733e-01,
            -5.0729e-01, -7.1612e-02, -7.1318e-01, -1.9664e-01,  1.3166e+00,
             7.2006e-01, -6.5808e-01,  4.3251e-01,  1.4288e-01,  2.9934e-01,
             1.2670e-01,  9.0248e-01,  2.8055e-01, -3.1241e-01,  3.9544e-04,
            -2.8367e-01, -4.4338e-02, -5.0243e-01, -6.1629e-01, -3.2395e-01,
            -5.0978e-01,  1.4667e+00, -8.2756e-01, -2.8187e-01,  3.0592e-01,
             7.1825e-01,  2.4872e-01,  2.1176e-01,  1.2193e+00, -8.3895e-02,
             4.2524e-01, -4.3993e-02,  3.0787e-01,  1.1929e-01, -1.3736e+00,
             4.1873e-01,  3.0964e-01,  1.6683e-01, -1.4263e+00,  3.2192e-01,
             1.0050e+00,  5.1096e-01, -4.7127e-02, -7.6868e-01,  6.2953e-01,
             5.9450e-01, -5.6120e-01,  6.6127e-02, -5.9528e-01,  8.1134e-01,
            -4.9986e-01, -3.4978e-02, -7.6352e-01,  1.2432e+00,  7.7016e-01,
             1.2787e+00, -1.1969e-01, -4.0689e-01, -1.7894e-01, -9.6513e-02,
             1.3725e-01, -3.9889e-01,  6.6768e-01,  1.8022e-01,  6.2237e-01,
             9.9088e-01, -8.9420e-01, -6.2595e-02,  3.0884e-01, -4.3949e-01,
            -4.1623e-01,  2.1973e-01,  1.2287e+00,  8.1508e-01, -2.0191e-01,
            -8.8593e-01, -1.8698e-01,  6.4800e-01, -3.5634e-01, -4.2804e-01,
            -1.0502e+00,  1.9172e-01, -6.9328e-02, -8.2889e-02, -7.5523e-02,
            -2.7242e-01, -2.7999e-02,  3.2653e-01,  5.8591e-01,  1.1815e-01,
            -1.8040e-01,  1.5010e+00, -4.9465e-01, -9.5648e-02,  5.0258e-01,
             7.7373e-01, -5.5028e-01, -4.5596e-01, -3.6429e-01,  2.1118e-01,
            -1.1527e+00, -4.6983e-01, -1.7028e+00,  3.1117e-02,  6.1104e-01,
             3.5977e-01,  4.0972e-01,  1.0623e+00, -2.5080e-02, -1.0282e+00,
            -2.6022e-01, -4.3293e-02,  3.9285e-01, -2.3603e-01,  1.0316e+00,
            -1.3446e-01,  1.2412e+00,  3.2096e-01, -4.9268e-01, -2.4120e-01,
             1.1690e-01, -5.7767e-01,  6.8544e-01, -1.6307e-01,  3.9976e-01,
             6.7640e-01,  5.8086e-01, -1.0646e+00,  2.6473e-01, -2.7217e-02,
             1.9808e+00, -2.5306e-01, -3.2833e-01, -8.1260e-01,  1.2640e+00,
            -4.8711e-01,  8.3085e-02,  7.9259e-01,  7.8186e-01, -7.4572e-02,
            -8.3598e-01, -1.2313e+00,  3.3803e-01, -4.7642e-01, -1.2260e-01,
            -1.3582e-01, -8.0896e-01,  6.4799e-02,  8.6550e-01, -4.1679e-02,
             3.3704e-01, -4.8273e-01,  1.6045e-01,  1.8452e-01, -5.2323e-01,
            -1.6845e-01, -1.1827e+00, -2.1527e-01, -9.8969e-01, -1.0710e+00,
            -1.7014e-01, -2.5989e-01,  7.0623e-01, -1.0838e-01,  7.1672e-01,
             4.3408e-01,  5.6733e-02,  1.1213e+00,  3.3819e-01, -5.2155e-01,
             2.9116e-01, -4.8303e-01,  7.1582e-02, -3.4038e-01,  2.5937e-01,
            -5.7021e-01, -9.0631e-01,  9.5214e-01,  6.1948e-01, -9.7434e-01,
            -2.5197e-01,  5.5948e-01,  2.0000e-01, -6.0864e-01, -9.4795e-01,
             6.6239e-02,  3.3827e-01, -1.0664e+00,  2.3797e-02, -3.7972e-01,
            -9.8886e-01,  2.7574e-01,  1.1230e-01, -3.6253e-01, -4.9694e-01,
             1.5233e-01,  5.6558e-01,  1.2435e-01, -2.8704e-01, -4.3305e-01,
            -8.2167e-01,  5.8055e-01,  5.4579e-01,  1.8591e-01,  8.0643e-02,
            -7.8623e-02, -5.5233e-01,  1.6703e-01,  2.7690e-02, -3.5083e-01,
             4.6103e-01, -8.3646e-01, -1.1215e-01, -3.3713e+00, -1.6447e-02,
            -5.7544e-01, -5.0604e-01,  5.0681e-01, -2.5461e-01, -4.8956e-01,
            -4.3603e-01, -4.4683e-01, -4.0966e-01, -1.9473e-01, -6.0733e-01,
            -9.8694e-03, -4.5296e-01, -3.5992e-01, -1.7254e-01, -1.9638e-01,
            -1.8347e-01,  1.7175e-01,  9.3256e-01,  4.2457e-01, -1.0718e-01,
            -6.4281e-02, -7.7383e-01,  3.1053e-01,  7.5111e-01,  2.9948e-01,
             1.6419e-01, -1.0887e+00, -2.8527e-02, -4.8389e-01,  2.9815e-01,
             1.9345e-01,  5.9602e-02,  2.8140e-01, -3.5876e-01,  6.7256e-01,
             4.9989e-02, -8.7744e-01, -9.2570e-01, -6.3284e-01, -4.0774e-01,
             4.4540e-01,  4.5781e-01,  1.4294e+00,  1.9854e-01,  2.4848e-01,
            -6.2588e-01, -6.0213e-01, -9.7980e-02,  3.9767e-02, -8.3530e-01,
            -4.3531e-01, -6.2393e-01,  6.0453e-02, -4.9405e-01,  9.5380e-01,
            -1.8543e-01,  2.6691e-02, -6.1839e-01,  2.4852e-01,  2.0723e-01,
            -5.8657e-01,  6.5213e-01,  3.7905e-01, -4.6848e-01,  2.6155e-02,
             1.1720e-01,  5.3758e-01,  9.1004e-01,  7.2143e-02, -6.5895e-01,
            -7.6769e-01, -1.4047e+00,  1.8063e-03, -1.7636e-01,  7.6328e-01,
            -4.5283e-01,  5.9941e-01, -8.4718e-02, -9.2966e-01, -5.1393e-01,
             4.1201e-02, -2.6627e-01,  3.6057e-01, -4.4083e-01,  4.0422e-01,
            -7.5810e-01,  3.2034e-01, -6.2238e-01, -1.7928e-01,  1.8183e-02,
             2.1498e-01,  8.6890e-01,  1.3720e-01,  3.6977e-01,  2.1444e-01,
            -4.8117e-01, -3.9211e-01,  4.7685e-01,  3.6097e-01,  2.3126e-02,
             5.7800e-02,  3.2309e-01,  4.4281e-02,  3.5813e-01, -2.2378e-01,
             6.4240e-01, -6.5882e-01, -4.8106e-01,  2.0839e-01, -1.0120e+00,
             7.8867e-01, -3.8934e-01, -8.3776e-01,  2.3194e-01,  1.3135e-02,
             4.4988e-01, -6.3563e-01,  3.5566e-01, -2.7372e-01,  2.5590e-01,
            -1.1083e+00, -4.8238e-01, -7.2263e-01, -2.6148e-01,  2.5796e-01,
            -7.7830e-01, -9.2944e-01, -2.3408e-01, -3.2489e-01, -3.0506e-01,
            -1.2610e-01,  9.1844e-01, -2.5831e-01,  4.1025e-01, -8.9134e-01,
            -4.0025e-01,  4.7123e-01,  3.4378e-01,  1.5121e-01,  7.5966e-01,
             8.6487e-01, -3.9274e-01, -3.3508e-01,  2.6371e-01, -2.7459e-01,
            -5.8591e-01, -1.0750e+00, -8.2166e-02, -5.8674e-02, -8.4190e-01,
             4.0197e-01, -4.8660e-02,  8.3937e-01,  1.0127e+00,  4.1971e-04,
             1.4465e-01, -1.1872e-01,  9.8069e-02,  4.7799e-01, -3.0566e-01,
             5.9884e-01,  5.8850e-01,  1.4710e-01,  4.0575e-01,  2.5158e-02,
            -2.5092e-02, -1.9944e-01, -1.9107e-01, -1.2030e-01, -9.5086e-01,
            -5.6823e-01, -3.3945e-01, -7.4155e-01,  7.9247e-01, -7.6073e-02,
            -4.8107e-01, -1.1135e+00, -5.4349e-01, -1.9628e-01,  1.8838e-01,
            -3.1978e-01, -5.7792e-01, -4.2503e-02, -5.2900e-01, -5.2176e-01,
            -1.1026e+00,  6.3782e-01,  3.7520e-01,  3.7599e-01,  3.1363e-01,
            -3.7918e-01, -1.3895e-01, -5.0375e-01, -1.0912e+00,  1.4397e-01,
             1.0140e+00, -1.7202e-01,  4.1315e-01,  8.6369e-01,  5.6016e-03,
             2.5844e-01, -5.6174e-01,  9.2305e-02, -1.8988e-01, -1.4879e-01,
            -8.7389e-01,  4.1599e-01,  7.1272e-01,  3.3806e-01, -1.9066e-01,
             2.1914e-01, -2.4186e-01,  6.7403e-01, -2.3120e-01, -5.1944e-01,
            -2.3993e-01, -2.6973e-01, -3.1342e-01,  5.5954e-01, -1.1575e-01,
            -2.4591e-02,  2.0432e-01, -1.6948e-01,  3.9610e-01, -3.1497e-01,
             1.2302e-01,  1.9395e-01, -3.6386e-01, -5.4253e-01, -8.8346e-01,
             3.0454e-01,  2.1850e-01,  7.3475e-02,  6.5967e-01, -4.0965e-02,
            -6.7097e-01, -2.3839e-01,  5.0320e-02, -5.7194e-02, -5.3244e-02,
            -3.8049e-02,  5.5760e-01, -6.9388e-02, -4.4298e-01,  5.9799e-03,
             9.9577e-01,  1.6562e-01,  2.6958e-01, -1.6535e-01,  3.1080e-01,
             9.7480e-02, -8.9362e-02,  1.5599e-01, -6.1339e-02, -2.3846e-01,
            -3.6978e-01,  7.5240e-01,  7.0144e-01, -5.4249e-03, -4.6894e-01,
            -1.3532e-01,  7.4245e-01, -7.9933e-03,  2.8255e-01,  5.2168e-01,
            -1.5013e-01,  3.3473e-01,  5.9204e-01,  1.3331e+00, -3.4627e-02,
            -7.5285e-01, -3.7455e-01, -5.7075e-01,  4.8481e-01, -2.4372e-01,
            -3.6413e-01,  5.2373e-01, -4.8319e-01, -4.4021e-01,  4.6229e-01,
             6.4885e-01,  2.4996e-01, -5.3338e-01,  7.0921e-01,  7.6941e-02,
             6.8833e-02, -2.9453e-01, -4.9120e-01,  1.9045e-01,  2.4790e-01,
            -1.3486e-01,  5.6321e-01, -2.0863e-01,  1.1069e+00,  8.1962e-02,
             1.3604e-01,  7.8395e-01,  1.1318e-01,  3.5500e-01, -2.1916e-01,
             5.4927e-01,  8.9384e-01, -3.8056e-01,  3.4227e-03,  1.7730e-01,
            -1.7196e-01,  4.2377e-01,  8.6134e-01, -3.4676e-01, -9.6534e-01,
            -3.2864e-02, -8.5711e-01, -4.1247e-01, -3.4631e-02, -3.0418e-01,
            -6.8806e-02,  1.1412e-01, -1.6320e-01, -5.5556e-01,  6.9288e-01,
             4.6784e-02, -2.2103e-01, -4.3264e-01,  4.5048e-01, -1.1525e+00,
            -4.6868e-01, -1.8338e-01,  1.5103e-01, -7.0392e-02, -6.9025e-01,
            -2.7734e-01,  5.8697e-01, -5.0644e-01,  5.3948e-01, -3.8664e-01,
             1.8947e-01,  1.3962e+00,  5.4602e-01,  4.4664e-01, -2.3063e-01,
            -1.7183e-01,  9.2532e-01,  3.5361e-01,  4.6997e-02,  7.3182e-01,
            -7.9908e-02,  1.1664e+00,  6.4117e-01, -5.9138e-02,  3.3898e-01,
             1.9703e-01, -9.8374e-01,  1.2435e+00,  1.6931e-01,  2.4686e-01,
             2.7717e-01,  3.1914e-01,  1.7802e-01, -1.3094e-01,  9.5111e-02,
             1.7774e-01, -1.2365e+00, -5.1270e-01,  4.4215e-01,  6.2962e-01,
             5.6389e-02, -3.2111e-01,  1.3623e-01, -7.6298e-02,  4.2042e-01,
             6.4718e-01, -3.3982e-01, -4.2825e-01,  6.8884e-01,  4.5062e-01,
            -2.2377e-01,  1.1648e+00, -3.9262e-02, -2.0337e-01,  7.0438e-01,
            -1.1873e+00,  3.8163e-01,  5.5157e-01, -3.3096e-01,  4.0442e-01,
            -2.9381e-02,  7.1922e-01, -3.6702e-01,  1.3203e+00, -3.1773e-01,
             9.2199e-01, -9.9956e-03, -7.1623e-01, -1.0239e+00,  4.6419e-02,
            -5.4726e-01,  1.3834e-01,  5.1538e-01, -1.9733e-02,  1.2030e+00,
            -2.7285e-01,  9.6586e-01,  1.0430e-01,  1.8040e-01, -3.9401e-01,
            -5.6534e-01,  1.2139e+00, -2.2596e-01,  1.3443e-01,  6.6717e-01,
            -4.8109e-01, -2.5209e-02, -6.7164e-01, -4.0653e-01,  4.4409e-01,
             2.6767e-01, -8.0575e-01,  1.4153e+00, -3.6506e-01,  2.5827e-01,
            -6.5003e-01,  6.7648e-01, -1.2115e+00, -2.5646e-01,  1.6612e-01,
             1.0894e+00,  4.1246e-01, -2.9842e-01, -5.5017e-01,  7.0980e-01,
             1.1451e+00, -7.4669e-01,  5.6189e-01, -2.1616e-01, -6.5712e-01,
             3.2280e-01, -4.1128e-01,  4.7093e-01,  4.0406e-01,  3.3855e-01,
             1.8100e-01, -1.8732e-01, -6.0756e-02, -5.0424e-01, -2.5015e-01,
            -1.0339e+00,  8.2525e-01,  3.1960e-01,  1.1254e-03,  8.4850e-01,
            -1.1779e-02, -4.5650e-01, -8.6922e-01, -1.4814e-01,  3.7370e-01,
            -3.1986e-01,  5.5159e-01, -3.2156e-01])
        assert(self.compare_tensors(expected, result, num_digits = 3))
    

if __name__ == "__main__":
	unittest.main()
