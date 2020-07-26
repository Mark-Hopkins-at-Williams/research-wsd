import unittest
import torch
from reed_wsd.mnist.networks import ConfidentFFN, inv_abstain_prob, max_nonabstain_prob


class TestMnistNetworks(unittest.TestCase):
    
    def build_simple_ffn(self):
        net = ConfidentFFN(input_size = 2, hidden_sizes = [2,2], output_size=3)
        for param in net.parameters():
            if param.shape == torch.Size([3]):
                param[0] = 1.4640
                param[1] = -0.3238
                param[2] = 0.7740
            elif param.shape == torch.Size([2, 2]):
                param[0][0] = 0.1940
                param[0][1] = 2.1614
                param[1][0] = -0.1721
                param[1][1] = -0.1721       
            elif param.shape == torch.Size([2]):
                param[0] = 0.1391
                param[1] = -0.1082
            elif param.shape == torch.Size([3, 2]):
                param[0][0] = -1.2682
                param[0][1] = -0.0383
                param[1][0] = -0.1029 
                param[1][1] = 1.4400
                param[2][0] = -0.4705
                param[2][1] = 1.1624
            else:
                torch.nn.init.ones_(param)        
        return net

    def test_ffn(self):
        net = self.build_simple_ffn()
        result, _ = net(torch.tensor([[-2., 1.], [5., 2.]]))
        result = result.to('cpu')
        expected = torch.tensor([[0.4551, 0.1621, 0.3828],
                                 [0.3366, 0.2261, 0.4372]]).to('cpu')
        assert result.shape == expected.shape
        assert torch.allclose(result, expected, atol=10**(-4))
    
    def test_ffn_with_inv_abstain_prob(self):
        net = self.build_simple_ffn()
        net.confidence_extractor = inv_abstain_prob
        result, conf = net(torch.tensor([[-2., 1.], [5., 2.]]))
        result = result.to('cpu')
        expected = torch.tensor([[0.4551, 0.1621, 0.3828],
                                 [0.3366, 0.2261, 0.4372]]).to('cpu')
        assert result.shape == expected.shape
        assert torch.allclose(result, expected, atol=10**(-4))
        conf = conf.to('cpu')
        expected_conf = torch.tensor([0.6172, 0.5628]).to('cpu')
        assert conf.shape == expected_conf.shape
        assert torch.allclose(conf, expected_conf, atol=10**(-4))
        
    def test_ffn_with_max_nonabstain_prob(self):
        net = self.build_simple_ffn()
        net.confidence_extractor = max_nonabstain_prob
        result, conf = net(torch.tensor([[-2., 1.], [5., 2.]]))
        result = result.to('cpu')
        expected = torch.tensor([[0.4551, 0.1621, 0.3828],
                                 [0.3366, 0.2261, 0.4372]]).to('cpu')
        assert result.shape == expected.shape
        assert torch.allclose(result, expected, atol=10**(-4))
        conf = conf.to('cpu')
        expected_conf = torch.tensor([0.4551, 0.3366]).to('cpu')
        assert conf.shape == expected_conf.shape
        assert torch.allclose(conf, expected_conf, atol=10**(-4))

        

    
if __name__ == "__main__":
	unittest.main()
