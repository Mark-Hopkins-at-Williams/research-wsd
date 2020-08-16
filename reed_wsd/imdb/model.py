import torch
from torch import nn
import torch.nn.functional as F
from reed_wsd.mnist.model import confidence_extractor_lookup

class SingleLayerFFN(nn.Module):
    def __init__(self,
                 input_size=768,
                 output_size=2,
                 confidence_extractor='max_prob'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size)
        self.confidence_extractor = confidence_extractor_lookup[confidence_extractor]
        torch.nn.init.xavier_uniform_(self.linear.weight)
        print('confidence:', self.confidence_extractor.__name__)

    def __call__(self, input_vec):
        nextout = self.linear(input_vec)
        confidence = self.confidence_extractor(nextout)
        nextout = F.softmax(nextout.clamp(min=-25, max=25), dim=1)
        return nextout, confidence

class AbstainingSingleLayerFFN(SingleLayerFFN):
    def __init__(self,
                 input_size=768,
                 output_size=2,
                 confidence_extractor='max_non_prob'):
        super().__init__(input_size, output_size+1, confidence_extractor)

class SingleLayerConfidentFFN(SingleLayerFFN):
    def __init__(self,
                 input_size=768,
                 output_size=2,
                 confidence_extractor='max_non_prob'):
        super().__init__(input_size, output_size+1, confidence_extractor)
        self.confidence_layer = nn.Linear(input_size, 1)
    
    def __call__(self, input_vec):
        nextout = self.linear(input_vec)
        nextout = F.softmax(nextout.clamp(min=-25, max=25), dim=1)
        confidence = self.confidence_layer(input_vec)
        return nextout, confidence
        
