from torch import nn
from reed_wsd.util import cudaify

def inv_abstain_prob(output_tensor):
    return (1. - output_tensor[:,-1])
    
def max_nonabstain_prob(output_tensor):
    return output_tensor[:,:-1].max(dim=1).values
    
class ConfidentFFN(nn.Module): 
 
    def __init__(self, 
                 input_size = 784, 
                 hidden_sizes = [128, 64], 
                 output_size = 11,
                 confidence_extractor = inv_abstain_prob):
        super(ConfidentFFN, self).__init__()
        self.confidence_extractor = confidence_extractor
        self.linear1 = cudaify(nn.Linear(input_size, hidden_sizes[0]))
        self.linear2 = cudaify(nn.Linear(hidden_sizes[0], hidden_sizes[1]))
        self.linear3 = cudaify(nn.Linear(hidden_sizes[1], output_size))
        self.softmax = cudaify(nn.Softmax(dim=1))

    def forward(self, input_vec):
        nextout = cudaify(input_vec)
        nextout = self.linear1(nextout).clamp(min=0)
        nextout = self.linear2(nextout).clamp(min=0)
        nextout = self.linear3(nextout)
        nextout = self.softmax(nextout)
        return nextout, self.confidence_extractor(nextout)

class PairedConfidentFFN(nn.Module): 
 
    def __init__(self, 
                 input_size = 784, 
                 hidden_sizes = [128, 64], 
                 output_size = 10,
                 confidence_extractor = inv_abstain_prob):
        super(PairedConfidentFFN, self).__init__()
        self.confidence_extractor = confidence_extractor
        self.linear1 = cudaify(nn.Linear(input_size, hidden_sizes[0]))
        self.linear2 = cudaify(nn.Linear(hidden_sizes[0], hidden_sizes[1]))
        self.final1 = cudaify(nn.Linear(hidden_sizes[1], output_size))
        self.final2 = cudaify(nn.Linear(hidden_sizes[1], 1))
        self.softmax = cudaify(nn.Softmax(dim=1))

    def forward(self, input_vec):
        nextout = cudaify(input_vec)
        nextout = self.linear1(nextout).clamp(min=0)
        nextout = self.linear2(nextout).clamp(min=0)
        savepoint = nextout
        nextout = self.final1(nextout)
        nextout = self.softmax(nextout)
        confidence = self.final2(savepoint)
        return nextout, confidence
    
    
