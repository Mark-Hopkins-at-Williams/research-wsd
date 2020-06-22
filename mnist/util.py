import torch

def cudaify(x):
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 2:
            cuda = torch.device('cuda:2')
        else:
            cuda = torch.device('cuda:0')
        return x.cuda(cuda)
    else: 
        return x
