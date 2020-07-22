import torch
import random

class PairLoader:
    def __init__(self, dataset, bsz=64, shuffle=True):
        self.bsz = bsz
        self.dataset = dataset
        self.single_img_loader1 = torch.utils.data.DataLoader(dataset, batch_size=bsz, shuffle=True)
        self.single_img_loader2 = torch.utils.data.DataLoader(dataset, batch_size=bsz, shuffle=True)

    def batch_iter(self):
        for ((imgs1, lbls1), (imgs2, lbls2)) in zip(self.single_img_loader1, self.single_img_loader2):
            imgs1 = imgs1.view(imgs1.shape[0], -1)
            imgs2 = imgs2.view(imgs2.shape[0], -1)
            
            yield imgs1, imgs2, lbls1, lbls2
            

