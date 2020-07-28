import torch

class MnistLoader:
    
    def __init__(self, dataset, batch_size=64, shuffle = True, confuser = lambda x: x):
        self.loader = torch.utils.data.DataLoader(dataset, 
                                                  batch_size=batch_size, 
                                                  shuffle=shuffle)
        self.confuser = confuser
        
    def __iter__(self):
        for images, labels in self.loader:
            images = images.view(images.shape[0], -1)
            labels = self.confuser(labels)
            yield images, labels

    def __len__(self):
        return len(self.loader)


def confuse(labels):
    labels = labels.clone()
    one_and_sevens = (labels == 1) + (labels == 7)
    one_seven_shape = labels[one_and_sevens].shape
    new_labels = torch.randint(0, 2, one_seven_shape) #change the second argument for different weights
    new_labels[new_labels == 0] = 7    
    labels[one_and_sevens] = new_labels
    #one_and_sevens = (labels == 2) + (labels == 3)
    #one_seven_shape = labels[one_and_sevens].shape
    #new_labels = torch.randint(0, 2, one_seven_shape)
    #new_labels[new_labels > 0] = 3
    #new_labels[new_labels == 0] = 2
    #labels[one_and_sevens] = new_labels
    #one_and_sevens = (labels == 4) + (labels == 5)
    #one_seven_shape = labels[one_and_sevens].shape
    #new_labels = torch.randint(0, 4, one_seven_shape)
    #new_labels[new_labels > 0] = 4
    #new_labels[new_labels == 0] = 5
    #labels[one_and_sevens] = new_labels
    return labels        
    
class ConfusedMnistLoader(MnistLoader):
    
    def __init__(self, dataset, batch_size=64, shuffle=True):
        super().__init__(dataset, batch_size, shuffle, confuse)
        
        
    
class PairLoader:
    def __init__(self, dataset, bsz=64, shuffle=True):
        self.bsz = bsz
        self.dataset = dataset
        self.single_img_loader1 = torch.utils.data.DataLoader(dataset, batch_size=bsz, shuffle=True)
        self.single_img_loader2 = torch.utils.data.DataLoader(dataset, batch_size=bsz, shuffle=True)

    def __iter__(self):
        return self.batch_iter()

    def batch_iter(self):
        for ((imgs1, lbls1), (imgs2, lbls2)) in zip(self.single_img_loader1, self.single_img_loader2):
            imgs1 = imgs1.view(imgs1.shape[0], -1)
            imgs2 = imgs2.view(imgs2.shape[0], -1)
            lbls1 = confuse(lbls1)
            lbls2 = confuse(lbls2)            
            yield imgs1, imgs2, lbls1, lbls2
            

