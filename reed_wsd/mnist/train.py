"""
Code adapted from:
https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627

"""

import torch
from reed_wsd.util import cudaify, predict_simple, predict_abs
from reed_wsd.train import Trainer, Decoder
from tqdm import tqdm

class MnistDecoder(Decoder):
    def __init__(self, predictor):
        self.predictor = predictor
    
    def __call__(self, net, data):
        net.eval()
        for images, labels in data:
            with torch.no_grad():
                outputs, conf = net(cudaify(images))
            preds = self.predictor(outputs)
            print(outputs[:10], preds[:10])
            for element in zip(preds, labels, conf, images):
                p, g, c, im = element
                yield {'evidence': im, 'pred': p.item(), 'gold': g.item(), 'confidence': c.item()}

class MnistSimpleDecoder(MnistDecoder):
    def __init__(self):
        super().__init__(predict_simple)

class MnistAbstainingDecoder(MnistDecoder):
    def __init__(self):
        super().__init__(predict_abs)
            

class PairwiseTrainer(Trainer):
         
    def _epoch_step(self, model):
        running_loss = 0.
        denom = 0
        for img_x, img_y, lbl_x, lbl_y in tqdm(self.train_loader, total=len(self.train_loader)):
            self.optimizer.zero_grad()                           
            output_x, conf_x = model(cudaify(img_x))
            output_y, conf_y = model(cudaify(img_y))
            loss = self.criterion(output_x, output_y, cudaify(lbl_x),
                                  cudaify(lbl_y), conf_x, conf_y)
            loss.backward()
            self.optimizer.step()                                                                                                 
            running_loss += loss.item()
            denom += 1
        return running_loss / denom
     
class SingleTrainer(Trainer):

    def _epoch_step(self, model):
        running_loss = 0.
        denom = 0
        for images, labels in tqdm(self.train_loader, total=len(self.train_loader)):
            self.optimizer.zero_grad()                       
            output, conf = model(cudaify(images))
            loss = self.criterion(output, conf, cudaify(labels))
            loss.backward()
            self.optimizer.step()                   
            running_loss += loss.item()
            denom += 1
        return running_loss / denom


