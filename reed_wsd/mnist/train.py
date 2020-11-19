"""
Code adapted from:
https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627

"""

import torch
import torch.nn.functional as F
from reed_wsd.util import cudaify, ABS
from reed_wsd.train import Trainer, Decoder
from tqdm import tqdm

class MnistSimpleDecoder(Decoder):
    def __call__(self, net, data, trust_model):
        net.eval()
        for images, labels in tqdm(data, total=len(data)):
            with torch.no_grad():
                outputs, conf = net(cudaify(images))
            preds = outputs.argmax(dim=1)
            if trust_model is not None:
                trust_score = trust_model.get_score(images.cpu().numpy(), 
                                                    preds.cpu().numpy())
            else:
                trust_score = [None] * labels.shape[0]
            for element in zip(preds, labels, conf, trust_score):
                p, g, c, t = element
                if t is not None:
                    yield {'pred': p.item(), 'gold': g.item(), 'confidence': c.item(), 'trustscore': t}
                else:
                    yield {'pred': p.item(), 'gold': g.item(), 'confidence': c.item()}
                    

class MnistAbstainingDecoder(Decoder):
    
    def __call__(self, net, data, trust_model=None):
	# note that trust_model here is a dummy argument
	# the function is not dependent on the variable
        net.eval()
        for images, labels in tqdm(data, total=len(data)):
            output, conf = net(cudaify(images))
            output = F.softmax(output.clamp(min=-25, max=25), dim=1)
            abs_i = output.shape[1] - 1
            preds = output.argmax(dim=-1)
            preds[preds == abs_i] = ABS
            for e in zip(preds, labels, conf):
                pred, gold, c = e
                result = {'pred': pred.item(), 'gold': gold.item(), 'confidence': c.item()} 
                yield result

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            self.optimizer.step()                   
            running_loss += loss.item()
            denom += 1
        return running_loss / denom


