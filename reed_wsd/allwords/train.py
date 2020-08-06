import torch
from reed_wsd.train import Trainer
from tqdm import tqdm
from reed_wsd.util import cudaify

class SingleEmbeddingTrainer(Trainer):
    def _epoch_step(self, model):
        running_loss = 0.0
        denom = 0
        for (_, _, evidence, response, zones) in tqdm(self.train_loader, total=len(self.train_loader)):
            self.optimizer.zero_grad()
            outputs, conf = model(cudaify(evidence), zones)
            loss_size = self.criterion(outputs, conf, cudaify(response))
            loss_size.backward()
            self.optimizer.step()
            running_loss += loss_size.data.item()
            denom += 1
        return running_loss / denom

class PairwiseEmbeddingTrainer(Trainer):
    def _epoch_step(self, model):
        running_loss = 0.0
        denom = 0
        for (pkg1, pkg2) in tqdm(self.train_loader, total=len(self.train_loader)):
            (_, _, evidence1, response1, zones1) = pkg1
            (_, _, evidence2, response2, zones2) = pkg2
            zones1 = cudaify(torch.tensor(zones1))
            zones2 = cudaify(torch.tensor(zones2))
            self.optimizer.zero_grad()
            outputs1, conf1 = model(cudaify(evidence1), zones1)
            outputs2, conf2 = model(cudaify(evidence2), zones2)
            loss_size = self.criterion(outputs1, outputs2, cudaify(response1), cudaify(response2), conf1, conf2)
            loss_size.backward()
            self.optimizer.step()
            running_loss += loss_size.data.item()
            denom += 1
        return running_loss / denom

class BEMTrainer(Trainer):
    def _epoch_step(self, model):
        running_loss = 0.0
        denom = 0
        for batch in tqdm(self.train_loader, total=len(self.train_loader)):
            contexts = batch['contexts']
            glosses = batch['glosses']
            span = batch['span']
            gold = batch['gold']
            scores = model(contexts, glosses, span)
            loss_size = self.criterion(scores, cudaify(torch.tensor(gold)))
            loss_size.backward()
            optimizer.step()
            running_loss += loss_size.data.item()
            denom += 1
        return running_loss / denom
