import torch
from reed_wsd.train import Trainer

class SingleEmbeddingTrainer(Trainer):
    def _epoch_step(self, model):
        running_loss = 0.0
	denom = 0
	batch_iter = self.train_loader.batch_iter()
	batch_iter = tqdm(batch_iter, total=len(self.train_loader))
	for i, (_, _, evidence, response, zones) in enumerate(batch_iter): 
	    self.optimizer.zero_grad()
	    outputs, conf = model(cudaify(evidence))
	    loss_size = self.critetion(outputs, response, confidence, zones)
	    loss_size.backward()
	    self.optimizer.step()
	    running_loss += loss_size.data.item()
            denom += 1
	return loss_size / denom

class TwinEmbeddingTrainer(Trainer):
    def _epoch_step(self, model):
	running_loss = 0.0
	denom = 0
	batch_iter = self.train_loader.batch_iter()
    	batch_iter = tqdm(batch_iter1, total=len(self.train_loader))
	for (pkg1, pkg2) in zip(batch_iter1, batch_iter2):
	    (_, _, evidence1, response1, zones1) = pkg1
	    (_, _, evidence2, response2, zones2) = pkg2
	    zones1 = cudaify(torch.tensor(zones1))
	    zones2 = cudaify(torch.tensor(zones2))
	    self.optimizer.zero_grad()
	    outputs1, conf1 = model(cudaify(evidence1))
	    outputs2, conf2 = model(cudaify(evidence2))
	    loss_size = loss(outputs1, outputs2, cudaify(response1), cudaify(response2), zones1, zones2)
	    loss_size.backward()
	    self.optimizer.step()
	    running_loss += loss_size.data.item()
	return running_loss / denom

class BEMTrainer(Trainer):
def _epoch_step(self, model):
        running_loss = 0.0
        denom = 0
        batch_iter = train_loader.batch_iter()
        batch_iter = tqdm(batch_iter, total=len(train_loader))
        for batch in batch_iter:
            contexts = batch['contexts']
            glosses = batch['glosses']
            span = batch['span']
            gold = batch['gold'] 
            scores = model(contexts, glosses, span)
            loss_size = loss(scores, cudaify(torch.tensor(gold)))
            loss_size.backward()
            optimizer.step()
            running_loss += loss_size.data.item()
        return running_loss
