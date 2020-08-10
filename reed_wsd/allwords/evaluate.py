import torch
import json
import os
from os.path import join
import torch.nn.functional as F
import matplotlib.pyplot as plt
from reed_wsd.util import cudaify


LARGE_NEGATIVE = 0
file_dir = os.path.dirname(os.path.realpath(__file__))

def predict(distribution):
    return distribution.argmax()

def accuracy(predicted_labels, gold_labels):
    assert(len(predicted_labels) == len(gold_labels))
    preds = torch.tensor(predicted_labels).double()
    gold = torch.tensor(gold_labels).double()
    n_correct = (preds == gold).double().sum().item()
    return n_correct, len(predicted_labels)

def yielde(predicted_labels, gold_labels):
    assert(len(predicted_labels) == len(gold_labels))
    preds = torch.tensor(predicted_labels).double()
    gold = torch.tensor(gold_labels).double()
    n_correct = (preds == gold).double().sum().item()
    return n_correct, len(predicted_labels)
    
def apply_zone_masks(outputs, zones):
    revised = torch.empty(outputs.shape, device=outputs.device)
    revised = revised.fill_(LARGE_NEGATIVE)
    for row in range(len(zones)):
        (start, stop) = zones[row]
        revised[row][start:stop] = outputs[row][start:stop]
    revised = F.normalize(revised, dim=-1, p=1)
    return revised

def predict_simple(output):
    return output.argmax(dim=1)

def predict_abs(output):
    return output[:, :-1].argmax(dim=1)

class AllwordsBEMDecoder:
    def __call__(self, net, data):
        net.eval()
        val_loader = data.batch_iter()
        with torch.no_grad():
            for batch in val_loader:
                contexts = batch['contexts']
                glosses = batch['glosses']
                span = batch['span']
                gold = batch['gold']
                scores = net(contexts, glosses, span)
                max_scores, preds = scores.max(dim=-1)
                for element in zip(max_scores,
                                   zip(preds,
                                       gold)):
                    (max_score, (pred, g)) = element
                    yield({'pred': pred.item(), 'gold': g, 'confidence': max_score.item()})

class AllwordsEmbeddingDecoder:
    def __init__(self, predictor):
        self.predictor = predictor

    def __call__(self, net, data):
        """
        Runs a trained neural network classifier on validation data, and iterates
        through the top prediction for each datum.
        
        TODO: write some unit tests for this function
        
        """
        net.eval()
        net = cudaify(net)
        with torch.no_grad():
            for inst_ids, targets, evidence, response, zones in data:
                output, conf = net(cudaify(evidence), zones)
                preds = self.predictor(output)
                for element in zip(preds, response, conf):
                    (pred, gold, c) = element
                    pkg = {'pred': pred, 'gold': gold.item(), 'confidence': c.item()}
                    yield pkg

class AllwordsSimpleEmbeddingDecoder(AllwordsEmbeddingDecoder):
    def __init__(self):
        super().__init__(predictor=predict_simple)

class AllwordsAbstainingEmbeddingDecoder(AllwordsEmbeddingDecoder):
    def __init__(self):
        super().__init__(predictor=predict_abs)

def evaluate(net, data, decoder):
    """
    The accuracy (i.e. percentage of correct classifications) is returned.
    net: trained network used to decode the data
    abstain: Boolean, whether the outout has an abstention class
    
    """
    decoded = list(decoder(net, data))
    predictions = [inst['pred'] for inst in decoded]
    gold = [inst['gold'] for inst in decoded]
    correct, total = accuracy(predictions, gold)
    acc = correct / total if total > 0 else 0
    print('correct, total: ', correct, total)
    return acc

