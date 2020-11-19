import torch
import os
import torch.nn.functional as F
from reed_wsd.util import cudaify, predict_abs, predict_simple, ABS


LARGE_NEGATIVE = 0
file_dir = os.path.dirname(os.path.realpath(__file__))

def apply_zone_masks(outputs, zones):
    revised = torch.empty(outputs.shape, device=outputs.device)
    revised = revised.fill_(LARGE_NEGATIVE)
    for row in range(len(zones)):
        (start, stop) = zones[row]
        revised[row][start:stop] = outputs[row][start:stop]
    revised = F.normalize(revised, dim=-1, p=1)
    return revised

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

class AllwordsAbstainingEmbeddingDecoder:
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
                abs_i = len(output.shape[1]) - 1 # last class is abstention class
                preds = output.argmax(dim=-1)
                preds[preds == abs_i] = -1
                for element in zip(preds, response, conf):
                    (pred, gold, c) = element
                    pkg = {'pred': pred, 'gold': gold.item(), 'confidence': c.item()}
                    yield pkg

class AllwordsSimpleEmbeddingDecoder:
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
                preds = output.argmax(dim=-1)
                for element in zip(preds, response, conf):
                    (pred, gold, c) = element
                    pkg = {'pred': pred, 'gold': gold.item(), 'confidence': c.item()}
                    yield pkg

