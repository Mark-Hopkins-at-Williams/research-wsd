import torch
from torch import nn
import torch.nn.functional as F
from reed_wsd.util import cudaify
from transformers import BertModel
from torch.nn.utils.rnn import pad_sequence

def zero_out_probs(input_vec, zones):
    new_vec = torch.Tensor.new_fill(input_vec.shape, torch.tensor(float('-inf')), device=input_vec.device)
    for row in range(len(zones)):
        start, stop = zones[row]
        new_vec[row, start: stop] = input_vec[row, start: stop]
    return new_vec

def max_prob(input_vec, zones):
    zoned_output = zero_out_probs(input_vec, zones)
    probs = F.softmax(zoned_output.clamp(min=-25, max=25), dim=1)
    confidence = probs.max(dim=1).values
    return zone_output, confidence

def max_non_abs(input_vec, zones):
    zoned_output = zero_out_probs(input_vec, zones)
    zoned_output[:, -1] = input_vec[:, -1]
    probs = F.softmax(zoned_output.clamp(min=-25, max=25), dim=1)
    confidence = probs[:, :-1].max(dim=1).values
    return zoned_output, confidence

def inv_abs(input_vec, zones):
    zoned_output = zero_out_probs(input_vec, zones)
    zoned_output[:, -1] = input_vec[:, -1]
    probs = F.softmax(zoned_output.clamp(min=-25, max=25), dim=1)
    confidence = 1. - probs[:, -1]
    return zoned_output, confidence

def abstention(input_vec, zones):
    abs_features = input_vec[:, -1]
    zoned_output = zero_out_probs(input_vec, zones)
    zoned_output[:, -1] = abs_features
    confidence = abs_features
    return zoned_output, confidence

apply_zones_lookup = {'max_prob': max_prob,
                      'max_non_abs': max_non_abs,
                      'inv_abs': inv_abs,
                      'abs': abstention}

class SingleLayerFFNWithZones(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 zone_applicant='max_prob'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = cudaify(nn.Linear(input_size, output_size))
        self.zone_applicant = apply_zones_lookup[zone_applicant]
        torch.nn.init.xavier_uniform_(self.linear.weight)
        print('confidence:', self.zone_applicant.__name__)

    def forward(self, input_vec, zones):
        nextout = self.linear(input_vec)
        return self.zone_applicant(nextout, zones)

class AbstainingSingleLayerFFNWithZones(SingleLayerFFNWithZones):
    def __init__(self, input_size,
                 output_size,
                 zone_applicant='max_non_abs'):
        super().__init__(input_size, output_size, zone_applicant)
        self.linear = cudaify(nn.Linear(input_size, output_size + 1))

class DropoutClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, num_labels):
        super(DropoutClassifier, self).__init__()
        self.dropout1 = nn.Dropout(p=0.2)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(hidden_size, num_labels)

    def forward(self, input_vec):
        nextout = input_vec
        nextout = self.linear1(nextout).clamp(min=0)
        nextout = self.dropout1(nextout)
        nextout = self.linear2(nextout).clamp(min=0)
        nextout = self.dropout2(nextout)
        nextout = self.linear3(nextout)
        return F.log_softmax(nextout, dim=1)

class BEMforWSD(nn.Module):
    """
    This is the Bi-encoder Model proposed by Blevins et al.
    in this paper: https://github.com/facebookresearch/wsd-biencoders
    """
    def __init__(self, gpu=False):
        super(BEMforWSD, self).__init__()
        self.gpu = gpu
        self.context_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.gloss_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.output_size = None

    def forward(self, contexts, glosses, pos):
        scores = []
        context_inputs = contexts['input_ids']
        if self.gpu:
            context_inputs = cudaify(context_inputs)
        context_masks = contexts['attention_mask']
        if self.gpu:
            context_masks = cudaify(context_masks)
        context_rep = self.context_encoder(input_ids=context_inputs,
                                           attention_mask=context_masks)[0] # last hidden state
        target_rep = self.target_representation(context_rep, pos)
        for i, g in enumerate(glosses):
            input_ids = g['input_ids']
            if self.gpu:
                input_ids = cudaify(input_ids)
            attention_mask = g['attention_mask']
            if self.gpu:
                attention_mask = cudaify(attention_mask)
            last_layer = self.gloss_encoder(input_ids=input_ids,
                                            attention_mask=attention_mask)[0]
            if 'span' not in g:
                gloss_reps = last_layer[:, 0, :] # the vector that corresponds to CLS
            else:
                span = g['span']
                gloss_reps = self.target_representation(last_layer, span)
            score = target_rep[i] * gloss_reps
            score = score.sum(dim=1)
            scores.append(score)
        result = pad_sequence(scores, batch_first=True)
        return result

    @staticmethod
    def target_representation(context_rep, pos):
        result = context_rep.clone()
        idx = torch.ones(context_rep.shape).bool()
        for i, p in enumerate(pos):
            idx[i, p[0]:p[1], :] = 0
        result[idx] = 0
        result = result.sum(dim=1)
        for i, p in enumerate(pos):
            result[i] /= (p[1] - p[0])
        return result
