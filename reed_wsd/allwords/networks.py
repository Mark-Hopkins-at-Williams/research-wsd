# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel
from reed_wsd.util import cudaify

class AffineClassifier(nn.Module): 
    
    def __init__(self, input_size, num_labels):
        super(AffineClassifier, self).__init__()
        print("input_size: {}; output_size: {}".format(input_size, num_labels))
        self.linear1 = nn.Linear(input_size, num_labels)
        torch.nn.init.xavier_uniform_(self.linear1.weight)

    def forward(self, input_vec):
        nextout = input_vec
        nextout = self.linear1(nextout)
        return nextout

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
