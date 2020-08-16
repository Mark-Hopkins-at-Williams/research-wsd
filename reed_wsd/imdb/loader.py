import torch
import os
import json
from torch.utils.data import Dataset
import random

class IMDBDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.ds = data

    @classmethod
    def from_json(cls, data_path, stage):
        assert(stage in ['train', 'test'])
        with open(data_path, 'r') as f:
            data = json.load(f)[stage]
        return cls(data)
        
    def __getitem__(self, i):
        return self.ds[i]

    def __len__(self):
        return len(self.ds)


class IMDBLoader:
    def __init__(self, dataset, bsz, shuffle=True):
        self.ds = dataset
        self.bsz = bsz
        self.index_list = list(range(len(self.ds)))
        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.index_list)

    def __len__(self):
        return len(self.ds) // self.bsz + 1
        
    def __iter__(self):
        evidence_batch = []
        gold_batch = []
        for i in self.index_list:
            evidence = self.ds[i]['vec']
            gold = self.ds[i]['gold']
            evidence_batch.append(evidence)
            gold_batch.append(gold)
            if len(evidence_batch) == self.bsz:
                yield torch.tensor(evidence_batch), torch.tensor(gold_batch)
                evidence_batch = []
                gold_batch = []
        if len(evidence_batch) > 0:
            yield torch.tensor(evidence_batch), torch.tensor(gold_batch)


class IMDBTwinLoader:
    def __init__(self, dataset, bsz):
        self.ds = dataset
        self.bsz = bsz
        self.loader1 = IMDBLoader(self.ds, self.bsz, shuffle=True)
        self.loader2 = IMDBLoader(self.ds, self.bsz, shuffle=True)
        assert(len(self.loader1) == len(self.loader2))

    def __len__(self):
        return len(self.ds) // self.bsz + 1

    def __iter__(self):
        for pkg1, pkg2 in zip(self.loader1, self.loader2):
            evidence_batch1, gold_batch1 = pkg1
            evidence_batch2, gold_batch2 = pkg2
            yield (evidence_batch1, evidence_batch2,
                   gold_batch1, gold_batch2) 

            
        
