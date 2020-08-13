import unittest
import torch
from reed_wsd.imdb.loader import IMDBDataset, IMDBLoader, IMDBTwinLoader

class TestLoader(unittest.TestCase):
    def setUp(self):
        self.data = [{'vec': [0.3, 0.4, 0.3], 'gold': 1},
                     {'vec': [0.1, 0.5, 0.4], 'gold': 0},
                     {'vec': [0.2, 0.6, 0.2], 'gold': 1}]
        self.ds = IMDBDataset(self.data)
        self.loader = IMDBLoader(self.ds, 2, shuffle=False)
        self.twin_loader = IMDBTwinLoader(self.ds, 2)

    def test_loader(self):
        batch_iter = iter(self.loader)
        evidence_batch1, gold_batch1 = next(batch_iter)
        evidence_batch2, gold_batch2 = next(batch_iter)
        assert(torch.equal(evidence_batch1, torch.tensor([[0.3, 0.4, 0.3],
                                                          [0.1, 0.5, 0.4]])))
        assert(torch.equal(gold_batch1, torch.tensor([1, 0])))
        assert(torch.equal(evidence_batch2, torch.tensor([[0.2, 0.6, 0.2]])))
        assert(torch.equal(gold_batch2, torch.tensor([1])))
        
        assert(len(self.loader) == 2)

    def test_twin_loader(self):
        batch_iter = self.twin_loader.__iter__()
        pkg1 = next(batch_iter)
        pkg2 = next(batch_iter)
        print(pkg1)
        print(pkg2)

if __name__ == '__main__':
    unittest.main()
