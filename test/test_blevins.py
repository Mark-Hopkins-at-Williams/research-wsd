import unittest
import json
from reed_wsd.allwords.blevins import BEMDataset, BEMLoader
from reed_wsd.allwords.wordsense import SenseTaggedSentences, SenseInventory
import torch



class TestBEMDataset(unittest.TestCase):
    def setUp(self):
        self.data = {'inventory': 
                         {'be': ['be%2:42:06::'],
                          'laugh_off': ['laugh_off%2:32:00::'],
                          'screen': ['screen%1:06:06::']},
                     'corpora': 
                         {'corpus1':
                            {'sents':
                                [{'sentid': 37163, 
                                 'words': [{'word': 'It', 'tag': 'PRP'}, 
                                           {'word': 'was', 'tag': 'VB', 
                                            'sense': 'be%2:42:06::', 'id': 'i1'}, 
                                           {'word': 'a', 'tag': 'DT'}, 
                                           {'word': 'disaster', 'tag': 'NN'}, 
                                           {'word': '!', 'tag': 'punc'}]},
                                {'sentid': 37165, 
                                 'words': [{'word': 'He', 'tag': 'PRP'}, 
                                           {'word': 'was', 'tag': 'VBD'}, 
                                           {'word': 'laughed_off', 'tag': 'VB', 
                                            'sense': 'laugh_off%2:32:00::', 'id': 'i1'}, 
                                           {'word': 'the', 'tag': 'DT'}, 
                                           {'word': 'screen', 'tag': 'NN', 
                                            'sense': 'screen%1:06:06::', 'id': 'i1'}, 
                                           {'word': '.', 'tag': 'punc'}]}],
                             'n_insts': 3                            
                            }
                            }}            
        self.sents = SenseTaggedSentences.from_json_str(json.dumps(self.data),
                                                        corpus_id = "corpus1")
    
    def test_BEMDataset_subset(self):
        ds = BEMDataset(self.sents, sense_sz=2, randomize_sents = False)
        assert(len(ds) == 2)
        ds = BEMDataset(self.sents, sense_sz=1, randomize_sents = False)
        assert(len(ds) == 1)
        ds = BEMDataset(self.sents, randomize_sents = False)
        inv_d = {'be': ['be%2:42:06::'],
                 'screen': ['screen%1:06:06::']}
        inv = SenseInventory(inv_d)
        ds.set_inventory(inv)
        sent_iter = ds.item_iter()
        sent1 = next(sent_iter)
        assert sent1['pos'] == [2, 3] # position of "be" in sentence 1
        # note that "laugh_off" is excluded from the iterator
        sent2 = next(sent_iter)
        assert sent2['pos'] == [7, 8] # position of "screen" in sentence 2
        assert(len(ds) == 2)

    def test_bem_dataset_default(self):
        ds = BEMDataset(self.sents, randomize_sents = False)
        inst0 = ds[0]
        expected_input_ids = torch.tensor([101, 2009, 2001, 1037, 7071, 999, 102])
        expected_range = [2, 3]
        expected_gloss = torch.tensor([[101, 2022, 7235, 2000, 1025, 2022, 2619, 2030, 2242, 102]])
        expected_gold = 0
        assert(torch.equal(expected_input_ids, inst0['input_ids']))
        assert(expected_range == inst0['pos'])
        assert(torch.equal(expected_gloss, inst0['glosses_ids']['input_ids']))
        assert(expected_gold == inst0['sense_id'])

    def test_bem_dataset_defn_tgt(self):
        ds = BEMDataset(self.sents, randomize_sents=False, gloss='defn_tgt')
        inst1 = ds[1]
        expected_context = torch.tensor([101, 2002, 2001, 4191, 1035, 2125, 1996, 3898, 1012, 102])
        expected_context_span = [3, 6]
        expected_gloss = torch.tensor([[101, 2000, 4756, 1035, 2125, 2003, 2000, 3066, 2007, 1037, 3291, 2011,
                                       5870, 2030, 12097, 2000, 2022, 11770, 2011, 2009, 102]])
        expected_gloss_span = [[2, 5]]
        assert(torch.equal(expected_context, inst1['input_ids']))
        assert(expected_context_span == inst1['pos'])
        assert(torch.equal(expected_gloss, inst1['glosses_ids']['input_ids']))
        assert(expected_gloss_span == inst1['glosses_ids']['span'])

    def test_bem_dataset_wneg(self):
        ds = BEMDataset(self.sents, randomize_sents=False, gloss='wneg')
        inst2 = ds[2]
        expected_context = torch.tensor([101, 2002, 2001, 4191, 1035, 2125, 1996, 3898, 1012, 102])
        expected_context_span = [7, 8]
        expected_gloss = torch.tensor([[101, 3898, 2003, 1037, 2317, 2030, 3165, 2098, 3302, 2073,
                                        4620, 2064, 2022, 11310, 2005, 10523, 102]])
        expected_gloss_span = [[1, 2]]
        assert(torch.equal(expected_context, inst2['input_ids']))
        assert(expected_context_span == inst2['pos'])
        assert(torch.equal(expected_gloss, inst2['glosses_ids']['input_ids']))
        assert(expected_gloss_span == inst2['glosses_ids']['span'])
        inst0 = ds[0]
        expected_gloss = torch.tensor([[ 101, 1996, 2343, 1997, 1996, 2194, 2003, 2198, 3044,  102]])
        expected_gloss_span = [[6, 7]]
        assert(torch.equal(expected_gloss, inst0['glosses_ids']['input_ids']))
        assert(expected_gloss_span == inst0['glosses_ids']['span'])

    def test_BEMLoader(self):
        ds = BEMDataset(self.sents, randomize_sents = False)
        loader = BEMLoader(ds, batch_size = 3)
        batch_iter = loader.batch_iter()
        pkg = next(batch_iter)
        expected_contexts = torch.tensor(
                             [[101, 2009, 2001, 1037, 7071, 999, 102, 0, 0, 0],
                             [101, 2002, 2001, 4191, 1035, 2125, 1996, 3898, 1012, 102], 
                             [101, 2002, 2001, 4191, 1035, 2125, 1996, 3898, 1012, 102]])
        expected_context_masks = torch.tensor(
                             [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        expected_span = [[2, 3], [3, 6], [7, 8]]
        be = {'input_ids': torch.tensor([[101, 2022, 7235, 2000, 1025, 2022, 2619, 2030, 2242, 102]]), 
              'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
              'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
        lo = {'input_ids': torch.tensor([[101, 3066, 2007, 1037, 3291, 2011, 5870, 2030, 12097, 2000, 2022, 11770, 2011, 2009, 102]]),
                'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}  
        screen = {'input_ids': torch.tensor([[101, 1037, 2317, 2030, 3165, 2098, 3302, 2073, 4620, 2064, 2022, 11310, 2005, 10523, 102]]),
                  'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                  'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}  
        expected_glosses = [be, lo, screen]
        expected_gold = [0, 0, 0]
        assert(torch.equal(expected_contexts, pkg['contexts']['input_ids']))
        assert(torch.equal(expected_context_masks, pkg['contexts']['attention_mask']))
        for i, inst in enumerate(expected_glosses):
            assert(torch.equal(inst['input_ids'], pkg['glosses'][i]['input_ids']))
        assert(expected_span == pkg['span'])
        assert(expected_gold == pkg['gold'])

        


        
if __name__ == "__main__":
	unittest.main()
