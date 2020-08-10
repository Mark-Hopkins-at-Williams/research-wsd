import unittest
import json
from torch import tensor
from reed_wsd.allwords import wordsense, vectorize

class TestWordsense(unittest.TestCase):
    
    def setUp(self):
        vec_map = {37163: {'sentid': 37163,
                           'vecs': [[11.0, 12.0, 13.0],
                                    [21.0, 22.0, 23.0],
                                    [31.0, 32.0, 33.0],
                                    [41.0, 42.0, 43.0],
                                    [51.0, 52.0, 53.0]],
                           'tokens': ['It', 'was', 'a', 'disaster', '!']},
                   37165: {'sentid': 37163,
                           'vecs':    [[11.1, 12.1, 13.1],
                                       [21.1, 22.1, 23.1],
                                       [31.1, 32.1, 33.1],
                                       [41.1, 42.1, 43.1],
                                       [51.1, 52.1, 53.1],
                                       [61.1, 62.1, 63.1],
                                       [71.1, 72.1, 73.1]],
                           'tokens': ['he', 'was', 'laughed', 'off', 
                                      'the', 'screen', '.']}}                         
        self.vec_mgr = vectorize.RamBasedVectorManager(vec_map)
        self.data = {'inventory': {'be': ['be%2:42:06::'],
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
        self.sents = wordsense.SenseTaggedSentences.from_json_str(json.dumps(self.data),
                                                        corpus_id = "corpus1")
     
    def compare_lists(self, list1, list2, num_digits = 2):
        assert(len(list1) == len(list2))
        compare = zip(list1, list2)
        for i, (exp, actual) in enumerate(compare):
            if abs(exp - actual) > 10**(-num_digits):
                print("Element {} is incorrect: {} vs {}.".format(i, 
                                                                  list1[i], 
                                                                  list2[i]))
                return False
        return True
        
    def compare_vectors(self, expected, result):
        return self.compare_lists(expected.tolist(), result.tolist())
        
    def compare_matrices(self, expected, result):
        compare = zip(expected.tolist(), result.tolist())     
        for vec1, vec2 in compare:
            if not self.compare_lists(vec1, vec2):
                return False
        return True
       
    def test_sense_inventory(self):
        inventory = wordsense.SenseInventory.from_sense_iter(['laugh_off%1', 'be%1', 
                                                    'laugh_off%2', 'laugh_off%3', 
                                                    'car%1', 'be%2'])
        assert inventory.sense_id('laugh_off%2') == 4
        assert inventory.sense_id('car%1') == 2
        assert inventory.sense_id('be%2') == 1
        assert inventory.lemma_ranges == {'be': (0, 2), 
                                          'car': (2, 3), 
                                          'laugh_off': (3, 6)}
        assert inventory.sense_range('laugh_off') == (3, 6)
    
    def test_sense_tagged_sentences(self):
        sents = self.sents
        data = self.data
        assert(len(sents) == 2)
        #assert(sents.num_senses() == 3)
        corpus1 = data['corpora']['corpus1']
        sent0 = corpus1['sents'][0]
        assert(sent0['sentid'] == 37163)
        sent1 = corpus1['sents'][1]
        assert(sent1['sentid'] == 37165)
        assert(sent1['words'][2]['word'] == 'laughed_off')
        assert(sent1['words'][2]['tag'] == 'VB')
        assert(sent1['words'][2]['sense'] == 'laugh_off%2:32:00::')
 
    def test_sense_instance_dataset(self):
        dataset = wordsense.SenseInstanceDataset(self.sents, self.vec_mgr,
                                                 randomize_sents = False)
        sense_inst0 = dataset[0]
        assert(sense_inst0.sense == 'be%2:42:06::')
        assert(sense_inst0.tokens ==  ['It', 'was', 'a', 'disaster', '!'])
        assert(sense_inst0.pos == 1)
        assert(self.compare_lists(sense_inst0.get_embedding('embed'), 
                                  [21.0, 22.0, 23.0]))
        sense_inst1 = dataset[1]
        assert(sense_inst1.sense == 'laugh_off%2:32:00::')
        assert(sense_inst1.tokens ==  ['He', 'was', 'laughed_off', 'the', 
                                       'screen', '.'])
        assert(sense_inst1.pos == 2)
        assert(self.compare_lists(sense_inst1.get_embedding('embed'), 
                                  [72.2, 74.2, 76.2]))      
        sense_inst2 = dataset[2]
        assert(sense_inst2.sense == 'screen%1:06:06::')
        assert(sense_inst2.tokens ==  ['He', 'was', 'laughed_off', 'the', 
                                       'screen', '.'])
        assert(sense_inst2.pos == 4)
        assert(self.compare_lists(sense_inst2.get_embedding('embed'), 
                                  [61.1, 62.1, 63.1]))       
 
    def test_sense_instance_loader(self):
        dataset = wordsense.SenseInstanceDataset(self.sents, self.vec_mgr,
                                                 randomize_sents = False)
        loader = wordsense.SenseInstanceLoader(dataset, batch_size = 1)
        batch_iter = loader.batch_iter()
        _, _, evid, resp, _ = next(batch_iter)
        expected_evid = tensor([[21., 22., 23.]])
        expected_resp = tensor([0])
        assert(self.compare_matrices(evid, expected_evid))
        assert(self.compare_vectors(resp.float(), expected_resp.float()))
        _, _, evid, resp, _ = next(batch_iter)
        expected_evid = tensor([[72.2, 74.2, 76.2]])
        expected_resp = tensor([1])
        assert(self.compare_matrices(evid, expected_evid))
        assert(self.compare_vectors(resp.float(), expected_resp.float()))
        _, _, evid, resp, _ = next(batch_iter)
        expected_evid = tensor([ [61.1, 62.1, 63.1]])
        expected_resp = tensor([2])
        assert(self.compare_matrices(evid, expected_evid))
        assert(self.compare_vectors(resp.float(), expected_resp.float()))
        
    def test_sense_instance_loader2(self):
        dataset = wordsense.SenseInstanceDataset(self.sents, self.vec_mgr,
                                                 randomize_sents = False)
        loader = wordsense.SenseInstanceLoader(dataset, batch_size = 2)
        batch_iter = loader.batch_iter()
        _, _, evid, resp, _ = next(batch_iter)
        expected_evid = tensor([[21., 22., 23.],
                                [72.2, 74.2, 76.2]])
        expected_resp = tensor([0, 1])
        assert(self.compare_matrices(evid, expected_evid))
        assert(self.compare_vectors(resp.float(), expected_resp.float()))

    def test_twin_sense_instance_loader(self):
        dataset = wordsense.SenseInstanceDataset(self.sents, self.vec_mgr,
                                                 randomize_sents = False)
        loader = wordsense.TwinSenseInstanceLoader(dataset, batch_size = 2)
        batch_iter = loader.batch_iter()
        ((_, _, evid1, resp1, _), (_, _, evid2, resp2, _)) = next(batch_iter)
        expected_evid = tensor([[21., 22., 23.],
                                [72.2, 74.2, 76.2]])
        expected_resp = tensor([0, 1])
        assert(self.compare_matrices(evid1, expected_evid))
        assert(self.compare_vectors(resp1.float(), expected_resp.float()))
        assert(self.compare_matrices(evid2, expected_evid))
        assert(self.compare_vectors(resp2.float(), expected_resp.float()))
        



        
if __name__ == "__main__":
	unittest.main()
