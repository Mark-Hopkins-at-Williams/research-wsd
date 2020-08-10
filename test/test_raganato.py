import unittest
from reed_wsd.allwords.raganato import harvest_data, create_sense_inventory
from reed_wsd.allwords.raganato import parse_raganato_gold
import os
from os.path import join

file_dir = os.path.dirname(os.path.realpath(__file__))

class TestRaganato(unittest.TestCase):
    def setUp(self):
        self.testdata_dir = join(file_dir, 'testdata')

          
    def test_harvest_data(self):
        inventory = create_sense_inventory([join(self.testdata_dir, 'raganato1.key.txt')])
        result, n_insts = harvest_data(join(self.testdata_dir, 'raganato1.xml'), 
                              join(self.testdata_dir, 'raganato1.key.txt'),
                              inventory)
        sents = result
        assert(len(sents) == 2)
        expected_words0 = [{'word': 'the', 'tag': 'DET'}, 
                           {'word': 'top', 'tag': 'NOUN', 
                            'sense': 'top%3:00:02::', 'id': 'd000.s000.t000'}, 
                           {'word': 'of', 'tag': 'ADP'}, 
                           {'word': 'the', 'tag': 'DET'}, 
                           {'word': 'top', 'tag': 'NOUN', 
                            'sense': 'top%2:42:03::', 'id': 'd000.s000.t001'}, 
                           {'word': 'is', 'tag': 'VERB'}, 
                           {'word': 'gold', 'tag': 'ADJ', 
                            'sense': 'gold%2:31:00::', 'id': 'd000.s000.t002'}]
        assert(sents[0]['words'] == expected_words0) 
        expected_words1 = [{'word': 'gold', 'tag': 'NOUN', 
                            'sense': 'gold%1:09:00::', 'id': 'd000.s001.t000'}, 
                           {'word': 'is', 'tag': 'DET'}, 
                           {'word': 'at', 'tag': 'ADP'}, 
                           {'word': 'the', 'tag': 'DET'}, 
                           {'word': 'mountain', 'tag': 'NOUN', 
                            'sense': 'mountain%1:21:00::', 'id': 'd000.s001.t001'}, 
                           {'word': 'top', 'tag': 'NOUN', 
                            'sense': 'top%3:00:02::', 'id': 'd000.s001.t002'}]
        assert(sents[1]['words'] == expected_words1)
        assert(n_insts == 6)

    def test_create_sense_inventory(self):
        inventory = create_sense_inventory([join(self.testdata_dir, 'raganato1.key.txt'), 
                                            join(self.testdata_dir, 'raganato2.key.txt')]) 
        expected = {'top': ['top%3:00:02::', 'top%2:42:03::'], 
                    'gold': ['gold%2:31:00::', 'gold%1:09:00::'], 
                    'mountain': ['mountain%1:21:00::'], 
                    'bottom': ['bottom%3:10:02::'], 
                    'silver': ['silver%2:11:00::', 'silver%1:29:00::']}
        assert inventory == expected

    def test_parse_raganato_gold(self):
        inventory = create_sense_inventory([join(self.testdata_dir, 'raganato1.key.txt'), 
                                            join(self.testdata_dir, 'raganato2.key.txt')])
        result = parse_raganato_gold(join(self.testdata_dir, 'raganato1.key.txt'), inventory)
        expected = {'d000.s000.t000': 'top%3:00:02::', 
                    'd000.s000.t001': 'top%2:42:03::', 
                    'd000.s000.t002': 'gold%2:31:00::', 
                    'd000.s001.t000': 'gold%1:09:00::', 
                    'd000.s001.t001': 'mountain%1:21:00::',
                    'd000.s001.t002': 'top%3:00:02::'}
        assert result == expected
        

if __name__ == "__main__":
	unittest.main()
