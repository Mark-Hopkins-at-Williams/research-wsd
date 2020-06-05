import unittest
from raganato import harvest_data, create_sense_inventory
from raganato import parse_raganato_gold, harvest_multi

class TestRaganato(unittest.TestCase):
          
          

    
    def test_harvest_data(self):
        inventory = create_sense_inventory(['testdata/raganato1.key.txt'])
        result = harvest_data('testdata/raganato1.xml', 
                              'testdata/raganato1.key.txt',
                              inventory)
        sents = result
        assert(len(sents) == 2)
        expected_words0 = [{'word': 'the', 'tag': 'DET'}, 
                           {'word': 'top', 'tag': 'NOUN', 'sense': 'top%3:00:02::'}, 
                           {'word': 'of', 'tag': 'ADP'}, 
                           {'word': 'the', 'tag': 'DET'}, 
                           {'word': 'top', 'tag': 'NOUN', 'sense': 'top%2:42:03::'}, 
                           {'word': 'is', 'tag': 'VERB'}, 
                           {'word': 'gold', 'tag': 'ADJ', 'sense': 'gold%2:31:00::'}]
        print(sents[0]['words'])
        assert(sents[0]['words'] == expected_words0)        
        expected_words1 = [{'word': 'gold', 'tag': 'NOUN', 'sense': 'gold%1:09:00::'}, 
                           {'word': 'is', 'tag': 'DET'}, 
                           {'word': 'at', 'tag': 'ADP'}, 
                           {'word': 'the', 'tag': 'DET'}, 
                           {'word': 'mountain', 'tag': 'NOUN', 'sense': 'mountain%1:21:00::'}, 
                           {'word': 'top', 'tag': 'NOUN', 'sense': 'top%3:00:02::'}]
        assert(sents[1]['words'] == expected_words1)

    def test_create_sense_inventory(self):
        inventory = create_sense_inventory(['testdata/raganato1.key.txt', 
                                            'testdata/raganato2.key.txt'])
        expected = {'top': ['3:00:02::', '2:42:03::'], 
                    'gold': ['2:31:00::', '1:09:00::'], 
                    'mountain': ['1:21:00::'], 
                    'bottom': ['3:10:02::'], 
                    'silver': ['2:11:00::', '1:29:00::']}
        assert inventory == expected

    def test_parse_raganato_gold(self):
        inventory = create_sense_inventory(['testdata/raganato1.key.txt', 
                                            'testdata/raganato2.key.txt'])
        result = parse_raganato_gold('testdata/raganato1.key.txt', inventory)
        expected = {'d000.s000.t000': 'top%3:00:02::', 
                    'd000.s000.t001': 'top%2:42:03::', 
                    'd000.s000.t002': 'gold%2:31:00::', 
                    'd000.s001.t000': 'gold%1:09:00::', 
                    'd000.s001.t001': 'mountain%1:21:00::',
                    'd000.s001.t002': 'top%3:00:02::'}
        print(result)
        assert result == expected
        
    def test_harvest_multi(self):
        result = harvest_multi(['testdata/raganato1.xml',
                                'testdata/raganato2.xml'],
                               ['testdata/raganato1.key.txt',
                                'testdata/raganato2.key.txt'])
        #print(result)
        
        
if __name__ == "__main__":
	unittest.main()
