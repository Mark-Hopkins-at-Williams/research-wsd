from reed_wsd.analysis import Prediction, Predictions
from reed_wsd.allwords.wordsense import SenseTaggedSentences
from reed_wsd.experiment import allwords_data_dir, corpus_id_lookup
from nltk.corpus import wordnet as wn


class AllwordsPrediction(Prediction):
    
    def __init__(self, pred):
        super.__init__(pred)
        self.inv = self._init_inv()
        self.convertors['gold'] = self._convert_sense
        self.convertors['pred'] = self._convert_sense
        self.convertors['confidence'] = lambda x: x
        self.convertors['abstained'] = lambda x: x


    @staticmethod
    def _init_inv():
        filename = join(allwords_data_dir, 'raganato.json')
        sents = SenseTaggedSentences.from_json(filename, corpus_id_lookup['semev07'])
        inv = sents.get_inventory()
        return inv

    def _convert_sense(self, sense_id):
        sense = self.inv.sense(sense_id)
        lemma = wn.lemma_from_key(sense)
        synset = lemma.synset()
        return lemma, synset

class AllwordsPredictions(Predictions):
    def __init__(self, data):
        super.__init__(data, AllwordsPrediction)

