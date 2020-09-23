from reed_wsd.allwords.model import SingleLayerFFNWithZones, AbstainingSingleLayerFFNWithZones, BEMforWSD
from reed_wsd.mnist.model import BasicFFN, AbstainingFFN
from reed_wsd.allwords.wordsense import SenseInstanceDataset, SenseTaggedSentences, SenseInstanceLoader, TwinSenseInstanceLoader
from reed_wsd.allwords.vectorize import DiskBasedVectorManager
from reed_wsd.allwords.blevins import BEMDataset, BEMLoader
from reed_wsd.mnist.train import MnistSimpleDecoder
from reed_wsd.mnist.train import MnistAbstainingDecoder
from reed_wsd.allwords.evaluate import AllwordsSimpleEmbeddingDecoder, AllwordsAbstainingEmbeddingDecoder, AllwordsBEMDecoder
from reed_wsd.loss import CrossEntropyLoss, NLLLoss, AbstainingLoss, ConfidenceLoss4, PairwiseConfidenceLoss
from reed_wsd.allwords.train import SingleEmbeddingTrainer, PairwiseEmbeddingTrainer, BEMTrainer
from reed_wsd.mnist.train import SingleTrainer as MnistSingleTrainer
from reed_wsd.mnist.train import PairwiseTrainer as MnistPairwiseTrainer
from reed_wsd.mnist.loader import ConfusedMnistLoader, MnistLoader, MnistPairLoader, ConfusedMnistPairLoader
from reed_wsd.imdb.loader import IMDBDataset, IMDBLoader, IMDBTwinLoader
from reed_wsd.imdb.model import SingleLayerFFN, AbstainingSingleLayerFFN, SingleLayerConfidentFFN
import os
from os.path import join
import torch.optim as optim
from torchvision import datasets, transforms
import json
import copy
import sys
from functools import reduce

file_dir = os.path.dirname(os.path.realpath(__file__))
mnist_dir = join(file_dir, 'mnist')
allwords_dir = join(file_dir, 'allwords')
allwords_data_dir = join(allwords_dir, 'data')
mnist_data_dir = join(mnist_dir, 'data')
mnist_train_dir = join(mnist_data_dir, 'train')
mnist_test_dir = join(mnist_data_dir, 'test')
imdb_dir = join(file_dir, 'imdb')

transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                               ])

corpus_id_lookup = {'semcor': 'data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml',
                        'semev07': 'data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.data.xml',
                        'semev13': 'data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2013/semeval2013.data.xml',
                        'semev15': 'data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2015/semeval2015.data.xml',
                        'sensev2': 'data/WSD_Evaluation_Framework/Evaluation_Datasets/senseval2/senseval2.data.xml',
                        'sensev3': 'data/WSD_Evaluation_Framework/Evaluation_Datasets/senseval3/senseval3.data.xml'}

criterion_lookup = {'crossentropy': CrossEntropyLoss,
                    'nll': NLLLoss,
                    'conf1': AbstainingLoss,
                    'conf4': ConfidenceLoss4,
                    'pairwise': PairwiseConfidenceLoss}

def add_list(this, other):
    assert(len(this) == len(other))
    return [(this[i] + other[i]) for i in range(len(this))]

def div_list(ls, divisor):
    return [element / divisor for element in ls]

def add_data_dict(this, other):
    assert(this.keys() == other.keys())
    return {key: add_list(this[key], other[key]) for key in this.keys()}

def div_data_dict(d, divisor):
    return {key: div_list(d[key], divisor) for key in d.keys()}

def add_analytics_dict(this, other):
    print('\n')
    print(this, other)
    assert(this.keys() == other.keys())
    return {key: (this[key] + other[key]) if key != 'prediction_by_class'
                                     else add_data_dict(this[key], other[key])
                                     for key in this.keys()}

def div_analytics_dict(d, divisor):
    return {key: (d[key] / divisor) if key != 'prediction_by_class'
                                     else div_data_dict(d[key], divisor)
                                     for key in d.keys()}


class TaskFactory:
    def __init__(self, config):
        #check dependency
        assert(config['task'] in ['mnist', 'allwords', 'imdb'])
        if config['task'] in ['mnist', 'imdb']:
            assert(config['architecture'] != 'bem')
        if config['architecture'] == 'simple':
            assert(config['confidence'] == 'max_prob')
            assert(config['criterion']['name'] == 'nll')
            assert(config['style'] == 'single')
        if config['architecture'] == 'abstaining':
            assert(config['confidence'] in ['max_non_abs', 'inv_abs', 'abs'])
            assert(config['criterion']['name'] in ['conf1', 'conf4', 'pairwise'])
        if config['architecture'] == 'bem':
            assert(config['task'] == 'allwords')
            assert(config['criterion']['name'] == 'crossentropy')
            assert(config['style'] == 'single')
        if config['style'] == 'single':
            assert(config['criterion']['name'] in ['crossentropy', 'nll', 'conf1', 'conf4'])
        if config['style'] == 'pairwise':
            assert(config['criterion']['name'] == 'pairwise')
            assert(config['architecture'] == 'abstaining')

        self.config = config
    
    def train_loader_factory(self):
        raise NotImplementedError("Cannot call on abstract class.")

    def val_loader_factory(self):
        raise NotImplementedError("Cannot call on abstract class.")

    def decoder_factory(self):
        raise NotImplementedError("Cannot call on abstract class.")

    def model_factory(self, data):
        raise NotImplementedError("Cannot call on abstract class.")
        
    def optimizer_factory(self):
        raise NotImplementedError("Cannot call on abstract class.")

    def criterion_factory(self):
        config = self.config['criterion']
        if config['name'] in ['conf1', 'conf4']:
            criterion = criterion_lookup[config['name']](alpha=config['alpha'], warmup_epochs=config['warmup_epochs'])
        if config['name'] in ['crossentropy', 'nll', 'pairwise']:
            criterion = criterion_lookup[config['name']]()
        return criterion
    
    def select_trainer(self):
        raise NotImplementedError("Cannot call on abstract class.")

    def trainer_factory(self):
        train_loader = self.train_loader_factory()
        val_loader = self.val_loader_factory()
        decoder = self.decoder_factory()
        model = self.model_factory(train_loader)
        optimizer = self.optimizer_factory(model)
        criterion = self.criterion_factory()
        n_epochs = self.config['n_epochs']
        trainer_class = self.select_trainer()
        trainer = trainer_class(criterion, optimizer, train_loader, 
                                val_loader, decoder, n_epochs)
    
        print('model:', type(model).__name__)
        print('criterion:', type(criterion).__name__)
        print('optimizer:', type(optimizer).__name__)
        print('train loader:', type(train_loader).__name__)
        print('val loader:', type(val_loader).__name__)
        print('decoder:', type(decoder).__name__)
        print('n_epochs', n_epochs)
        if hasattr(criterion, 'warmup_epochs'):
            print('warmup epochs:', criterion.warmup_epochs)
        else:
            print('warmup epochs: N/A')
    
        return trainer, model        


class AllwordsTaskFactory(TaskFactory):
    
    def __init__(self, config):
        super().__init__(config)
        
        self._model_lookup = {'simple': SingleLayerFFNWithZones,
                              'abstaining': AbstainingSingleLayerFFNWithZones,
                              'bem': BEMforWSD}
        self._decoder_lookup = {'simple': AllwordsSimpleEmbeddingDecoder,
                                 'abstaining': AllwordsAbstainingEmbeddingDecoder,
                                 'bem': AllwordsBEMDecoder}
        assert(self.config['task'] == 'allwords')


    @staticmethod
    def init_loader(stage, architecture, style, corpus_id, bsz):
        data_dir = allwords_data_dir
        filename = join(data_dir, 'raganato.json')
        sents = SenseTaggedSentences.from_json(filename, corpus_id)
        if architecture == "bem":
            ds = BEMDataset(sents)
            loader = BEMLoader(ds, bsz)
        if architecture == 'simple' or architecture == 'abstaining': 
            vecmgr = DiskBasedVectorManager(join(join(data_dir, 'vecs'), corpus_id))
            ds = SenseInstanceDataset(sents, vecmgr)
            if stage == 'train':
                if style == 'single':
                    loader = SenseInstanceLoader(ds, batch_size=bsz)
                if style == 'pairwise':
                    loader = TwinSenseInstanceLoader(ds, batch_size=bsz)
            if stage == 'test':
                loader = SenseInstanceLoader(ds, batch_size=bsz)
        return loader
        
    def train_loader_factory(self):
        if self.config['architecture'] == 'bem' or self.config['architecture'] == 'simple':
            assert(self.config['style'] == 'single')
        return AllwordsTaskFactory.init_loader('train', 
                                    self.config['architecture'], 
                                    self.config['style'], 
                                    corpus_id_lookup['semcor'],
                                    self.config['bsz'])

    def val_loader_factory(self):
        if self.config['architecture'] == 'bem' or self.config['architecture'] == 'simple':
            assert(self.config['style'] == 'single')
        return AllwordsTaskFactory.init_loader('test', 
                                    self.config['architecture'], 
                                    self.config['style'], 
                                    corpus_id_lookup[self.config['dev_corpus']],
                                    self.config['bsz'])

    def decoder_factory(self):
        return self._decoder_lookup[self.config['architecture']]()
    
    def model_factory(self, data):
        if self.config['architecture'] == 'bem':
            model = self._model_lookup[self.config['architecture']](gpu=True)
        else:
            model = self._model_lookup[self.config['architecture']](input_size=768,
                                                                  output_size=data.num_senses(),
                                                                  zone_applicant=self.config['confidence'])
        return model
 
    def optimizer_factory(self, model):
        return optim.Adam(model.parameters(), lr=0.001)
 
    def select_trainer(self):
        if self.config['architecture'] == 'bem':
            trainer = BEMTrainer
        else:
            if self.config['style'] == 'single':
                trainer = SingleEmbeddingTrainer
            if self.config['style'] == 'pairwise':
                trainer = PairwiseEmbeddingTrainer
        return trainer       

 
class MnistTaskFactory(TaskFactory):
    
    def __init__(self, config):
        super().__init__(config)
        self._model_lookup = {'simple': BasicFFN,
                              'abstaining': AbstainingFFN}
        self._decoder_lookup = {'simple': MnistSimpleDecoder,
                                'abstaining': MnistAbstainingDecoder}

    @staticmethod
    def init_loader(stage, style, confuse, bsz):
        if stage == 'train':
            ds = datasets.MNIST(mnist_train_dir, download=True, train=True, transform=transform)
            if style == 'single':
                if confuse != False:
                    loader = ConfusedMnistLoader(ds, bsz, confuse, shuffle=True)
                else:
                    loader = MnistLoader(ds, bsz, shuffle=True)
            if style == 'pairwise':
                if confuse != False:
                    loader = ConfusedMnistPairLoader(ds, bsz, confuse, shuffle=True)
                else:
                    loader = MnistPairLoader(ds, bsz, shuffle=True)
        if stage == 'test':
            ds = datasets.MNIST(mnist_test_dir, download=True, train=False, transform=transform)
            if confuse != False:
                loader = ConfusedMnistLoader(ds, bsz, confuse, shuffle=True)
            else:
                loader = MnistLoader(ds, bsz, shuffle=True)
        return loader
        
    def train_loader_factory(self):
        return MnistTaskFactory.init_loader('train', 
                                            self.config['style'], 
                                            self.config['confuse'], 
                                            self.config['bsz'])
    
    def val_loader_factory(self):
        return MnistTaskFactory.init_loader('test', 
                                            self.config['style'], 
                                            self.config['confuse'], 
                                            self.config['bsz'])

    def decoder_factory(self):
        return self._decoder_lookup[self.config['architecture']]()

    def model_factory(self, data):
        return self._model_lookup[self.config['architecture']](confidence_extractor=self.config['confidence'])

    def optimizer_factory(self, model):
        return optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    def select_trainer(self):
        if self.config['style'] == 'single':
            trainer = MnistSingleTrainer
        elif self.config['style'] == 'pairwise':
            trainer = MnistPairwiseTrainer
        return trainer


class IMDBTaskFactory(TaskFactory):
    def __init__(self, config):
        super().__init__(config)
        self._model_lookup = {'simple': SingleLayerFFN,
                              'abstaining': AbstainingSingleLayerFFN}
        self._decoder_lookup = {'simple': MnistSimpleDecoder,
                                'abstaining': MnistAbstainingDecoder}
    def train_loader_factory(self):
        ds = IMDBDataset.from_json(join(imdb_dir, 'data/aclImdb/imdb.json'), 'train')
        bsz = self.config['bsz']
        if self.config['style'] == 'single':
            loader = IMDBLoader(ds, bsz, shuffle=True)
        if self.config['style'] == 'pairwise':
            loader = IMDBTwinLoader(ds, bsz)
        return loader

    def val_loader_factory(self):
        ds = IMDBDataset.from_json(join(imdb_dir, 'data/aclImdb/imdb.json'), 'test')
        bsz = self.config['bsz']
        loader = IMDBLoader(ds, bsz, shuffle=True)
        return loader

    def decoder_factory(self):
        return self._decoder_lookup[self.config['architecture']]()

    def model_factory(self, data):
        return self._model_lookup[self.config['architecture']](confidence_extractor=self.config['confidence'])
        
    def optimizer_factory(self, model):
        return optim.Adam(model.parameters(), lr=0.0005)
    
    def select_trainer(self):
        if self.config['style'] == 'single':
            trainer = MnistSingleTrainer
        elif self.config['style'] == 'pairwise':
            trainer = MnistPairwiseTrainer
        return trainer

task_factories = {'mnist': MnistTaskFactory,
                  'allwords': AllwordsTaskFactory,
                  'imdb': IMDBTaskFactory}

def run_experiment(config):
    task_factory = task_factories[config['task']](config)
    trainer, model = task_factory.trainer_factory()
    best_model = trainer(model)
    return best_model

class Experiment:
    def __init__(self, config, reps=10):
        self.config = config
        self.reps = reps
        self.task_factory = task_factories[config['task']](config)

    def run(self):
        measurements = []
        for i in range(self.reps):
            print('\nTRIAL {}'.format(i))
            trainer, model = self.task_factory.trainer_factory()    
            _, results = trainer(model)
            measurements.append(results)
        measurement_sum = reduce(add_analytics_dict, measurements)
        avg_measurement = div_analytics_dict(measurement_sum, self.reps)
        self.result = avg_measurement
        return results
    
    def return_analytics(self):
        result = copy.deepcopy(self.result)
        return result

class ExperimentSequence:
    def __init__(self, experiments, reps):
        self.experiments = experiments
        self.reps = reps
    
    @classmethod
    def from_json(cls, configs_path, reps=10):
        with open(configs_path, 'r') as f:
            configs = json.load(f)
        experiments = []
        for config in configs:
            experiments.append(Experiment(config, reps))
        return cls(experiments, reps)

    def run_and_save(self, out_path):
        results = []
        for experiment in self.experiments:
            experiment.run()
            results.append(experiment.return_analytics())
        with open(out_path, 'w') as f:
            json.dump(results, f)

            

if __name__ == "__main__":
    config_path = sys.argv[1]
    output_path = sys.argv[2]

    exp_seq = ExperimentSequence.from_json(config_path)
    exp_seq.run_and_save(output_path)
    
