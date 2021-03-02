from reed_wsd.mnist.model import BasicFFN, AbstainingFFN, ConfidenceFFN
from reed_wsd.allwords.wordsense import SenseInstanceDataset, SenseTaggedSentences, SenseInstanceLoader, TwinSenseInstanceLoader
from reed_wsd.allwords.vectorize import DiskBasedVectorManager
from reed_wsd.allwords.blevins import BEMDataset, BEMLoader
from reed_wsd.allwords.model import SingleLayerFFNWithZones, AbstainingSingleLayerFFNWithZones, BEMforWSD
from reed_wsd.mnist.train import MnistSimpleDecoder
from reed_wsd.mnist.train import MnistAbstainingDecoder
from reed_wsd.allwords.evaluate import AllwordsSimpleEmbeddingDecoder, AllwordsAbstainingEmbeddingDecoder, AllwordsBEMDecoder
from reed_wsd.loss import CrossEntropyLoss, NLLLoss, AbstainingLoss, ConfidenceLoss4, PairwiseConfidenceLoss, DACLoss, ConsciousLoss, WeightedNLLLoss
from reed_wsd.allwords.train import SingleEmbeddingTrainer, PairwiseEmbeddingTrainer, BEMTrainer
from reed_wsd.mnist.train import SingleTrainer as MnistSingleTrainer
from reed_wsd.mnist.train import PairwiseTrainer as MnistPairwiseTrainer
from reed_wsd.mnist.loader import ConfusedMnistLoader, MnistLoader, MnistPairLoader, ConfusedMnistPairLoader
from reed_wsd.imdb.loader import IMDBDataset, IMDBLoader, IMDBTwinLoader
from reed_wsd.imdb.model import SingleLayerFFN, AbstainingSingleLayerFFN, SingleLayerConfidenceFFN
import os
from os.path import join
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import json
import copy
import sys
from functools import reduce
from trustscore import TrustScore
from reed_wsd.util import logger_config, log
from datetime import datetime
import statistics
import argparse

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
                    'batch_softmax': WeightedNLLLoss,
                    'nll': NLLLoss,
                    'conf1': AbstainingLoss,
                    'conf4': ConfidenceLoss4,
                    'pairwise': PairwiseConfidenceLoss,
                    'dac': DACLoss,
                    'conscious': ConsciousLoss}

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
        if config['trustscore'] == True:
            assert(config['confidence'] == 'max_prob')
        if config['architecture'] == 'simple':
            assert(config['confidence'] == 'max_prob')
            assert(config['criterion']['name'] in ['nll', 'batch_softmax'])
            assert(config['style'] == 'single')
        if config['architecture'] == 'abstaining':
            assert(config['confidence'] in ['max_non_abs', 'inv_abs', 'abs', 'entropy', 'norm'])
            assert(config['criterion']['name'] in ['conf1', 'conf4', 'pairwise', 'dac'])
        if config['architecture'] == 'confidence':
            assert(config['confidence'] == 'conscious')
            assert(config['criterion']['name'] == 'conscious')
        if config['architecture'] == 'bem':
            assert(config['task'] == 'allwords')
            assert(config['criterion']['name'] == 'crossentropy')
            assert(config['style'] == 'single')
        if config['style'] == 'single':
            assert(config['criterion']['name'] in ['crossentropy', 'nll', 'batch_softmax', 'conf1', 'conf4', 'dac', 'conscious'])
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
        if config['name'] in ['batch_softmax']:
            criterion = criterion_lookup[config['name']](warmup_epochs=config['warmup_epochs'], horizon=config['horizon'])
        if config['name'] in ['conf1', 'conf4']:
            criterion = criterion_lookup[config['name']](alpha=config['alpha'], warmup_epochs=config['warmup_epochs'])
        if config['name'] in ['crossentropy', 'nll', 'pairwise', 'conscious']:
            criterion = criterion_lookup[config['name']]()
        if config['name'] in ['dac']:
            criterion = criterion_lookup[config['name']](target_alpha=config['alpha'],
                                                         warmup_epochs=config['warmup_epochs'],
                                                         total_epochs=self.config['n_epochs'],
                                                         alpha_init_factor=config['alpha_init_factor'])
        return criterion

    def trust_model_factory(self, train_loader):
        if self.config['trustscore']:
            trust_model = TrustScore()
            train_instances = list(train_loader)
            evidences = []
            labels = []
            for pkg in train_instances:
                if self.config['task'] == "allwords":
                    (_, _, evidence, label, _) = pkg
                else:
                    evidence, label = pkg
                evidences.append(evidence)
                labels.append(label)
            train_evidence = torch.cat(evidences).numpy()
            train_label = torch.cat(labels).numpy()
            print(train_evidence.shape)
            print(train_label.shape)
            trust_model.fit(train_evidence, train_label)
            return trust_model
        else:
            return None

    def select_trainer(self):
        raise NotImplementedError("Cannot call on abstract class.")

    def trainer_factory(self):
        train_loader = self.train_loader_factory()
        if self.config['trustscore']:
            trustmodel = self.trust_model_factory(train_loader)
        else:
            trustmodel = None
        val_loader = self.val_loader_factory()
        decoder = self.decoder_factory()
        model = self.model_factory(train_loader)
        optimizer = self.optimizer_factory(model)
        scheduler = self.scheduler_factory(optimizer)
        criterion = self.criterion_factory()
        n_epochs = self.config['n_epochs']
        trainer_class = self.select_trainer()
        trainer = trainer_class(criterion, optimizer, train_loader, 
                                val_loader, decoder, n_epochs, trustmodel, scheduler)
    
        log('model: ' + type(model).__name__)
        log('criterion: ' + type(criterion).__name__)
        log('optimizer: ' + type(optimizer).__name__)
        log('train loader: ' + type(train_loader).__name__)
        log('val loader: ' + type(val_loader).__name__)
        log('decoder: ' + type(decoder).__name__)
        log('n_epochs: ' + str(n_epochs))
        if trustmodel is not None:
            log('trustscore: True')
        else:
            log('trustscore: False')
        if hasattr(criterion, 'warmup_epochs'):
            log('warmup epochs: ' + str(criterion.warmup_epochs))
        else:
            log('warmup epochs: N/A')
    
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
 
    def scheduler_factory(self, optimizer):
        if self.config['criterion']['name'] == 'dac':
            return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7, 20, 30], gamma=0.5)
        else:
            return None
        
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
                              'abstaining': AbstainingFFN,
                              'confidence': ConfidenceFFN}
        self._decoder_lookup = {'simple': MnistSimpleDecoder,
                                'abstaining': MnistAbstainingDecoder,
                                'confidence': MnistSimpleDecoder}

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
        if self.config['criterion']['name'] == 'dac':
            return optim.SGD(model.parameters(), lr=0.02, weight_decay=5e-4, nesterov=True, momentum=0.9)
        else:
            return optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    def scheduler_factory(self, optimizer):
        if self.config['criterion']['name'] == 'dac':
            return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80, 120], gamma=0.5)
        else:
            return None

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
                              'abstaining': AbstainingSingleLayerFFN,
                              'confidence': SingleLayerConfidenceFFN}
        self._decoder_lookup = {'simple': MnistSimpleDecoder,
                                'abstaining': MnistAbstainingDecoder,
                                'confidence': MnistSimpleDecoder}
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
        if self.config['criterion']['name'] == 'dac':
            return optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, nesterov=True, momentum=0.9)
        else:
            return optim.Adam(model.parameters(), lr=0.0005)
    
    def scheduler_factory(self, optimizer):
        if self.config['criterion']['name'] == 'dac':
            return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 60, 120], gamma=0.1)
        else:
            return None

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

class Result:
    """
    represent results for a single metric

    """

    def __init__(self, results):
        self._results = results

    def raw(self):
        return self._results

    def max(self):
        return max(self._results)
    
    def mean(self):
        return statistics.mean(self._results)

    def stdev(self):
        return statistics.pstdev(self._results)

    def summary(self):
        return {'max': self.max(),
                'mean': self.mean(),
                'stdev': self.stdev()}


class Results:

    def __init__(self, trial_results):
        self._trial_results = trial_results
        self._dict = dict()
        for k in trial_results[0].keys():
            """
            we assume that each element in trial_results
            has the same keys
            
            """
            self._dict[k] = Result([r[k] for r in trial_results])
    
    def __getitem__(self, key):
        return self._dict[key]

    def summary(self):
        s = {k: self._dict[k].summary() for k in self._dict.keys()}
        return s

    def as_json(self):
        return self._trial_results
            

class Experiment:
    def __init__(self, config, reps):
        self.config = config
        self.reps = reps
        self.task_factory = task_factories[config['task']](config)

    def run(self):
        measurements = []
        validations = []
        for i in range(self.reps):
            log('\nTRIAL {}'.format(i))
            trainer, model = self.task_factory.trainer_factory()    
            _, results, validation = trainer(model)
            measurements.append(results)
            validations.append(validation)
            log('\n')
        self.result = Results(measurements)
        self.validations = validations
        return self.result
    
    def return_experiment_data(self):
        return self.result.as_json(), self.result.summary(), self.validations

class ExperimentSequence:
    def __init__(self, configs_path, experiments, log_file, reps=1):
        self.configs_path = os.path.abspath(configs_path) # returns type str
        self.experiments = experiments
        self.reps = reps
        logger_config(log_file)
        # load checkpoint, set index to 0 if no checkpoint
        if os.path.exists("experiment_checkpoints.json"):
            with open("experiment_checkpoints.json", "r") as f:
                d = json.load(f)
                if self.configs_path in d:
                    self.start_i = d[self.configs_path]
                else:
                    self.start_i = 0
        else:
            self.start_i = 0
        # clear the log file
        if log_file is not None:
            with open(log_file, 'w') as f:
                pass
    
    @classmethod
    def from_json(cls, configs_path, log_path=None, reps=1):
        with open(configs_path, 'r') as f:
            configs = json.load(f)
        experiments = []
        for config in configs:
            experiments.append(Experiment(config, reps))
        return cls(configs_path, experiments, log_path, reps)

    def run_and_save(self, result_path, valid_path):
        results = []
        valids = []
        for i, experiment in enumerate(self.experiments, start=self.start_i):
            try:
                experiment.run()
                result, summary, valid = experiment.return_experiment_data()
                results.append([experiment.config, result, summary])
                valids.append([experiment.config, valid])
                with open(result_path, 'w') as f:
                    json.dump(results, f)
                if valid_path is not None:
                    with open(valid_path, 'w') as f:
                        json.dump(valids, f)
                log('\n')
            except:
                if os.path.exists("experiment_checkpoints.json"):
                    with open("experiment_checkpoints.json", "a") as f:
                        d = json.load(f)
                        d[self.configs_path] = i
                        json.dump(d, f)
                else:
                    with open("experiment_checkpoints.json", "w") as f:
                        d = dict()
                        d[self.configs_path] = i
                        json.dump(d, f)

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="path to experiment configs", type=str)
    parser.add_argument("result_path", help="path where experimental results are stored", type=str)
    parser.add_argument("-vl", "--valid_path", help="path to which validation results are stored", type=str)
    parser.add_argument("-lg", "--log_path", help="path to the log file", type=str)
    parser.add_argument("-r", "--reps", help="number of repititions for each experiment", default=1, type=int)

    args = parser.parse_args()

    config_path = args.config_path
    output_path = args.result_path
    valid_path = args.valid_path if args.valid_path else None
    log_file = args.log_path if args.log_path else None
    reps = args.reps

    exp_seq = ExperimentSequence.from_json(config_path, log_file, reps)
    exp_seq.run_and_save(output_path, valid_path)
    
