from reed_wsd.allwords.model import SimpleFFN, SimpleAbstainingFFN, BEMforWSD
from reed_wsd.mnist.model import BasicFFN, AbstainingFFN
from reed_wsd.allwords.wordsense import SenseInstanceDataset, SenseTaggedSentences, SenseInstanceLoader, TwinSenseInstanceLoader
from reed_wsd.allwords.vectorize import DiskBasedVectorManager
from reed_wsd.allwords.blevins import BEMDataset, BEMLoader
from reed_wsd.mnist.train import SimpleDecoder as MnistSimpleDecoder
from reed_wsd.mnist.train import AbstainingDecoder as MnistAbstainingDecoder
from reed_wsd.allwords.evaluate import AllwordsSimpleEmbeddingDecoder, AllwordsAbstainingEmbeddingDecoder, AllwordsBEMDecoder
from reed_wsd.loss import CrossEntropyLoss, NLLLoss, ConfidenceLoss1, ConfidenceLoss4, PairwiseConfidenceLoss
from reed_wsd.allwords.train import SingleEmbeddingTrainer, PairwiseEmbeddingTrainer, BEMTrainer
from reed_wsd.mnist.train import MnistSingleTrainer, MnistPairwiseTrainer
from reed_wsd.mnist.loader import ConfusedMnistLoader, MnistLoader, MnistPairLoader, ConfusedMnistPairLoader
import os
from os.path import join
import torch.optim as optim
from torchvision import datasets, transforms

file_dir = os.path.dirname(os.path.realpath(__file__))
mnist_dir = join(file_dir, 'mnist')
allwords_dir = join(file_dir, 'allwords')
allwords_data_dir = join(allwords_dir, 'data')
mnist_data_dir = join(mnist_dir, 'data')
mnist_train_dir = join(mnist_data_dir, 'train')
mnist_test_dir = join(mnist_data_dir, 'test')

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
                    'conf1': ConfidenceLoss1,
                    'conf4': ConfidenceLoss4,
                    'pairwise': PairwiseConfidenceLoss}

class TaskFactory:
    def __init__(self, config):
        #check dependency
        assert(config['task'] in ['mnist', 'allwords'])
        if config['task'] == 'mnist':
            assert(config['architecture'] != 'bem')
        if config['architecture'] == 'simple':
            assert(config['confidence'] == 'max_prob')
            assert(config['criterion']['name'] == 'nll')
            assert(config['style'] == 'single')
        if config['architecture'] == 'abstaining':
            assert(config['confidence'] in ['max_non_abs', 'inv_abs', 'abs'])
            assert(config['criterion']['name'] in ['conf1', 'conf4', 'pairwise'])
        if config['architecture'] == 'bem':
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
            criterion = criterion_lookup[config['name']](p0=config['p0'], warmup_epochs=config['warmup_epochs'])
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
        n_epochs = config['n_epochs']
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
        
        self._model_lookup = {'simple': SimpleFFN,
                                 'abstaining': SimpleAbstainingFFN,
                                 'bem': BEMforWSD}
        self._decoder_lookup = {'simple': AllwordsSimpleEmbeddingDecoder,
                                 'abstaining': AllwordsAbstainingEmbeddingDecoder,
                                 'bem': AllwordsBEMDecoder}
        assert(self.config['task'] == 'allwords')


    @staticmethod
    def init_loader(stage, architecture, style, corpus_id):
        data_dir = allwords_data_dir
        filename = join(data_dir, 'raganato.json')
        sents = SenseTaggedSentences.from_json(filename, corpus_id)
        if architecture == "bem":
            ds = BEMDataset(sents)
            loader = BEMLoader(ds, config['bsz'])
        if architecture == 'simple' or architecture == 'abstaining': 
            vecmgr = DiskBasedVectorManager(join(join(data_dir, 'vecs'), corpus_id))
            ds = SenseInstanceDataset(sents, vecmgr)
            if stage == 'train':
                if style == 'single':
                    loader = SenseInstanceLoader(ds, batch_size=config['bsz'])
                if style == 'pairwise':
                    loader = TwinSenseInstanceLoader(ds, batch_size=config['bsz'])
            if stage == 'test':
                loader = SenseInstanceLoader(ds, batch_size=config['bsz'])
        return loader
        
    def train_loader_factory(self):
        if config['architecture'] == 'bem' or config['architecture'] == 'simple':
            assert(config['style'] == 'single')
        return AllwordsTaskFactory.init_loader('train', 
                                    self.config['architecture'], 
                                    self.config['style'], 
                                    corpus_id_lookup['semcor'])

    def val_loader_factory(self):
        if self.config['architecture'] == 'bem' or self.config['architecture'] == 'simple':
            assert(self.config['style'] == 'single')
        return AllwordsTaskFactory.init_loader('test', 
                                    self.config['architecture'], 
                                    self.config['style'], 
                                    corpus_id_lookup[config['dev_corpus']])

    def decoder_factory(self):
        return self._decoder_lookup[self.config['architecture']]()
    
    def model_factory(self, data):
        if config['architecture'] == 'bem':
            model = self._model_lookup[self.config['architecture']](gpu=True)
        else:
            model = self._model_lookup[self.config['architecture']](input_size=768,
                                                                  output_size=data.num_senses(),
                                                                  zone_applicant=self.config['confidence'])
        return model
 
    def optimizer_factory(self, model):
        return optim.Adam(model.parameters(), lr=0.001)
 
    def select_trainer(self):
        if config['architecture'] == 'bem':
            trainer = BEMTrainer
        else:
            if config['style'] == 'single':
                trainer = SingleEmbeddingTrainer
            if config['style'] == 'pairwise':
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
    def init_loader(stage, style, confused, bsz):
        if stage == 'train':
            ds = datasets.MNIST(mnist_train_dir, download=True, train=True, transform=transform)
            if style == 'single':
                if confused:
                    loader = ConfusedMnistLoader(ds, bsz, shuffle=True)
                if not confused:
                    loader = MnistLoader(ds, bsz, shuffle=True)
            if style == 'pairwise':
                if confused:
                    loader = ConfusedMnistPairLoader(ds, bsz, shuffle=True)
                if not confused:
                    loader = MnistPairLoader(ds, bsz, shuffle=True)
        if stage == 'test':
            ds = datasets.MNIST(mnist_test_dir, download=True, train=False, transform=transform)
            if confused:
                loader = ConfusedMnistLoader(ds, bsz, shuffle=True)
            else:
                loader = MnistLoader(ds, bsz, shuffle=True)
        return loader
        
    def train_loader_factory(self):
        return MnistTaskFactory.init_loader('train', 
                                            self.config['style'], 
                                            self.config['confused'], 
                                            self.config['bsz'])
    
    def val_loader_factory(self):
        return MnistTaskFactory.init_loader('test', 
                                            self.config['style'], 
                                            self.config['confused'], 
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

task_factories = {'mnist': MnistTaskFactory,
                  'allwords': AllwordsTaskFactory}

def run_experiment(config):
    task_factory = task_factories[config['task']](config)
    trainer, model = task_factory.trainer_factory()
    best_model = trainer(model)
    return best_model

if __name__ == "__main__":
    config =  {'task': 'allwords',
	       'architecture': 'abstaining',
	       'confidence': 'inv_abs',
               'criterion': { 'name': 'conf4', 'p0': 0.5, 'warmup_epochs': 2},
	       'confused': None,
	       'style': 'single',
	       'dev_corpus': 'semev07',
	       'bsz': 16,
	       'n_epochs': 20
	     }    
    
    run_experiment(config)
