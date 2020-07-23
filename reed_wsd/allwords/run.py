import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from os.path import join
import torch
from torch import optim
from reed_wsd.allwords.evaluate import evaluate, decode_gen, decode_BEM
from reed_wsd.allwords.networks import AffineClassifier, DropoutClassifier, BEMforWSD
from reed_wsd.util import cudaify, Logger
from reed_wsd.allwords.wordsense import SenseTaggedSentences, SenseInstanceDataset, SenseInstanceLoader
from reed_wsd.allwords.blevins import BEMDataset, BEMLoader
from reed_wsd.allwords.vectorize import DiskBasedVectorManager
from reed_wsd.allwords.loss import NLLLossWithZones
from tqdm import tqdm
import copy

file_dir = os.path.dirname(os.path.realpath(__file__))

def FNN_batch_trainer(net, train_loader, optimizer, loss, max_grad_norm):
    running_loss = 0.0
    batch_iter = train_loader.batch_iter()
    batch_iter = tqdm(batch_iter, total=len(train_loader))
    for i, (_, _, evidence, response, zones) in enumerate(batch_iter): 
        optimizer.zero_grad()
        outputs = net(cudaify(evidence))
        loss_size = loss(outputs, response, zones)
        loss_size.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
        optimizer.step()
        running_loss += loss_size.data.item()
    return loss_size

def BEM_batch_trainer(net, train_loader, optimizer, loss, max_grad_norm):
    running_loss = 0.0
    batch_iter = train_loader.batch_iter()
    batch_iter = tqdm(batch_iter, total=len(train_loader))
    for batch in batch_iter:
       	contexts = batch['contexts']
       	glosses = batch['glosses']
       	span = batch['span']
       	gold = batch['gold'] 
        scores = net(contexts, glosses, span)
        loss_size = loss(scores, cudaify(torch.tensor(gold)))
        loss_size.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
        optimizer.step()
        running_loss += loss_size.data.item()
    return running_loss

def train_all_words_classifier(net, train_loader, dev_loader, loss, optimizer, batch_trainer, decoder, n_epochs, max_grad_norm=1.0, logger=Logger(verbose=True)):
    logger('Training classifier.\n')   
    best_net = net
    best_acc = 0.0
    net = cudaify(net)
    for epoch in range(n_epochs): 
        logger("  Epoch {} Accuracy = ".format(epoch)) 
        net.train()
        net.zero_grad() #reset the grads
        running_loss = batch_trainer(net, train_loader, optimizer, loss, max_grad_norm)
        net.eval()
        acc = evaluate(net, dev_loader, decoder)
        if acc > best_acc:
            best_net = copy.deepcopy(net)
            best_acc = acc
        logger("{:.2f}\n".format(acc))
        logger("    Running Loss = {:.3f}\n".format(running_loss))
    logger("The best dev acc is {:.3f}\n".format(best_acc))
    return best_net, acc

def init_loader(data_dir, stage, style, batch_size = 16, sense_sz=-1, gloss='defn_cls'):
    assert(stage == "train" or stage == "dev")
    assert(style == "bem" or stage == "fnn")
    if gloss is not None:
        assert(style == "bem")
    if stage == "dev":
        corpus_id = 'data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.data.xml'
    elif stage == "train":
        corpus_id = 'data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml'

    filename = join(data_dir, 'raganato.json')
    sents = SenseTaggedSentences.from_json(filename, corpus_id) 
    if style == "bem":
        ds = BEMDataset(sents, sense_sz=sense_sz, gloss=gloss)
        loader = BEMLoader(ds, batch_size)
    if style == 'fnn':
        vecmgr = DiskBasedVectorManager(join(join(data_dir, 'vecs'), corpus_id))
        ds = SenseInstanceDataset(sents, vecmgr)
        loader = SenseInstanceLoader(ds, batch_size)
    return loader

if __name__ == "__main__":
    net = BEMforWSD(True)
    print(type(net))
    batch_trainer = BEM_batch_trainer
    train_loader = init_loader("./data", stage="train", style="bem", batch_size=4, sense_sz=-1, gloss='defn_cls')
    inv = train_loader.get_inventory()
    dev_loader = init_loader("./data", stage="dev", style="bem", batch_size=4)
    dev_loader.set_inventory(inv)
    loss = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(net.parameters(), lr=5 * 10**(-5))
    decoder = decode_BEM
    n_epochs = 10

    best_net, acc = train_all_words_classifier(net, train_loader, dev_loader, loss, optimizer, batch_trainer, decoder, n_epochs)
    with open("trained_models/bem.pt", "w") as f:
        torch.save(best_net.state_dict(), 'trained_models/bem.pt')

