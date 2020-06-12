import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from os.path import join
import torch
from torch import optim
from evaluate import evaluate, decode
from allwords.networks import AffineClassifier, DropoutClassifier
from allwords.util import cudaify, Logger
from allwords.wordsense import SenseTaggedSentences, SenseInstanceDataset, SenseInstanceLoader
from allwords.vectorize import DiskBasedVectorManager
from allwords.loss import NLLLossWithZones


def train_all_words_classifier(train_loader, dev_loader, logger):
    n_epochs = 10
    learning_rate = 0.001
    logger('Training classifier.\n')   
    input_size = 768 # TODO: what is it in general?
    output_size = train_loader.num_senses()
    net = AffineClassifier(input_size, output_size)
    #net = DropoutClassifier(input_size, 300, output_size)
    net = cudaify(net)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)    
    best_net = net
    best_acc = 0.0
    for epoch in range(n_epochs):      
        train_batches = train_loader.batch_iter()
        logger("  Epoch {} Accuracy = ".format(epoch))       
        running_loss = 0.0
        total_train_loss = 0.0
        net.train()
        loss = NLLLossWithZones() 
        for i, (_, _, evidence, response, zones) in enumerate(train_batches): 
            optimizer.zero_grad()
            outputs = net(evidence)
            loss_size = loss(outputs, response, zones)
            loss_size.backward()
            #torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
            optimizer.step()
            running_loss += loss_size.data.item()
            total_train_loss += loss_size.data.item()   
            if i % 1000 == 0:
                acc = evaluate(net, dev_loader)
                print(acc)                
        net.eval()
        acc = evaluate(net, dev_loader)
        if acc > best_acc:
            best_net = net
            best_acc = acc
        net.train()
        logger("{:.2f}\n".format(acc))
    logger("  ...best accuracy = {:.2f}".format(best_acc))
    return best_net



if __name__ == '__main__':
    batch_size = 16
    logger = Logger(verbose = True)
    data_dir = sys.argv[1]
    train_corpus_id = 'data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml'
    dev_corpus_id = 'data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.data.xml'
    filename = join(data_dir, 'raganato.json')
    st_sents = SenseTaggedSentences.from_json(filename, train_corpus_id)
    vecmgr = DiskBasedVectorManager(join(join(data_dir, 'vecs'), train_corpus_id))
    ds = SenseInstanceDataset(st_sents, vecmgr)
    dev_sents = SenseTaggedSentences.from_json(filename, dev_corpus_id) 
    dev_vecmgr = DiskBasedVectorManager(join(join(data_dir, 'vecs'), dev_corpus_id))
    dev_ds = SenseInstanceDataset(dev_sents, dev_vecmgr)
    train_loader = SenseInstanceLoader(ds, batch_size)
    dev_loader = SenseInstanceLoader(dev_ds, batch_size)
    net = train_all_words_classifier(train_loader, dev_loader, logger)  
    predictions = decode(net, dev_loader)
    results = []
    for inst_id, target, predicted_sense_index, _ in predictions:
        results.append((inst_id, st_sents.inventory.sense(predicted_sense_index)))
    with open('foo.txt', 'w') as writer:
        for (inst_id, sense) in results:
            writer.write('{} {}\n'.format(inst_id, sense))
            