import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from os.path import join
import torch
from torch import optim
from reed_wsd.allwords.evaluate import evaluate, decode_gen
from reed_wsd.allwords.networks import AffineClassifier, DropoutClassifier
from reed_wsd.util import cudaify, Logger
from reed_wsd.allwords.wordsense import SenseTaggedSentences, SenseInstanceDataset, SenseInstanceLoader
from reed_wsd.allwords.vectorize import DiskBasedVectorManager
from reed_wsd.allwords.loss import NLLLossWithZones

file_dir = os.path.dirname(os.path.realpath(__file__))

def train_all_words_classifier(net, train_loader, dev_loader, loss, n_epochs, logger, abstain):
    learning_rate = 0.001
    logger('Training classifier.\n')   
    input_size = 768 # TODO: what is it in general?
    output_size = train_loader.num_senses()
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
        for i, (_, _, evidence, response, zones) in enumerate(train_batches): 
            optimizer.zero_grad()
            outputs = net(cudaify(evidence))
            loss_size = loss(outputs, response, zones)
            loss_size.backward()
            optimizer.step()
            running_loss += loss_size.data.item()
            total_train_loss += loss_size.data.item()   
            if i % 1000 == 0: # change to True if you want detailed logging
                if abstain:
                    acc = evaluate(net, dev_loader, abs_class=output_size - 1)
                else:
                    acc = evaluate(net, dev_loader)
                print(acc)                
        net.eval()
        acc = evaluate(net, dev_loader, abs_class=output_size - 1)
        if acc > best_acc:
            best_net = net
            best_acc = acc
        net.train()
        logger("{:.2f}\n".format(acc))
    logger("  ...best accuracy = {:.2f}".format(best_acc))
    return best_net

def init_loader(data_dir, stage, batch_size = 16):
    assert(stage == "train" or stage == "dev")
    if stage == "dev":
        corpus_id = 'data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.data.xml'
    elif stage == "train":
        corpus_id = 'data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml'

    filename = join(data_dir, 'raganato.json')
    sents = SenseTaggedSentences.from_json(filename, corpus_id) 
    vecmgr = DiskBasedVectorManager(join(join(data_dir, 'vecs'), corpus_id))
    ds = SenseInstanceDataset(sents, vecmgr)
    loader = SenseInstanceLoader(ds, batch_size)
    return loader


"""
if __name__ == '__main__':
    batch_size = 16
    logger = Logger(verbose = True)
    data_dir = sys.argv[1]
    train_corpus_id = 'data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml'
    filename = join(data_dir, 'raganato.json')
    st_sents = SenseTaggedSentences.from_json(filename, train_corpus_id)
    vecmgr = DiskBasedVectorManager(join(join(data_dir, 'vecs'), train_corpus_id))
    ds = SenseInstanceDataset(st_sents, vecmgr)
    train_loader = SenseInstanceLoader(ds, batch_size)
    dev_loader = init_dev_loader(data_dir, batch_size)
    net = train_all_words_classifier(train_loader, dev_loader, logger)  
    save_path = join(file_dir, "saved")
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    torch.save(net.state_dict(), join(save_path, "bert_simple.py")) 
    predictions = decode(net, dev_loader) 
    results = []
    for inst_id, target, predicted_sense_index, _ in predictions:
        results.append((inst_id, st_sents.inventory.sense(predicted_sense_index)))
    with open('foo.txt', 'w') as writer:
        for (inst_id, sense) in results:
            writer.write('{} {}\n'.format(inst_id, sense))
"""

