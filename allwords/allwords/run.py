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


file_dir = os.path.dirname(os.path.realpath(__file__))

def train_all_words_classifier(train_loader, dev_loader, logger):
    n_epochs = 20
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
        loss = torch.nn.CrossEntropyLoss() 
        for i, (_, _, evidence, response, zones) in enumerate(train_batches): 
            optimizer.zero_grad()
            outputs = net(evidence)
            loss_size = loss(outputs, response)
            loss_size.backward()
            optimizer.step()
            running_loss += loss_size.data.item()
            total_train_loss += loss_size.data.item()   
            """
            if i % 100 == 0:
                acc = evaluate(net, dev_loader)
                print(acc)                
            """
        net.eval()
        acc = evaluate(net, dev_loader)
        if acc > best_acc:
            best_net = net
            best_acc = acc
        net.train()
        logger("{:.2f}\n".format(acc))
    logger("  ...best accuracy = {:.2f}".format(best_acc))
    return best_net

def init_dev_loader(data_dir, batch_size = 16):
    dev_corpus_id = 'data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.data.xml'
    filename = join(data_dir, 'raganato.json')
    dev_sents = SenseTaggedSentences.from_json(filename, dev_corpus_id) 
    dev_vecmgr = DiskBasedVectorManager(join(join(data_dir, 'vecs'), dev_corpus_id))
    dev_ds = SenseInstanceDataset(dev_sents, dev_vecmgr)
    dev_loader = SenseInstanceLoader(dev_ds, batch_size)
    return dev_loader


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
    torch.save(net.state_dict(), join(file_dir, "../saved/bert_simple.pt")) 
    predictions = decode(net, dev_loader) 
    results = []
    for inst_id, target, predicted_sense_index, _ in predictions:
        results.append((inst_id, st_sents.inventory.sense(predicted_sense_index)))
    with open('foo.txt', 'w') as writer:
        for (inst_id, sense) in results:
            writer.write('{} {}\n'.format(inst_id, sense))
    # TODO: move his to another file (say, confidence.py or somethings)
    """
    batch_size = 16
    data_dir = sys.argv[1]
    dev_corpus_id = 'data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.data.xml'
    filename = join(data_dir, 'raganato.json')
    dev_sents = SenseTaggedSentences.from_json(filename, dev_corpus_id) 
    dev_vecmgr = DiskBasedVectorManager(join(join(data_dir, 'vecs'), dev_corpus_id))
    dev_ds = SenseInstanceDataset(dev_sents, dev_vecmgr)
    dev_loader = SenseInstanceLoader(dev_ds, batch_size)
    input_size = 768 # TODO: what is it in general?
    output_size = dev_loader.num_senses()
    net = AffineClassifier(input_size, output_size)
    net.load_state_dict(torch.load(join(file_dir, "../saved/bert_simple.pt")))
    net = cudaify(net)
    confidence_analysis(net, dev_loader)
    """
