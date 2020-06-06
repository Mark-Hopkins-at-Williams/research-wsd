import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch import optim
from allwords.networks import AffineClassifier
from allwords.util import cudaify, Logger
from allwords.wordsense import SenseTaggedSentences, SenseInstanceDataset, SenseInstanceLoader
from allwords.vectorize import DiskBasedVectorManager


def decode(net, data):
    """
    Runs a trained neural network classifier on validation data, and iterates
    through the top prediction for each datum.
    
    """        
    net.eval()
    val_loader = data.batch_iter()
    for inst_ids, targets, evidence, response, spans in val_loader:#TODO: choose the max of valid entries
        val_outputs = net(evidence)
        revised = torch.empty(val_outputs.shape)
        revised = revised.fill_(-10000000)
        for row in range(len(spans)):
            (start, stop) = spans[row]
            revised[row][start:stop] = val_outputs[row][start:stop]
        for i, (inst_id, (target, output)) in enumerate(zip(inst_ids, zip(targets, revised))):        
            yield inst_id, target, output.argmax()
    

def evaluate(net, data):
    """

    The accuracy (i.e. percentage of correct classifications) is returned.
    
    """        
    def accuracy(outputs, labels):
        correct = 0
        total = 0
        for (i, output) in enumerate(outputs):
            total += 1
            if labels[i] == output.argmax():        
                correct += 1            
        return correct, total
    net.eval()
    correct = 0
    total = 0
    val_loader = data.batch_iter()
    for _, _, evidence, response, spans in val_loader: #TODO: choose the max of valid entries
        val_outputs = net(evidence)
        revised = torch.empty(val_outputs.shape)
        revised = revised.fill_(-10000000)
        for row in range(len(spans)):
            (start, stop) = spans[row]
            revised[row][start:stop] = val_outputs[row][start:stop]
        correct_inc, total_inc = accuracy(revised, response)
        correct += correct_inc
        total += total_inc
    return correct/total    


def train_all_words_classifier(train_loader, dev_loader, logger):
    n_epochs = 30
    learning_rate = 0.001
    logger('Training classifier.\n')   
    input_size = 768 # TODO: what is it in general?
    output_size = train_loader.num_senses()
    net = AffineClassifier(input_size, output_size)
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
        loss = torch.nn.NLLLoss() 
        for i, (_, _, evidence, response, zones) in enumerate(train_batches): 
            optimizer.zero_grad()
            outputs = net(evidence)
            loss_size = loss(outputs, response)
            loss_size.backward()
            optimizer.step()
            running_loss += loss_size.data.item()
            total_train_loss += loss_size.data.item()                          
        net.eval()
        acc = evaluate(net, dev_loader)
        if acc > best_acc:
            best_net = net
            best_acc = acc
        logger("{:.2f}\n".format(acc))
    logger("  ...best accuracy = {:.2f}".format(best_acc))
    return best_net



if __name__ == '__main__':
    batch_size = 16
    logger = Logger(verbose = True)
    train_corpus_id = 'data/raganato/Training_Corpora/SemCor/semcor.data.xml'
    dev_corpus_id = 'data/raganato/Evaluation_Datasets/semeval2007/semeval2007.data.xml'
    filename = '../data/raganato.json'
    st_sents = SenseTaggedSentences.from_json(filename, train_corpus_id)
    vecmgr = DiskBasedVectorManager('../vecs/bertvecs2')
    ds = SenseInstanceDataset(st_sents, vecmgr)
    dev_sents = SenseTaggedSentences.from_json(filename, dev_corpus_id) 
    dev_vecmgr = DiskBasedVectorManager('../vecs/bertvecs-se07')
    dev_ds = SenseInstanceDataset(dev_sents, dev_vecmgr)
    train_loader = SenseInstanceLoader(ds, batch_size)
    dev_loader = SenseInstanceLoader(dev_ds, batch_size)
    net = train_all_words_classifier(train_loader, dev_loader, logger)  
    predictions = decode(net, dev_loader)
    results = []
    for inst_id, target, predicted_sense_index in predictions:
        results.append((inst_id, st_sents.inventory.sense(predicted_sense_index)))
    with open('foo.txt', 'w') as writer:
        for (inst_id, sense) in results:
            writer.write('{} {}\n'.format(inst_id, sense))
            