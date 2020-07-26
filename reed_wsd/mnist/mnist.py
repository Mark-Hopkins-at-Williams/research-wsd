"""
Code adapted from:
https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627

"""

import torch
import copy
from time import time
from torch import optim
from reed_wsd.util import cudaify
from collections import defaultdict
from reed_wsd.plot import PYCurve, plot_curves
from reed_wsd.mnist.networks import ConfidentFFN, PairedConfidentFFN


def decoder(net, data):
    net.eval()
    for images, labels in data:
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            # Turn off gradients to speed up this part
            with torch.no_grad():
                ps, conf = net(cudaify(img))                
            ps = ps.squeeze(dim=0)
            c = conf.squeeze(dim=0).item()
            pred = ps.argmax(dim=0).item()
            gold = labels[i].item()
            yield {'pred': pred, 'gold': gold, 'confidence': c}
    
def validate_and_analyze(model, val_loader, output_size = 11):
    results = decoder(model, val_loader)
    correct_count, n_confident = 0, 0
    avg_err_conf = 0
    avg_crr_conf = 0
    n_error = 0
    data_dict = {}
    error_dict = defaultdict(int)
    for i in range(output_size):
        data_dict[i] = [0, 0, 0] # [n_correct, n_wrong, n_abstain]
    for result in results:
        prediction = result['pred']
        gold = result['gold']
        confidence = result['confidence']
        if prediction == gold:
            data_dict[gold][0] += 1
            avg_crr_conf += confidence
            correct_count += 1
            n_confident += 1            
        else:
            #print("mistook {} for {}".format(gold, prediction))
            data_dict[gold][1] += 1
            avg_err_conf += confidence
            error_dict[gold] += 1
            n_confident += 1            
            n_error += 1            
    for lbl in error_dict:
        error_dict[lbl] /= n_error
    print('average error confidence:', avg_err_conf / n_error)
    print('average correct confidence:', avg_crr_conf / correct_count)
    return data_dict, correct_count / n_confident




def train_pair(criterion, train_loader, val_loader, model_file, n_epochs = 20):
    model = PairedConfidentFFN()
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    time0 = time()
    best_model = None
    best_model_score = float('-inf')
    for e in range(n_epochs):
        running_loss = 0
        denom = 0.
        batch_iter = train_loader.batch_iter()
        for img_x, img_y, lbl_x, lbl_y in batch_iter:
            optimizer.zero_grad()                           
            output_x, conf_x = model(cudaify(img_x))
            output_y, conf_y = model(cudaify(img_y))
            loss = criterion(output_x, output_y, cudaify(lbl_x), cudaify(lbl_y), conf_x, conf_y)
            loss.backward()
            optimizer.step()                                                                                                 
            running_loss += loss.item()
            denom += 1
        _, precision = validate_and_analyze(model, val_loader)
        if precision > best_model_score:
            print("Updating best model.")
            best_model = copy.deepcopy(model)
            best_model_score = precision
        print("Epoch {} - Training loss: {}; Dev precision: {}".format(e, 
                                                                      running_loss/denom,
                                                                      precision))
        print("\nTraining Time (in minutes) =",(time()-time0)/60)
    torch.save(best_model.state_dict(), model_file)
    return best_model
    
def train(criterion, train_loader, val_loader, model_file, n_epochs = 10):
    model = ConfidentFFN()
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    time0 = time()
    best_model = None
    best_model_score = float('-inf')
    for e in range(n_epochs):
        criterion.notify(e)
        running_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()                       
            output, conf = model(cudaify(images))
            loss = criterion(output, cudaify(labels))
            loss.backward()
            optimizer.step()                                                                                             
            running_loss += loss.item()
        _, precision = validate_and_analyze(model, val_loader)
        if precision > best_model_score:
            print("Updating best model.")
            best_model = copy.deepcopy(model)
            best_model_score = precision
        print("Epoch {} - Training loss: {}; Dev precision: {}".format(e, 
                                                                      running_loss/len(train_loader),
                                                                      precision))
        print("\nTraining Time (in minutes) =",(time()-time0)/60)    
    torch.save(best_model.state_dict(), model_file)
    return best_model




def plot_saved_models(val_loader, 
                      filenames = ['saved/pair_baseline.pt', 
                                   'saved/pair_neg_abs.pt']):
    curves = []
    for filename in filenames:
        net_base = ConfidentFFN()
        net_base.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
        decoded = list(decoder(net_base, val_loader))
        pyc_base = PYCurve.from_data(decoded)
        curves.append((pyc_base, filename))
    plot_curves(*curves)
    
