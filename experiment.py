# -*- coding: utf-8 -*-
import torch
import pandas as pd
from train import train_net
from networks import SimpleClassifier, DropoutClassifier, BertForSenseDisambiguation
from util import cudaify
from lemmas import all_sense_histograms, sample_sense_pairs, sample_inputids_pairs, sample_sense_pairs_with_vec, sample_cross_lemma
from compare import getExampleSentencesBySense
from collections import defaultdict
from pytorch_transformers import BertConfig, BertTokenizer, BertModel

def tensor_batcher(t, batch_size):
    def shuffle_rows(a):
        return a[torch.randperm(a.size()[0])]        
    neg = t[(t[:, 0] == 0).nonzero().squeeze(1)] # only negative rows
    pos = t[(t[:, 0] == 1).nonzero().squeeze(1)] # only positive rows
    neg = shuffle_rows(neg)
    pos = shuffle_rows(pos)
    min_row_count = min(neg.shape[0], pos.shape[0])
    epoch_data = torch.cat([neg[:min_row_count], pos[:min_row_count]])    
    epoch_data = shuffle_rows(epoch_data)
    for i in range(0, len(epoch_data), batch_size):
        yield epoch_data[i:i+batch_size]
        
def test_training(d, k):
    def nth_dim_positive_data(n, d, k):
        data = torch.randn(d, k)
        u = torch.cat([torch.clamp(torch.sign(data[2:3]), min=0), data])
        return u.t()

    train = nth_dim_positive_data(2, d, k)
    dev = nth_dim_positive_data(2, d, 500)
    #test = nth_dim_positive_data(2, d, 500)   
    classifier = SimpleClassifier(d,100,2)
    train_net(classifier, train, dev, tensor_batcher,
              batch_size=96, n_epochs=30, learning_rate=0.001,
              verbose=True)


def create_and_train_net(net, training_data, test_data, verb):
    training_data = cudaify(training_data)
    test_data = cudaify(test_data)
    if verb:
        print("training size:", training_data.shape)
        print("testing size:", test_data.shape)
    classifier = cudaify(net)
    best_net, best_acc = train_net(classifier, training_data, test_data, tensor_batcher,
                batch_size=96, n_epochs=30, learning_rate=0.001,
                verbose=verb)
    return best_acc
    
def train_from_csv(train_csv, dev_csv):
    print('loading train')
    train = torch.tensor(pd.read_csv(train_csv).values).float()
    print('train size: {}'.format(train.shape[0]))
    print('loading dev')
    dev = torch.tensor(pd.read_csv(dev_csv).values).float()
    print('dev size: {}'.format(dev.shape[0]))
    return create_and_train_net(DropoutClassifier(1536, 100, 2), train, dev)

def train_lemma_classifiers(min_sense2_freq, max_sense2_freq, n_fold, max_sample_size, verbose=True):
    lemma_info_dict = defaultdict(tuple)
    for (lemma, sense_hist) in all_sense_histograms():
        if len(sense_hist) > 1 and sense_hist[1][0] >= min_sense2_freq and sense_hist[1][0] <= max_sense2_freq:
            sense1 = sense_hist[0][1]
            sense2 = sense_hist[1][1]   
            print(lemma)                    
            data = sample_sense_pairs(max_sample_size//2, lemma, sense1, sense2, n_fold)

            sum_acc = 0
            fold_count = 0
            for training_data, test_data in data:
                sum_acc += create_and_train_net(DropoutClassifier(1536, 100, 2), training_data, test_data, verbose)
                fold_count += 1
            avg_acc = sum_acc / fold_count
            lemma_info_dict[lemma] = (avg_acc, sense1, sense2)
            print("  Best Epoch Accuracy Average = {:.2f}".format(avg_acc))
    return dict(lemma_info_dict)

def train_lemma_classifiers_with_vec(vectorization, min_sense2_freq, max_sense2_freq, n_fold, max_sample_size, verbose=True):
    lemma_info_dict = defaultdict(tuple)
    for (lemma, sense_hist) in all_sense_histograms():
        if len(sense_hist) > 1 and sense_hist[1][0] >= min_sense2_freq and sense_hist[1][0] <= max_sense2_freq:
            sense1 = sense_hist[0][1]
            sense2 = sense_hist[1][1]   
            print(lemma)                    
            data = sample_sense_pairs_with_vec(vectorization, max_sample_size//2, lemma, sense1, sense2, n_fold)

            sum_acc = 0
            fold_count = 0
            for training_data, test_data in data:
                sum_acc += create_and_train_net(DropoutClassifier(1536, 100, 2), training_data, test_data, verbose)
                fold_count += 1
            avg_acc = sum_acc / fold_count
            lemma_info_dict[lemma] = (avg_acc, sense1, sense2)
            print("  Best Epoch Accuracy Average = {:.2f}".format(avg_acc))
    return dict(lemma_info_dict)    


def train_finetune(min_sense2_freq, max_sense2_freq, n_fold, max_sample_size, verbose=True):
    config = BertConfig.from_pretrained('bert-base-uncased')
    config.output_hidden_states = True

    lemma_info_dict = defaultdict(tuple)
    for (lemma, sense_hist) in all_sense_histograms():
        if len(sense_hist) > 1 and sense_hist[1][0] >= min_sense2_freq and sense_hist[1][0] <= max_sense2_freq:
            sense1 = sense_hist[0][1]
            sense2 = sense_hist[1][1]   
            print(lemma)                    
            
            data = sample_inputids_pairs(max_sample_size//2, lemma, sense1, sense2, n_fold)

            sum_acc = 0
            fold_count = 0
            for training_data, test_data in data:
                sum_acc += create_and_train_net(BertForSenseDisambiguation(config), training_data, test_data, verbose)
                fold_count += 1
            avg_acc = sum_acc / fold_count

            lemma_info_dict[lemma] = (avg_acc, sense1, sense2)
            print("  Best Epoch Accuracy Average = {:.2f}".format(avg_acc))
    return dict(lemma_info_dict)

def train_cross_lemmas(threshold, n_fold, n_pairs_per_lemma, verbose=True):
    config = BertConfig.from_pretrained('bert-base-uncased')
    config.output_hidden_states = True

    data = sample_cross_lemma(threshold, n_fold, n_pairs_per_lemma)
    sum_acc = 0
    for training_data, test_data in data:
        sum_acc += create_and_train_net(DropoutClassifier(1536, 100, 2), training_data, test_data, verbose)
    avg_acc = sum_acc / n_fold
    print("  Best Epoch Accuracy Average = {:.2f}".format(avg_acc))
    with open("generality_result.txt", "w") as f:
        f.write("accuracy across lemmas is: " + str(avg_acc))

def train_with_neighbors(specification, threshold, n_fold, max_sample_size, verbose=True):
    specification_space = ["avg_both", "avg_left", "avg_right", "concat_both", "concat_left", "concat_right"]
    assert specification in specification_space, "parameter specification can only be one of the following: " + str(specification_space)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    config = BertConfig.from_pretrained('bert-base-uncased')
    config.output_hidden_states = True
    bert = BertModel.from_pretrained('bert-base-uncased')

    def vectorize_avg_both(instance):
        position = instance.pos + 1
        tokens = instance.tokens
        tokens = ["CLS"] + tokens + ["SEP"]
        input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)
        results = bert(input_ids)
        final_hidden_layers = results[0].squeeze(0)
        prev_token_vec = final_hidden_layers[position - 1]
        token_vec = final_hidden_layers[position]
        next_token_vec = final_hidden_layers[position + 1]
        avged = torch.mean(torch.stack([prev_token_vec, token_vec, next_token_vec]), dim=0)
        return avged.detach()
    
    def vectorize_avg_left(instance):
        tokens = instance.tokens
        tokens = ["CLS"] + tokens + ["SEP"]
        position = instance.pos + 1
        input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)
        results = bert(input_ids)
        final_hidden_layers = results[0].squeeze(0)
        prev_token_vec = final_hidden_layers[position - 1]
        token_vec = final_hidden_layers[position]
        avged = torch.mean(torch.stack([prev_token_vec, token_vec]), dim=0)
        return avged.detach()
    
    def vectorize_avg_right(instance):
        tokens = instance.tokens
        position = instance.pos + 1
        tokens = ["CLS"] + tokens + ["SEP"]
        input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)
        results = bert(input_ids)
        final_hidden_layers = results[0].squeeze(0)
        token_vec = final_hidden_layers[position]
        next_token_vec = final_hidden_layers[position + 1]
        avged = torch.mean(torch.stack([token_vec, next_token_vec]), dim=0)
        return avged.detach()
    
    def vectorize_concat_both(instance):
        position = instance.pos + 1
        tokens = instance.tokens
        tokens = ["CLS"] + tokens + ["SEP"]
        input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)
        results = bert(input_ids)
        final_hidden_layers = results[0].squeeze(0)
        prev_token_vec = final_hidden_layers[position - 1]
        token_vec = final_hidden_layers[position]
        next_token_vec = final_hidden_layers[position + 1]
        concat = torch.cat([prev_token_vec, token_vec, next_token_vec])
        return concat.detach()

    def vectorize_concat_left(instance):
        position = instance.pos + 1
        tokens = instance.tokens
        tokens = ["CLS"] + tokens + ["SEP"]
        input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)
        results = bert(input_ids)
        final_hidden_layers = results[0].squeeze(0)
        prev_token_vec = final_hidden_layers[position - 1]
        token_vec = final_hidden_layers[position]
        concat = torch.cat([prev_token_vec, token_vec])
        return concat.detach()

    def vectorize_concat_right(instance):
        position = instance.pos + 1
        tokens = instance.tokens
        tokens = ["CLS"] + tokens + ["SEP"]
        input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)
        results = bert(input_ids)
        final_hidden_layers = results[0].squeeze(0)
        token_vec = final_hidden_layers[position]
        next_token_vec = final_hidden_layers[position + 1]
        concat = torch.cat([token_vec, next_token_vec])
        return concat.detach()
    
    def get_lemmas(threshold):
        data = pd.read_csv("data/classifier_data8_20-max.csv")
        lemmas = []
        for i in data.index:
            if data.iloc[i]["best_avg_acc"] >= threshold:
                lemmas.append(data.iloc[i]["lemma"])
        return lemmas

    if specification.startswith("avg"):
        net = DropoutClassifier(768*2, 100, 2)
        if specification == "avg_both": vectorization = vectorize_avg_both
        if specification == "avg_left": vectorization = vectorize_avg_left
        if specification == "avg_right": vectorization = vectorize_avg_right
    else:
        if not specification.endswith("both"):
            net = DropoutClassifier(768*2*2, 100, 2)
        else:
            net = DropoutClassifier(768*3*2, 100, 2)
        if specification == "concat_both": vectorization = vectorize_concat_both
        if specification == "concat_left": vectorization = vectorize_concat_left
        if specification == "concat_right": vectorization = vectorize_concat_right
    
    lemmas = get_lemmas(0.7)
    lemma_info_dict = defaultdict(tuple)
    for (lemma, sense_hist) in all_sense_histograms():
        if not lemma in lemmas: continue
        if len(sense_hist) > 1 and sense_hist[1][0] >= 21:
            sense1 = sense_hist[0][1]
            sense2 = sense_hist[1][1]
            print(lemma)
            
            data = sample_sense_pairs_with_vec(vectorization, max_sample_size//2, lemma, sense1, sense2, n_fold)

            sum_acc = 0
            fold_count = 0
            for training_data, test_data in data:
                if specification.startswith("avg"):
                    net = DropoutClassifier(768*2, 100, 2)
                else:
                    if not specification.endswith("both"):
                        net = DropoutClassifier(768*2*2, 100, 2)
                    else:
                        net = DropoutClassifier(768*3*2, 100, 2)
                sum_acc += create_and_train_net(net, training_data, test_data, verbose)
                fold_count += 1
            avg_acc = sum_acc / fold_count

            lemma_info_dict[lemma] = (avg_acc, sense1, sense2)
            print("  Best Epoch Accuracy Average = {:.2f}".format(avg_acc))
    return dict(lemma_info_dict)


def neighbors_test():
    spec_acc_dict = defaultdict(int)
    specification_space = ["avg_both", "avg_left", "avg_right", "concat_both", "concat_left", "concat_right"]
        
    with open("neighbor_test_result.json", "w") as f:
        f.write("")

    for spec in specification_space:
        print()
        print("training spec: " + spec)
        print()
        lemma_info_dict = train_with_neighbors(spec, 0.7, 10, 2000, verbose=True)
        score = 0
        for lemma in lemma_info_dict.keys():
            score += lemma_info_dict[lemma][0]
        score /= len(lemma_info_dict.keys())
        spec_acc_dict[spec] = score
        
        with open("neighbor_test_result.json", "w") as f:
            json.dump(spec_acc_dict, f)
    
    return spec_acc_dict

