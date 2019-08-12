from util import delete_last_line
<<<<<<< HEAD
import pprint
=======
import pandas as pd
import os
import random
import seaborn as sns
from compare import getExampleSentencesBySense
import matplotlib.pyplot as plt


#Note: 12 is the largest valid indice
def generate_random_layers_i(max_num_layers_to_average):
    """
    Randomly creates a list of layers to be averaged together. This function creates and
    returns a valid instance of the layers_i parameter from train_lemma_classifier_with_vec.
    Note it will also return a value for add_sent_encoding.
    """
    layers_to_average = random.randrange(1, max_num_layers_to_average+1)
    layers_i = []
    add_sent_encoding = False
    i = 0
    while i < layers_to_average:
        #If a value of 13 is generated it will be interpreted as add_sent_encoding=True
        layer = random.randrange(14)
        if layer == 13 and add_sent_encoding == False:
            add_sent_encoding = True
        elif not layer in layers_i:
            layers_i.append(layer)
        else:
            continue
        i += 1
    return layers_i, add_sent_encoding


def train_lemma_classifier_with_vec(layers_i, min_sense2_freq, max_sense2_freq, n_fold, max_sample_size, 
                                    verbose=True, add_sent_encoding=False):
    """
    train the lemma classifier with a specified vectorization patter

    layers_i is a list of indices that specifies the layers of output the vectorization is going to average
    add_sent_encoding is a boolean specifying whether including the sentence vector in the averaging

    For the rest of the parameters see train_lemma_classifiers in experiment.py
    """
    with open("bert.py", "a") as f:
        f.write("\nvectorize_instance = generate_vectorization(" + str(layers_i) + ", " +  str(add_sent_encoding) + ")")
    try:
        from experiment import train_lemma_classifiers
        lemma_info_dict = train_lemma_classifiers(min_sense2_freq, max_sense2_freq, n_fold, max_sample_size, verbose)
    finally:
        delete_last_line("bert.py")
    if add_sent_encoding == True:
            layers_i.append(True)
    return lemma_info_dict, layers_i

def test_single_layer(min_sense2, max_sense2):
    layers = []
    for i in range(13):
        curr_dict = train_lemma_classifier_with_vec([i], min_sense2, max_sense2, 10, 600, verbose = False)
        accs = [lemma[0] for lemma in list(curr_dict.values())]
        layers.append(accs)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(layers)
    return layers
    
def train_lemma_classifier_with_diff_layers(max_layers_to_average, num_layer_combos, min_sense2_freq, max_sense2_freq, n_fold, max_sample_size):
    """
    Calls train_lemma_classifier with a variety of values for layers_i and add_sent_encoding.
    It then writes the relevant data to a file.
    """
    layers_i_list = [ ([x], False) for x in range(1)]
    i = 0
    while i < num_layer_combos:
        layers_i_tuple = generate_random_layers_i(max_layers_to_average)
        if not layers_i_tuple in layers_i_list:
            layers_i_list.append(layers_i_tuple)
        else:
            continue
        i += 1
    
    data = []
    for layers_i_tuple in layers_i_list:
        lemma_info_dict, layers_i = train_lemma_classifier_with_vec(layers_i_tuple[0], min_sense2_freq, max_sense2_freq, n_fold, max_sample_size, 
                                        verbose=True, add_sent_encoding=layers_i_tuple[1])
        for key in lemma_info_dict.keys():
            lemma_info = lemma_info_dict[key]
            data.append([layers_i, key, lemma_info[0], lemma_info[1], lemma_info[2]])
    df = pd.DataFrame(data, columns=["spec", "lemma", "best_avg_acc", "sense1", "sense2"])

    num = 1
    while os.path.exists("classifier_data"+str(num)+".csv"):
        num += 1

    df.to_csv("classifier_data"+str(num)+".csv", index=False)

def store_csv_DF_from_lemma_classifier(lemma_info_dict, layers_i):
    """
    Takes the return values from the lemma_info_dict and converts them to 
    a pandas df before writing them to a file for later use.
    """
    
    data = []
    for key in lemma_info_dict.keys():
        lemma_info = lemma_info_dict[key]
        data.append([layers_i, key, lemma_info[0], lemma_info[1], lemma_info[2]])
    df = pd.DataFrame(data, columns=["spec", "lemma", "best_avg_acc", "sense1", "sense2"])
    
    num = 1
    while os.path.exists("classifier_data"+str(num)+".csv"):
        num += 1

    df.to_csv("classifier_data"+str(num)+".csv", index=False)

def present_csv_DF_data(good_threshold, bad_threshold, num_example_sentences):
    """
    num_lemmas controls the total number of lemmas that will be included
    good_threshold controls the minimum average accuracy a lemma must have to 
    """
    
    
    
    df = pd.read_csv("classifier_data3.csv")
    sns.set(style="whitegrid")

    specs = sns.barplot(x="spec", y="best_avg_acc", data=df)
    plt.xticks(rotation=45)
    plt.show()

    lemmas = sns.barplot(x="lemma", y="best_avg_acc", data=df)
    plt.xticks(rotation=45)
    plt.show()


    df = pd.read_csv("classifier_data4.csv")
    
    best = df[df["best_avg_acc"] >= 0.85]
    print(best.values.tolist())
    worst = df[df["best_avg_acc"] <= 0.54]
    combined = pd.concat([best, worst])
    comb_lemmas = sns.barplot(x="lemma", y="best_avg_acc",data=combined)
    plt.xticks(rotation=45)
    plt.show()
    
    df_list = combined.values.tolist()

    def printSentences(sents):
        for sent in sents:
            print(sent)
        print()

    while True:
        lemma = input("\nenter a lemma to see its sentences:\n")
        for row in df_list:
            if row[1] == lemma:
                printSentences(getExampleSentencesBySense(row[3], 3))
                printSentences(getExampleSentencesBySense(row[4], 3))





if __name__ == "__main__":
    present_csv_DF_data(1,1,1)
    #train_lemma_classifier_with_diff_layers(1, 0, 40, 100, 10, 1000)
