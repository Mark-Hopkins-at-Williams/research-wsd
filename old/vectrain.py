from util import delete_last_line
import pprint
import pandas as pd
import os
import random
import seaborn as sns
from compare import getExampleSentencesBySense
import matplotlib.pyplot as plt
import json
from lemmas import lemmadata_iter
from  bert import generate_vectorization
from elmo import elmo_vectorize


#Note: 12 is the largest valid indice



def train_lemma_classifier_with_vec(layers_i, add_sent, min_sense2_freq, max_sense2_freq, n_fold, max_sample_size, 
                                    verbose=True):
    """
    train the lemma classifier with a specified vectorization pattern

    layers_i is a list of indices that specifies the layers of output the vectorization is going to average
    add_sent_encoding is a boolean specifying whether including the sentence vector in the averaging

    For the rest of the parameters see train_lemma_classifiers in experiment.py
    """
    assert isinstance(layers_i, list), "layers_i must be a list, instead got " + str(type(layers_i))
    vectorize = generate_vectorization(layers_i, add_sent)
    lemma_info_dict = train_lemma_classifiers_with_vec(vectorize, min_sense2_freq, max_sense2_freq, n_fold, max_sample_size, verbose)
    return lemma_info_dict, layers_i

def train_lemma_classifiers_with_elmo(min_sense2_freq, max_sense2_freq, n_fold, max_sample_size):
    vectorize = elmo_vectorize
    lemma_info_dict = train_lemma_classifiers_with_vec(vectorize, min_sense2_freq, max_sense2_freq, n_fold, max_sample_size)
    
    for key in lemma_info_dict.keys():
        lemma_info = lemma_info_dict[key]
        data.append(["elmo", key, lemma_info[0], lemma_info[1], lemma_info[2]])
    df = pd.DataFrame(data, columns=["spec", "lemma", "best_avg_acc", "sense1", "sense2"])

    num = 1
    while os.path.exists("elmo_2000_"+str(num)+".csv"):
        num += 1
    df = update_df_format(df, max_sample_size)
    df.to_csv("elmo_2000"+str(num)+".csv", index=False)



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

    df.to_csv("classifier_datat"+str(num)+".csv", index=False)

def get_list_learnable_lemmas(good_threshold, file_name):
    """
    Returns a list containing the filenames of all the lemmas at or above the
    specified accuracy threshold in the specified data_file. 
    Note: The only valid datafiles are the classifier_data* files.
    """
    df = pd.read_csv(file_name)
    df = df[df["best_avg_acc"] >= good_threshold]
    df = df["lemma"]
    lemma_list = df.values.tolist()
    return lemma_list

def present_csv_DF_data(good_threshold, bad_threshold, num_example_sentences):
    """
    num_lemmas controls the total number of lemmas that will be included
    good_threshold controls the minimum average accuracy a lemma must have to 
    """
    
    df = pd.read_csv("classifier_data_spec8.csv")
    sns.set(style="whitegrid")

    specs = sns.barplot(x="spec", y="best_avg_acc", data=df)
    plt.xticks(rotation=45)
    plt.show()

    lemmas = sns.barplot(x="lemma", y="best_avg_acc", data=df)
    plt.xticks(rotation=45)
    plt.show()


    df = pd.read_csv("classifier_data8_20-max.csv")
    
    best = df[df["best_avg_acc"] >= 0.7]
    worst = df[df["best_avg_acc"] <= 0.53]
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
                
def getLemmaPosLists(filename):
    df = pd.read_csv(filename)
    print(len(df))
    with open("data/sense_to_pofs_dict.json") as f:
        sense_pos_dict = json.load(f)
    
    def senses_have_same_pos(row):
        sense1_pos = sense_pos_dict[row[3]]
        sense2_pos = sense_pos_dict[row[4]]
        return sense1_pos == sense2_pos

    def senses_have_dif_pos(row):
        sense1_pos = sense_pos_dict[row[3]]
        sense2_pos = sense_pos_dict[row[4]]
        return sense1_pos != sense2_pos

    df_as_list = df.values.tolist()
    lemmas_with_same_pos = [ row[1] for row in df_as_list if senses_have_same_pos(row) ]
    lemmas_with_dif_pos = [ row[1] for row in df_as_list if senses_have_dif_pos(row) ]
    
    assert len(lemmas_with_same_pos) + len(lemmas_with_dif_pos) == len(df_as_list)

    return lemmas_with_same_pos, lemmas_with_dif_pos

def getLemmaPosPairsList(filename):
    df = pd.read_csv(filename)
    
    with open("data/sense_to_pofs_dict.json") as f:
        sense_pos_dict = json.load(f)

    pairs_dict = {}

    def get_pair_type(row):
        sense1_pos = sense_pos_dict[row[3]]
        sense2_pos = sense_pos_dict[row[4]]
        if sense1_pos < sense2_pos:
            sense1_pos, sense2_pos = sense2_pos, sense1_pos
        return sense1_pos, sense2_pos

    for row in df.values.tolist():
        key = get_pair_type(row)
        if not key in pairs_dict.keys():
            pairs_dict[key] = []
        pairs_dict[key].append(row[1])

    return pairs_dict

def update_df_format(df, max_samp):
    """
    Returns a dataframe with more columns: pos1, pos2, sense1_freq, sense2_freq, and max_samp.
    pos1 is the part of speech of sense1
    pos2 is the part of speech of sense2
    sense1_freq is the number of instances of the first sense
    sense2_freq is the number of instances of the second sense
    max_samp is the value of max samp this data was generated with
    Note: this function does not write the df to a file, If this is desired, 
    the caller will be responsible for this. 
    """
    df = add_pos_columns(df)
    df = add_sense_frequency_cols(df)
    if not "max_samp" in df.columns:
        df["max_samp"] = max_samp
    return df


def add_pos_columns(df):
    """
    Returns a new df that has two additional columns: pos1 and pos2.
    pos1 is the part of speech of sense1
    pos2 is the part of speech of sense2
    """
    with open("data/sense_to_pofs_dict.json") as f:
        sense_pos_dict = json.load(f)
    def get_pos(sense):
        return sense_pos_dict[sense]
    df = pd.concat([df, df["sense1"].apply(get_pos).rename("pos1")], axis=1)
    df = pd.concat([df, df["sense2"].apply(get_pos).rename("pos2")], axis=1)
    return df

def add_sense_frequency_cols(df):
    """
    Returns a dataframe with two new columns sense1_freq and sense2_freq
    """
    def create_sense_freq_dict(lemmadir):
        result = defaultdict(int)
        for (_, instances) in lemmadata_iter(lemmadir):
            for instance in instances:
                result[instance.sense] += 1
        return result

    freq_dict = create_sense_freq_dict()
    def get_freq(sense):
        return freq_dict[sense]
    df = pd.concat([df, df["sense1"].apply(get_freq).rename("sense1_freq")], axis=1)
    df = pd.concat([df, df["sense2"].apply(get_freq).rename("sense2_freq")], axis=1)
    return df

def manually_convert_format():
    df = pd.read_csv("classifier_data8_20-max.csv")
    df = update_df_format(df, 1000)
    df.to_csv("all_lemmas_20-max_layer_0.csv")

if __name__ == "__main__":
    train_lemma_classifiers_with_elmo(43,43, 1, 20)
