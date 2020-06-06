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


def train_lemma_classifier_with_vec(layers_i, add_sent, min_sense2_freq, max_sense2_freq, n_fold, max_sample_size, 
                                    verbose=True):
    """
    train the lemma classifier with a specified vectorization patter

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

def train_lemma_classifier_with_vec_specific_lemmas(layers_i, lemmas_list, n_fold, max_sample_size, 
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
        from experiment import train_lemma_classifiers_on_specific_lemmas
        lemma_info_dict = train_lemma_classifiers_on_specific_lemmas(lemmas_list, n_fold, max_sample_size, verbose)
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
    
def train_lemma_classifier_with_diff_layers_specific_lemmas(max_layers_to_average, num_layer_combos, lemmas, n_fold, max_sample_size):
    """
    Calls train_lemma_classifier with a variety of values for layers_i and add_sent_encoding.
    It then writes the relevant data to a file.
    """
    layers_i_list = [ ([x], False) for x in range(13)]
    i = 0
    while i < num_layer_combos:
        layers_i_tuple = generate_random_layers_i(max_layers_to_average)
        if not layers_i_tuple in layers_i_list:
            layers_i_list.append(layers_i_tuple)
        else:
            continue
        i += 1
    
    data = []
    for index, layers_i_tuple in enumerate(layers_i_list):
        print(str(index)+" of "+str(len(layers_i_list)))
        lemma_info_dict, layers_i = train_lemma_classifier_with_vec_specific_lemmas(layers_i_tuple[0], lemmas, n_fold, max_sample_size, 
                                        verbose=True, add_sent_encoding=layers_i_tuple[1])
        for key in lemma_info_dict.keys():
            lemma_info = lemma_info_dict[key]
            data.append([layers_i, key, lemma_info[0], lemma_info[1], lemma_info[2]])
    df = pd.DataFrame(data, columns=["spec", "lemma", "best_avg_acc", "sense1", "sense2"])

    num = 1
    while os.path.exists("classifier_data"+str(num)+".csv"):
        num += 1

    df = update_df_format(df, max_sample_size)

    df.to_csv("classifier_data_spec"+str(num)+".csv", index=False)

def train_lemma_classifier_with_diff_layers(max_layers_to_average, num_layer_combos, min_sense2_freq, max_sense2_freq, n_fold, max_sample_size):
    """
    Calls train_lemma_classifier with a variety of values for layers_i and add_sent_encoding.
    It then writes the relevant data to a file.
    """
    """ [ ([x], False) for x in range(13)] """
    layers_i_list =  [([12], False)]
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
        lemma_info_dict, layers_i = train_lemma_classifier_with_vec(layers_i_tuple[0], layers_i_tuple[1], min_sense2_freq, max_sense2_freq, n_fold, max_sample_size, 
                                        verbose=True)
        for key in lemma_info_dict.keys():
            lemma_info = lemma_info_dict[key]
            data.append([layers_i, key, lemma_info[0], lemma_info[1], lemma_info[2]])
    df = pd.DataFrame(data, columns=["spec", "lemma", "best_avg_acc", "sense1", "sense2"])

    num = 1
    while os.path.exists("last_layer_all_lemmas_2000_"+str(num)+".csv"):
        num += 1
    df = update_df_format(df, max_sample_size)
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


def initial_info():
    input("In this test you are asked to tell if two words with shared lemma have the same or different meanings in their respective sentences. (press enter to continue)")
    input("simply type in \"y\" or \"n\" when prompted. (press enter to continue)")
    input("Note that lemma means the root-form of a word. e.g. could, can't, canned have the same lemma can. (press enter to continue)")
    input("Ready? press enter to continue to the test!")
    print()

def create_test_folders_if_dne():
    if not os.path.exists("data/human_test_logs"):
        os.makedirs("data/human_test_logs")

    if not os.path.exists("data/human_test_results"):
        os.makedirs("data/human_test_results")

def get_valid_answer():
    answer = input("Do the sentences have the same definition for the lemma word?(y/n) ")
    while not(answer == "y" or answer == "n"):
        print("answer input can only be y or n")
        answer = input("Do the sentences have the same definition for the lemma word?(y/n) ")
    return answer == "y"
#classifier_data_
#spec, lemma, best_avg_acc, sense1, sense2
def human_acc_test(threshold_high, threshold_low, filename, test_size):

    initial_info()

    create_test_folders_if_dne()


    df = pd.read_csv(filename)

    high_acc_df = df[df["best_avg_acc"] >= threshold_high]
    low_acc_df = df[df["best_avg_acc"] <= threshold_low]

    cutoff = min(test_size, len(high_acc_df.index), len(low_acc_df.index))

    high_acc_df = high_acc_df.iloc[:cutoff]
    low_acc_df = low_acc_df.iloc[:cutoff]

    data = pd.concat([high_acc_df, low_acc_df]).sample(frac=1)


    with open("data/sense_to_pofs_dict.json") as f:
        sense_pos_dict = json.load(f)

    correct_count_high = 0
    correct_count_low = 0
    diff_pos_count_high = 0
    diff_pos_count_low = 0
    exports = []

    username = input("input your username: ")
    while username == "" or len(username) < 3:
        username = input("please enter your full-length username:")
    
    lemma_num = 1
    for i in data.index:
        acc = data["best_avg_acc"][i]
        lemma = data["lemma"][i]
        sense1 = data["sense1"][i]
        sense2 = data["sense2"][i]

        pos1 = sense_pos_dict[sense1]
        pos2 = sense_pos_dict[sense2]
        same_pos = True if pos1 == pos2 else False

        if not same_pos:
            if acc >= threshold_high:
                diff_pos_count_high += 1
            if acc <= threshold_low:
                diff_pos_count_low += 1

        sentences1 = getExampleSentencesBySense(sense1, 3)
        sentences2 = getExampleSentencesBySense(sense2, 3)

        is_same = random.random() >= 0.5

        if is_same:
            if random.random() >= 0.5:
                pair = random.sample(sentences1, 2)
            else:
                pair = random.sample(sentences2, 2)
        else:
            pair = random.sample(sentences1, 1) + random.sample(sentences2, 1)

        correct_answer = is_same

        print()
        print("progress: word " + str(lemma_num) + "/" + str(test_size*2))
        print(lemma + ": ")
        print("sentence 1: " + pair[0]+"\n")
        print("sentence 2: " + pair[1]+"\n")
        
        answer = get_valid_answer()

        is_correct = answer == correct_answer

        export = [lemma, pair[0], pair[1], is_correct, sense1, sense2, is_same]
        exports.append(export)

        if is_correct:
            if acc >= threshold_high: correct_count_high += 1
            if acc <= threshold_low: correct_count_low += 1
        lemma_num += 1


    if len(high_acc_df.index) > 0:
        human_acc_high = correct_count_high / len(high_acc_df.index)
        diff_perc_high = diff_pos_count_high / len(high_acc_df.index)
    else:
        human_acc_high = 0
        diff_perc_high = 0
    if len(low_acc_df.index) > 0:
        human_acc_low = correct_count_low / len(low_acc_df.index)
        diff_perc_low = diff_pos_count_low / len(low_acc_df.index)
    else:
        human_acc_low = 0
        diff_perc_low = 0

    print("high accuracy stats:")
    print("percentage of different POS: {:.3f}".format(diff_perc_high))
    print("human disambiguation accuracy: {:.3f}".format(human_acc_high))

    print("low accuracy stats:")
    print("percentage of different POS: {:.3f}".format(diff_perc_low))
    print("human disambiguation accuracy: {:.3f}".format(human_acc_low))

    exports = pd.DataFrame(exports, columns = ["lemma", "sent1", "sent2", "is_correct", "sense1", "sense2", "is_same"])
    exports.to_csv("data/human_test_logs/" + username + ".csv", index=False)

    result_d = {}
    result_d["human_acc_high"] = human_acc_high
    result_d["diff_perc_high"] = diff_perc_high
    result_d["human_acc_low"] = human_acc_low
    result_d["diff_perc_low"] = diff_perc_low
    with open("data/human_test_results/" + username + ".json", "w") as f:
        json.dump(result_d, f)

def human_acc_test_same_pos(threshold_high, threshold_low, filename, test_size):

    initial_info()

    create_test_folders_if_dne()

    high_acc_list, low_acc_list = get_low_and_high_acc_lists_same_pos(threshold_high, threshold_low, filename)    
    cutoff = min(test_size, len(high_acc_list), len(low_acc_list))
    high_acc_list, low_acc_list = high_acc_list[:cutoff], low_acc_list[:cutoff]
    high_and_low_list = high_acc_list + low_acc_list
    random.shuffle(high_and_low_list)

    # The test ends up having twice as many questions as 
    # specificied by test_size. The assertion will fail when the
    # specified thresholds do not contain enough instances.
    assert len(high_and_low_list) == test_size*2

    with open("data/sense_to_pofs_dict.json") as f:
        sense_pos_dict = json.load(f)

    def prepare_next_question(sense1, sense2):
        """
        Returns a pair of sentences and a flag indicating if the
        two sentences have the same sense.
        """
        pos1, pos2 = sense_pos_dict[sense1], sense_pos_dict[sense2]

        sentences1 = getExampleSentencesBySense(sense1, 3)
        sentences2 = getExampleSentencesBySense(sense2, 3)

        if random.random() >= 0.5:
            if random.random() >= 0.5:
                sentence_pair = random.sample(sentences1, 2)
            else:
                sentence_pair = random.sample(sentences2, 2)
            return sentence_pair, True
        else:
            sentence_pair = random.sample(sentences1, 1) + random.sample(sentences2, 1)
            return sentence_pair, False


    correct_count_high = 0
    correct_count_low = 0
    exports = []

    username = input("Input your username: ")
    while username == "" or len(username) < 3:
        username = input("Please enter your full-length username:")
    
    for i in range(len(high_and_low_list)):
        if i == 0:
            _, lemma, acc, sense1, sense2 = high_and_low_list[i]
            sentence_pair, correct_answer = prepare_next_question(sense1, sense2)
        
        print("\n\nYou have completed: " + str(i) + "/" + str(test_size*2))
        print(lemma + ": \n")
        print("Sentence 1: " + sentence_pair[0]+"\n")
        print("Sentence 2: " + sentence_pair[1]+"\n")
        print("Start determining if the words have the same definition.")
        print("We are currently loading the next question.")
        # Preparing the next question while the user reads the current one
        # makes the test feel much faster.
        i += 1
        if i < len(high_and_low_list):
            _, n_lemma, n_acc, n_sense1, n_sense2 = high_and_low_list[i]
            n_sentence_pair, n_correct_answer = prepare_next_question(n_sense1, n_sense2)

        is_correct = correct_answer == get_valid_answer()

        export = [lemma, acc, sentence_pair[0], sentence_pair[1], is_correct, sense1, sense2, correct_answer]
        exports.append(export)

        if is_correct:
            if acc >= threshold_high:
                correct_count_high += 1
            else:
                correct_count_low += 1
        lemma, acc, sense1, sense2 = n_lemma, n_acc, n_sense1, n_sense2
        sentence_pair, correct_answer = n_sentence_pair, n_correct_answer

       

    print("high accuracy stats:")
    human_acc_high = correct_count_high/test_size
    print("human disambiguation accuracy: {:.3f}".format(human_acc_high))

    print("low accuracy stats:")
    human_acc_low = correct_count_low/test_size
    print("human disambiguation accuracy: {:.3f}".format(human_acc_low))

    exports = pd.DataFrame(exports, columns = ["lemma", "acc", "sent1", "sent2", "is_correct", "sense1", "sense2", "is_same"])
    exports.to_csv("data/human_test_logs/" + username + ".csv", index=False)

    result_d = {}
    result_d["human_acc_high"] = human_acc_high
    result_d["human_acc_low"] = human_acc_low
    with open("data/human_test_results/" + username + ".json", "w") as f:
        json.dump(result_d, f)

def get_low_and_high_acc_lists_same_pos(threshold_high, threshold_low, filename):
    df = pd.read_csv(filename)
    high_acc_df = df[df["best_avg_acc"] >= threshold_high]
    low_acc_df = df[df["best_avg_acc"] <= threshold_low]

    with open("data/sense_to_pofs_dict.json") as f:
        sense_pos_dict = json.load(f)
    
    def senses_have_same_pos(row):
        sense1_pos = sense_pos_dict[row[3]]
        sense2_pos = sense_pos_dict[row[4]]
        return sense1_pos == sense2_pos

    high_acc_list = high_acc_df.values.tolist()
    low_acc_list = low_acc_df.values.tolist()

    high_acc_list = list(filter(senses_have_same_pos, high_acc_list))
    low_acc_list = list(filter(senses_have_same_pos, low_acc_list))
 
    return high_acc_list, low_acc_list


def diff_pos_perc(threshold_high, threshold_low, filename):
    df = pd.read_csv(filename)
    high_acc_df = df[df["best_avg_acc"] >= threshold_high]
    low_acc_df = df[df["best_avg_acc"] <= threshold_low]
    data = pd.concat([high_acc_df, low_acc_df]).sample(frac=1)
    with open("data/sense_to_pofs_dict.json") as f:
        sense_pos_dict = json.load(f)
    
    diff_pos_count_high = 0
    diff_pos_count_low = 0
    
    for i in data.index:
        sense1 = data["sense1"][i]
        sense2 = data["sense2"][i]
        acc = data["best_avg_acc"][i]
        pos1 = sense_pos_dict[sense1]
        pos2 = sense_pos_dict[sense2]
        same_pos = True if pos1 == pos2 else False

        if not same_pos:
            if acc >= threshold_high:
                diff_pos_count_high += 1
            if acc <= threshold_low:
                diff_pos_count_low += 1

    
    diff_perc_high = diff_pos_count_high / len(high_acc_df.index)
    diff_perc_low = diff_pos_count_low / len(low_acc_df.index)
    print(diff_perc_high)
    print(diff_perc_low)

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