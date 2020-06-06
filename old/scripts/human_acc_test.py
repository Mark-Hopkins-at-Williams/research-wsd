#from vectrain import human_acc_test

import pandas as pd
import os
import random
import json

def getExampleSentencesBySense(sense, num_examples):
    with open("data/completedata.json") as data:
        file_data = json.load(data)   

    examples = []
    for document in file_data:
        doc = document["doc"]
        for sent_object in doc:
            for word_with_sense in sent_object["senses"]:
                if word_with_sense["sense"] == sense:
                    examples.append(sent_object["natural_sent"])
    if len(examples) >= num_examples:
        examples = random.sample(examples, num_examples)
    return examples


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

if __name__ == "__main__":
    human_acc_test(0.7, 0.58, "data/classifier_data8_20-max.csv", 25)