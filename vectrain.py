from util import delete_last_line

def train_lemma_classifier_with_vec(layers_i, min_sense2_freq, max_sense2_freq, n_fold, 
                                    verbose=True, add_sent_encoding=False):
    with open("bert.py", "a") as f:
        f.write("\nvectorize_instance = generate_vectorization(" + str(layers_i) + ", " +  str(add_sent_encoding) + ")")
    try:
        from experiment import train_lemma_classifiers
        lemma_avg_acc_dict = train_lemma_classifiers(min_sense2_freq, max_sense2_freq, n_fold, verbose)
    finally:
        delete_last_line("bert.py")
    return lemma_avg_acc_dict