from util import delete_last_line
import pprint

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
    return lemma_info_dict

def test_single_layer(min_sense2, max_sense2):
    layers = []
    for i in range(13):
        curr_dict = train_lemma_classifier_with_vec([i], min_sense2, max_sense2, 10, 600, verbose = False)
        accs = [lemma[0] for lemma in list(curr_dict.values())]
        layers.append(accs)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(layers)
    return layers
