from experiment import train_finetune, train_cross_lemmas, neighbors_test, train_lemma_classifiers_with_vec_elmo
from elmo import elmo_vectorize
import json
def run_main():
    train_finetune(21,21, 10, 20)

if __name__ == "__main__":
    train_lemma_classifiers_with_vec_elmo(elmo_vectorize, 21, 10000000000, 10, 2000)
