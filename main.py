from experiment import train_finetune, train_cross_lemmas, neighbors_test, train_lemma_classifiers_with_vec_elmo
from elmo import elmo_vectorize
def run_main():
    train_finetune(21, 10000000000, 5, 2000)

if __name__ == "__main__":
<<<<<<< HEAD
    #d = train_lemma_classifiers_with_vec_elmo(elmo_vectorize, 21, 100000000000, 5, 2000)
    run_main()
=======
    neighbors_test("elmo")

>>>>>>> 0ae1d4bb2f5e8626ca15d521393bd65d39680b42
