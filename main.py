from bert import vectorize_instance
from experiment import train_cross_lemmas
from elmo import elmo_vectorize
if __name__ == "__main__":
	train_cross_lemmas(elmo_vectorize, 0.7, 5, 2000, verbose=True)

