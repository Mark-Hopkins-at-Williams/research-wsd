Disambiguation
--------------

To train lemma-specific sense classifiers, do the following:

    from experiment import *
    train_lemma_classifiers(vectorize_instance,40,42,1,2000,'lemmadata', cached=True, verbose=True)    
    
This will train sense classifiers for all lemmas whose second sense has
between 40 and 42 instances using n-fold cross-validation with all logs on.

To run the general lemma test(in contrast to single-lemma training and testing) on words with 0.7 accuracy or higher with 5-fold cross-validation and 2000 training examples per lemma:
1. Make sure there is "word_lemma_dict.json" under a the "data" folder.
For ELMo:
    	
	from elmo import elmo_vectorize
	from experiment import train_cross_lemmas
	from compare import createLemmaData_elmo
	createLemmaData_elmo()
	train_cross_lemmas(elmo_vectorize, 0.7, 5, 2000, verbose=True)
    
For BERT:

	from bert import vectorize_instance
	from experiment import train_cross_lemmas
	from compare import createLemmaData
	createLemmaData()
	train_cross_lemmas(vectorize_instance, 0.7, 5, 2000, verbose=True)

TEST TAKERS! Read Here!

Follow the exact instruction below to make sure your test is bugless:

1. Only take the test on a Linux or Mac machine.

2. Before starting your test, make sure to download the mandatory data files into the right directory. Download "completedata.json", "human_acc_test" data and "sense_to_pofs_dict.json" into "data" folder. Do not rename the download!

3. Create two folders: "human_test_logs" and "human_test_results" in the data folder.

4. type "python3 human_acc_test.py" and enter under the repo directory to start the test


     
    
Data Links
----------
## googledata.json
https://drive.google.com/file/d/1VP1Z0KYYJecMrUE4VelX3ujyjd5wh2WO/view?usp=sharing

## completedata.json:
https://drive.google.com/file/d/1ekfegaI2Cn4TmDck--X6vIbvEkDp2yMJ/view?usp=sharing

## Precomputed bert vecs and lemmadata
https://drive.google.com/file/d/195BOf-qxJHz-ybTORvYO26F59zP3zrCc/view?usp=sharing

## human_acc_test data
https://drive.google.com/file/d/1f547LZsS5CAYPWTNwRl9HGpf00T0TknH/view?usp=sharing

## sense_to_pofs_dict.json
https://drive.google.com/file/d/1RCWEWwCu4i1vXZLPtbZtLcoh8EUMpT31/view?usp=sharing

## word_lemma_dict.json
https://drive.google.com/file/d/1_2Z-rYVX4pDAp1pfDp0s3YAnHBvi2F7Q/view?usp=sharing

##lemmadata
https://drive.google.com/drive/folders/1NWGKiqEH8dCToj_gY4EpmH_V5N3yZJ4v?usp=sharing

## lemmadata_elmo
https://drive.google.com/drive/folders/1crGo0SLR1mVNDzmOUsmdvy9ealUGY17V?usp=sharing
