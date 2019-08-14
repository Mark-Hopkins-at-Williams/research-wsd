Disambiguation
--------------

To train lemma-specific sense classifiers, do the following:

    from experiment import *
    train_lemma_classifiers(40, 42, n, verbose=True)
    
This will train sense classifiers for all lemmas whose second sense has
between 40 and 42 instances using n-fold cross-validation with all logs on.


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
