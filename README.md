Disambiguation
--------------

To train lemma-specific sense classifiers, do the following:

    from experiment import *
    train_lemma_classifiers(40, 42, n, verbose=True)
    
This will train sense classifiers for all lemmas whose second sense has
between 40 and 42 instances using n-fold cross-validation with all logs on.
<<<<<<< HEAD
     
=======
    
    
>>>>>>> c46b2dfef6e24c1061bce55d74022053d0c9a62e
    
Data Links
----------
## googledata.json
https://drive.google.com/file/d/1VP1Z0KYYJecMrUE4VelX3ujyjd5wh2WO/view?usp=sharing

## completedata.json:
https://drive.google.com/file/d/1ekfegaI2Cn4TmDck--X6vIbvEkDp2yMJ/view?usp=sharing

## Precomputed bert vecs and lemmadata
https://drive.google.com/file/d/195BOf-qxJHz-ybTORvYO26F59zP3zrCc/view?usp=sharing
