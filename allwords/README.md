Reed Word Sense Disambiguation
------------------------------

### To download and preprocess the Raganato 2017 data:

From the top-level directory, run:

    bash ./install.sh

### To run the main training script

From the top-level directory, run:

    python allwords/run.py ./data

### To run all unit tests

From the top-level directory, run: 
    
    python -m unittest

### To run a particular unit test module (e.g. test/test_align.py)

From the top-level directory, run:

    python -m unittest test.test_align
    
    
    