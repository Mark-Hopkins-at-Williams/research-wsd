Reed Word Sense Disambiguation
------------------------------

### To download and preprocess the Raganato 2017 data:

From the top-level directory, run:

    cd allwords
    bash ./install.sh

### To run the main training script

From the top-level directory, run:

    cd allwords
    python allwords/run.py ./data

### To run the official scoring script

The scorer ("Scorer.java") is provided by Raganato et al (2017).
To use the scorer, you first need to compile:

	javac allwords/data/WSD_Evaluation_Framework/Evaluation_Datasets/Scorer.java

Then, evaluate your system by typing the following commands: 

    cd allwords/data/WSD_Evaluation_Framework/Evaluation_Datasets
    java Scorer [gold-standard] [system-output]

Example of usage:

	cd allwords/data/WSD_Evaluation_Framework/Evaluation_Datasets
	java Scorer semeval2007/semeval2007.gold.key.txt output.txt

Please note that the official scoring programs are case sensitive and may be
sensitive if a different default character encoding is used.  The answer
format is the same of the gold-standard format. 

### To run all unit tests

From the top-level directory, run: 
    
    cd allwords
    python -m unittest

### To run a particular unit test module (e.g. test/test_align.py)

From the top-level directory, run:

    cd allwords
    python -m unittest test.test_align
    
    
    