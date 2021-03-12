Selective Prediction Research, Reed College
------------------------------
## Project Structure

### old

Folder `old` stores the research of summer 2019. The project explores the WSD task with BERT and ELMo.

### reed_wsd

Folder `reed_wsd` stores the research on selective prediction. The instructions below are for the selective prediction project.

## Installing the reed_wsd package:
The repo only works when it is installed as a package

From the top-level directory:

    pip install -e .

## Preprocessing Data
Before doing any training, it is necessary to download and preprocess the data first.
### To download and preprocess the Raganato 2017 data:

From the top-level directory, run:

    cd reed_wsd/allwords
    bash ./install.sh

### To download and preprocess the imdb data:

From top-level directory, run:

    cd reed_wsd/imdb
    bash ./install.sh

### To Run Experiments
You can specify the following arguments to `python3 experiment.py` to run experiments.

	positional arguments:
	  config_path           path to experiment configs
	  result_path           path where experimental results are stored

	optional arguments:
	  -h, --help            show this help message and exit
	  -vl VALID_PATH, --valid_path VALID_PATH
				path to which validation results are stored
	  -lg LOG_PATH, --log_path LOG_PATH
				path to the log file
	  -r REPS, --reps REPS  number of repititions for each experiment

    
For example,

    python3 experiment.py config_path results_path -vl valid/test.json -lg logs/test.log

`-r` flag has default value 1.

A configuration dictionary looks like this:

    config =  {'task': 'mnist'/'allwords'/'imdb',
               'architecture': 'abstaining'/'simple'/'bem',
               'confidence': 'inv_abs'/'max_prob'/'max_non_abs',
               'criterion': {'name': 'pairwise'/'nll'/'crossentropy'/'conf1'/'conf4',
			     'alpha': float,
			     'warmup_epochs': int},
               'confused': bool,
               'style': 'pairwise'/'single',
               'dev_corpus': corpus_id,
               'bsz': int,
               'n_epochs': int
             }


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

## Testing

**Note that some unit tests are deprecated now.** One won't pass the tests. Clean-up in the unit test is needed.

### To run all unit tests

From the top-level directory, run: 
    
    cd allwords
    python3 -m unittest


### To run a particular unit test module (e.g. test/test_align.py)

From the top-level directory, run:

    cd allwords
    python3 -m unittest test.test_align
 
