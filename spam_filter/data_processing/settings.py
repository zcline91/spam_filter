from pathlib import Path


# Locations of csv files created using `extract_from_corpus.py`
# The keys are also the 'names' of the corpora used for indices in Pandas 
# dataframes.   
CORPORA_CSV_PATH = Path("")
CORPUS_FILENAMES = {
    'enron': 'enron.csv',
    'ling':'lingspam_public.csv',
    'trec05': 'trec05p-1.csv',
    'trec06': 'trec06p.csv',
    'trec07': 'trec07p.csv',
}

# Locations to store docbins for test and training sets.
DOCBIN_PATH = Path("")
DOCBIN_FILENAMES = {
    'train': {'body': 'trainbody', 'subject': 'trainsubject'},
    'test': {'body': 'testbody', 'subject': 'testsubject'},
}

# Locations to store 'spam' Series as CSV (after running the corpora CSV data 
# through the preprocessor, eliminating duplicates, etc.) Saving this means 
# being able to load the classes without rerunning the preprocessing pipeline
# on all the corpora data every time.
SPAM_CLASS_PATH = Path("")
SPAM_CLASS_FILENAMES = {
    'train': 'train_classes.csv', 
    'test': 'test_classes.csv'
}

# Don't change this after starting to train models.
TEST_RATIO = 0.2
