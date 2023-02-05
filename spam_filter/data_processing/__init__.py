from .settings import CORPORA_CSV_PATH, CORPUS_FILENAMES, DOCBIN_PATH, \
    DOCBIN_FILENAMES, SPAM_CLASS_PATH, SPAM_CLASS_FILENAMES, TEST_RATIO
from .preprocessing import load_train_test_csvs, load_train_test_docs, \
    load_train_test_classes, email_cleaning
from .spacy import create_docbins, Lemmatizer, DocCreator
