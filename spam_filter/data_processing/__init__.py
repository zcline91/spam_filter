from .settings import CORPORA_CSV_PATH, CORPUS_FILENAMES, DOCBIN_PATH, \
    DOCBIN_FILENAMES, SPAM_CLASS_PATH, SPAM_CLASS_FILENAMES, TEST_RATIO
from .spacy import create_docbins, Lemmatizer, DocCreator
from .emailextract import email_to_df