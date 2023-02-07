import logging

import pandas as pd

from .email_cleaning_pipelines import corpus_prep
from .test_set_creation import split_train_test_by_id
from ..spacy import load_docbins, DocBinError
from ..settings import CORPORA_CSV_PATH, CORPUS_FILENAMES, TEST_RATIO, \
    DOCBIN_PATH, DOCBIN_FILENAMES, SPAM_CLASS_PATH, SPAM_CLASS_FILENAMES


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def load_corpora_csvs(path=CORPORA_CSV_PATH, corpus_names=CORPUS_FILENAMES):
    """Load email corpora in path (from csv files with fields 
    `path` (str), `spam` (bool encoded as 0, 1), `subject` (str), 
    `body` (str)) as a pandas dataframe. The argument `corpus_names`
    is a dict-like object of corpus names and filenames in `path`.
    The returned dataframe has indices (corpus, path)."""
    logger.info(f"Loading all csvs at {path} with names and files "
        f"{corpus_names} and concatenating them into a single pandas "
        "DataFrame.\nDepening on the size of the csvs, this could take "
        "a while...")
    def corpora_gen():
        for corpus_name, filename in corpus_names.items():
            df = pd.read_csv(path / filename, 
                dtype={"path": "string", "spam": "bool", 
                       "subject": "string", "body": "string"}
                )
            df.insert(0, 'corpus', corpus_name)
            df['corpus'] = df['corpus'].astype('string')
            df.set_index(['corpus', 'path'], inplace=True)
            yield df
    return pd.concat(corpora_gen())


def cleaned_corpora_csvs(path=CORPORA_CSV_PATH, corpus_names=CORPUS_FILENAMES):
    """Load email corpora, transformed with the corpus_prep pipeline"""
    return corpus_prep.transform(load_corpora_csvs(path, corpus_names))


def load_train_test_csvs(path=CORPORA_CSV_PATH, corpus_names=CORPUS_FILENAMES, 
        test_ratio=TEST_RATIO):
    """Load email corpora, transformed with the corpus_prep pipeline,
    in two sets: a training set and a test set."""
    train_set, test_set = split_train_test_by_id(
        cleaned_corpora_csvs(path, corpus_names), test_ratio, "path", 
        string_id=True, id_from_index=True
    )
    return train_set, test_set


def load_train_test_classes(path=SPAM_CLASS_PATH, 
        filenames=SPAM_CLASS_FILENAMES):
    """Load a Series of corpora 'spam' classes (1 for spam, 0 for ham)
    indexed by 'corpus' and 'path'."""
    train_path = path / filenames['train']
    test_path = path / filenames['test']
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} does not exists. (Have you run "
            "create_classes() yet?)")
    if not test_path.exists():
        raise FileNotFoundError(f"{test_path} does not exists. (Have you run "
            "create_classes() yet?)")
    train_classes, test_classes = (
        pd.read_csv(
            path, 
            dtype={'corpus': 'string', 'path': 'string', 
                'spam': 'boolean'},
            index_col=('corpus', 'path')
        )
        for path in (train_path, test_path)
    )
    return train_classes['spam'], test_classes['spam']


def load_train_test_docs(train_classes, test_classes, path=DOCBIN_PATH, 
        docbin_names=DOCBIN_FILENAMES):
    """Load email corpora data as two DataFrames of spacy Docs."""
    params = {'index_names': ('corpus', 'path'), 
        'index_dtypes': ('string', 'string')}
    # Load the training set docs
    train_sub_docs = load_docbins(path / docbin_names['train']['subject'], 
        **params)
    train_body_docs = load_docbins(path / docbin_names['train']['body'], 
        **params)
    # Check that the docs indeed match the index of the training classes series
    for docset in [train_sub_docs, train_body_docs]:
        if not (docset.index.sort_values().equals(
                train_classes.index.sort_values())):
            raise DocBinError("The train subject and/or body docbin does not"
                "have an index matching that of the passed train_classes "
                "Series.")

    # Load the test set docs
    test_sub_docs = load_docbins(path / docbin_names['test']['subject'], 
        **params)
    test_body_docs = load_docbins(path / docbin_names['test']['body'], 
        **params)
    # Check that the docs indeed match the index of the test classes series
    for docset in [test_sub_docs, test_body_docs]:
        if not (docset.index.sort_values().equals(
                test_classes.index.sort_values())):
            raise DocBinError("The test subject and/or body docbin does not"
                "have an index matching that of the passed test_classes "
                "Series.")

    train_set = pd.DataFrame({'subject_doc': train_sub_docs, 
        'body_doc': train_body_docs}).reindex(train_classes.index)
    test_set = pd.DataFrame({'subject_doc': test_sub_docs, 
        'body_doc': test_body_docs}).reindex(test_classes.index)
    return train_set, test_set
