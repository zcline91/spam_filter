import logging

import pandas as pd

from .email_cleaning_pipelines import corpus_prep
from .test_set_creation import split_train_test_by_id
from ..spacy.dochandling import DocBinLoader
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
                dtype={"path": "string[pyarrow]", "spam": "bool", 
                       "subject": "string[pyarrow]", "body": "string[pyarrow]"}
                )
            df.insert(0, 'corpus', corpus_name)
            df['corpus'] = df['corpus'].astype('string[pyarrow]')
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
            dtype={'corpus': 'string[pyarrow]', 'path': 'string[pyarrow]', 
                'spam': 'boolean'},
            index_col=('corpus', 'path')
        )
        for path in (train_path, test_path)
    )
    return train_classes['spam'], test_classes['spam']


def load_train_test_docs(train_classes, test_classes, path=DOCBIN_PATH, 
        docbin_names=DOCBIN_FILENAMES):
    """Load email corpora data as two Series of spacy Docs."""
    params = {'index_names': ('corpus', 'path'), 
        'index_dtypes': ('string[pyarrow]', 'string[pyarrow]')}
    docs = {
        data_set: {
            field: DocBinLoader(path / filename, **params)
            for field, filename in set_field_files.items()
        }
        for data_set, set_field_files in docbin_names.items()
    }
    train_set = pd.DataFrame({
        'body_doc': docs['train']['body'].transform(train_classes), 
        'subject_doc': docs['train']['subject'].transform(train_classes),
    })
    test_set = pd.DataFrame({
        'body_doc': docs['test']['body'].transform(test_classes), 
        'subject_doc': docs['test']['subject'].transform(test_classes),
    })
    return train_set, test_set
