from pathlib import Path
import logging

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import spacy
from spacy.tokens import Doc, DocBin


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DocBinError(Exception):
    pass


# Create a Doc attribute for storing the index of the email
Doc.set_extension("identifier", default=None)


class DocCreator(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
    
    def fit(self, X, y=None):
        # No fitting of the transformer is necessary
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.to_numpy()
        docs = list(self.nlp.pipe(X.flat, disable=["parser", "ner"],
            n_process=-1, batch_size=250))
        return np.array(docs, dtype=object).reshape(X.shape)



def create_docbins(pd_series, path, batch_size=2000):
    """Saves .spacy DocBin file(s) to `path`, where the DocBin
    consists of Doc objects created from the entries of a string
    pandas Series `pd_series`. 
    NOTE: Do not include the extension '.spacy' in `docbin_path`.
    If the series is larger than the batch size, a directory will be 
    created at the `path` and spacy files saved within. If the 
    docs will fit in one file, the '.spacy' extension will be added 
    appropriately."""
    path = Path(path)
    if not path.parent.exists():
        raise FileNotFoundError(f"{path.parent} does not exist")
    mem_size = pd_series.memory_usage()
    msg = (f"Running spacy pipeline on Series '{pd_series.name}' of length "
        f"{len(pd_series)} (size in memory: {mem_size / 1_000_000.0:.3f} MB).")
    if mem_size > 50 * 10**6: # Over 50 MB
        msg += " This may take a while..."
    logger.info(msg)
    # If fewer than batch_size docs are in the Series, save to a single file.
    if pd_series.size <= batch_size:
        parent = path.parent
    else:
        parent = path.with_suffix('')
        parent.mkdir(exist_ok=True)
    for i in range(0, pd_series.size, batch_size):
        batch_num = (i+1)//batch_size
        logger.info(f"Running batch {batch_num}: docs {i+1} through "
            f"{min([i + batch_size, pd_series.size])}")
        dc = DocCreator()
        subseries = pd_series[i: i + batch_size]
        docs = dc.transform(subseries)
        for idx, doc in zip(subseries.index, docs):
            doc._.identifier = idx
        docbin = DocBin(store_user_data=True, docs=docs)
        if pd_series.size <= batch_size:
            docbin_path = parent / path.stem.with_suffix('.spacy')
        else:
            docbin_path = parent / f"{batch_num}.spacy"
        logger.info(f"Saving to {docbin_path}")
        docbin.to_disk(docbin_path)


def load_docbins(path, index_names=None, index_dtypes=None):
    """Load (a collection of) .spacy DocBin files into a pandas
    Series. To retrieve the index stored in the docbin (which would 
    have been set when they were created), specify a tuple of 
    index names and another tuple of index data types. Otherwise,
    the Series is indexed by integers."""
    if (index_names is not None and index_dtypes is not None 
            and len(index_names) != len(index_dtypes)):
        raise DocBinError("The number of provided index names "
            f"({len(index_names)}) and index dtypes ({len(index_dtypes)}) "
            "do not match.")
    path = Path(path)
    if path.exists() and path.is_dir():
        multi=True
    elif path.with_suffix('.spacy').exists():
        multi=False
    else:
        raise DocBinError(f"{path} does not exist")
    nlp = spacy.load("en_core_web_sm")
    if multi:
        logger.info(f"Loading docbins at {path}")
        docs = []
        for child in path.iterdir():
            if child.suffix == '.spacy':
                docbin = DocBin().from_disk(child)
                docs.extend(docbin.get_docs(nlp.vocab))
    else:
        path = path.with_suffix('.spacy')
        logger.info(f"Loading docbin at {path}")
        docbin = DocBin().from_disk(path)
        docs = list(docbin.get_docs(nlp.vocab))
    # Create index, if specified, using each doc's 'identifier' attribute.
    if index_names is not None:
        logger.debug(f"Creating index for docs using names {index_names}")
        if len(index_names) == 1:
            name = index_names[0]
        else:
            name = tuple(index_names)
        index = pd.Index(
            [doc._.identifier for doc in docs], 
            name=name
        )
        # Change index dtypes as specified
        logger.debug(f"Overriding index dtype(s) to {index_dtypes}")
        if isinstance(index, pd.MultiIndex):
            index = index.set_levels(
                [level.astype(index_dtypes[i]) 
                    for i, level in enumerate(index.levels)]
            )
        else: # Just a single index
            index = index.astype(index_dtypes[0])
        return pd.Series(docs, index=index)
    else:
        return pd.Series(docs)
