from pathlib import Path
import logging

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import spacy
from spacy.tokens import Doc, DocBin


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


MAX_DOCBIN_SIZE = 50_000


# class DocBinError(Exception):
#     pass


Doc.set_extension("identifier", default=None)


nlp = spacy.load("en_core_web_sm")


class DocBaseMixin:
    """A class providing common functionality for the other Doc-based
    classes."""
    def __init__(self, docbin_path=None, index_names=None, index_dtypes=None):
        if (index_names is not None and index_dtypes is not None 
                and len(index_names) != len(index_dtypes)):
            raise DocBinError("The number of provided index names "
                f"({len(index_names)}) and index dtypes ({len(index_dtypes)}) "
                "do not match.")
        self.docbin_path = Path(docbin_path) if docbin_path is not None else None
        self.index_names = index_names
        self.index_dtypes = index_dtypes

    def create_index(self, docs):
        """Create a pandas Index based on an identifier stored in the 
        iterable of docs. If the identifiers are tuples, a MultiIndex is
        created. If `index_names` was set on initialization, those names
        will be used for the index columns. Similarly, if `index_dtypes` 
        was set, those will be used for the index."""
        logger.debug(f"Creating index for docs using names {self.index_names}")
        if len(self.index_names) == 1:
            name = self.index_names[0]
        else:
            name = tuple(self.index_names)
        self.index = pd.Index(
            [doc._.identifier for doc in docs], 
            name=name
        )
        # Change index dtypes is specified
        if self.index_dtypes is not None:
            logger.debug(f"Overriding index dtype(s) to {self.index_dtypes}")
            if isinstance(self.index, pd.MultiIndex):
                self.index = self.index.set_levels(
                    [level.astype(self.index_dtypes[i]) 
                        for i, level in enumerate(self.index.levels)]
                )
            else: # Just a single index
                self.index = self.index.astype(self.index_dtypes[0])
    
    def check_index_args_against_series(self, X):
        """Check that the index levels of the pandas Series X 
        and self.index match, based on the initialization parameters."""
        if not isinstance(X, pd.Series):
            raise DocBinError(f"{X!r} is not a pandas.Series object")
        X_index_levels = len(X.index.levels) if \
                isinstance(X.index, pd.MultiIndex) else 1
        if (self.index_names is not None 
                and X_index_levels != len(self.index_names)):
            raise DocBinError(f"The index of the dataframe ({X_index_levels}) "
                "does not match the number of given index names "
                f"({len(self.index_names)}: {', '.join(self.index_names)})")
        if (self.index_dtypes is not None 
                and X_index_levels != len(self.index_dtypes)):
            raise DocBinError(f"The index of the dataframe ({X_index_levels}) "
                "does not match the number of given index dtypes "
                f"({len(self.index_dtypes)}: {', '.join(self.index_dtypes)})")

    def check_index_against_series(self, X):
        """Ensure self.index is the same as that of the pandas Series X.
        This is a way to guarantee that a loaded docbin matches the 
        correct series, or that before saving a docbin, it will match
        the series in the future."""
        if not X.index.identical(self.index):
            raise DocBinError(f"The index of the series and the index of the "
                "docs do not match. Be sure the index_names and index_dtypes "
                "are set correctly.")


class DocBinLoader(BaseEstimator, TransformerMixin, DocBaseMixin):
    """A transfomer that returns a pandas Series of spacy Doc objects 
    read from a spacy DocBin file(s) at `docbin_path` for the Series X. 
    If X has a named index, the index_names, and index_dtypes should 
    be specified, each as a tuple or list of strings. This is 
    necessary to ensure the returned Series of Docs matches the order
    of the input Series.
    NOTE: `docbin_path` can be a path to a single file or to a folder
    containing multiple '.spacy' docbin files. In the case of the former,
    do not include the extension '.spacy'."""
    def __init__(self, docbin_path, index_names=None, index_dtypes=None):
        super().__init__(docbin_path, index_names, index_dtypes)
        if self.docbin_path.exists() and self.docbin_path.is_dir():
            self.multi = True
        elif self.docbin_path.with_suffix('.spacy').exists():
            self.multi = False
        else:
            raise DocBinError(f"{docbin_path} does not exist")
    
    def fit(self, X, y=None):
         # No fitting of the transformer is necessary
        return self
    
    def transform(self, X, y=None):
        self.check_index_args_against_series(X)
        if self.multi:
            logger.info(f"Loading docbins at {self.docbin_path}")
            docs = []
            for child in self.docbin_path.iterdir():
                if child.suffix == '.spacy':
                    docbin = DocBin().from_disk(child)
                    docs.extend(docbin.get_docs(nlp.vocab))
        else:
            path = self.docbin_path.with_suffix('.spacy')
            logger.info(f"Loading docbin at {path}")
            docbin = DocBin().from_disk(path)
            docs = list(docbin.get_docs(nlp.vocab))
        self.create_index(docs)
        self.check_index_against_series(X)
        return pd.Series(docs, index=self.index).reindex_like(X, copy=False)


class DocCreator(BaseEstimator, TransformerMixin, DocBaseMixin):
    """A transfomer that creates and returns a pandas Series of spacy 
    Doc objects for the pandas Series X. If `docbin_path` is set, a
    spacy DocBin will be saved there, which can be loaded with 
    DocBinLoader on future runs. In that case, if X has a named index, 
    the index_names, and index_dtypes should be specified, each as a 
    tuple or list of strings.
    NOTE: Do not include the extension '.spacy' in `docbin_path`.
    If the docs are too large for one file, a directory will be 
    created at the `docbin_path` and spacy files saved within. If the 
    docs will fit in one file, the '.spacy' extension will be added 
    appropriately."""
    def __init__(self, docbin_path=None, index_names=None, index_dtypes=None):
        if docbin_path is not None and not Path(docbin_path).parent.exists():
            raise DocBinError(f"{Path(docbin_path).parent} does not exist")
        super().__init__(docbin_path, index_names, index_dtypes)

    def fit(self, X, y=None):
        # No fitting of the transformer is necessary
        return self
    
    def set_output(self, *arg, **kwarg):
        pass

    def transform(self, X, y=None):
        # self.check_index_args_against_series(X)
        try:
            mem_size = X.memory_usage()
            msg = (f"Running spacy pipeline on Series '{X.name}' of length "
                f"{len(X)} (size in memory: {mem_size / 1_000_000.0:.3f} MB).")
            if mem_size > 50 * 10**6: # Over 50 MB
                msg += " This may take a while..."
            logger.info(msg)
        except AttributeError:
            pass
        docs = []
        for doc, context in nlp.pipe(zip(X, X.index), as_tuples=True, 
                disable=["parser", "ner"]):
            doc._.identifier = context
            docs.append(doc)
        if self.index_names is not None:
            logger.info("Creating an index for the created docs")
            self.create_index(docs)
            logger.info("Checking that the index matches that of Series "
            f"'{X.name}'")
            self.check_index_against_series(X)
        # Save the docbin to disk for later use
        if self.docbin_path is not None:
            logger.info(f"Creating docbin(s) for Series '{X.name}'")
            docbins = [
                DocBin(store_user_data=True, 
                    docs=docs[i: i + MAX_DOCBIN_SIZE])
                for i in range(0, len(docs), MAX_DOCBIN_SIZE)
            ]
            if len(docbins) == 1:
                new_path = self.docbin_path.with_suffix('.spacy')
                logger.info(f"Saving docbin to to {self.docbin_path}")
                docbins[0].to_disk(new_path)
            else:
                new_folder = self.docbin_path.with_suffix('')
                new_folder.mkdir(exist_ok=True)
                logger.warning("There are too many docs for one docbin "
                    "file. Multiple docbins will be created at "
                    f"{new_folder}")
                for i, docbin in enumerate(docbins):
                    new_path = new_folder / f"{i+1}.spacy"
                    logger.info(f"Saving docbin {i+1} of {len(docbins)} to "
                        f"{new_path}")
                    docbin.to_disk(new_path)
        try:
            return pd.DataFrame({'doc': docs}, index=self.index)
        except AttributeError: # if there's no index attribute
            return pd.DataFrame({'doc': docs})


class DocCreatorB(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
    
    def fit(self, X, y=None):
        # No fitting of the transformer is necessary
        return self

    def transform(self, X, y=None):
        return np.array(list(
            self.nlp.pipe(X.to_numpy().flat, disable=["parser", "ner"])
            ), dtype=object).reshape(X.shape)
