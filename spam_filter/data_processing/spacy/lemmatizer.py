import copy

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import spacy 
from spacy.attrs import LEMMA


nlp = spacy.load("en_core_web_sm")

class Lemmatizer(BaseEstimator, TransformerMixin):
    """A transformer that returns a pandas Series of dictionaries
    of the form {<lemma>: <count>} for an input pandas Series of
    spacy Doc objects. Only lemmas in the default spacy vocab 
    stringstore (for "en_core_web_sm") are returned in the counts."""

    # Make a copy of the initial string store in order to test if lemmas 
    # are known words or not.
    stringstore = copy.copy(nlp.vocab.strings)

    def __init__(self, del_stop=False, del_punct=True, del_num=False):
        self.del_stop = del_stop
        self.del_punct = del_punct
        self.del_num = del_num

    def fit(self, X, y=None):
         # No fitting of the transformer is necessary
        return self
    
    def transform(self, X, y=None):
        assert isinstance(X, pd.Series)
        lemma_counts = [
            {self.stringstore[lemma_hash]: count 
                for lemma_hash, count in 
                    doc.count_by(LEMMA, exclude=self.exclude_token).items()
            }
            for doc in X
        ]
        return pd.Series(lemma_counts, index=X.index)

    def exclude_token(self, token):
        conditions = [
            token.lemma not in self.stringstore,
            token.like_email,
            token.like_url,
        ]
        if any(conditions):
            return True
        if self.del_stop and token.is_stop:
            return True
        if self.del_punct and token.is_punct:
            return True
        if self.del_num and token.like_num:
            return True
        return False