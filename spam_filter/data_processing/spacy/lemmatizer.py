import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import spacy 
from spacy.attrs import LEMMA


class Lemmatizer(BaseEstimator, TransformerMixin):
    """A transformer that takes spacy Doc objects as input and 
    returns dicts of the form {<lemma>: <count>}. Only lemmas in 
    the default spacy vocab stringstore (for "en_core_web_sm") 
    are returned in the counts."""

    # Store the model as a class attribute in order to check the 
    # stringstore for known lemmas.
    nlp = spacy.load("en_core_web_sm")

    def __init__(self, del_stop=False, del_punct=True, del_num=False):
        self.del_stop = del_stop
        self.del_punct = del_punct
        self.del_num = del_num

    def fit(self, X, y=None):
         # No fitting of the transformer is necessary
        return self
    
    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.to_numpy()
        lemma_counts = [{
            self.nlp.vocab.strings[lemma_hash]: count 
                for lemma_hash, count in 
                    doc.count_by(LEMMA, exclude=self.exclude_token).items()
            } for doc in X.flat
        ]
        return np.array(lemma_counts).reshape(X.shape)

    def exclude_token(self, token):
        conditions = [
            token.lemma not in self.nlp.vocab.strings,
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